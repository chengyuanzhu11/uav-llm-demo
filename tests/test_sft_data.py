"""Lightweight integrity/masking guards; processor integration runs on training host."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from training.data import DemonstrationDataset, QwenCompletionCollator, completion_labels


def _export(root):
    from PIL import Image

    (root / "images").mkdir()
    for name, color in (("first", "red"), ("second", "blue")):
        Image.new("RGB", (32, 32), color).save(root / "images" / (name + ".png"))
    text = dict(id="text:1", split="pilot", loss_scope="completion_only", images=[], tools=[],
        prompt=[dict(role="user", content="Perform the task"),
                dict(role="assistant", content="bad earlier action"),
                dict(role="user", content="Correct this using the diagnostic")],
        completion=[dict(role="assistant", content="correct action")],
        metadata=dict(stage="plan", task_id="text"))
    visual = dict(id="visual:1", split="pilot", loss_scope="completion_only",
        images=["images/first.png", "images/second.png"],
        tools=[dict(type="function", function=dict(name="observe", parameters=dict(type="object")))],
        prompt=[dict(role="user", content=[dict(type="text", text="Inspect the target"),
            dict(type="image", image="images/first.png")]),
            dict(role="assistant", content=None, tool_calls=[dict(type="function",
                function=dict(name="observe", arguments="{}"))]),
            dict(role="tool", content=[dict(type="image", image="images/second.png")])],
        completion=[dict(role="assistant", content=None, tool_calls=[dict(type="function",
            function=dict(name="observe", arguments="{}"))])],
        metadata=dict(stage="visual", task_id="visual"))
    (root / "pilot.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (text, visual)))
    _refresh_manifest(root)
    return text, visual


def _refresh_manifest(root):
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "audit.json")
    refs = [dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 bytes=path.stat().st_size) for path in paths]
    (root / "audit.json").write_text(json.dumps(dict(files=refs, decisions=2,
        episodes=2, approved_episodes=2, loss_scope="completion_only")))


class SFTDataTests(unittest.TestCase):
    def test_export_loads_without_changing_recovery_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text, visual = _export(root)
            dataset = DemonstrationDataset(root)
            self.assertEqual(dataset.records, [text, visual])
            self.assertEqual(dataset.summary["image_references"], 2)
            self.assertEqual(dataset.summary["image_decisions"], 1)
            self.assertEqual(dataset[0]["prompt"][1]["content"], "bad earlier action")

    def test_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _export(root)
            (root / "images/first.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after approval"):
                DemonstrationDataset(root)

    def test_image_order_and_directory_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text, visual = _export(root)
            visual["images"].reverse()
            (root / "pilot.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (text, visual)))
            _refresh_manifest(root)
            with self.assertRaisesRegex(ValueError, "frame order"):
                DemonstrationDataset(root)
            audit = json.loads((root / "audit.json").read_text())
            audit["files"][0]["path"] = "../outside.png"
            (root / "audit.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "leaves the exported dataset"):
                DemonstrationDataset(root)

    def test_masks_every_prior_turn_including_previous_eos_and_images(self):
        # Prior assistant responses include EOS too; only the current response
        # and its EOS belong in the labels, regardless of conversational role.
        prefix = [1, 10, 999, 901, 901, 901, 11, 999, 12]
        target = [20, 21, 999]
        labels = completion_labels(prefix, prefix + target, eos_token_id=999,
                                   image_token_ids=[901], max_length=30)
        self.assertEqual(labels, [-100] * len(prefix) + target)

    def test_boundary_length_and_eos_fail_closed(self):
        cases = [([1, 2], [1, 3, 999], "exact prefix", 20),
                 ([1, 2], [1, 2, 3, 999], "truncation is forbidden", 3),
                 ([1], [1, 2], "EOS", 20),
                 ([1], [1, 901, 999], "Visual tokens", 20)]
        for prefix, full, error, limit in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                completion_labels(prefix, full, eos_token_id=999, image_token_ids=[901], max_length=limit)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch belongs to isolated training environment")
    def test_mixed_batch_masks_padding_and_preserves_image_patch_order(self):
        import torch
        from types import SimpleNamespace

        class ProcessorDouble:
            # This double exercises only collation/guard logic. It is not a
            # replacement for the real-model processor preflight in train_sft.
            image_token_id = 901
            video_token_id = 902
            vision_start_token_id = 903
            vision_end_token_id = 904
            tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=999)
            image_processor = SimpleNamespace(merge_size=1)

            def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
                assert not tokenize
                assert all(message.get("content") is not None for message in messages)
                self.last_tools = tools
                return "prompt" if add_generation_prompt else "prompttarget"

            def __call__(self, *, text, images, return_tensors, padding, truncation, add_special_tokens):
                assert not padding and not truncation and not add_special_tokens
                count = len(images or [])
                ids = [1, 2, 999] + [901] * (count * 4)
                if text == ["prompttarget"]:
                    ids += [25, 26, 999]
                result = dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))
                if images:
                    result.update(image_grid_thw=torch.tensor([[1, 2, 2]] * count),
                        pixel_values=torch.tensor([[list(image.getpixel((0, 0)))] * 4 for image in images]).reshape(-1, 3))
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _export(root)
            dataset = DemonstrationDataset(root)
            processor = ProcessorDouble()
            collator = QwenCompletionCollator(processor, root, max_length=30)
            output = collator([dataset[0], dataset[1]])
            self.assertEqual(output["input_ids"].shape, (2, 14))
            self.assertTrue(bool((output["labels"][0, 6:] == -100).all()))
            self.assertTrue(bool((output["attention_mask"][0, 6:] == 0).all()))
            self.assertEqual(output["labels"][0, :6].tolist(), [-100, -100, -100, 25, 26, 999])
            self.assertEqual(output["labels"][1, :11].tolist(), [-100] * 11)
            self.assertEqual(output["pixel_values"][0].tolist(), [255, 0, 0])
            self.assertEqual(output["pixel_values"][4].tolist(), [0, 0, 255])
            self.assertEqual(processor.last_tools, dataset[1]["tools"])
            self.assertIsNone(dataset[1]["completion"][0]["content"])
            self.assertEqual(collator.diagnostics(dataset[1])["image_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
