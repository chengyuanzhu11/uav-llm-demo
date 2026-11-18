"""Verified exported decisions and Qwen processor based completion-only batches.

The inference service and its dependencies remain independent. Importing this
module does not import torch or transformers; those are needed only to collate.
"""
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected a dataset-relative artifact path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Artifact is missing or leaves the exported dataset")
    return path


def _manifest(root):
    audit_path = root / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    refs = audit.get("files")
    if not isinstance(refs, list) or not refs:
        raise ValueError("Dataset lacks an export-integrity manifest")
    files = {}
    for ref in refs:
        relative = ref.get("path")
        path = _contained(root, relative)
        if relative in files:
            raise ValueError("Duplicate artifact in integrity manifest")
        if _sha256(path) != ref.get("sha256") or path.stat().st_size != ref.get("bytes"):
            raise ValueError("Exported artifact changed after approval: " + relative)
        files[relative] = ref
    if audit.get("loss_scope") != "completion_only":
        raise ValueError("Dataset audit must specify completion_only")
    if not audit.get("approved_episodes") or audit.get("approved_episodes") != audit.get("episodes"):
        raise ValueError("Export must contain only approved episodes")
    return audit, files


def _image_references(messages):
    """Read actual message order, including repeated frames and tool results."""
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Invalid multimodal message part")
            kind = part.get("type")
            if kind == "image":
                images.append(part.get("image"))
            elif kind == "image_url":
                images.append(part.get("image_url", {}).get("url"))
            elif kind != "text":
                raise ValueError("This exported image-decision dataset does not support " + str(kind))
    return images


def _validate_record(record, root, files):
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError("Every decision needs a stable ID")
    prompt, completion = record.get("prompt"), record.get("completion")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Decision prompt must contain messages")
    if record.get("loss_scope") != "completion_only" or not isinstance(completion, list) or len(completion) != 1:
        raise ValueError("Expected one completion-only target")
    if completion[0].get("role") != "assistant":
        raise ValueError("Only the final assistant decision can be supervised")
    if not completion[0].get("content") and not completion[0].get("tool_calls"):
        raise ValueError("Assistant target is empty")
    if _image_references(completion):
        raise ValueError("Completion must be a decision, not generated image content")
    references = _image_references(prompt)
    if references != record.get("images", []):
        raise ValueError("Image list must exactly match message frame order")
    for relative in references:
        _contained(root, relative)
        if relative not in files:
            raise ValueError("Image is not covered by the export manifest")
    if not isinstance(record.get("tools", []), list):
        raise ValueError("tools must be a list")


class DemonstrationDataset(Sequence):
    """Load one exported split, validating the whole export before sampling."""

    def __init__(self, dataset_dir, split="pilot"):
        self.root = Path(dataset_dir).resolve()
        audit, files = _manifest(self.root)
        jsonl_paths = sorted(self.root.glob("*.jsonl"))
        if not jsonl_paths or any(path.name not in files for path in jsonl_paths):
            raise ValueError("Dataset contains unmanifested decision files")
        all_records, ids = [], set()
        for path in jsonl_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                _validate_record(record, self.root, files)
                if record["id"] in ids:
                    raise ValueError("Duplicate decision ID")
                ids.add(record["id"])
                all_records.append(record)
        if len(all_records) != audit.get("decisions"):
            raise ValueError("Decision count differs from the export audit")
        self.records = [r for r in all_records if r.get("split") == split]
        if not self.records:
            raise ValueError("Requested split is empty: " + str(split))
        self.summary = dict(
            split=split, decisions=len(self.records), export_decisions=len(all_records),
            episodes=len({r.get("metadata", {}).get("task_id") for r in self.records}),
            image_decisions=sum(bool(r.get("images")) for r in self.records),
            image_references=sum(len(r.get("images", [])) for r in self.records),
            stages=dict(Counter(r.get("metadata", {}).get("stage", "unknown") for r in self.records)),
            audit_sha256=_sha256(self.root / "audit.json"),
            dataset_file_sha256={path.name: files[path.name]["sha256"] for path in jsonl_paths},
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def completion_labels(prompt_ids, full_ids, *, eos_token_id, image_token_ids=(), max_length):
    """Validate the true processor boundary and mask every historical turn."""
    if len(full_ids) > max_length:
        raise ValueError("Processed decision exceeds max_length; truncation is forbidden")
    boundary = len(prompt_ids)
    if not boundary or full_ids[:boundary] != prompt_ids:
        raise ValueError("Processed prompt is not an exact prefix of the full conversation")
    target = full_ids[boundary:]
    if not target or eos_token_id is None or eos_token_id not in target:
        raise ValueError("Completion must include the template end-of-turn EOS token")
    image_token_ids = set(image_token_ids)
    if any(token in image_token_ids for token in target):
        raise ValueError("Visual tokens are not valid supervised action targets")
    return [-100] * boundary + target


class QwenCompletionCollator:
    """Use the model's own tools template and image expansion; never truncate.

    Images are loaded in message order and encoded identically for prefix and
    full conversation. Each row has one current target even if the prompt has
    earlier failed assistant decisions. Right padding avoids shifting boundaries.
    """

    def __init__(self, processor, dataset_root, max_length=16384):
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        self.processor = processor
        self.root = Path(dataset_root).resolve()
        _, self.files = _manifest(self.root)
        self.max_length = max_length
        if processor.tokenizer.pad_token_id is None:
            raise ValueError("Processor tokenizer must define its native pad token")
        self.image_token_ids = tuple(
            value for value in (getattr(processor, name, None) for name in
                                ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"))
            if isinstance(value, int)
        )
        if not self.image_token_ids:
            raise ValueError("Expected a Qwen vision-language processor with visual token IDs")

    def _messages_and_images(self, row):
        from PIL import Image

        _validate_record(row, self.root, self.files)
        prompt = deepcopy(row["prompt"])
        completion = deepcopy(row["completion"])
        # OpenAI tool messages may explicitly contain null. Qwen's native
        # template iterates non-string content, so represent absence as "".
        for message in prompt + completion:
            if message.get("content") is None:
                message["content"] = ""
        images = []
        for relative in _image_references(prompt):
            path = _contained(self.root, relative)
            if _sha256(path) != self.files[relative]["sha256"]:
                raise ValueError("Image changed after dataset validation")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return prompt, completion, images

    def encode(self, row):
        import torch

        prompt, completion, images = self._messages_and_images(row)
        tools = row.get("tools") or None
        texts = [self.processor.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=generation,
        ) for messages, generation in ((prompt, True), (prompt + completion, False))]
        if not texts[1].startswith(texts[0]):
            raise ValueError("Native chat template changed the prompt when appending its completion")
        encoded = [self.processor(
            text=[text], images=images or None, return_tensors="pt",
            padding=False, truncation=False, add_special_tokens=False,
        ) for text in texts]
        prefix, full = encoded
        prompt_ids = prefix["input_ids"][0].tolist()
        full_ids = full["input_ids"][0].tolist()
        labels = completion_labels(prompt_ids, full_ids,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            image_token_ids=self.image_token_ids, max_length=self.max_length)
        if not bool(full["attention_mask"].all()):
            raise ValueError("Unbatched processor output unexpectedly contains padding")
        for key in ("pixel_values", "image_grid_thw"):
            if (key in prefix) != (key in full) or (key in full and not torch.equal(prefix[key], full[key])):
                raise ValueError("Prompt/full image processing is inconsistent: " + key)
        grids = full.get("image_grid_thw")
        if len(images) != (len(grids) if grids is not None else 0):
            raise ValueError("Frame count does not match image_grid_thw")
        image_tokens = full_ids.count(self.processor.image_token_id)
        if grids is not None:
            expected = int((grids.prod(dim=-1) // self.processor.image_processor.merge_size ** 2).sum())
            if expected != image_tokens:
                raise ValueError("Expanded image tokens do not match processed image grids")
        elif image_tokens:
            raise ValueError("Image tokens have no corresponding image pixels")
        full = dict(full)
        full["labels"] = torch.tensor([labels], dtype=torch.long)
        diagnostic = dict(
            id=row["id"], stage=row.get("metadata", {}).get("stage"),
            prompt_tokens=len(prompt_ids), total_tokens=len(full_ids),
            supervised_tokens=sum(token != -100 for token in labels),
            image_count=len(images), image_tokens=image_tokens,
            image_grid_thw=grids.tolist() if grids is not None else [],
            image_sizes=[list(image.size) for image in images],
            prompt_fully_masked=all(token == -100 for token in labels[:len(prompt_ids)]),
            completion_only=True, eos_supervised=self.processor.tokenizer.eos_token_id in labels[len(prompt_ids):],
            prior_assistant_turns=sum(message.get("role") == "assistant" for message in prompt),
            truncation=False,
        )
        return full, diagnostic

    def diagnostics(self, row):
        return self.encode(row)[1]

    def __call__(self, rows):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        if not rows:
            raise ValueError("Cannot collate an empty batch")
        features = [self.encode(row)[0] for row in rows]
        expected_keys = {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
        extra = set().union(*(set(feature) for feature in features)) - expected_keys
        if extra:
            raise ValueError("Unexpected processor fields need explicit collation: " + ", ".join(sorted(extra)))
        result = {}
        for key, padding in (("input_ids", self.processor.tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = pad_sequence([feature[key][0] for feature in features], batch_first=True, padding_value=padding)
        # Qwen stores concatenated patches/grids, not [batch, image, ...].
        # Text-only rows contribute no images; concatenation keeps sample order.
        for key in ("pixel_values", "image_grid_thw"):
            values = [feature[key] for feature in features if key in feature]
            if values:
                result[key] = torch.cat(values, dim=0)
        if bool((result["labels"][result["attention_mask"] == 0] != -100).any()):
            raise ValueError("Padding must never be supervised")
        return result
