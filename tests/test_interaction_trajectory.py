import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from uav_llm_demo.interaction_trajectory import InteractionRecorder, restore_payload, validate_trajectory


def frame():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00\x80\x00" * 2)) + chunk(b"IEND", b""))
    depth = struct.pack("<2f", 3.0, 4.0)
    return dict(frame_id="frame-a", width=2, height=1, revision=1, execution_id="step-a",
                simulation_time=2.0, position_grid=[1, 2, 3], yaw_deg=0, pitch_deg=-30,
                fx=2, fy=2, cx=1, cy=.5, source="gazebo_rgbd", depth_convention="optical_axis_metres",
                rgb="data:image/png;base64," + base64.b64encode(png).decode(), depth_m=[3.0, 4.0],
                rgb_sha256=hashlib.sha256(png).hexdigest(), depth_sha256=hashlib.sha256(depth).hexdigest())


def episode(path, *, terminal=True):
    recorder = InteractionRecorder(path, metadata={"model": "test", "api_key": "dont-save"}, secrets=("test-secret-value",))
    image = frame()
    recorder.record("decision_input", decision_id="run-1:1:step-a:1", context={"instruction": "find a tree"}, frames=[image])
    request = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image["rgb"]}},
                                                         {"type": "text", "text": "test-secret-value"}]}],
               "headers": {"Authorization": "Bearer dont-save"}, "max_tokens": 512}
    recorder.record("model_request", decision_id="run-1:1:step-a:1", request=request)
    recorder.record("model_response", decision_id="run-1:1:step-a:1", response={"tool": "look"}, http_status=200)
    if terminal:
        recorder.record("decision_result", decision_id="run-1:1:step-a:1", status="completed",
                        action={"tool": "ground", "arguments": {"frame_id": "frame-a", "pixel": [.5, .5], "confidence": .9}},
                        feedback={"ok": True})
    result = recorder.close(outcome="failed", oracle={"target_id": "private-answer", "passed": False})
    return recorder, result, request


class InteractionTrajectoryTests(unittest.TestCase):
    def test_complete_episode_roundtrips_images_and_keeps_oracle_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episode"
            recorder, result, request = episode(path)
            self.assertTrue(result["complete"], result["errors"])
            self.assertEqual(result["outcome"], "failed")
            self.assertTrue(validate_trajectory(path)["complete"])
            events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
            restored = restore_payload(path, events[1]["payload"])["request"]
            self.assertEqual(restored["messages"][0]["content"][0], request["messages"][0]["content"][0])
            self.assertEqual(restored["messages"][0]["content"][1]["text"], "[redacted]")
            self.assertEqual(restored["max_tokens"], 512)
            self.assertNotIn("headers", restored)
            self.assertNotIn("private-answer", (path / "events.jsonl").read_text())
            self.assertNotIn("dont-save", (path / "manifest.json").read_text())
            self.assertEqual(len(list((path / "artifacts").glob("*.png"))), 1)
            self.assertEqual(recorder.close()["decisions"], 1)
            with self.assertRaises(FileExistsError):
                InteractionRecorder(path)

    def test_text_decisions_require_explicit_stage_and_cannot_hide_images(self):
        with tempfile.TemporaryDirectory() as temp:
            for variant in ("valid", "no-stage", "no-context", "missing-visual-frame", "text-image"):
                with self.subTest(variant=variant):
                    recorder = InteractionRecorder(Path(temp) / variant)
                    data = dict(decision_id="task:1", modality="text", stage="requirements", context={}, frames=[])
                    if variant == "no-stage":
                        data.pop("stage")
                    elif variant == "no-context":
                        data.pop("context")
                    elif variant == "missing-visual-frame":
                        data.pop("modality")
                    recorder.record("decision_input", **data)
                    messages = [{"role": "user", "content": "Inspect then return"}]
                    if variant == "text-image":
                        messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame()["rgb"]}}]})
                    recorder.record("model_request", decision_id="task:1", request={"messages": messages})
                    recorder.record("model_response", decision_id="task:1", response={"choices": []})
                    recorder.record("decision_result", decision_id="task:1", status="applied")
                    result = recorder.close(outcome={"status": "completed"})
                    self.assertEqual(result["complete"], variant == "valid", result["errors"])

    def test_cancelled_inflight_reply_preserves_complete_evidence(self):
        # The HTTP observer can emit both records after reset, or only the
        # response can arrive late when the request was already recorded.
        cases = [("cancelled", True, True), ("discarded", False, True),
                 ("cancelled", True, False), ("completed", False, True)]
        with tempfile.TemporaryDirectory() as temp:
            for index, (status, late_request, response) in enumerate(cases):
                with self.subTest(status=status, late_request=late_request, response=response):
                    recorder = InteractionRecorder(Path(temp) / str(index))
                    recorder.record("decision_input", decision_id="d", frames=[frame()])
                    if not late_request:
                        recorder.record("model_request", decision_id="d", request={"messages": []})
                    recorder.record("decision_result", decision_id="d", status=status)
                    if late_request:
                        recorder.record("model_request", decision_id="d", request={"messages": []})
                    if response:
                        recorder.record("model_response", decision_id="d", response=None,
                                        outcome="error", error="connection closed")
                    result = recorder.close(outcome="cancelled")
                    expected = status in {"cancelled", "discarded"} and response
                    self.assertEqual(result["complete"], expected, result["errors"])
                    if not response:
                        self.assertTrue(any("unpaired" in error for error in result["errors"]))
                    elif not expected:
                        self.assertTrue(any("precedes its model response" in error for error in result["errors"]))

    def test_motion_requires_its_original_decision_input(self):
        with tempfile.TemporaryDirectory() as temp:
            for terminal_decision in ("d", "other", "missing"):
                with self.subTest(terminal_decision=terminal_decision):
                    recorder = InteractionRecorder(Path(temp) / terminal_decision)
                    for decision_id in ("d", "other"):
                        recorder.record("decision_input", decision_id=decision_id, frames=[frame()])
                        recorder.record("decision_result", decision_id=decision_id, status="cancelled")
                    recorder.record("motion_started", motion_id="m", decision_id="d")
                    recorder.record("motion_aborted", motion_id="m", decision_id=terminal_decision)
                    result = recorder.close(outcome="cancelled")
                    self.assertEqual(result["complete"], terminal_decision == "d", result["errors"])
                    if terminal_decision == "other":
                        self.assertTrue(any("different originating decision" in error for error in result["errors"]))
                    elif terminal_decision == "missing":
                        self.assertTrue(any("earlier decision input" in error for error in result["errors"]))

    def test_missing_terminal_or_damaged_evidence_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing-result"
            _, result, _ = episode(path, terminal=False)
            self.assertFalse(result["complete"])
            self.assertTrue(any("terminal result" in error for error in result["errors"]))
            self.assertFalse(validate_trajectory(path)["complete"])
            path = Path(temp) / "unfinished-motion"
            recorder = InteractionRecorder(path)
            image = frame()
            recorder.record("decision_input", decision_id="d", frames=[image])
            recorder.record("decision_result", decision_id="d", status="cancelled")
            recorder.record("motion_started", decision_id="d", motion_id="m")
            result = recorder.close(outcome="failed")
            self.assertFalse(result["complete"])
            self.assertTrue(any("completion/abort" in error for error in result["errors"]))
            path = Path(temp) / "damaged-image"
            episode(path)
            image_path = next((path / "artifacts").glob("*.png"))
            image_path.write_bytes(b"corrupt")
            result = validate_trajectory(path)
            self.assertFalse(result["complete"])
            self.assertTrue(any("Hash mismatch" in error for error in result["errors"]))
            image_path.unlink()
            self.assertFalse(validate_trajectory(path)["complete"])


if __name__ == "__main__":
    unittest.main()
