"""Asynchronous, opt-in evidence for task and visual decisions.

Completeness means reproducible records, not task success or approved SFT
targets. Finish inference before closing the recorder.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from queue import Full, Queue
import struct
from threading import Lock, Thread
import time

from .onboard_observation import frame_metadata

_CREDENTIALS = {
    "headers", "authorization", "proxyauthorization", "apikey", "key", "token",
    "accesstoken", "refreshtoken", "password", "secret", "secrets", "clientsecret",
    "privatekey", "cookie", "cookies",
}
_STATUSES = {"applied", "completed", "rejected", "discarded", "cancelled", "error"}
_NORMAL = {"applied", "completed", "rejected"}
_MIME_SUFFIX = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def _sensitive_field(key):
    name = str(key).lower().replace("_", "").replace("-", "")
    return name in _CREDENTIALS or name.endswith(("apikey", "accesstoken", "refreshtoken",
                                                  "clientsecret", "privatekey", "password", "headers"))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Artifact path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path leaves the episode directory")
    return path


class InteractionRecorder:
    """Create a new episode directory, never overwrite a previous collection."""

    def __init__(self, directory, metadata=None, secrets=()):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self._secrets = tuple(s for s in secrets if isinstance(s, str) and s)
        self.metadata = self._scrub(deepcopy(metadata or {}))
        self.created_at, self._started = _now(), time.monotonic()
        self._queue, self._lock = Queue(maxsize=64), Lock()
        self._sequence, self._closed, self._artifacts = 0, False, {}
        self.manifest = None
        self.dropped = self.errors = self.records = 0
        self._error_messages = []
        self._stream = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self._thread = Thread(target=self._write, name="uav-interaction-recorder", daemon=True)
        self._thread.start()

    def _scrub(self, value):
        if isinstance(value, str):
            if value.startswith("data:image/"):
                return value
            for secret in self._secrets:
                value = value.replace(secret, "[redacted]")
            return value
        if isinstance(value, dict):
            return {str(k): self._scrub(v) for k, v in value.items()
                    if not _sensitive_field(k)}
        if isinstance(value, (list, tuple)):
            return [self._scrub(v) for v in value]
        return value

    def record(self, kind, **payload):
        if not isinstance(kind, str) or not kind:
            raise ValueError("Event kind must be a nonempty string")
        # Callers provide immutable snapshots; only the writer traverses RGB-D.
        with self._lock:
            if self._closed:
                raise RuntimeError("Interaction recorder is closed")
            self._sequence += 1
            event = dict(sequence=self._sequence, timestamp=_now(),
                         wall_seconds=time.monotonic() - self._started, kind=kind, payload=payload)
            try:
                self._queue.put_nowait(event)
            except Full:
                self.dropped += 1
                return False
        return True

    def mark_error(self, code="observer_failed"):
        """Record instrumentation failure without interrupting flight execution."""
        with self._lock:
            self.errors += 1
            if len(self._error_messages) < 20:
                self._error_messages.append(self._scrub(str(code))[:100])

    def _artifact(self, data, suffix, mime_type, **extra):
        digest = _sha(data)
        relative = "artifacts/" + digest + suffix
        if relative not in self._artifacts:
            path = self.directory / relative
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            self._artifacts[relative] = dict(path=relative, sha256=digest, bytes=len(data), mime_type=mime_type)
        return dict(self._artifacts[relative], **extra)

    def _image(self, url):
        if not isinstance(url, str) or ";base64," not in url:
            raise ValueError("RGB must be a base64 image data URL")
        header, encoded = url.split(",", 1)
        mime_type = header.removeprefix("data:").removesuffix(";base64")
        if mime_type not in _MIME_SUFFIX:
            raise ValueError("Unsupported image MIME type")
        data = base64.b64decode(encoded, validate=True)
        if not data:
            raise ValueError("Image is empty")
        return self._artifact(data, _MIME_SUFFIX[mime_type], mime_type)

    def _frame(self, frame):
        width, height = frame.get("width"), frame.get("height")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("Invalid frame dimensions")
        depth = frame["depth_m"]
        if len(depth) != width * height or not all(math.isfinite(v) for v in depth):
            raise ValueError("Depth must contain one finite value per pixel")
        rgb = self._image(frame["rgb"])
        raw = struct.pack("<" + str(len(depth)) + "f", *depth)
        if frame.get("rgb_sha256") != rgb["sha256"] or frame.get("depth_sha256") != _sha(raw):
            raise ValueError("Frame RGB/depth hashes do not match")
        stored = self._artifact(gzip.compress(raw, mtime=0), ".f32.gz", "application/gzip",
                                dtype="float32_le", shape=[height, width], content_sha256=_sha(raw))
        return dict(frame_metadata(frame), rgb=rgb, depth=stored)

    def _externalize(self, value):
        if isinstance(value, str) and value.startswith("data:image/"):
            return {"artifact": self._image(value)}
        if isinstance(value, dict):
            if {"frame_id", "rgb", "depth_m"}.issubset(value):
                return self._frame(value)
            return {k: self._externalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._externalize(v) for v in value]
        return value

    def _write(self):
        while True:
            event = self._queue.get()
            try:
                if event is None:
                    return
                event["payload"] = self._externalize(self._scrub(event["payload"]))
                self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                self.records += 1
            except Exception as exc:
                self.errors += 1
                if len(self._error_messages) < 20:
                    self._error_messages.append(self._scrub(type(exc).__name__ + ": " + str(exc))[:500])
            finally:
                self._queue.task_done()

    def close(self, outcome=None, oracle=None):
        with self._lock:
            if self._closed:
                return validate_trajectory(self.directory)
            self._closed = True
        self._queue.put(None)
        self._thread.join()
        self._stream.flush()
        self._stream.close()
        manifest = dict(version=1, created_at=self.created_at, closed_at=_now(), metadata=self.metadata,
                        outcome=self._scrub(deepcopy(outcome)), records=self.records, dropped=self.dropped,
                        errors=self.errors, error_messages=self._error_messages,
                        events=dict(path="events.jsonl", sha256=_sha((self.directory / "events.jsonl").read_bytes())),
                        artifacts=list(self._artifacts.values()), complete=True)
        if oracle is not None:
            data = (json.dumps(self._scrub(deepcopy(oracle)), ensure_ascii=False, allow_nan=False) + "\n").encode()
            (self.directory / "oracle.json").write_bytes(data)
            manifest["oracle"] = dict(path="oracle.json", sha256=_sha(data))
        result = _validate(self.directory, manifest)
        manifest.update(complete=result["complete"], validation_errors=result["errors"])
        self.manifest = manifest
        with (self.directory / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        return result


def _validate(root, manifest):
    if not isinstance(manifest, dict):
        return dict(complete=False, errors=["Manifest must be an object"], decisions=0, records=0)
    errors, decisions, motions, checked = [], {}, {}, {}
    if manifest.get("version") != 1:
        errors.append("Unsupported or missing manifest version")
    if manifest.get("dropped") or manifest.get("errors"):
        errors.append("Recorder dropped events or encountered write errors")

    def read_reference(ref):
        if not isinstance(ref, dict) or not isinstance(ref.get("sha256"), str):
            raise ValueError("Invalid artifact reference")
        relative = ref.get("path")
        path = _safe_path(root, relative)
        data = checked.get(relative)
        if data is None:
            data = path.read_bytes()
            checked[relative] = data
        if _sha(data) != ref["sha256"]:
            raise ValueError("Hash mismatch: " + relative)
        if "bytes" in ref and len(data) != ref["bytes"]:
            raise ValueError("Size mismatch: " + relative)
        return data

    def walk(value):
        if isinstance(value, dict):
            if "artifact" in value:
                read_reference(value["artifact"])
            if {"frame_id", "rgb", "depth", "width", "height"}.issubset(value):
                rgb = read_reference(value["rgb"])
                ref = value["depth"]
                depth = gzip.decompress(read_reference(ref))
                shape = [value["height"], value["width"]]
                if ref.get("dtype") != "float32_le" or ref.get("shape") != shape or len(depth) != 4 * shape[0] * shape[1]:
                    raise ValueError("Depth shape/dtype mismatch for " + str(value["frame_id"]))
                if _sha(depth) != ref.get("content_sha256") or _sha(depth) != value.get("depth_sha256"):
                    raise ValueError("Depth content hash mismatch")
                if _sha(rgb) != value.get("rgb_sha256"):
                    raise ValueError("RGB content hash mismatch")
                if value["rgb"].get("mime_type") == "image/png":
                    if rgb[:8] != b"\x89PNG\r\n\x1a\n" or len(rgb) < 24 or list(struct.unpack(">II", rgb[16:24])) != shape[::-1]:
                        raise ValueError("PNG dimensions do not match frame metadata")
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith("data:image/"):
            raise ValueError("Inline image was not externalized")

    try:
        for ref in manifest.get("artifacts", []):
            read_reference(ref)
        if manifest.get("oracle"):
            read_reference(manifest["oracle"])
        data = read_reference(manifest["events"])
        if data and not data.endswith(b"\n"):
            errors.append("Events file has an incomplete final line")
        events = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(events) != manifest.get("records"):
            errors.append("Event count does not match manifest")
        for number, event in enumerate(events, 1):
            if event.get("sequence") != number:
                errors.append("Event sequence is missing, duplicated, or reordered")
            payload = event.get("payload", {})
            walk(payload)
            kind, decision_id = event.get("kind"), payload.get("decision_id")
            if kind in {"motion_started", "motion_completed", "motion_aborted"}:
                motion_id = payload.get("motion_id", decision_id)
                if not isinstance(motion_id, str) or not motion_id:
                    errors.append("Motion event lacks motion_id or decision_id")
                else:
                    motion = motions.setdefault(motion_id, {"kinds": [], "decision_id": decision_id})
                    motion["kinds"].append(kind)
                    if not isinstance(decision_id, str) or not decision_id:
                        errors.append("Motion event lacks its originating decision_id: " + motion_id)
                    elif decisions.get(decision_id, {}).get("kinds", []).count("decision_input") != 1:
                        errors.append("Motion event is not bound to an earlier decision input: " + motion_id)
                    if motion["decision_id"] != decision_id:
                        errors.append("Motion terminal has a different originating decision: " + motion_id)
            if kind not in {"decision_input", "model_request", "model_response", "decision_result"}:
                continue
            if not isinstance(decision_id, str) or not decision_id:
                errors.append("Decision event lacks decision_id")
                continue
            record = decisions.setdefault(decision_id, {"kinds": [], "frames": set(), "images": set(), "request_images": set(), "result": None, "modality": None})
            record["kinds"].append(kind)
            if kind == "decision_input":
                frames = payload.get("frames")
                modality = payload.get("modality", "vision")
                record["modality"] = modality
                if modality == "text":
                    if payload.get("stage") not in {"requirements", "audit", "plan", "review"}:
                        errors.append("Text decision lacks a supported task stage: " + decision_id)
                    if frames != [] or not isinstance(payload.get("context"), dict):
                        errors.append("Text decision requires context and an empty frames list: " + decision_id)
                elif modality != "vision":
                    errors.append("Unknown decision modality: " + decision_id)
                elif not isinstance(frames, list) or not frames:
                    errors.append("Decision input lacks its image frames: " + decision_id)
                else:
                    for frame in frames:
                        if not isinstance(frame, dict) or not {"frame_id", "rgb", "depth"}.issubset(frame):
                            errors.append("Decision frame lacks saved RGB/depth: " + decision_id)
                        else:
                            record["frames"].add(frame["frame_id"])
                            record["images"].add(frame["rgb"]["sha256"])
            elif kind == "model_request":
                def request_images(value):
                    if isinstance(value, dict):
                        if "image_url" in value:
                            url = value["image_url"].get("url") if isinstance(value["image_url"], dict) else value["image_url"]
                            if not isinstance(url, dict) or "artifact" not in url:
                                errors.append("Model image is not a saved data-URL artifact: " + decision_id)
                            else:
                                record["request_images"].add(url["artifact"]["sha256"])
                        for item in value.values():
                            request_images(item)
                    elif isinstance(value, list):
                        for item in value:
                            request_images(item)
                request_images(payload)
            elif kind == "decision_result":
                record["result"] = payload
        for motion_id, motion in motions.items():
            kinds = motion["kinds"]
            terminals = kinds.count("motion_completed") + kinds.count("motion_aborted")
            if kinds.count("motion_started") != 1 or terminals != 1 or kinds[0] != "motion_started":
                errors.append("Motion requires one start and one completion/abort: " + motion_id)
        if not decisions:
            errors.append("Episode contains no decisions")
        for decision_id, record in decisions.items():
            kinds, result = record["kinds"], record["result"]
            if kinds.count("decision_input") != 1 or kinds.count("decision_result") != 1:
                errors.append("Decision requires one input and one terminal result: " + decision_id)
            if kinds[0] != "decision_input":
                errors.append("Decision events must begin with their input: " + decision_id)
            if result is None:
                continue
            status = result.get("status")
            if status not in _STATUSES:
                errors.append("Unknown decision result status: " + decision_id)
            # Reset/discard may finish a decision while its HTTP worker is still
            # running. Its observer can record the in-flight request and reply
            # afterwards; cancellation must not silently lose those records.
            if status not in {"cancelled", "discarded"} and kinds[-1] != "decision_result":
                errors.append("Decision result precedes its model response: " + decision_id)
            balance, response_before_request = 0, False
            for kind in kinds:
                balance += (kind == "model_request") - (kind == "model_response")
                response_before_request |= balance < 0
            if response_before_request:
                errors.append("Model response precedes its request: " + decision_id)
            if balance:
                errors.append("Decision has unpaired model requests and responses: " + decision_id)
            if not record["request_images"].issubset(record["images"]):
                errors.append("Model request image is absent from decision input frames: " + decision_id)
            if status in _NORMAL and not kinds.count("model_request"):
                errors.append("Applied/rejected decision lacks model requests and responses: " + decision_id)
            action = result.get("action")
            if isinstance(action, dict) and action.get("tool") in {"probe", "ground"}:
                if action.get("arguments", {}).get("frame_id") not in record["frames"] and result.get("status") in {"applied", "completed"}:
                    errors.append("Applied pixel action is not bound to an input frame: " + decision_id)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, struct.error) as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
    if manifest.get("complete") is False and not errors:
        errors.append("Manifest marks this episode incomplete")
    return dict(complete=not errors, errors=errors, decisions=len(decisions),
                records=manifest.get("records", 0), motions=len(motions), outcome=manifest.get("outcome"))


def validate_trajectory(path):
    """Validate saved evidence; complete never implies task or SFT success."""
    root = Path(path).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return _validate(root, manifest)
    except (OSError, ValueError, TypeError) as exc:
        return dict(complete=False, errors=[type(exc).__name__ + ": " + str(exc)], decisions=0, records=0)


def restore_payload(path, payload):
    """Restore model-message image data URLs from artifact references."""
    root = Path(path).resolve()
    def restore(value):
        if isinstance(value, dict):
            if set(value) == {"artifact"}:
                ref = value["artifact"]
                data = _safe_path(root, ref["path"]).read_bytes()
                if _sha(data) != ref["sha256"]:
                    raise ValueError("Image artifact hash mismatch")
                return "data:" + ref["mime_type"] + ";base64," + base64.b64encode(data).decode("ascii")
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value
    return restore(payload)
