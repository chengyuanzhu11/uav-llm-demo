"""Evidence-reviewed decision samples; never an automatic training trigger.

An intact journal is only a source. Explicit response selection, independent
task standards and re-evaluated execution evidence are required to approve
labels. Earlier assistant messages remain in the prompt, outside the loss.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import shutil
from pathlib import Path
import re

from .interaction_trajectory import validate_trajectory
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from .run_log import source_fingerprint

VERSION = 1
_SPLITS = {"pilot", "train", "validation", "test"}
_PRIVATE_KEYS = {"oracle", "oracle_target", "ground_truth", "reference_plan",
                 "expected_response", "future_observations", "task_standard", "execution_result"}
_SECRET_KEYS = {"authorization", "apikey", "password", "secret", "privatekey", "accesstoken", "refreshtoken"}
_SECRET = re.compile(r"(?i)\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{16,}|\bgh[pousr]_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY-----")


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Evidence paths must be relative to their episode")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path leaves its episode")
    return path


def file_reference(root, relative):
    path = _path(Path(root), relative)
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _read_ref(root, ref):
    if not isinstance(ref, dict) or not isinstance(ref.get("sha256"), str):
        raise ValueError("Missing evidence hash")
    data = _path(root, ref.get("path")).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("Evidence hash mismatch: " + str(ref.get("path")))
    return json.loads(data)


def _write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _safe_training_value(value):
    """Reject credential/privileged fields, including JSON embedded in messages."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub("[^a-z]", "", str(key).lower())
            if normalized in _SECRET_KEYS:
                raise ValueError("Credential field in training material")
            if str(key).lower() in _PRIVATE_KEYS:
                raise ValueError("Privileged oracle field in training material")
            _safe_training_value(item)
    elif isinstance(value, list):
        for item in value:
            _safe_training_value(item)
    elif isinstance(value, str):
        if _SECRET.search(value) or "[redacted]" in value:
            raise ValueError("Secret or redacted content cannot be a training sample")
        for secret_key in _SECRET_KEYS:
            if re.search("[" + chr(34) + chr(39) + "]" + r"[_-]*".join(secret_key) + "[" + chr(34) + chr(39) + "]" + r"\s*:", value, re.I):
                raise ValueError("Credential field embedded in model messages")
        for key in _PRIVATE_KEYS:
            if re.search(r'["\x27]' + re.escape(key) + r'["\x27]\s*:', value, re.I):
                raise ValueError("Privileged oracle field embedded in model messages")


def _standard(value):
    if not isinstance(value, dict):
        raise ValueError("Task standard must be an object")
    for key in ("id", "family", "scenario", "instruction", "annotation_source"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError("Task standard requires " + key)
    if value["annotation_source"] not in {"agent_assisted", "human_reviewed"}:
        raise ValueError("Unknown annotation provenance")
    if not isinstance(value.get("intent"), list) or not value["intent"] or not all(
            (isinstance(item, str) and item.strip()) or (isinstance(item, dict) and set(item) == {"id", "text"}
             and all(isinstance(item[k], str) and item[k].strip() for k in ("id", "text"))) for item in value["intent"]):
        raise ValueError("Task standard requires separately authored intent clauses")
    if value.get("evaluation_kind", "semantic") not in {"semantic", "visual_rgbd"}:
        raise ValueError("Unsupported task evaluation kind")
    from .semantic_benchmark import validate_requirement
    validate_requirement(value.get("oracle"))
    checks = value.get("trajectory_checks", [])
    if not isinstance(checks, list):
        raise ValueError("Trajectory checks must be a list")
    for rule in checks:
        if not isinstance(rule, dict) or set(rule) != {"kind", "target", "max_distance_m"} or rule["kind"] != "horizontal_radius":
            raise ValueError("Unsupported trajectory check")
        if not isinstance(rule["target"], str) or not rule["target"]:
            raise ValueError("Trajectory check requires a named reference target")
        limit = rule["max_distance_m"]
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("Trajectory radius must be finite and positive")
    return value


def _execution_checks(standard, execution):
    """Recompute objective satisfaction; a saved success flag is insufficient."""
    if not isinstance(execution, dict):
        raise ValueError("Execution evidence must be an object")
    from .semantic_benchmark import evaluate_oracle
    for key in ("evidence", "telemetry"):
        if not isinstance(execution.get(key), list) or not execution[key]:
            raise ValueError("Execution requires measured " + key)
    if execution.get("status") != "completed" or execution.get("safety_violations") != 0:
        raise ValueError("Execution did not complete safely")
    if execution.get("service_success") is not True or execution.get("safe_execution") is not True:
        raise ValueError("Service did not accept the completed safe execution")
    if execution.get("static_path_blocked") is not False:
        raise ValueError("Execution lacks an unblocked-path check")
    if execution.get("source_unchanged") is not True or not execution.get("source_sha256"):
        raise ValueError("Execution source changed or was not fingerprinted")
    if execution.get("instruction", standard["instruction"]) != standard["instruction"]:
        raise ValueError("Execution instruction differs from the task standard")
    result = evaluate_oracle(standard["oracle"], evidence=execution["evidence"],
        telemetry=execution["telemetry"], world=execution["world"], mission_id=execution["mission_id"],
        condition_evaluations=execution.get("condition_evaluations"),
        condition_evaluations_dropped=execution.get("condition_evaluations_dropped", 0))
    if result.get("satisfied") is not True:
        raise ValueError("Independent task oracle rejects measured execution")
    trajectory = []
    for rule in standard.get("trajectory_checks", []):
        reference = execution["world"][rule["target"]]["center"]
        values = []
        for row in execution["telemetry"]:
            point = row.get("position")
            if not isinstance(point, list) or len(point) != 3 or any(type(v) not in (float,int) or not math.isfinite(v) for v in point):
                raise ValueError("Trajectory check requires finite 3D telemetry positions")
            values.append(math.hypot(3*(point[0]-reference[0]), 3*(point[1]-reference[1])))
        measured = max(values)
        if measured > rule["max_distance_m"]:
            raise ValueError("Measured trajectory violates the specified horizontal radius")
        trajectory.append(dict(**rule, measured_max_distance_m=measured))
    def requires_final_landing(requirement, measured):
        if requirement["type"] == "goal":
            return requirement["state"] == "landed"
        if requirement["type"] in {"all", "sequence"}:
            return any(requires_final_landing(a,b) for a,b in zip(requirement["items"], measured["children"]))
        if requirement["type"] == "if":
            branch = measured["measured"]["selected_branch"]
            return requires_final_landing(requirement[branch], measured["measured"]["child"])
        return False
    if execution.get("runtime") == "gazebo" and requires_final_landing(standard["oracle"], result):
        final = execution.get("runtime_final", {})
        if final.get("landed") is not True or final.get("armed") is not False:
            raise ValueError("Gazebo landing requires PX4 landed and disarmed evidence")
    if standard.get("evaluation_kind") == "visual_rgbd":
        _visual_execution_checks(standard, execution)
    return dict(kind=standard.get("evaluation_kind", "semantic"), oracle=result, trajectory=trajectory,
                source_sha256=execution["source_sha256"], safety_violations=0,
                annotation_source=standard["annotation_source"])


def _visual_execution_checks(standard, execution):
    # A deliberately stricter shape than historical verify_gazebo success flags.
    # The reviewer supplies visible-interior annotations for selected decisions.
    for key in ("landed", "disarmed", "safe_execution"):
        if execution.get(key) is not True:
            raise ValueError("Visual episode requires verified " + key)
    if execution.get("status") != "completed" or execution.get("safety_violations") != 0:
        raise ValueError("Visual execution did not complete safely")
    if not isinstance(execution.get("localizations"), list) or not execution["localizations"]:
        raise ValueError("Visual episode lacks measured localization evidence")
    required = standard.get("visual_requirement")
    if required:
        matches = [item for item in execution["evidence"] if item.get("action") == "visual"
                   and item.get("source") == "onboard_rgbd_model"
                   and item.get("result", {}).get("observed") is True
                   and item["result"].get("description") == required["description"]]
        if len(matches) != 1:
            raise ValueError("Visual result does not match the separately authored task target")
        visual = matches[0]
        views = visual["result"].get("views", [])
        if len(views) != 2 or len({v.get("frame_id") for v in views}) != 2:
            raise ValueError("Visual completion requires two distinct confirmation frames")
        limits = standard["visual_limits"]
        for view in views:
            found = [loc for loc in execution["localizations"] if loc.get("tool", "ground") == "ground"
                     and loc.get("frame_id") == view.get("frame_id")
                     and loc.get("rgb_sha256") == view.get("rgb_sha256")]
            if (len(found) != 1 or found[0].get("source") != "gazebo_rgbd"
                    or found[0]["surface_error_m"] > limits["max_surface_error_m"]
                    or not 0 <= found[0]["frame_age_seconds"] <= limits["max_frame_age_seconds"]
                    or found[0]["distance_m"] > limits["max_distance_m"]):
                raise ValueError("Final confirmation fails independent target/range review")
        if required.get("before_dwell") or required.get("after_dwell"):
            from .semantic_benchmark import evaluate_oracle
            # Restrict measured dwell to the required side of visual completion,
            # rather than counting incidental hovering while taking photographs.
            when = visual["simulation_time"]
            boundary = when if required.get("before_dwell") else min(v["simulation_time"] for v in views)
            telemetry = [row for row in execution["telemetry"] if
                         (row["t"] >= boundary if required.get("before_dwell") else row["t"] <= boundary)]
            poses = [item for item in standard["oracle"]["items"] if item["type"] == "pose"]
            if len(poses) != 1 or not evaluate_oracle(poses[0], evidence=execution["evidence"], telemetry=telemetry,
                    world=execution["world"],mission_id=execution["mission_id"])["satisfied"]:
                raise ValueError("Visual confirmation and dwell do not follow the required order")
    return dict(kind="visual_rgbd", annotation_source=standard["annotation_source"])


def _journal(root):
    validation = validate_trajectory(root)
    if not validation["complete"]:
        raise ValueError("Incomplete trajectory: " + "; ".join(validation["errors"][:5]))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return manifest, events


def decision_candidates(episode):
    """List response candidates without assuming local acceptance means correct."""
    root = Path(episode).resolve()
    _, events = _journal(root)
    inputs, requests, results, rows = {}, {}, {}, []
    for event in events:
        payload, kind = event["payload"], event["kind"]
        decision = payload.get("decision_id")
        if kind == "decision_input":
            inputs[decision] = event
        elif kind == "model_request":
            requests[decision] = event
        elif kind == "model_response":
            rows.append(dict(decision_id=decision, response_sequence=event["sequence"],
                             input=inputs.get(decision), request=requests.get(decision), response=event))
        elif kind == "decision_result":
            results[decision] = event
    for row in rows:
        row["result"] = results.get(row["decision_id"])
    return rows


def _assistant(response, model):
    body = response.get("response")
    scripted = (response.get("source") == "provider_override" and isinstance(model, str)
                and ("scripted" in model.lower() or "expert" in model.lower())
                and response.get("http_status") is None)
    if not (response.get("http_status") == 200 or scripted) or response.get("outcome") not in {None, "ok"}:
        raise ValueError("Response is not a successful provider exchange")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected one complete recorded assistant choice")
    choice = choices[0]
    if choice.get("finish_reason") not in {"stop", "tool_calls"}:
        raise ValueError("Truncated, filtered or unfinished responses cannot be labels")
    message = deepcopy(choice.get("message"))
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Response lacks an assistant message")
    if message.get("refusal") or not (message.get("content") or message.get("tool_calls")):
        raise ValueError("Empty or refused responses cannot be approved demonstrations")
    return {key: value for key, value in message.items() if key in {"role", "content", "tool_calls"}}


def _decoded_action(message, current_frame_id=None, tools=None):
    if message.get("tool_calls"):
        from .direct_vision import DirectVisionAgent
        schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools} if tools else None
        return DirectVisionAgent._decision(
            {"choices": [{"message": message, "finish_reason": "tool_calls"}]}, current_frame_id, schemas=schemas)
    return json.loads(message["content"])


def _inside(point, polygon):
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("Visible target annotation requires a polygon")
    coords = [point, *polygon]
    if any(not isinstance(p, list) or len(p) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in p) for p in coords):
        raise ValueError("Pixel and visible-region polygon use normalized coordinates")
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        # Strict interior; exact boundary can be ambiguous due to antialiasing.
        cross = (x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0])
        if abs(cross) < 1e-10 and min(a[0],b[0]) <= x <= max(a[0],b[0]) and min(a[1],b[1]) <= y <= max(a[1],b[1]):
            return False
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
    return inside


def _visual_target(candidate, action, standard, execution, annotations):
    # Pixel identity is reviewed independently of world-space localization error.
    args = action.get("arguments", {})
    frame_id, pixel = args.get("frame_id"), args.get("pixel")
    frames = candidate["input"]["payload"].get("frames", [])
    frame = next((item for item in frames if item.get("frame_id") == frame_id), None)
    if frame is None:
        raise ValueError("Visual target refers to a non-input frame")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Visual labels require independently annotated visible target regions")
    region = next((item for item in annotations
                   if item.get("frame_id") == frame_id and item.get("rgb_sha256") == frame.get("rgb_sha256")
                   and _inside(pixel, item.get("polygon"))), None)
    if region is None or not region.get("reviewer") or not _inside(pixel, region.get("polygon")):
        raise ValueError("Selected pixel is not inside the reviewed visible target")
    loc = next((item for item in execution["localizations"]
                if item.get("decision_id") == candidate["decision_id"] and item.get("frame_id") == frame_id), None)
    if loc is None or loc.get("source") != "gazebo_rgbd" or loc.get("pixel") != pixel:
        raise ValueError("Selected visual response lacks its actual RGB-D localization")
    if loc.get("rgb_sha256") != frame.get("rgb_sha256"):
        raise ValueError("Localization is tied to a different image")
    for key in ("surface_error_m", "frame_age_seconds", "distance_m"):
        if type(loc.get(key)) not in (int, float) or not math.isfinite(loc[key]) or loc[key] < 0:
            raise ValueError("Missing measured visual " + key)
    limits = dict(standard.get("visual_limits", {}))
    if action["tool"] == "probe" or candidate["result"]["payload"].get("status") != "completed":
        limits["max_distance_m"] = limits.get("max_acquisition_distance_m", limits.get("max_distance_m"))
    for field, bound in (("surface_error_m", "max_surface_error_m"), ("frame_age_seconds", "max_frame_age_seconds"),
                         ("distance_m", "max_distance_m")):
        if type(limits.get(bound)) not in (int,float) or not math.isfinite(limits[bound]) or limits[bound] <= 0 or loc[field] > limits[bound]:
            raise ValueError("Visual evidence violates reviewed " + bound)


def _visual_action(root, candidate, action, standard, execution, annotations):
    """Approve an actual tool effect and its useful, observed continuation.

    The final episode oracle is checked separately. This check follows recorded
    frame/decision/candidate identities; a successful episode cannot approve an
    unused preview, an unexecuted move or a repeated camera request by itself.
    """
    current = candidate["input"]["payload"]["frames"][-1]
    if current.get("source") != "gazebo_rgbd":
        raise ValueError("Visual labels require native RGB-D input frames")
    tool = action["tool"]
    proof = dict(tool=tool, input_frame_id=current["frame_id"], rgb_sha256=current["rgb_sha256"])
    if tool in {"ground", "probe"}:
        _visual_target(candidate, action, standard, execution, annotations)
        if candidate["result"]["payload"].get("status") == "completed":
            views = [view for item in execution["evidence"] for view in item.get("result", {}).get("views", [])]
            if not any(view.get("frame_id") == current["frame_id"] and
                       view.get("rgb_sha256") == current["rgb_sha256"] for view in views):
                raise ValueError("Completed ground lacks its final visual evidence")
        return proof

    _, events = _journal(root)
    candidates = decision_candidates(root)

    def accepted(row, name):
        result = (row.get("result") or {}).get("payload", {})
        if result.get("status") not in {"applied", "completed"} or result.get("action", {}).get("tool") != name:
            return False
        frames = row["input"]["payload"].get("frames", [])
        if not frames or any(frames[-1].get(key) != current.get(key) for key in ("revision", "execution_id")):
            return False
        request = row["request"]["payload"]["request"]
        message = _assistant(row["response"]["payload"], request.get("model"))
        return _decoded_action(message, frames[-1]["frame_id"], request.get("tools")) == result["action"]

    def later_ground(sequence, predicate):
        for row in candidates:
            if row["input"]["sequence"] <= sequence or not accepted(row, "ground"):
                continue
            image = row["input"]["payload"]["frames"][-1]
            if image["frame_id"] == current["frame_id"] or image.get("source") != "gazebo_rgbd" or not predicate(image):
                continue
            try:
                _visual_target(row, row["result"]["payload"]["action"], standard, execution, annotations)
            except ValueError:
                continue
            return dict(continuation_decision_id=row["decision_id"],
                        continuation_frame_id=image["frame_id"], continuation_rgb_sha256=image["rgb_sha256"])
        raise ValueError("Visual action lacks a fresh, independently verified grounding continuation")

    def metric(a, b):
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != 3 or len(b) != 3:
            raise ValueError("Visual motion lacks measured 3D positions")
        value = math.sqrt(sum(((x-y)*scale)**2 for x,y,scale in zip(a,b,(3,3,5))))
        if not math.isfinite(value):
            raise ValueError("Visual motion position must be finite")
        return value

    def movement(submit):
        result = submit["result"]["payload"]
        identifier = result["action"]["arguments"]["candidate_id"]
        previews = [row for row in candidates if row["result"] and
                    row["result"]["sequence"] < submit["input"]["sequence"] and accepted(row, "preview_move") and
                    row["result"]["payload"].get("feedback", {}).get("candidate_id") == identifier]
        if len(previews) != 1:
            raise ValueError("Submitted move lacks its unique accepted preview")
        preview = previews[0]
        endpoint = preview["result"]["payload"]["feedback"].get("endpoint_grid")
        motion = [event for event in events if event["payload"].get("decision_id") == submit["decision_id"]
                  and event["kind"] in {"motion_started", "motion_completed", "motion_aborted"}]
        if [event["kind"] for event in motion] != ["motion_started", "motion_completed"]:
            raise ValueError("Submitted move lacks an uninterrupted completed motion")
        start, end = motion
        if not submit["result"]["sequence"] < start["sequence"] < end["sequence"]:
            raise ValueError("Motion completion precedes its submitted action")
        reached = end["payload"].get("position_grid")
        tolerance = standard.get("visual_limits", {}).get("max_motion_endpoint_error_m", 1.)
        if type(tolerance) not in (int,float) or not 0 < tolerance <= 3:
            raise ValueError("Motion endpoint tolerance must be in (0, 3] metres")
        if metric(reached, endpoint) > tolerance or metric(result.get("state", {}).get("position_grid"), reached) < .25:
            raise ValueError("Submitted move did not reach its previewed endpoint")
        following = later_ground(end["sequence"], lambda image:
            image["simulation_time"] >= end["payload"]["simulation_time"] and
            metric(image["position_grid"], reached) <= tolerance)
        return dict(preview_decision_id=preview["decision_id"], submit_decision_id=submit["decision_id"],
                    candidate_id=identifier, motion_completed_sequence=end["sequence"], **following)

    if tool == "look":
        args = action["arguments"]
        angle = lambda a,b: abs((a-b+180) % 360-180)
        if angle(current["yaw_deg"], args["yaw_deg"]) < .5 and abs(current["pitch_deg"]-args["pitch_deg"]) < .5:
            raise ValueError("Repeated camera angles are not a useful look label")
        result = candidate["result"]["payload"]
        proof.update(later_ground(candidate["result"]["sequence"], lambda image:
            image["simulation_time"] >= result["state"]["simulation_time"] and
            angle(image["yaw_deg"], args["yaw_deg"]) <= 1 and abs(image["pitch_deg"]-args["pitch_deg"]) <= 1))
    elif tool == "preview_move":
        identifier = candidate["result"]["payload"].get("feedback", {}).get("candidate_id")
        following = [row for row in candidates if row["input"]["sequence"] > candidate["result"]["sequence"] and
                     accepted(row, "submit_move") and row["result"]["payload"]["action"]["arguments"]["candidate_id"] == identifier]
        if not identifier or len(following) != 1:
            raise ValueError("Preview was not consumed by one accepted submit_move")
        proof.update(movement(following[0]))
    elif tool == "submit_move":
        proof.update(movement(candidate))
    elif tool == "wait":
        state = candidate["result"]["payload"].get("state", {})
        if math.sqrt(sum(v*v for v in state.get("velocity", []))) <= .05:
            raise ValueError("Stationary wait is context, not a useful training label")
        active = [row for row in candidates if row["result"] and row["result"]["sequence"] < candidate["input"]["sequence"]
                  and accepted(row, "submit_move")]
        if not active:
            raise ValueError("Wait lacks a previously submitted motion")
        proof.update(movement(active[-1]))
        if proof["motion_completed_sequence"] <= candidate["result"]["sequence"]:
            raise ValueError("Wait occurred after motion completed")
    else:
        raise ValueError("Unsupported visual tool for demonstration review")
    return proof


def _recorded_schema_validator(schema):
    # Stored response_format is a complete decoder schema, whose anyOf wrappers
    # are outside the optimized runtime validator's narrower vocabulary.
    def local_refs(node):
        if isinstance(node, dict):
            if "$ref" in node and (not isinstance(node["$ref"], str) or not node["$ref"].startswith("#/")):
                raise ValueError("Recorded schemas may reference local definitions only")
            for value in node.values():
                local_refs(value)
        elif isinstance(node, list):
            for value in node:
                local_refs(value)
    local_refs(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        raise ValueError("Recorded response schema is invalid") from None
    return Draft202012Validator(schema)


def _normalize_json_content(content, schema):
    """Only reorder existing keys/whitespace; never change an action or add fields."""
    decoded = json.loads(content)
    validator = _recorded_schema_validator(schema)
    error = next(validator.iter_errors(decoded), None)
    if error is not None:
        # jsonschema's normal exception string includes the raw response; never
        # emit it because a malformed output could contain protected content.
        raise ValueError("Response violates recorded JSON Schema at " +
                         "/".join(str(part) for part in error.absolute_path))
    def ordered(value, current, depth=0):
        if depth > 80:
            raise ValueError("Response normalization exceeds supported nesting")
        if isinstance(current, bool):
            return value
        if "$ref" in current:
            resolved = schema
            for part in current["$ref"][2:].split("/"):
                resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
            current = {**resolved, **{key:item for key,item in current.items() if key != "$ref"}}
        for kind in ("oneOf", "anyOf"):
            if kind in current:
                alternatives = [variant for variant in current[kind]
                                if validator.evolve(schema=variant).is_valid(value)]
                if not alternatives:
                    raise ValueError("No valid recorded decoder alternative")
                parent = {key:item for key,item in current.items() if key != kind}
                child = alternatives[0]
                merged = {**parent, **child}
                if "properties" in parent and "properties" in child:
                    merged["properties"] = {**parent["properties"], **child["properties"]}
                return ordered(value, merged, depth+1)
        if isinstance(value, dict):
            properties = current.get("properties", {})
            keys = [key for key in properties if key in value] + [key for key in value if key not in properties]
            return {key:ordered(value[key], properties.get(key, {}), depth+1) for key in keys}
        if isinstance(value, list):
            return [ordered(item, current.get("items", {}), depth+1) for item in value]
        return value
    content_out = json.dumps(ordered(decoded, schema), ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
    if json.loads(content_out) != decoded:
        raise ValueError("Serialization normalization changed response meaning")
    return content_out, decoded


def _sample(root, candidate, standard, execution, target, annotations=None):
    if not candidate.get("input") or not candidate.get("request") or not candidate.get("result"):
        raise ValueError("Decision lacks its input, request or terminal feedback")
    request = candidate["request"]["payload"].get("request")
    result = candidate["result"]["payload"]
    if result.get("status") not in {"applied", "completed"}:
        raise ValueError("Rejected or aborted actions are context, not approved labels")
    if not (candidate["input"]["sequence"] < candidate["request"]["sequence"] < candidate["response_sequence"]
            < candidate["result"]["sequence"]):
        raise ValueError("Decision input/response/acceptance causal order is invalid")
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list) or not request["messages"]:
        raise ValueError("Missing exact model prompt")
    message = _assistant(candidate["response"]["payload"], request.get("model"))
    _safe_training_value(request["messages"])
    _safe_training_value(message)
    _safe_training_value(request.get("tools", []))
    original_content = message.get("content")
    normalization = "none"
    frames = candidate["input"]["payload"].get("frames", [])
    decoded = _decoded_action(message, frames[-1]["frame_id"] if frames else None, request.get("tools"))
    schema = request.get("response_format", {}).get("json_schema", {}).get("schema")
    if schema and not message.get("tool_calls"):
        message["content"], normalized_value = _normalize_json_content(original_content, schema)
        if normalized_value != decoded:
            raise ValueError("Normalized response differs from accepted action")
        normalization = "decoder_property_order_only"
    action = result.get("action")
    if action is None or decoded != action:
        raise ValueError("Approved response differs from the action actually accepted")
    visual_verification = None
    if candidate["input"]["payload"].get("modality") != "text":
        if standard.get("evaluation_kind") != "visual_rgbd":
            raise ValueError("Visual labels require visual-specific evidence review")
        visual_verification = _visual_action(root, candidate, action, standard, execution, annotations)
    images, refs = [], []
    def convert(value):
        if isinstance(value, dict):
            if set(value) == {"artifact"}:
                ref = value["artifact"]
                path = _path(root, ref["path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                    raise ValueError("Image hash mismatch")
                if ref.get("mime_type") not in {"image/png", "image/jpeg", "image/webp"}:
                    raise ValueError("Only recorded RGB images are model inputs")
                images.append(str(path))
                refs.append(deepcopy(ref))
                return str(path)
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value
    prompt = convert(deepcopy(request["messages"]))
    stage = candidate["input"]["payload"].get("stage", "visual")
    label_issue = ("requirements_audit_missing_assertions" if stage == "audit"
                   and not decoded.get("assertions") else None)
    return dict(id=standard["id"] + ":" + str(candidate["response_sequence"]), prompt=prompt,
        completion=[message], images=images, tools=deepcopy(request.get("tools", [])),
        loss_scope="completion_only", metadata=dict(episode=str(root), task_id=standard["id"],
            family=standard["family"], scenario=standard["scenario"],
            trajectory_group=standard.get("trajectory_group", standard["id"]),
            scene_geometry_sha256=standard.get("scene_geometry_sha256"), frozen_split=standard.get("frozen_split"),
            decision_id=candidate["decision_id"], response_sequence=candidate["response_sequence"],
            stage=stage, training_eligible=label_issue is None, recollection_reason=label_issue,
            model=candidate["request"]["payload"].get("model", request.get("model")),
            source=candidate["response"]["payload"].get("source", "provider"),
            response_format=deepcopy(request.get("response_format")),
            target_normalization=normalization,
            original_response_content_sha256=hashlib.sha256(original_content.encode()).hexdigest() if isinstance(original_content, str) else None,
            annotation_source=standard["annotation_source"], review_reason=target["reason"],
            visual_verification=visual_verification,
            image_references=refs, task_standard_sha256=canonical_sha256(standard),
            execution_source_sha256=execution.get("source_sha256")))


def _bind_execution(manifest, events, standard, execution):
    if execution.get("trajectory_events_sha256") != manifest["events"]["sha256"]:
        raise ValueError("Execution is not bound to this interaction journal")
    if execution.get("task_standard_sha256") != canonical_sha256(standard):
        raise ValueError("Execution is not bound to this source task standard")
    metadata = manifest.get("metadata", {})
    for field, expected in (("source_sha256", execution.get("source_sha256")),
                            ("task_standard_sha256", canonical_sha256(standard)),
                            ("task_id", standard["id"]), ("scenario", standard["scenario"])):
        if not expected or metadata.get(field) != expected:
            raise ValueError("Manifest does not bind execution " + field)
    if standard.get("scene_sha256") and any(value != standard["scene_sha256"] for value in
            (metadata.get("scene_asset_content_sha256"), execution.get("scene_asset_content_sha256"))):
        raise ValueError("Execution scene differs from the frozen task layout")
    run_ids = {event["payload"].get("run_id") for event in events if event["payload"].get("run_id")}
    if not execution.get("run_id") or run_ids != {execution["run_id"]}:
        raise ValueError("Execution run ID differs from its decision journal")
    if execution.get("instruction") != standard["instruction"] or execution.get("mission_id") is None:
        raise ValueError("Execution lacks its source task instruction/mission")
    for event in events:
        if event["kind"] == "decision_input":
            context = event["payload"].get("context", {})
            if context.get("instruction") != standard["instruction"]:
                raise ValueError("Recorded decision belongs to another source instruction")
            if event["payload"].get("modality") == "text" and context.get("mission_id") != execution["mission_id"]:
                raise ValueError("Recorded decision belongs to another mission")
    for row in execution.get("evidence", []) + execution.get("telemetry", []):
        if row.get("mission_id") != execution["mission_id"]:
            raise ValueError("Execution evidence crosses mission identities")


def build_review(episode, *, standard, execution_path="execution.json", reviewer,
                 response_sequences=None, target_reviews=None, visual_annotations=None):
    """Create an approval sidecar; no selection returns a read-only pending review.

    target_reviews maps selected response sequences to explanations relating
    them to the source intent. Corrections require a newly executed journal:
    this API deliberately offers no arbitrary replacement assistant target.
    """
    root = Path(episode).resolve()
    manifest, events = _journal(root)
    standard = _standard(deepcopy(standard))
    standard_path, review_path = root / "standard.json", root / "review.json"
    if review_path.exists():
        raise FileExistsError("Review already exists; existing approval is immutable")
    if standard_path.exists() and canonical_sha256(json.loads(standard_path.read_text(encoding="utf-8"))) != canonical_sha256(standard):
        raise ValueError("Existing source task standard differs from the proposed review")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("An identifiable reviewer is required")
    candidates = {row["response_sequence"]: row for row in decision_candidates(root)}
    selected = [] if response_sequences is None else list(response_sequences)
    if len(set(selected)) != len(selected) or any(type(seq) is not int or seq not in candidates for seq in selected):
        raise ValueError("Selections must be unique recorded model response sequences")
    execution_ref = file_reference(root, execution_path)
    execution = _read_ref(root, execution_ref)
    _bind_execution(manifest, events, standard, execution)
    checks = _execution_checks(standard, execution)
    targets = []
    for sequence in selected:
        reason = (target_reviews or {}).get(sequence, (target_reviews or {}).get(str(sequence)))
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Each selected target requires a specific review reason")
        target = dict(response_sequence=sequence, decision_id=candidates[sequence]["decision_id"], reason=reason)
        _sample(root, candidates[sequence], standard, execution, target, visual_annotations)
        targets.append(target)
    review = dict(version=VERSION, status="approved" if targets else "pending", reviewer=reviewer,
        review_kind="evidence_verified_agent_assisted" if standard["annotation_source"] == "agent_assisted" else "human_reviewed",
        events_sha256=manifest["events"]["sha256"], task_standard_sha256=canonical_sha256(standard),
        evaluator_source_sha256=source_fingerprint(),
        execution=execution_ref, checks=checks, targets=targets, visual_annotations=deepcopy(visual_annotations),
        limitations=["Standards may be agent-assisted; provenance is not human approval.",
                     "Execution verification does not prove all unselected decisions correct."])
    if targets:
        if not standard_path.exists():
            _write_new(standard_path, standard)
        review["standard"] = file_reference(root, "standard.json")
        _write_new(review_path, review)
    return review


def audit_episode(episode):
    root = Path(episode).resolve()
    report = dict(episode=str(root), approved=False, samples=[], errors=[])
    try:
        manifest, events = _journal(root)
        review = json.loads((root / "review.json").read_text(encoding="utf-8"))
        standard = _standard(_read_ref(root, review["standard"]))
        if review.get("version") != VERSION or review.get("status") != "approved":
            raise ValueError("Review is missing, pending or unsupported")
        if not review.get("reviewer") or not review.get("targets"):
            raise ValueError("Review has no reviewer or selected targets")
        if review.get("events_sha256") != manifest["events"]["sha256"]:
            raise ValueError("Review refers to different interaction events")
        if review.get("task_standard_sha256") != canonical_sha256(standard):
            raise ValueError("Task standard changed after review")
        execution = _read_ref(root, review["execution"])
        _bind_execution(manifest, events, standard, execution)
        checks = _execution_checks(standard, execution)
        candidates = {row["response_sequence"]: row for row in decision_candidates(root)}
        seen, samples = set(), []
        for target in review["targets"]:
            seq = target["response_sequence"]
            if seq in seen or seq not in candidates or not isinstance(target.get("reason"), str) or not target["reason"].strip():
                raise ValueError("Invalid or duplicated reviewed target")
            if target.get("decision_id") != candidates[seq]["decision_id"]:
                raise ValueError("Reviewed target is bound to another decision")
            samples.append(_sample(root, candidates[seq], standard, execution, target, review.get("visual_annotations")))
            seen.add(seq)
        report.update(approved=True, samples=samples, checks=checks, standard=standard,
                      training_eligible_decisions=sum(row["metadata"]["training_eligible"] for row in samples),
                      review_kind=review["review_kind"],
                      review_evaluator_source_sha256=review["evaluator_source_sha256"],
                      evaluator_source_sha256=source_fingerprint())
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    return report


def export_dataset(episodes, output, *, splits=None):
    """Export only approved decisions; reject family/scenario split leakage.

    A single-campus pilot remains pilot. This deliberately does not create a
    misleading train/test split out of neighboring or rephrased episodes.
    """
    output = Path(output).resolve()
    roots = [Path(path).resolve() for path in episodes]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Provide nonempty, unique episode directories")
    audits = [audit_episode(root) for root in roots]
    assignments, seen_ids, family_split, scene_split, lineage_split = {}, set(), {}, {}, {}
    samples, excluded_labels = [], []
    for audit in audits:
        if not audit["approved"]:
            continue
        standard = audit["standard"]
        if standard["id"] in seen_ids:
            raise ValueError("Duplicate task id; group repeats under one unique episode identity")
        seen_ids.add(standard["id"])
        split = (splits or {}).get(standard["id"], "pilot")
        if split not in _SPLITS:
            raise ValueError("Unsupported dataset split")
        if standard.get("frozen_split", split) != split:
            raise ValueError("Requested split differs from the frozen source task assignment")
        for groups, key in ((family_split, standard["family"]),
                            (scene_split, standard.get("scene_geometry_sha256", standard["scenario"])),
                            (lineage_split, standard.get("trajectory_group", standard["id"]))):
            if key in groups and groups[key] != split:
                raise ValueError("Leakage: task family or scenario crosses dataset splits")
            groups[key] = split
        assignments[standard["id"]] = split
        for sample in audit["samples"]:
            if not sample["metadata"]["training_eligible"]:
                excluded_labels.append(dict(id=sample["id"], task_id=standard["id"],
                    stage=sample["metadata"]["stage"], reason=sample["metadata"]["recollection_reason"]))
                continue
            sample["split"] = split
            samples.append(sample)
    output.mkdir(parents=True, exist_ok=False)
    for sample in samples:
        paths = {}
        for absolute, ref in zip(sample["images"], sample["metadata"]["image_references"]):
            suffix = Path(ref["path"]).suffix
            relative = "images/" + ref["sha256"] + suffix
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(absolute, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != ref["sha256"]:
                raise ValueError("Copied dataset image failed integrity check")
            paths[absolute] = relative
        def portable(value):
            if isinstance(value, dict):
                if value.get("type") == "image_url":
                    url = value.get("image_url")
                    url = url.get("url") if isinstance(url, dict) else url
                    if url not in paths:
                        raise ValueError("Dataset image is not a validated episode artifact")
                    return {"type": "image", "image": paths[url]}
                return {key: portable(item) for key, item in value.items()}
            if isinstance(value, list):
                return [portable(item) for item in value]
            return value
        sample["prompt"] = portable(sample["prompt"])
        sample["images"] = [paths[path] for path in sample["images"]]
    for split in sorted({sample["split"] for sample in samples}):
        with (output / (split + ".jsonl")).open("x", encoding="utf-8") as stream:
            for sample in samples:
                if sample["split"] == split:
                    stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    files = [dict(file_reference(output, path.relative_to(output).as_posix()), bytes=path.stat().st_size)
             for path in sorted(output.rglob("*")) if path.is_file()]
    report = dict(version=VERSION, files=files, episodes=len(audits), approved_episodes=sum(a["approved"] for a in audits),
        decisions=len(samples), excluded_labels=excluded_labels, splits=assignments, loss_scope="completion_only",
        training_started=False, image_paths="Relative to this dataset directory; resolve before processor collation.", processor_validation="Required before training; do not truncate image tokens or target completions.",
        audits=[{key: value for key, value in audit.items() if key not in {"samples", "standard"}} for audit in audits])
    _write_new(output / "audit.json", report)
    return report
