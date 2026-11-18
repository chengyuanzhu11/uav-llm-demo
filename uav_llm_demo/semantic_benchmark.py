"""Independent, evidence-grounded campus task evaluation.

The oracle is separately authored engineering data, never the evaluated model's
program. Fixtures are agent-assisted and have not been independently reviewed by
humans. Offline responses test the execution/evaluation pipeline only; remote
interpretation is opt-in.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import TimeoutError as FutureTimeout
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from .agents import DeepSeekAgent, PlannerProviderError
from .mission_service import MissionService, SEMANTIC_POLICIES
from .html_runtime import HtmlRuntime
from .motion import norm
from .provider_config import load_planner_config
from .run_log import source_fingerprint
from .safety import SCALE, metric_distance
from .scenarios import load_scenario
from .spatial_motion import spatial_catalog
from .task_protocol import PROTOCOL_VERSION


BENCHMARK_VERSION = "campus-semantic-v1"
LOADED_SOURCE_FINGERPRINT = source_fingerprint()
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "campus_semantic_v1.json"
TERMINAL_FAILURES = {"needs_clarification", "generation_failed", "safety_wait", "failed"}
_SOURCES = {"inspect": "simulated_inspection", "thermal_scan": "simulated_thermal_sensor",
            "capture": "simulated_capture", "motion": "runtime_trajectory_tracking",
            "review": "validated_execution_review", "control": "runtime_telemetry",
            "visit": "runtime_telemetry", "return_home": "runtime_telemetry"}
_SUCCESS = {"inspect": "inspected", "thermal_scan": "simulated", "capture": "captured",
            "motion": "completed", "review": "reviewed", "control": "completed",
            "visit": "visited", "return_home": "returned"}


def redact(value, secrets=()):
    """Defensive report sanitization, including instructions and provider errors."""
    if isinstance(value, dict):
        hidden = {"apikey", "token", "authorization", "password", "secret", "privatekey"}
        return {str(k): "[redacted]" if re.sub(r"[^a-z]", "", str(k).lower()) in hidden
                else redact(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        value = re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [redacted]", value)
        return re.sub(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,})", "[redacted]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric data")
    if value < 0 or (positive and value == 0):
        raise ValueError(name + " must be positive" if positive else name + " must be nonnegative")
    return float(value)


def validate_requirement(req, depth=0):
    if depth > 12 or not isinstance(req, dict):
        raise ValueError("Invalid or over-nested oracle requirement")
    kind = req.get("type")
    allowed = {
        "all": {"type", "items"}, "sequence": {"type", "items"},
        "evidence": {"type", "action", "target", "count", "min_count", "result", "min_duration", "occurrence"},
        "if": {"type", "observation", "field", "op", "value", "then", "else", "decision_occurrence"},
        "circuit": {"type", "target", "direction", "laps", "tolerance_laps", "heading", "after_update", "before_update"},
        "goal": {"type", "target", "anchor", "state", "altitude_m", "dwell_seconds", "yaw_deg"},
        "pose": {"type", "target", "anchor", "state", "altitude_m", "dwell_seconds", "yaw_deg", "occurrence"},
    }
    if kind not in allowed or set(req) - allowed[kind]:
        raise ValueError("Unknown oracle predicate or field")
    if kind in {"all", "sequence"}:
        if not isinstance(req.get("items"), list) or not req["items"]:
            raise ValueError("Oracle composition requires nonempty items")
        for child in req["items"]:
            validate_requirement(child, depth + 1)
    elif kind == "if":
        if req.get("op") not in {">", ">=", "<", "<=", "==", "!="}:
            raise ValueError("Unsupported oracle comparison")
        if not isinstance(req.get("field"), str) or "value" not in req:
            raise ValueError("Oracle condition requires field and value")
        if type(req.get("decision_occurrence", 1)) is not int or req.get("decision_occurrence", 1) < 1:
            raise ValueError("Condition decision occurrence must be a positive integer")
        validate_requirement(req.get("observation"), depth + 1)
        if req["observation"]["type"] != "evidence":
            raise ValueError("Condition observation must select measured evidence")
        for arm in ("then", "else"):
            validate_requirement(req.get(arm), depth + 1)
    else:
        targetless_review = (kind == "evidence" and req.get("action") == "review"
                             and "target" in req and req["target"] is None)
        if not targetless_review and (not isinstance(req.get("target"), str) or not req["target"]):
            raise ValueError("Oracle target is required")
        if kind == "evidence":
            if req.get("action") not in _SOURCES:
                raise ValueError("Unsupported evidence action")
            for key in ("count", "min_count"):
                if key in req and (type(req[key]) is not int or req[key] < 0):
                    raise ValueError("Evidence counts must be nonnegative integers")
            if "count" in req and "min_count" in req:
                raise ValueError("Specify one count rule")
            if "result" in req and not isinstance(req["result"], dict):
                raise ValueError("Evidence result must be an object")
            if "occurrence" in req and (type(req["occurrence"]) is not int or req["occurrence"] < 1):
                raise ValueError("Evidence occurrence must be a positive integer")
            _number(req.get("min_duration", 0), "min_duration")
        elif kind == "circuit":
            if req.get("direction") not in {"clockwise", "counterclockwise"}:
                raise ValueError("Circuit direction is required")
            if (isinstance(req.get("laps"), bool) or not isinstance(req.get("laps"), (int, float))
                    or not math.isfinite(req["laps"]) or req["laps"] <= 0):
                raise ValueError("Circuit laps must be finite positive numbers")
            _number(req.get("tolerance_laps", .06), "tolerance_laps")
            if req.get("heading") not in {None, "face_center"}:
                raise ValueError("Unsupported heading oracle")
            for bound in ("after_update", "before_update"):
                if bound in req and (type(req[bound]) is not int or req[bound] < 0):
                    raise ValueError("Circuit update bounds must be nonnegative integers")
            if "before_update" in req and req["before_update"] <= req.get("after_update", 0):
                raise ValueError("Circuit update range must not be empty")
        else:
            if req.get("state") not in {"landed", "airborne"}:
                raise ValueError("Goal state is required")
            if req.get("anchor", "center") not in {"center", "arrival"}:
                raise ValueError("Unknown goal anchor")
            _number(req.get("dwell_seconds", 0), "dwell_seconds")
            if "altitude_m" in req:
                _number(req["altitude_m"], "altitude_m")
            if "yaw_deg" in req and (isinstance(req["yaw_deg"], bool) or not isinstance(req["yaw_deg"], (int, float))
                                      or not math.isfinite(req["yaw_deg"])):
                raise ValueError("Goal yaw must be finite")
            if "occurrence" in req and (type(req["occurrence"]) is not int or req["occurrence"] < 1):
                raise ValueError("Pose occurrence must be positive")
    return req


def load_dataset(path=DEFAULT_DATASET):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != BENCHMARK_VERSION or not data.get("cases"):
        raise ValueError("Unknown or empty campus benchmark")
    ids, families = set(), {}
    for case in data["cases"]:
        if "task_world" in data:
            case["task_world"] = {**deepcopy(data["task_world"]), **case.get("task_world", {})}
        if case.get("id") in ids or not isinstance(case.get("id"), str):
            raise ValueError("Case IDs must be unique")
        ids.add(case["id"])
        if case.get("split") not in {"dev", "holdout"}:
            raise ValueError("Case split must be dev or holdout")
        family = case.get("combination")
        if not family or families.setdefault(family, case["split"]) != case["split"]:
            raise ValueError("Task combinations must not cross dev/holdout splits")
        if not isinstance(case.get("instruction"), str) or not 1 <= len(case["instruction"]) <= 2000:
            raise ValueError("Case requires a user instruction")
        validate_requirement(case.get("oracle"))
        if not isinstance(case.get("offline_responses"), list) or not case["offline_responses"]:
            raise ValueError("Offline execution fixtures are required")
        for stage in ("offline_requirements", "offline_audits"):
            if not isinstance(case.get(stage), list) or not case[stage]:
                raise ValueError("Independent offline requirements and audit fixtures are required")
        for update in case.get("updates", []):
            validate_requirement(update.get("after"))
            if not isinstance(update.get("instruction"), str) or not update["instruction"]:
                raise ValueError("Task update requires an instruction")
        if "task_world" in case or "task_observation" in case:
            from .task_observation import TaskObservationConfig, TaskObservationWorld
            if "task_world" in case:
                TaskObservationWorld.from_dict(case["task_world"])
            options = case.get("task_observation", {})
            if "seed" in options:
                raise ValueError("Sensor seeds come from the paired run, not case overrides")
            TaskObservationConfig(**options)
        if "physics" in case:
            raise ValueError("Local physics cases are no longer supported; remove physics settings for HTML regression or migrate the experiment to Gazebo.")
    return data


def _sensor_aliases(records):
    """Recognize only forward, same-action/target refreshes of measured records."""
    known, aliases, refreshes = {}, {}, set()
    for row in records:
        identity, original_id = row.get("evidence_id"), row.get("refresh_of")
        original = known.get(original_id)
        if (identity and identity not in known and original is not None
                and row["action"] in {"thermal_scan", "inspect", "capture"}
                and row.get("observation", {}).get("status") == "valid"
                and row["simulation_time"] > original["simulation_time"]
                and all(row.get(key) == original.get(key) for key in ("action", "target_id", "mission_id", "vehicle_id", "source"))):
            for root, current in list(aliases.items()):
                if current["evidence_id"] == original_id:
                    aliases[root] = row
            aliases[original_id] = row
            refreshes.add(row["execution_id"])
        if identity and identity not in known:
            known[identity] = row
    return aliases, refreshes


def _evidence(records, req, mission_id, *, include_refresh=False):
    selected, seen, evidence_ids = [], set(), set()
    for row in records:
        eid = row.get("execution_id")
        if (not eid or eid in seen or row.get("mission_id") != mission_id
                or row.get("vehicle_id") != "uav-1"
                or row.get("evidence_id") and row["evidence_id"] in evidence_ids):
            continue
        if row.get("action") != req["action"] or row.get("target_id") != req["target"]:
            continue
        result = row.get("result", {})
        if (row.get("source") != _SOURCES[req["action"]]
                or result.get(_SUCCESS[req["action"]]) is not True):
            continue
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (row.get("simulation_time"), row.get("duration_seconds", 0))):
            continue
        observation = row.get("observation")
        if observation is not None:
            measured_at = row.get("simulation_time")
            if (not isinstance(observation, dict) or observation.get("status") != "valid"
                    or any(isinstance(observation.get(field), bool) or
                           not isinstance(observation.get(field), (float, int)) or
                           not math.isfinite(observation[field])
                           for field in ("observed_at", "valid_until", "confidence"))
                    or not isinstance(measured_at, (float, int))
                    or not observation["observed_at"] <= measured_at <= observation["valid_until"]
                    or not 0 <= observation["confidence"] <= 1):
                continue
        seen.add(eid)
        if row.get("evidence_id"):
            evidence_ids.add(row["evidence_id"])
        selected.append(row)
    selected.sort(key=lambda r: r["simulation_time"])
    if not include_refresh:
        _, refreshes = _sensor_aliases(selected)
        selected = [row for row in selected if row["execution_id"] not in refreshes]
    if "occurrence" in req:
        selected = selected[req["occurrence"] - 1:req["occurrence"]]
    return [row for row in selected if row.get("duration_seconds", 0) + 1e-6 >= req.get("min_duration", 0)
            and all(row.get("result", {}).get(k) == v for k, v in req.get("result", {}).items())]


def _condition_observation(records, original, mission_id, field, evaluations=None, *, occurrence=1, dropped=0):
    if evaluations is None:
        # Historical fixtures have no decision trace. Never let a later refresh
        # rewrite the reading they originally used.
        return original, original["simulation_time"]
    if dropped:
        return None, None
    matches = [event for event in evaluations if event.get("requested_evidence_id") == original.get("evidence_id")
               and event.get("field") == field]
    if len(matches) < occurrence:
        return None, None
    event = matches[occurrence - 1]
    when = event.get("simulation_time")
    if isinstance(when, bool) or not isinstance(when, (int, float)) or not math.isfinite(when):
        return None, None
    rows = _evidence(records, dict(action=original["action"], target=original["target_id"]),
                     mission_id, include_refresh=True)
    rows = [row for row in rows if row["simulation_time"] <= when]
    aliases, _ = _sensor_aliases(rows)
    reading = aliases.get(original.get("evidence_id"), original)
    observed = reading.get("observation")
    if (reading["simulation_time"] > when or reading.get("evidence_id") != event.get("resolved_evidence_id")
            or field not in reading["result"] or event.get("observed_value") != reading["result"][field]
            or observed is not None and not observed["observed_at"] <= when <= observed["valid_until"]):
        return None, None
    return reading, when


def _compare(left, op, right):
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    try:
        return {">": lambda: left > right, ">=": lambda: left >= right,
                "<": lambda: left < right, "<=": lambda: left <= right,
                "==": lambda: left == right, "!=": lambda: left != right}[op]()
    except TypeError:
        return False


def _boundary_error(position, boundary):
    """Horizontal metric distance from measured pose to a fixed scene perimeter."""
    if not boundary:
        return math.inf
    px, py = position[0] * SCALE[0], position[1] * SCALE[1]
    points = [(p[0] * SCALE[0], p[1] * SCALE[1]) for p in boundary]
    distances = []
    for a, b in zip(points, points[1:] + points[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        fraction = min(1., max(0., ((px - a[0]) * dx + (py - a[1]) * dy) / max(dx * dx + dy * dy, 1e-12)))
        distances.append(math.hypot(px - a[0] - fraction * dx, py - a[1] - fraction * dy))
    return min(distances)


def _track_distance(samples):
    return sum(metric_distance(a["position"], b["position"]) for a, b in zip(samples, samples[1:]))


def _motion_tracks(completions, telemetry, target, *, after_update=0, before_update=math.inf):
    """Associate actual tracking with completed continuation chains and orphans.

    The scope uses explicit user updates. Autonomous revisions never erase
    physical motion. Legacy samples lacking action/target can only be linked
    through a completed execution's identity.
    """
    chains, valid, claimed = [], True, set()
    for record in completions:
        chain = record["result"].get("tracking_executions", [record["execution_id"]])
        if (not isinstance(chain, list) or not chain or any(not isinstance(eid, str) or not eid for eid in chain)
                or len(set(chain)) != len(chain) or chain[-1] != record["execution_id"] or claimed.intersection(chain)):
            valid = False
            chain = [record["execution_id"]]
        chains.append((record, chain))
        claimed.update(chain)
    runs, previous, episode = {}, None, None
    for sample in sorted(telemetry, key=lambda item: item["t"]):
        eid = sample.get("execution_id")
        if (not eid or sample.get("phase") != "tracking"
                or not after_update <= sample.get("user_update_index", 0) < before_update):
            previous = None
            continue
        explicit = sample.get("action") == "motion" and sample.get("target_id") == target
        legacy = sample.get("action") is None and sample.get("target_id") is None and eid in claimed
        if explicit or legacy:
            if previous != eid:
                episode = []
                runs.setdefault(eid, []).append(episode)
            episode.append(sample)
            previous = eid
        else:
            previous = None
    groups, covered = [], set()
    for record, chain in chains:
        # A completion after a user's selected time window cannot retroactively
        # certify an earlier unfinished execution inside that window.
        if record["execution_id"] not in runs:
            if after_update == 0 and before_update == math.inf:
                groups.append(dict(samples=[], segments=[], execution_ids=chain, completed=True, time=record["simulation_time"]))
            continue
        parts = [part for eid in chain for part in runs.get(eid, [])]
        ordered = all(a[-1]["t"] <= b[0]["t"] for a, b in zip(parts, parts[1:]))
        valid = valid and ordered and parts[-1][-1]["t"] <= record["simulation_time"] + 1e-6
        samples = [sample for part in parts for sample in part]
        groups.append(dict(samples=samples, segments=parts, execution_ids=chain, completed=True, time=record["simulation_time"]))
        covered.update(chain)
    for eid, parts in runs.items():
        if eid not in covered:
            samples = [sample for part in parts for sample in part]
            groups.append(dict(samples=samples, segments=parts, execution_ids=[eid], completed=False, time=samples[-1]["t"]))
    return groups, sorted((part for parts in runs.values() for part in parts), key=lambda samples: samples[0]["t"]), valid


def evaluate_oracle(requirement, *, evidence, telemetry, world, mission_id,
                    condition_evaluations=None, condition_evaluations_dropped=0):
    """Evaluate only measured evidence/poses and fixed scene geometry, never a plan.

    Telemetry positions use scene GRID units; speed is m/s, yaw radians, time
    simulation seconds. Clockwise has positive winding because scene y is south.
    """
    validate_requirement(requirement)
    telemetry = [row for row in telemetry if row.get("mission_id", mission_id) == mission_id
                 and row.get("vehicle_id", "uav-1") == "uav-1"]

    def check(req, path, after=-math.inf):
        kind = req["type"]
        measured, first, last = {}, None, None
        if kind in {"all", "sequence"}:
            children = []
            for i, child in enumerate(req["items"]):
                result = check(child, f"{path}.{i}", after)
                children.append(result)
                if kind == "sequence" and result["last"] is not None:
                    after = result["last"]
            passed = all(c["satisfied"] for c in children)
            timed = [c for c in children if c["first"] is not None]
            if kind == "sequence":
                passed = passed and all(a["last"] <= b["first"] for a, b in zip(timed, timed[1:]))
            if timed:
                first, last = min(c["first"] for c in timed), max(c["last"] for c in timed)
            return dict(path=path, type=kind, satisfied=passed, children=children, first=first, last=last)
        if kind == "evidence":
            rows = _evidence(evidence, req, mission_id)
            count = len(rows)
            passed = count == req["count"] if "count" in req else count >= req.get("min_count", 1)
            measured = dict(count=count)
            if rows:
                first, last = rows[0]["simulation_time"], rows[-1]["simulation_time"]
        elif kind == "if":
            rows = _evidence(evidence, req["observation"], mission_id)
            observation = check(req["observation"], path + ".observation")
            reading, read_at = (_condition_observation(evidence, rows[-1], mission_id, req["field"],
                condition_evaluations, occurrence=req.get("decision_occurrence", 1),
                dropped=condition_evaluations_dropped) if rows else (None, None))
            available = observation["satisfied"] and reading is not None and req["field"] in reading["result"]
            if available:
                value = reading["result"][req["field"]]
                branch = "then" if _compare(value, req["op"], req["value"]) else "else"
                child = check(req[branch], path + "." + branch)
                passed = child["satisfied"] and (child["first"] is None or child["first"] >= read_at)
                measured = dict(observed_value=value, selected_branch=branch, child=child,
                                observation_evidence_id=reading.get("evidence_id"), decision_time=read_at)
                first, last = read_at, child["last"] or read_at
            else:
                reason = "condition_trace_truncated" if condition_evaluations_dropped else "observation_or_decision_missing"
                passed, measured = False, dict(reason=reason)
        elif kind == "circuit":
            center = world[req["target"]]["center"]
            completions = _evidence(evidence, dict(action="motion", target=req["target"]), mission_id)
            tracks, runs, bindings_valid = _motion_tracks(completions, telemetry, req["target"],
                after_update=req.get("after_update", 0), before_update=req.get("before_update", math.inf))
            groups = []
            direction = 1 if req["direction"] == "clockwise" else -1
            for track in tracks:
                samples = track["samples"]
                winding = reverse = 0.0
                for segment in track["segments"]:
                    angles = [math.atan2(s["position"][1] - center[1], s["position"][0] - center[0]) for s in segment]
                    changes = [(b - a + math.pi) % (2 * math.pi) - math.pi for a, b in zip(angles, angles[1:])]
                    winding += sum(changes)
                    reverse += sum(max(0, -change * direction) for change in changes)
                heading_errors = []
                for s in samples:
                    p = s["position"]
                    desired = math.atan2(center[0] - p[0], -(center[1] - p[1]))
                    heading_errors.append(abs((s["yaw"] - desired + math.pi) % (2 * math.pi) - math.pi))
                groups.append(dict(laps=winding / (2 * math.pi), samples=len(samples),
                                   reverse_laps=reverse / (2 * math.pi),
                                   completed=track["completed"], execution_ids=track["execution_ids"],
                                   tracking_distance_m=sum(_track_distance(segment) for segment in track["segments"]),
                                   heading_error_max=max(heading_errors, default=math.pi),
                                   boundary_error_max_m=max((_boundary_error(s["position"], world[req["target"]].get("boundary"))
                                                             for s in samples), default=math.inf),
                                   time=track["time"], start=samples[0]["t"] if samples else None,
                                   start_position=samples[0]["position"] if samples else None,
                                   end_position=samples[-1]["position"] if samples else None))
            wanted = req["laps"] * direction
            measured_laps = sum(g["laps"] for g in groups)
            # One program can express two laps as one motion, two consecutive
            # motions or a repeat body. Accept the same measured physical path.
            active = [g for g in groups if g["completed"] or g["tracking_distance_m"] > .02]
            uncredited = sum(g["tracking_distance_m"] for g in active if not g["completed"])
            valid = bindings_valid and bool(active) and all(g["completed"] and g["samples"] >= 2 and g["laps"] * wanted > 0
                                        and g["boundary_error_max_m"] <= 1.0
                                        and (req.get("heading") != "face_center" or g["heading_error_max"] <= .2)
                                        for g in active)
            continuous = all(metric_distance(a[-1]["position"], b[0]["position"]) <= 1.0
                             for a, b in zip(runs, runs[1:]))
            passed = (valid and continuous and sum(g["samples"] for g in groups) >= 20
                      and sum(g["reverse_laps"] for g in groups) <= req.get("tolerance_laps", .06)
                      and abs(measured_laps - wanted) <= req.get("tolerance_laps", .06))
            measured = dict(executions=groups, expected_signed_laps=wanted, measured_signed_laps=measured_laps,
                            uncredited_tracking_distance_m=uncredited,
                            total_tracking_distance_m=sum(_track_distance(run) for run in runs),
                            after_update=req.get("after_update", 0), before_update=req.get("before_update"))
            if groups:
                first = min((group["start"] for group in groups if group["start"] is not None), default=None)
                last = max(group["time"] for group in groups)
        elif kind in {"goal", "pose"}:
            point = list(world[req["target"]][req.get("anchor", "center")])
            if req["state"] == "landed":
                point[2] = 0.0
            if "altitude_m" in req:
                point[2] = req["altitude_m"] / SCALE[2]
            # Pose obligations describe measured behavior, not planner node boundaries.
            # A sequence searches only after the preceding obligation finished.
            runs, held = [], []
            # Virtual sampling permits at most 0.1 s; realtime polls at 0.02 s.
            # A missing interval cannot prove continuous physical dwell.
            max_sample_gap = .1 + 1e-6
            previous_active = False
            terminal_matches = False
            for sample in telemetry:
                yaw_ok = ("yaw_deg" not in req or abs((sample["yaw"] - math.radians(req["yaw_deg"]) + math.pi)
                                                     % (2 * math.pi) - math.pi) <= .06)
                contact = sample["position"][2] * SCALE[2] <= .05
                if sample.get("runtime") == "gazebo" and req["state"] == "landed":
                    contact = contact and sample.get("landed") is True and sample.get("armed") is False
                matches = (sample["t"] >= after and metric_distance(sample["position"], point) <= .22
                           and sample["speed_mps"] <= .03 and yaw_ok
                           and (req["state"] == "landed") == contact)
                terminal_matches = matches
                active = bool(sample.get("execution_id") and sample.get("action"))
                if matches and active:
                    if held and (not previous_active or not 0 < sample["t"] - held[-1]["t"] <= max_sample_gap):
                        runs.append(held)
                        held = []
                    held.append(sample)
                elif matches and kind == "goal" and not active:
                    # Terminal telemetry can confirm the final pose, but earns
                    # no dwell after the last executing action.
                    previous_active = False
                    continue
                elif held:
                    runs.append(held)
                    held = []
                previous_active = matches and active
            if kind == "pose":
                if held:
                    runs.append(held)
                qualifying = [run for run in runs
                              if run[-1]["t"] - run[0]["t"] + .11 >= req.get("dwell_seconds", 0)]
                index = req.get("occurrence", 1) - 1
                held = qualifying[index] if index < len(qualifying) else []
                # Credit only the earliest sufficient prefix; later obligations
                # may use the rest of the same uninterrupted physical hold.
                if held:
                    end = next((i for i, sample in enumerate(held)
                                if sample["t"] - held[0]["t"] + 1e-6 >= req.get("dwell_seconds", 0)),
                               len(held) - 1)
                    held = held[:end + 1]
            dwell = held[-1]["t"] - held[0]["t"] if held else 0.0
            passed = bool(held) and dwell + .11 >= req.get("dwell_seconds", 0)
            if kind == "goal" and req.get("dwell_seconds", 0) == 0:
                # A final-state obligation does not require an active program
                # node: PX4 may disarm after the completed action was cleared.
                # Only the final measured pose counts; an earlier matching pose
                # cannot hide a later departure. Idle time still earns no dwell.
                passed = bool(telemetry) and terminal_matches
                if passed:
                    first = last = telemetry[-1]["t"]
            elif held:
                first, last = held[0]["t"], held[-1]["t"]
            measured = dict(dwell_seconds=dwell, target_position=point,
                            final_position=telemetry[-1]["position"] if telemetry else None)
        return dict(path=path, type=kind, satisfied=passed, measured=measured, first=first, last=last)

    return check(requirement, "oracle")


class _Journal:
    dropped = errors = 0

    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(deepcopy(record))


class _BudgetedAgent(DeepSeekAgent):
    def __init__(self, *, replies=None, requirements=None, audits=None, max_calls=6, deadline=math.inf, **kwargs):
        super().__init__(**kwargs)
        self._replies = deque(deepcopy(replies)) if replies is not None else None
        self._requirements = deque(deepcopy(requirements or []))
        self._audits = deque(deepcopy(audits or []))
        self.max_calls, self.deadline, self.calls = max_calls, deadline, 0
        self.stage_calls = Counter()
        self.budget_exhausted = False
        self._budget_lock = threading.Lock()
        if self._replies is not None:
            self.model = "offline-scripted-responses"

    def _take_budget(self, stage):
        with self._budget_lock:
            if self.calls >= self.max_calls or time.monotonic() >= self.deadline:
                self.budget_exhausted = True
                raise PlannerProviderError("Benchmark interpretation budget exhausted.", retryable=False)
            self.calls += 1
            self.stage_calls[stage] += 1

    @staticmethod
    def _fixture(queue, stage):
        if not queue:
            raise PlannerProviderError("Offline " + stage + " fixture exhausted.", retryable=False)
        return json.dumps(queue.popleft())

    def _request_json_content(self, messages, result_kind, **kwargs):
        # Intercept only the provider boundary: offline runs use the same source
        # filtering, parsers and bounded correction diagnostics as real calls.
        stage = {"task requirements": "requirements", "requirements audit": "audit",
                 "semantic task plan": "plan"}[result_kind]
        self._take_budget(stage)
        if self._replies is not None:
            queue = {"requirements": self._requirements, "audit": self._audits, "plan": self._replies}[stage]
            return self._fixture(queue, stage)
        return super()._request_json_content(messages, result_kind, **kwargs)

    def plan(self, *args, **kwargs):
        # Campus routes must use the local planner; never accidentally issue an
        # extra remote navigation request outside the interpretation call budget.
        raise PlannerProviderError("Campus benchmark forbids remote route generation.", retryable=False)


def _world(scenario):
    objects = {o["id"]: o for o in spatial_catalog(scenario)["objects"]}
    output = {}
    for poi in scenario.points_of_interest:
        arrival = [poi["position"][axis] for axis in "xyz"]
        center = list(objects.get(poi["id"], {}).get("center_grid", arrival))
        if len(center) == 2:
            center.append(arrival[2])
        output[poi["id"]] = dict(arrival=arrival, center=center,
                                 boundary=deepcopy(objects.get(poi["id"], {}).get("boundary_grid", [])))
    output["home"] = dict(arrival=list(scenario.base), center=list(scenario.base))
    return output


def _sample(service):
    with service._lock:
        vehicle = service._vehicle
        if vehicle is None:
            return None  # A failed native startup must not invent a ground-truth pose.
        step = service.semantic.current_step
        return dict(t=service._time, position=list(vehicle.position), speed_mps=norm(vehicle.velocity),
                    runtime=service.runtime.backend_name,
                    landed=service._runtime_status.get("landed"), armed=service._runtime_status.get("armed"),
                    mission_id=service.semantic.mission_id, vehicle_id=service.semantic.vehicle_id,
                    yaw=vehicle.yaw, execution_id=step["execution_id"] if step else None,
                    phase=service._motion_phase, action=step["action"] if step else None,
                    target_id=step.get("target_id") if step else None)


def _measurement(service):
    """Copy the evaluation view, not the large browser-rendering scene each tick."""
    with service._lock:
        snapshot = service._snapshot
    # Published snapshots are immutable, as in service.state(). Copying the
    # evidence outside the lock keeps scoring from delaying native control.
    return deepcopy({key: snapshot[key] for key in ("status", "semantic_mission", "flight", "milestones", "metrics")})


class _DelayedRuntime:
    """Benchmark-only actuator latency; production scheduling/protection stay intact."""
    def __init__(self, runtime, delay):
        self.runtime, self.delay = runtime, delay

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def drive(self, *args, **kwargs):
        time.sleep(self.delay)
        return self.runtime.drive(*args, **kwargs)


def recovery_metrics(requirement, evidence, events, mission_id, *, condition_evaluations=None,
                     condition_evaluations_dropped=0, telemetry=()):
    """Count excess successful actions after review against independent exact counts.

    Required repetitions and useful retries with no successful evidence are not
    duplicates. Open/minimum-count requirements deliberately have no upper cap.
    """
    caps = {}
    telemetry = [row for row in telemetry if row.get("mission_id", mission_id) == mission_id
                 and row.get("vehicle_id", "uav-1") == "uav-1"]

    def collect(req):
        if req["type"] in {"all", "sequence"}:
            for child in req["items"]:
                collect(child)
        elif req["type"] == "if":
            collect(req["observation"])
            rows = _evidence(evidence, req["observation"], mission_id)
            reading, _ = (_condition_observation(evidence, rows[-1], mission_id, req["field"],
                condition_evaluations, occurrence=req.get("decision_occurrence", 1),
                dropped=condition_evaluations_dropped) if rows else (None, None))
            if reading is not None and req["field"] in reading["result"]:
                branch = "then" if _compare(reading["result"][req["field"]], req["op"], req["value"]) else "else"
                collect(req[branch])
        elif req["type"] == "evidence":
            key = req["action"], req["target"]
            if "min_count" in req:
                caps[key] = None
            elif key not in caps or caps[key] is not None:
                caps[key] = max(caps.get(key, 0), req.get("occurrence", req.get("count", 1)))

    collect(requirement)
    reviews = [event for event in events if event.get("event") == "semantic_execution_feedback"
               and event.get("feedback", {}).get("kind") == "review_requested"]
    after = min((e["simulation_seconds"] for e in reviews if e.get("simulation_seconds") is not None), default=math.inf)
    duplicates = []
    for (action, target), cap in caps.items():
        if cap is None:
            continue
        rows = _evidence(evidence, dict(action=action, target=target), mission_id)
        duplicates.extend(dict(action=action, target_id=target, execution_id=row["execution_id"],
                               simulation_time=row["simulation_time"])
                          for row in rows[cap:] if row["simulation_time"] >= after)
    uncredited_distance = tracked_distance = 0.0
    targets = {row["target_id"] for row in evidence if row.get("action") == "motion"}
    targets.update(sample.get("target_id") for sample in telemetry if sample.get("action") == "motion")
    for target in targets:
        completed = _evidence(evidence, dict(action="motion", target=target), mission_id)
        tracks, runs, _ = _motion_tracks(completed, telemetry, target)
        tracked_distance += sum(_track_distance(run) for run in runs)
        uncredited_distance += sum(_track_distance(segment) for track in tracks if not track["completed"]
                                   for segment in track["segments"])
    return dict(review_requests=len(reviews), repeated_completed_actions=len(duplicates),
                duplicate_evidence=duplicates,
                total_tracking_distance_m=tracked_distance, uncredited_tracking_distance_m=uncredited_distance,
                duplicate_definition="successful actions beyond independently annotated exact counts after review")


def _validate_prices(prices):
    if prices is not None:
        if (not isinstance(prices, dict) or set(prices) != {"input_miss", "input_hit", "output", "currency"}
                or not isinstance(prices["currency"], str) or not prices["currency"]):
            raise ValueError("Prices require input_miss, input_hit, output and currency")
        for field in ("input_miss", "input_hit", "output"):
            _number(prices[field], field)


def _usage_cost(diagnostics, prices, *, offline):
    usage = diagnostics["usage"]
    complete = not offline and not usage["usage_missing"] and not usage["unreported_requests"]
    result = dict(usage=usage, complete_usage=complete, estimated_cost=None, currency=None)
    if prices is not None:
        result["currency"] = prices["currency"]
        # Missing usage stays unknown, never becomes a misleading zero-cost run.
        if complete:
            result["estimated_cost"] = (usage["prompt_cache_miss_tokens"] * prices["input_miss"]
                + usage["prompt_cache_hit_tokens"] * prices["input_hit"]
                + usage["completion_tokens"] * prices["output"]) / 1_000_000
    return result


def run_case(case, *, mode="virtual", backend="offline", seed=17, max_calls=6,
             simulation_budget=600.0, wall_budget=300.0, planner_timeout=30.0,
             traffic="none", provider=None, policy="contract",
             control_dt=MissionService.CONTROL_DT, control_delay_ms=0.0, prices=None,
             world_mode="fixture", planner_latency=0.0, runtime_backend="html"):
    """Run an isolated MissionService, never the browser's Flask instance.

    Budgets apply per case. Every selected case yields a record, including
    clarification, timeout and provider/generation failures. A blocked HTTP
    call can finish after the wall deadline, bounded by provider timeout; its
    late result is never counted as in-budget success.
    """
    validate_requirement(case["oracle"])
    if mode not in {"virtual", "realtime"} or backend not in {"offline", "deepseek"}:
        raise ValueError("Unsupported benchmark mode/backend")
    if backend == "deepseek" and mode != "realtime":
        raise ValueError("Model inference requires explicit mode=realtime and backend=deepseek")
    if runtime_backend not in {"html", "gazebo"}:
        raise ValueError("Unsupported execution runtime")
    if runtime_backend == "gazebo" and (mode != "realtime" or traffic != "none"):
        raise ValueError("Gazebo requires realtime mode and traffic=none; native dynamic traffic is not implemented")
    if type(max_calls) is not int or not 1 <= max_calls <= 30:
        raise ValueError("max_calls must be an integer in [1, 30]")
    for name, value in (("simulation_budget", simulation_budget), ("wall_budget", wall_budget), ("planner_timeout", planner_timeout)):
        _number(value, name, positive=True)
    if type(seed) is not int or traffic not in {"none", "fixed"}:
        raise ValueError("Invalid seed or traffic policy")
    if policy not in SEMANTIC_POLICIES:
        raise ValueError("Unknown semantic policy")
    if world_mode not in {"fixture", "seeded"}:
        raise ValueError("Task world must use fixture values or seeded generation")
    if "physics" in case:
        raise ValueError("The retired physics settings cannot be applied to HTML or Gazebo.")
    _number(control_dt, "control_dt", positive=True)
    _number(control_delay_ms, "control_delay_ms")
    _number(planner_latency, "planner_latency")
    if control_delay_ms > 1000:
        raise ValueError("Control-delay stress is bounded to 1000 ms per actuator call")
    if control_dt > .1 or (mode == "realtime" and control_dt != MissionService.CONTROL_DT):
        raise ValueError("Virtual control_dt must be <= .1; realtime uses the production control period")
    if control_delay_ms and mode != "realtime":
        raise ValueError("Control-delay stress is a separate realtime experiment")
    if planner_latency and mode != "virtual":
        raise ValueError("Explicit planner latency belongs to virtual experiments; realtime measures actual latency")
    _validate_prices(prices)
    started = time.monotonic()
    config = dict(api_key="offline-benchmark-placeholder")
    if backend == "deepseek":
        config = load_planner_config() if provider is None else dict(provider)
    config["timeout_seconds"] = min(planner_timeout, wall_budget)
    agent = _BudgetedAgent(replies=case["offline_responses"] if backend == "offline" else None,
                           requirements=case.get("offline_requirements"), audits=case.get("offline_audits"),
                           max_calls=max_calls, deadline=started + wall_budget, **config)
    if backend == "deepseek" and not agent.configured:
        raise ValueError("Configure the planner model and endpoint; non-local endpoints also require an API key")
    secret = agent.api_key
    scenario = load_scenario("semantic_campus")
    if traffic == "none":
        scenario = replace(scenario, dynamic_objects=())
    journal, telemetry = _Journal(), []
    session = None
    if runtime_backend == "gazebo":
        from .gazebo_session import GazeboSession
        session = GazeboSession(scenario, seed=seed)
        runtime = session.runtime
    else:
        runtime = HtmlRuntime()
        runtime.reset(scenario.base)
    runtime_description = runtime.status()
    if control_delay_ms:
        runtime = _DelayedRuntime(runtime, control_delay_ms / 1000)
    from .task_observation import TaskObservationConfig, TaskObservationWorld
    observation = dict(task_observation_config=TaskObservationConfig(seed=seed, **case.get("task_observation", {})))
    if world_mode == "seeded":
        observation["task_world"] = TaskObservationWorld.generate(scenario, seed)
    elif "task_world" in case:
        observation["task_world"] = TaskObservationWorld.from_dict(case["task_world"])
    else:
        raise ValueError("Fixture task world is absent; select seeded generation explicitly")
    service = MissionService(scenario=scenario, agent=agent, target_seed=seed,
                             traffic_seed=seed + 1, journal=journal, planner_latency_seconds=planner_latency,
                             semantic_policy=policy, runtime=runtime, **observation)
    world = _world(scenario)
    applied_updates = []
    user_update_index = 0
    status, reason = "timeout", "simulation_budget"
    pending_updates = deque(case.get("updates", []))
    try:
        try:
            if session:
                session.start(timeout=min(180, wall_budget))
                runtime_description = runtime.status()
                service.advance(control_dt)  # Initialize from native pose before interpreting language.
            service.submit_task(case["instruction"], background=mode == "realtime")
            while True:
                state = _measurement(service)
                sample = _sample(service)
                sample["user_update_index"] = user_update_index
                if not telemetry or sample["t"] > telemetry[-1]["t"]:
                    telemetry.append(sample)
                semantic = state["semantic_mission"]
                decision_context = dict(condition_evaluations=semantic.get("condition_evaluations"),
                    condition_evaluations_dropped=semantic.get("condition_evaluations_dropped", 0))
                elapsed = time.monotonic() - started
                if elapsed >= wall_budget:
                    reason = "wall_budget"
                    break
                if pending_updates:
                    update = pending_updates[0]
                    trigger = evaluate_oracle(update["after"], evidence=semantic["evidence"], telemetry=telemetry,
                                              world=world, mission_id=semantic["mission_id"], **decision_context)
                    if trigger["satisfied"]:
                        service.submit_task(update["instruction"], background=mode == "realtime")
                        user_update_index += 1
                        applied_updates.append(dict(instruction=update["instruction"], simulation_time=sample["t"],
                                                    user_update_index=user_update_index))
                        pending_updates.popleft()
                        continue
                if state["status"] == "completed":
                    status, reason = "completed", "service_completed"
                    break
                if semantic["status"] in TERMINAL_FAILURES or state["flight"]["error"]:
                    status = "clarification" if semantic["status"] == "needs_clarification" else "failed"
                    reason = "call_budget" if agent.budget_exhausted else semantic["status"]
                    break
                if sample["t"] >= simulation_budget:
                    break
                if mode == "realtime":
                    time.sleep(.02)
                else:
                    future = service._future
                    if future is not None:
                        try:
                            future.result(timeout=min(planner_timeout, max(.01, wall_budget - elapsed)))
                        except FutureTimeout:
                            if not future.done():
                                reason = "local_planner_timeout"
                                break
                        except Exception:
                            pass  # Service classifies the completed request on advance.
                    service.advance(min(control_dt, simulation_budget - service._time))
        except Exception as exc:
            status, reason = "failed", "runtime_startup" if session and service._vehicle is None else "runtime_error"
            service.trace.event("benchmark_runtime_failed", service._time, error_type=type(exc).__name__)
        with service._lock:
            service._shutdown.set()
            state = _measurement(service)
            endpoint = _sample(service)
            if endpoint is not None:
                endpoint["user_update_index"] = user_update_index
                if not telemetry or endpoint["t"] > telemetry[-1]["t"]:
                    telemetry.append(endpoint)
            semantic = state["semantic_mission"]
            decision_context = dict(condition_evaluations=semantic.get("condition_evaluations"),
                condition_evaluations_dropped=semantic.get("condition_evaluations_dropped", 0))
            oracle = evaluate_oracle(case["oracle"], evidence=semantic["evidence"], telemetry=telemetry,
                                     world=world, mission_id=semantic["mission_id"], **decision_context)
            service_success = bool(state["milestones"].get("task_completed"))
            violations = state["metrics"].get("collision_or_separation_events", 0)
            safe_execution = not violations and bool(state["milestones"].get("static_path_unblocked"))
            success = status == "completed" and oracle["satisfied"] and safe_execution and not pending_updates
            diagnostics = service.trace.snapshot()
            result = dict(case_id=case["id"], split=case["split"], combination=case["combination"],
                          instruction=case["instruction"], updates=applied_updates, status=status,
                          termination_reason=reason, oracle_success=success, requirements_satisfied=oracle["satisfied"],
                          service_success=service_success, false_completion=service_success and not oracle["satisfied"],
                          safe_execution=safe_execution,
                          safety_violations=violations,
                          static_path_blocked=not bool(state["milestones"].get("static_path_unblocked")),
                          oracle=oracle, evidence=semantic["evidence"], telemetry=telemetry,
                          observations=semantic.get("task_observations", []),
                          **decision_context,
                          simulation_seconds=service._time, wall_seconds=time.monotonic() - started,
                          metrics=state["metrics"], diagnostics=diagnostics,
                          runtime_final=runtime.status(),
                          simulator_logs=str(session.log_dir) if session and session.log_dir else None,
                          recovery=recovery_metrics(case["oracle"], semantic["evidence"], journal.records, semantic["mission_id"], telemetry=telemetry, **decision_context),
                          cost=_usage_cost(diagnostics, prices, offline=backend == "offline"),
                          interpretation_calls=agent.calls,
                          interpretation_stage_calls=dict(agent.stage_calls),
                          regression_pass=service_success and success,
                          protocol=dict(benchmark_version=BENCHMARK_VERSION, task_protocol_version=PROTOCOL_VERSION,
                                        case_sha256=hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                                        source_sha256=LOADED_SOURCE_FINGERPRINT,
                                        source_unchanged=LOADED_SOURCE_FINGERPRINT == service._source_version == source_fingerprint(),
                                        mode=mode, backend=backend,
                                        policy=policy, runtime=runtime_description["backend"],
                                        requirement_source="independent_user_requirements" if policy == "contract" else "plan_derived_baseline",
                                        model=agent.model, seed=seed, traffic_seed=seed + 1, traffic=traffic,
                                        simulation_budget=simulation_budget, wall_budget=wall_budget,
                                        max_calls=max_calls, planner_timeout=planner_timeout,
                                        planner_latency=planner_latency,
                                        control_dt=control_dt, control_delay_ms=control_delay_ms,
                                        task_observation=case.get("task_observation", {}),
                                        task_world_mode=world_mode,
                                        runtime_model=runtime_description.get("model_version", "kinematic"),
                                        runtime_config=runtime_description.get("config", {}),
                                        simulator=session.manifest if session else None,
                                        sensor_source="simulated task sensors",
                                        pose_source=runtime_description.get("position_source", "html_kinematic_integration"),
                                        camera_source=runtime_description.get("onboard_source", "browser_rgbd"),
                                        reset_policy="fresh_process_and_px4_data_per_case" if session else "fresh_in_memory_runtime",
                                        oracle_source="independent engineering fixtures (agent-assisted; not human-reviewed)"))
    finally:
        try:
            service.close()
        finally:
            if session:
                session.close()
    result["events"] = journal.records
    return redact(result, (secret,))


def summarize(results):
    count = len(results)
    successes = sum(bool(r["oracle_success"]) for r in results)
    service_successes = sum(bool(r["service_success"]) for r in results)
    false_completions = sum(bool(r["false_completion"]) for r in results)
    costs = [r.get("cost", {}) for r in results]
    currencies = {c.get("currency") for c in costs}
    priced = bool(costs) and len(currencies) == 1 and all(c.get("estimated_cost") is not None for c in costs)
    return dict(runs=count, successes=successes, success_rate=successes / count if count else None,
                service_successes=service_successes,
                false_completions=false_completions,
                false_completion_rate=false_completions / count if count else None,
                false_claim_rate=false_completions / service_successes if service_successes else None,
                regression_passes=sum(bool(r.get("regression_pass")) for r in results),
                outcomes=dict(Counter(r["status"] for r in results)),
                total_interpretation_calls=sum(r["interpretation_calls"] for r in results),
                total_simulation_seconds=sum(r.get("simulation_seconds", 0) for r in results),
                total_wall_seconds=sum(r["wall_seconds"] for r in results),
                safety_violations=sum(r.get("safety_violations", 0) for r in results),
                static_blocked_runs=sum(bool(r.get("static_path_blocked")) for r in results),
                repeated_completed_actions=sum(r.get("recovery", {}).get("repeated_completed_actions", 0) for r in results),
                uncredited_tracking_distance_m=sum(r.get("recovery", {}).get("uncredited_tracking_distance_m", 0) for r in results),
                total_estimated_cost=sum(c["estimated_cost"] for c in costs) if priced else None,
                cost_currency=next(iter(currencies)) if priced else None,
                complete_usage=bool(costs) and all(c.get("complete_usage", False) for c in costs),
                note="Descriptive finite-suite results; offline fixtures do not measure language-model ability; no population CI implied.")


def compare_policies(results, *, baseline="single_pass"):
    """Pair by independently specified case + seed, rejecting changed conditions.

    Unmatched runs are exposed rather than silently removed. Finite public
    fixtures are not independent samples of arbitrary language understanding.
    """
    if baseline not in SEMANTIC_POLICIES:
        raise ValueError("Unknown baseline")
    grouped = {}
    for row in results:
        protocol = row["protocol"]
        policy = protocol["policy"]
        key = row["case_id"], protocol["seed"]
        group = grouped.setdefault(policy, {})
        if key in group:
            raise ValueError("Duplicate case/seed for policy")
        if not protocol["source_unchanged"]:
            raise ValueError("Source changed during an experiment")
        group[key] = row
    if baseline not in grouped:
        raise ValueError("Baseline has no results")
    # Retain the retired physics field when comparing historical result files.
    invariant_fields = ("benchmark_version", "task_protocol_version", "case_sha256", "source_sha256",
        "mode", "backend", "runtime", "model", "seed", "traffic_seed", "traffic", "simulation_budget",
        "wall_budget", "max_calls", "planner_timeout", "control_dt", "control_delay_ms", "task_observation", "physics",
        "runtime_model", "runtime_config", "task_world_mode", "planner_latency", "simulator", "reset_policy")
    comparisons = {}
    before = grouped[baseline]
    for policy, after in grouped.items():
        if policy == baseline:
            continue
        paired, recovered, regressed = [], [], []
        deltas = {name: [] for name in ("interpretation_calls", "simulation_seconds", "wall_seconds",
                                       "false_completion", "safety_violations", "repeated_completed_actions",
                                       "uncredited_tracking_distance_m", "total_tracking_distance_m")}
        for key in sorted(before.keys() & after.keys()):
            a, b = before[key], after[key]
            mismatches = [field for field in invariant_fields if a["protocol"].get(field) != b["protocol"].get(field)]
            if mismatches:
                raise ValueError("Unpaired experiment conditions: " + ", ".join(mismatches))
            identity = dict(case_id=key[0], seed=key[1])
            paired.append(identity)
            if not a["oracle_success"] and b["oracle_success"]:
                recovered.append(identity)
            if a["oracle_success"] and not b["oracle_success"]:
                regressed.append(identity)
            for metric in deltas:
                recovery_metric = metric in {"repeated_completed_actions", "uncredited_tracking_distance_m", "total_tracking_distance_m"}
                source_a = a.get("recovery", {}) if recovery_metric else a
                source_b = b.get("recovery", {}) if recovery_metric else b
                deltas[metric].append(float(source_b.get(metric, 0)) - float(source_a.get(metric, 0)))
        comparisons[policy] = dict(paired_runs=len(paired), recovered=recovered, regressed=regressed,
            unmatched_baseline=[dict(case_id=k[0], seed=k[1]) for k in sorted(before.keys() - after.keys())],
            unmatched_candidate=[dict(case_id=k[0], seed=k[1]) for k in sorted(after.keys() - before.keys())],
            mean_paired_delta={k: sum(v) / len(v) if v else None for k, v in deltas.items()})
    return dict(baseline=baseline, summaries={policy: summarize(list(rows.values())) for policy, rows in grouped.items()},
                comparisons=comparisons, note="Paired finite-suite contrasts; positive deltas mean candidate minus baseline.")
