"""Independent, evidence-grounded campus task evaluation.

The oracle is human-authored data, never a model program. Offline responses test
the execution/evaluation pipeline only; remote interpretation is opt-in.
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
from .mission_service import MissionService
from .motion import norm
from .run_log import source_fingerprint
from .safety import SCALE, metric_distance
from .scenarios import load_scenario
from .semantic_tasks import semantic_plan_from_json_text
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
        "if": {"type", "observation", "field", "op", "value", "then", "else"},
        "circuit": {"type", "target", "direction", "laps", "tolerance_laps", "heading"},
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
        validate_requirement(req.get("observation"), depth + 1)
        if req["observation"]["type"] != "evidence":
            raise ValueError("Condition observation must select measured evidence")
        for arm in ("then", "else"):
            validate_requirement(req.get(arm), depth + 1)
    else:
        if not isinstance(req.get("target"), str) or not req["target"]:
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
            if type(req.get("laps")) is not int or req["laps"] < 1:
                raise ValueError("Circuit laps must be positive integers")
            _number(req.get("tolerance_laps", .06), "tolerance_laps")
            if req.get("heading") not in {None, "face_center"}:
                raise ValueError("Unsupported heading oracle")
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
        if type(case.get("offline_expected_success", True)) is not bool:
            raise ValueError("Expected offline outcome must be boolean")
        for update in case.get("updates", []):
            validate_requirement(update.get("after"))
            if not isinstance(update.get("instruction"), str) or not update["instruction"]:
                raise ValueError("Task update requires an instruction")
    return data


def _evidence(records, req, mission_id):
    selected, seen = [], set()
    for row in records:
        eid = row.get("execution_id")
        if (not eid or eid in seen or row.get("mission_id") != mission_id
                or row.get("vehicle_id") != "uav-1"):
            continue
        if row.get("action") != req["action"] or row.get("target_id") != req["target"]:
            continue
        result = row.get("result", {})
        if (row.get("source") != _SOURCES[req["action"]]
                or result.get(_SUCCESS[req["action"]]) is not True):
            continue
        seen.add(eid)
        selected.append(row)
    selected.sort(key=lambda r: r["simulation_time"])
    if "occurrence" in req:
        selected = selected[req["occurrence"] - 1:req["occurrence"]]
    return [row for row in selected if row.get("duration_seconds", 0) + 1e-6 >= req.get("min_duration", 0)
            and all(row.get("result", {}).get(k) == v for k, v in req.get("result", {}).items())]


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


def evaluate_oracle(requirement, *, evidence, telemetry, world, mission_id):
    """Evaluate only measured evidence/poses and fixed scene geometry, never a plan.

    Telemetry positions use scene GRID units; speed is m/s, yaw radians, time
    simulation seconds. Clockwise has positive winding because scene y is south.
    """
    validate_requirement(requirement)

    def check(req, path):
        kind = req["type"]
        measured, first, last = {}, None, None
        if kind in {"all", "sequence"}:
            children = [check(child, f"{path}.{i}") for i, child in enumerate(req["items"])]
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
            available = observation["satisfied"] and bool(rows) and req["field"] in rows[-1]["result"]
            if available:
                value = rows[-1]["result"][req["field"]]
                branch = "then" if _compare(value, req["op"], req["value"]) else "else"
                child = check(req[branch], path + "." + branch)
                passed = child["satisfied"] and (child["first"] is None or child["first"] >= rows[-1]["simulation_time"])
                measured = dict(observed_value=value, selected_branch=branch, child=child)
                first, last = rows[-1]["simulation_time"], child["last"] or rows[-1]["simulation_time"]
            else:
                passed, measured = False, dict(reason="observation_missing")
        elif kind == "circuit":
            center = world[req["target"]]["center"]
            completions = _evidence(evidence, dict(action="motion", target=req["target"]), mission_id)
            groups = []
            for record in completions:
                samples = [s for s in telemetry if s.get("execution_id") == record["execution_id"]
                           and s.get("phase") == "tracking"]
                angles = [math.atan2(s["position"][1] - center[1], s["position"][0] - center[0]) for s in samples]
                winding = sum((b - a + math.pi) % (2 * math.pi) - math.pi for a, b in zip(angles, angles[1:]))
                heading_errors = []
                for s in samples:
                    p = s["position"]
                    desired = math.atan2(center[0] - p[0], -(center[1] - p[1]))
                    heading_errors.append(abs((s["yaw"] - desired + math.pi) % (2 * math.pi) - math.pi))
                groups.append(dict(laps=winding / (2 * math.pi), samples=len(samples),
                                   heading_error_max=max(heading_errors, default=math.pi),
                                   boundary_error_max_m=max((_boundary_error(s["position"], world[req["target"]].get("boundary"))
                                                             for s in samples), default=math.inf),
                                   time=record["simulation_time"], start=samples[0]["t"] if samples else None,
                                   start_position=samples[0]["position"] if samples else None,
                                   end_position=samples[-1]["position"] if samples else None))
            wanted = req["laps"] * (1 if req["direction"] == "clockwise" else -1)
            measured_laps = sum(g["laps"] for g in groups)
            # One program can express two laps as one motion, two consecutive
            # motions or a repeat body. Accept the same measured physical path.
            valid = bool(groups) and all(g["samples"] >= 2 and g["laps"] * wanted > 0
                                        and g["boundary_error_max_m"] <= 1.0
                                        and (req.get("heading") != "face_center" or g["heading_error_max"] <= .2)
                                        for g in groups)
            continuous = all(metric_distance(a["end_position"], b["start_position"]) <= 1.0
                             for a, b in zip(groups, groups[1:]) if a["end_position"] and b["start_position"])
            passed = (valid and continuous and sum(g["samples"] for g in groups) >= 20
                      and abs(measured_laps - wanted) <= req.get("tolerance_laps", .06))
            measured = dict(executions=groups, expected_signed_laps=wanted, measured_signed_laps=measured_laps)
            if groups:
                first, last = groups[0]["start"], groups[-1]["time"]
        elif kind in {"goal", "pose"}:
            point = list(world[req["target"]][req.get("anchor", "center")])
            if req["state"] == "landed":
                point[2] = 0.0
            if "altitude_m" in req:
                point[2] = req["altitude_m"] / SCALE[2]
            candidates = telemetry
            if kind == "pose":
                matching = _evidence(evidence, dict(action="control", target=req["target"],
                                     occurrence=req.get("occurrence", 1)), mission_id)
                candidates = [s for s in telemetry if matching and s.get("execution_id") == matching[0]["execution_id"]]
            held = []
            for sample in reversed(candidates):
                yaw_ok = ("yaw_deg" not in req or abs((sample["yaw"] - math.radians(req["yaw_deg"]) + math.pi)
                                                     % (2 * math.pi) - math.pi) <= .06)
                contact = sample["position"][2] * SCALE[2] <= .05
                if (metric_distance(sample["position"], point) > .22 or sample["speed_mps"] > .03 or not yaw_ok
                        or (req["state"] == "landed") != contact):
                    break
                held.append(sample)
            dwell = held[0]["t"] - held[-1]["t"] if held else 0.0
            passed = bool(held) and dwell + .11 >= req.get("dwell_seconds", 0)
            measured = dict(dwell_seconds=dwell, target_position=point,
                            final_position=telemetry[-1]["position"] if telemetry else None)
            if held:
                first, last = held[-1]["t"], held[0]["t"]
        return dict(path=path, type=kind, satisfied=passed, measured=measured, first=first, last=last)

    return check(requirement, "oracle")


class _Journal:
    dropped = errors = 0

    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(deepcopy(record))


class _BudgetedAgent(DeepSeekAgent):
    def __init__(self, *, replies=None, max_calls=6, deadline=math.inf, **kwargs):
        super().__init__(**kwargs)
        self._replies = deque(deepcopy(replies)) if replies is not None else None
        self.max_calls, self.deadline, self.calls = max_calls, deadline, 0
        self.budget_exhausted = False
        self._budget_lock = threading.Lock()
        if self._replies is not None:
            self.model = "offline-scripted-responses"

    def interpret_task(self, catalog, instruction, feedback="", *, repair=None):
        with self._budget_lock:
            if self.calls >= self.max_calls or time.monotonic() >= self.deadline:
                self.budget_exhausted = True
                raise PlannerProviderError("Benchmark interpretation budget exhausted.", retryable=False)
            self.calls += 1
        if self._replies is not None:
            if not self._replies:
                raise PlannerProviderError("Offline response fixture exhausted.", retryable=False)
            return semantic_plan_from_json_text(json.dumps(self._replies.popleft()), catalog)
        return super().interpret_task(catalog, instruction, feedback, repair=repair)

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
        step = service.semantic.current_step
        return dict(t=service._time, position=list(vehicle.position), speed_mps=norm(vehicle.velocity),
                    yaw=vehicle.yaw, execution_id=step["execution_id"] if step else None,
                    phase=service._motion_phase, action=step["action"] if step else None,
                    target_id=step.get("target_id") if step else None)


def _measurement(service):
    """Copy the evaluation view, not the large browser-rendering scene each tick."""
    with service._lock:
        snapshot = service._snapshot
        return deepcopy({key: snapshot[key] for key in ("status", "semantic_mission", "flight", "milestones", "metrics")})


def run_case(case, *, mode="virtual", backend="offline", seed=17, max_calls=6,
             simulation_budget=600.0, wall_budget=300.0, planner_timeout=30.0,
             traffic="none", provider=None):
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
        raise ValueError("Remote APIs require explicit mode=realtime and backend=deepseek")
    if type(max_calls) is not int or not 1 <= max_calls <= 30:
        raise ValueError("max_calls must be an integer in [1, 30]")
    for name, value in (("simulation_budget", simulation_budget), ("wall_budget", wall_budget), ("planner_timeout", planner_timeout)):
        _number(value, name, positive=True)
    if type(seed) is not int or traffic not in {"none", "fixed"}:
        raise ValueError("Invalid seed or traffic policy")
    started = time.monotonic()
    config = dict(provider or {}) if backend == "deepseek" else dict(api_key="offline-benchmark-placeholder")
    config["timeout_seconds"] = min(planner_timeout, wall_budget)
    agent = _BudgetedAgent(replies=case["offline_responses"] if backend == "offline" else None,
                           max_calls=max_calls, deadline=started + wall_budget, **config)
    if backend == "deepseek" and not agent.api_key:
        raise ValueError("Set DEEPSEEK_API_KEY in the environment for real API evaluation")
    secret = agent.api_key
    scenario = load_scenario("semantic_campus")
    if traffic == "none":
        scenario = replace(scenario, dynamic_objects=())
    journal, telemetry = _Journal(), []
    service = MissionService(scenario=scenario, agent=agent, target_seed=seed,
                             traffic_seed=seed + 1, journal=journal, planner_latency_seconds=0)
    world = _world(scenario)
    applied_updates = []
    status, reason = "timeout", "simulation_budget"
    pending_updates = deque(case.get("updates", []))
    try:
        try:
            service.submit_task(case["instruction"], background=mode == "realtime")
            while True:
                state = _measurement(service)
                sample = _sample(service)
                if not telemetry or sample["t"] > telemetry[-1]["t"]:
                    telemetry.append(sample)
                semantic = state["semantic_mission"]
                elapsed = time.monotonic() - started
                if elapsed >= wall_budget:
                    reason = "wall_budget"
                    break
                if pending_updates:
                    update = pending_updates[0]
                    trigger = evaluate_oracle(update["after"], evidence=semantic["evidence"], telemetry=telemetry,
                                              world=world, mission_id=semantic["mission_id"])
                    if trigger["satisfied"]:
                        service.submit_task(update["instruction"], background=mode == "realtime")
                        applied_updates.append(dict(instruction=update["instruction"], simulation_time=sample["t"]))
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
                    service.advance(min(.1, simulation_budget - service._time))
        except Exception as exc:
            status, reason = "failed", "runtime_error"
            service.trace.event("benchmark_runtime_failed", service._time, error_type=type(exc).__name__)
        with service._lock:
            service._shutdown.set()
            state = _measurement(service)
            endpoint = _sample(service)
            if not telemetry or endpoint["t"] > telemetry[-1]["t"]:
                telemetry.append(endpoint)
            semantic = state["semantic_mission"]
            oracle = evaluate_oracle(case["oracle"], evidence=semantic["evidence"], telemetry=telemetry,
                                     world=world, mission_id=semantic["mission_id"])
            service_success = bool(state["milestones"].get("safe_success"))
            violations = state["metrics"].get("collision_or_separation_events", 0)
            safe_execution = not violations and bool(state["milestones"].get("static_path_unblocked"))
            success = status == "completed" and oracle["satisfied"] and safe_execution and not pending_updates
            result = dict(case_id=case["id"], split=case["split"], combination=case["combination"],
                          instruction=case["instruction"], updates=applied_updates, status=status,
                          termination_reason=reason, oracle_success=success, requirements_satisfied=oracle["satisfied"],
                          service_success=service_success, false_completion=service_success and not oracle["satisfied"],
                          safe_execution=safe_execution,
                          oracle=oracle, evidence=semantic["evidence"], telemetry=telemetry,
                          simulation_seconds=service._time, wall_seconds=time.monotonic() - started,
                          metrics=state["metrics"], diagnostics=service.trace.snapshot(),
                          interpretation_calls=agent.calls,
                          expected_oracle_success=case.get("offline_expected_success", True) if backend == "offline" else True,
                          regression_pass=(status == "completed" and service_success
                                           and success == (case.get("offline_expected_success", True) if backend == "offline" else True)),
                          protocol=dict(benchmark_version=BENCHMARK_VERSION, task_protocol_version=PROTOCOL_VERSION,
                                        case_sha256=hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                                        source_sha256=LOADED_SOURCE_FINGERPRINT,
                                        source_unchanged=LOADED_SOURCE_FINGERPRINT == service._source_version == source_fingerprint(),
                                        mode=mode, backend=backend,
                                        model=agent.model, seed=seed, traffic_seed=seed + 1, traffic=traffic,
                                        simulation_budget=simulation_budget, wall_budget=wall_budget,
                                        max_calls=max_calls, planner_timeout=planner_timeout,
                                        control_dt=MissionService.CONTROL_DT if mode == "realtime" else .1,
                                        sensor_source="HTML simulated observations", oracle_source="human annotated requirements"))
    finally:
        service.close()
    result["events"] = journal.records
    return redact(result, (secret,))


def summarize(results):
    count = len(results)
    successes = sum(bool(r["oracle_success"]) for r in results)
    return dict(runs=count, successes=successes, success_rate=successes / count if count else None,
                service_successes=sum(bool(r["service_success"]) for r in results),
                false_completions=sum(bool(r["false_completion"]) for r in results),
                expected_negative_fixtures=sum(not r.get("expected_oracle_success", True) for r in results),
                regression_passes=sum(bool(r.get("regression_pass")) for r in results),
                outcomes=dict(Counter(r["status"] for r in results)),
                total_interpretation_calls=sum(r["interpretation_calls"] for r in results),
                total_wall_seconds=sum(r["wall_seconds"] for r in results),
                note="Descriptive finite-suite results; offline fixtures do not measure language-model ability; no population CI implied.")
