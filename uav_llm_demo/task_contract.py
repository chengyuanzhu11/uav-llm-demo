"""Preserve accepted task requirements across reviews and measured execution."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import operator
import uuid

from .task_protocol import TaskProtocolError


_MAX_EXPANDED_NODES = 4096
_MAX_MATCH_ATTEMPTS = 20000
_MOTION_HEADING_TOLERANCE_DEG = math.degrees(.2)
_OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
        "<=": operator.le, ">": operator.gt, ">=": operator.ge}
_ACTION_FIELDS = {
    "visit": ("target_id", "duration_seconds"),
    "inspect": ("target_id", "duration_seconds"),
    "thermal_scan": ("target_id", "duration_seconds"),
    "capture": ("target_id", "duration_seconds"),
    "return_home": ("target_id", "duration_seconds"),
    "motion": ("target_id", "motion"),
    "control": ("target_id", "control"),
    "review": ("instruction",),
}
_SOURCES = {"visit": "runtime_telemetry", "return_home": "runtime_telemetry",
            "control": "runtime_telemetry", "inspect": "simulated_inspection",
            "thermal_scan": "simulated_thermal_sensor", "capture": "simulated_capture",
            "motion": "runtime_trajectory_tracking", "review": "validated_execution_review"}


class TaskContractError(TaskProtocolError):
    """Bounded diagnostics feed the existing schema-correction channel."""

    def __init__(self, code, message, path="$.steps", **details):
        super().__init__(path, code, message, **details)


def _program(plan):
    data = plan.to_dict() if hasattr(plan, "to_dict") else plan
    if not isinstance(data, dict) or data.get("status") != "ready" or not data.get("steps"):
        raise TaskContractError("contract_program_required", "A contract requires an accepted ready program.")
    return deepcopy(data)


def _action_value(node):
    action = node.get("action")
    return {"action": action, **{key: deepcopy(node.get(key)) for key in _ACTION_FIELDS.get(action, ())}}


def _walk(nodes):
    for node in nodes:
        yield node
        if node["action"] == "repeat":
            yield from _walk(node["steps"])
        elif node["action"] == "branch":
            yield from _walk(node["then"])
            yield from _walk(node["else"])


def _finite(value, minimum=0):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= minimum)


def _ref_expression(expression, references):
    if expression is None:
        return None
    if "all" in expression or "any" in expression:
        key = "all" if "all" in expression else "any"
        return {key: [_ref_expression(item, references) for item in expression[key]]}
    if "not" in expression:
        return {"not": _ref_expression(expression["not"], references)}
    result = deepcopy(expression)
    if "step_id" in result and result["step_id"] in references:
        reference = references[result.pop("step_id")]
        result.update(reference)
    return result


def _fold_condition(expression, constants):
    """Fold proven review booleans; never invent a sensor or physical result."""
    if expression is None:
        return None
    if "all" in expression or "any" in expression:
        key = "all" if "all" in expression else "any"
        items = [_fold_condition(item, constants) for item in expression[key]]
        decisive = False if key == "all" else True
        if any(item is decisive for item in items):
            return decisive
        unresolved = [item for item in items if not isinstance(item, bool)]
        if not unresolved:
            return not decisive
        return unresolved[0] if len(unresolved) == 1 else {key: unresolved}
    if "not" in expression:
        inner = _fold_condition(expression["not"], constants)
        return not inner if isinstance(inner, bool) else {"not": inner}
    reference = next(((kind, expression[kind]) for kind in ("step_id", "evidence_id") if kind in expression), None)
    result = constants.get(reference, {})
    if expression.get("field") in result:
        return bool(_OPS[expression["op"]](result[expression["field"]], expression["value"]))
    return deepcopy(expression)


def _fold_review_conditions(nodes, constants):
    result = []
    for original in nodes:
        node = deepcopy(original)
        guard = _fold_condition(node.get("when"), constants)
        if guard is False:
            continue
        if "when" in node:
            node["when"] = None if guard is True else guard
        if node["action"] == "branch":
            predicate = _fold_condition(node["condition"], constants)
            if isinstance(predicate, bool):
                result.extend(_fold_review_conditions(node["then" if predicate else "else"], constants))
                continue
            node["condition"] = predicate
            node["then"] = _fold_review_conditions(node["then"], constants)
            node["else"] = _fold_review_conditions(node["else"], constants)
        elif node["action"] == "repeat":
            body = _fold_review_conditions(node["steps"], constants)
            # Keep a control-only loop representable rather than claiming an
            # unbounded empty loop completed. Its runtime budget remains active.
            node["steps"] = body or node["steps"]
            until = _fold_condition(node.get("until"), constants)
            if until is True:
                node["count"], node["until"] = 1, None
            else:
                node["until"] = None if until is False else until
        result.append(node)
    return result


def _normal_form(nodes, references=None, origins=None, *, expand_counts=True, constants=None):
    """Alpha-rename occurrences and expand only finite, unconditional count loops."""
    counter = [0]

    def walk(items, env):
        output = []
        for raw in items:
            counter[0] += 1
            if counter[0] > _MAX_EXPANDED_NODES:
                raise TaskContractError("contract_expansion_limit", "Task contract expansion exceeds its verification budget.")
            node = deepcopy(raw)
            old_id, new_id = node["id"], "requirement_" + str(counter[0])
            node["id"] = new_id
            if origins is not None:
                origins[new_id] = old_id
            action = node["action"]
            if (expand_counts and action == "repeat" and node.get("until") is None
                    and node.get("when") is None and node.get("count") is not None):
                for _ in range(node["count"]):
                    body, env = walk(node["steps"], env)
                    output.extend(body)
                continue
            if action == "branch":
                node["condition"] = _ref_expression(node["condition"], env)
                node["then"], _ = walk(node["then"], dict(env))
                node["else"], _ = walk(node["else"], dict(env))
            elif action == "repeat":
                node["when"] = _ref_expression(node.get("when"), env)
                node["steps"], body_env = walk(node["steps"], dict(env))
                node["until"] = _ref_expression(node.get("until"), body_env)
                if node.get("when") is None:
                    env.update(body_env)
            else:
                node["when"] = _ref_expression(node.get("when"), env)
                if node.get("when") is None:
                    env[old_id] = {"step_id": new_id}
            output.append(node)
        return output, env

    return walk(_fold_review_conditions(nodes, constants or {}), dict(references or {}))[0]


def _matches(expected, candidate):
    """Order-preserving structural inclusion with shared alpha-renamed references.

    No trailing action is admitted: reaching home and then leaving it would not
    preserve a terminal landing. Additional actions inside loops/branches are
    checked recursively, so no unselected arm is counted as a mandatory action.
    """
    attempts = [0]

    def without_reviews(nodes, terminal=True):
        result = []
        for index, item in enumerate(nodes):
            at_end = terminal and index == len(nodes) - 1
            if item["action"] == "review" and not at_end:
                continue
            node = deepcopy(item)
            if node["action"] == "repeat":
                node["steps"] = without_reviews(node["steps"], at_end)
            elif node["action"] == "branch":
                node["then"] = without_reviews(node["then"], at_end)
                node["else"] = without_reviews(node["else"], at_end)
            result.append(node)
        return result

    expected, candidate = without_reviews(expected), without_reviews(candidate)

    def node_match(left, right, bindings):
        attempts[0] += 1
        if attempts[0] > _MAX_MATCH_ATTEMPTS:
            raise TaskContractError("contract_matching_limit", "Task contract comparison exceeds its verification budget.")
        if left["action"] != right["action"]:
            return None
        mapping = {**bindings, left["id"]: {"step_id": right["id"]}}
        action = left["action"]
        if action == "branch":
            if _ref_expression(left["condition"], mapping) != right["condition"]:
                return None
            yes = sequence(left["then"], right["then"], mapping)
            no = sequence(left["else"], right["else"], mapping)
            return {**yes, **no} if yes is not None and no is not None else None
        if action == "repeat":
            if left.get("count") != right.get("count") or _ref_expression(left.get("when"), mapping) != right.get("when"):
                return None
            body_mapping = sequence(left["steps"], right["steps"], mapping)
            if body_mapping is None or _ref_expression(left.get("until"), body_mapping) != right.get("until"):
                return None
            return body_mapping
        if _action_value(left) != _action_value(right):
            return None
        if _ref_expression(left.get("when"), mapping) != right.get("when"):
            return None
        return mapping

    def sequence(left, right, bindings):
        pending = [(0, 0, dict(bindings))]
        while pending:
            li, ri, mapping = pending.pop()
            if li == len(left):
                if ri == len(right):
                    return mapping
                continue
            remaining = len(left) - li
            for ci in range(ri, len(right) - remaining + 1):
                updated = node_match(left[li], right[ci], mapping)
                if updated is not None:
                    pending.append((li, ci + 1, mapping))
                    pending.append((li + 1, ci + 1, updated))
                    break
        return None

    return sequence(expected, candidate, {})


class TaskContractMonitor:
    """Immutable accepted obligations plus a private evidence-driven interpreter.

    Construct a NEW instance for an explicit user revision. Autonomous changes
    must call validate_revision, complete/record the current review, and then
    accept_revision. All input programs must already pass the public DSL parser
    and local geometry/safety validation. ``record`` receives the actual runtime
    node and its newly created evidence, never model-supplied completion claims.
    """

    def __init__(self, plan, mission_id, vehicle_id="uav-1", revision=1, *,
                 telemetry=None, evidence=(), scenario=None):
        data = _program(plan)
        self.contract_id = uuid.uuid4().hex
        self.mission_id, self.vehicle_id, self.revision = mission_id, vehicle_id, revision
        self._constraints = deepcopy(data["constraints"])
        self._scenario = scenario
        self._stack = [{"kind": "sequence", "nodes": deepcopy(data["steps"]), "index": 0}]
        self._next = None
        self._telemetry = deepcopy(telemetry or {})
        self._step_evidence, self._evidence = {}, {}
        self._projected_review_results = {}
        self._seen_execution_ids, self._seen_evidence_ids = set(), set()
        self._matched_count = self._skipped_count = 0
        self._failures = []
        self._open_review = False
        self._binding = {node["id"]: deepcopy(node) for node in _walk(data["steps"])}
        self._obligation_binding = {node_id: {node_id} for node_id in self._binding}
        self._motion_geometry = {}
        self._cache_geometry(data["steps"])
        for item in evidence:
            if item.get("mission_id") == mission_id and item.get("vehicle_id") == vehicle_id:
                self._evidence[item["evidence_id"]] = deepcopy(item)

    def _evaluate(self, expression):
        if "all" in expression:
            return all(self._evaluate(item) for item in expression["all"])
        if "any" in expression:
            return any(self._evaluate(item) for item in expression["any"])
        if "not" in expression:
            return not self._evaluate(expression["not"])
        field = expression["field"]
        if expression.get("source") == "telemetry":
            values = self._telemetry
        else:
            if "step_id" in expression and expression["step_id"] in self._projected_review_results:
                values = self._projected_review_results[expression["step_id"]]
            else:
                item = (self._evidence.get(expression["evidence_id"]) if "evidence_id" in expression
                        else self._step_evidence.get(expression["step_id"]))
                values = item.get("result", {}) if item else {}
        if field not in values:
            raise TaskContractError("contract_missing_observation", "A required condition has no bound observation.")
        try:
            return _OPS[expression["op"]](values[field], expression["value"])
        except (KeyError, TypeError, ValueError):
            raise TaskContractError("contract_invalid_observation", "A required condition has an invalid observation.") from None

    def _advance(self, telemetry=None):
        if telemetry is not None:
            self._telemetry = deepcopy(telemetry)
        if self._next is not None:
            return
        transitions = 0
        while self._stack:
            transitions += 1
            if transitions > _MAX_EXPANDED_NODES:
                raise TaskContractError("contract_execution_limit", "Task contract cannot progress within its execution budget.")
            frame = self._stack[-1]
            if frame["kind"] == "loop":
                node = frame["node"]
                until = node.get("until") is not None and self._evaluate(node["until"])
                exhausted = node.get("count") is not None and frame["iteration"] >= node["count"]
                if until or exhausted:
                    self._stack.pop()
                else:
                    frame["iteration"] += 1
                    self._stack.append({"kind": "sequence", "nodes": node["steps"], "index": 0})
                continue
            if frame["index"] >= len(frame["nodes"]):
                self._stack.pop()
                continue
            node = frame["nodes"][frame["index"]]
            frame["index"] += 1
            action = node["action"]
            if action == "branch":
                selected = self._evaluate(node["condition"])
                self._stack.append({"kind": "sequence", "nodes": node["then" if selected else "else"], "index": 0})
                continue
            if node.get("when") is not None:
                selected = self._evaluate(node["when"])
                if not selected:
                    self._skipped_count += 1
                    continue
            if action == "repeat":
                self._stack.append({"kind": "loop", "node": node, "iteration": 1})
                self._stack.append({"kind": "sequence", "nodes": node["steps"], "index": 0})
                continue
            self._next = node
            return

    def _raw_remaining(self):
        result = []
        if self._next is not None:
            current = deepcopy(self._next)
            current["when"] = None  # This guard has already been selected using evidence.
            result.append(current)
        for frame in reversed(self._stack):
            if frame["kind"] == "sequence":
                result.extend(deepcopy(frame["nodes"][frame["index"]:]))
                continue
            node = deepcopy(frame["node"])
            count = node.get("count")
            remaining_count = None if count is None else count - frame["iteration"]
            if remaining_count == 0:
                continue
            node["count"], node["when"] = remaining_count, None
            if node.get("until") is not None:
                # The unfinished current body executes before the next until test.
                result.append({"id": node["id"] + "_continue", "action": "branch",
                               "condition": {"not": deepcopy(node["until"])}, "then": [node], "else": []})
            else:
                result.append(node)
        return result

    def _projection(self, telemetry, after_review):
        clone = deepcopy(self, {id(self._scenario): self._scenario})
        clone._advance(telemetry)
        if after_review:
            # A locally accepted auxiliary review may precede a required action.
            # It grants no authority to discard that pending action.
            if clone._next is not None and clone._next["action"] == "review":
                # This projection is conditional on a successful validated
                # review. It is not an evidence record and cannot satisfy a
                # physical obligation or leak an invented evidence identifier.
                clone._projected_review_results[clone._next["id"]] = {"reviewed": True}
                clone._next = None
                clone._advance(telemetry)
        return clone

    def _review_constants(self):
        values = {("step_id", key): result for key, result in self._projected_review_results.items()}
        for key, item in self._step_evidence.items():
            if item.get("action") == "review" and item.get("source") == _SOURCES["review"]:
                values[("step_id", key)] = {"reviewed": item["result"]["reviewed"]}
        for key, item in self._evidence.items():
            if item.get("action") == "review" and item.get("source") == _SOURCES["review"]:
                values[("evidence_id", key)] = {"reviewed": item["result"]["reviewed"]}
        return values

    def remaining_steps(self, telemetry=None, *, after_review=False):
        clone = self._projection(telemetry, after_review)
        references = {step_id: {"evidence_id": item["evidence_id"]}
                      for step_id, item in clone._step_evidence.items()}
        return _normal_form(clone._raw_remaining(), references, expand_counts=False,
                            constants=clone._review_constants())

    def validate_revision(self, plan, telemetry=None, *, after_review=True):
        data = _program(plan)
        if data["constraints"] != self._constraints:
            raise TaskContractError("contract_constraints_changed", "Autonomous revision must preserve the accepted hard constraints.", "$.constraints")
        clone = self._projection(telemetry, after_review)
        references = {step_id: {"evidence_id": item["evidence_id"]}
                      for step_id, item in clone._step_evidence.items()}
        expected_origins, candidate_origins = {}, {}
        expected = _normal_form(clone._raw_remaining(), references, expected_origins,
                                constants=clone._review_constants())
        candidate = _normal_form(data["steps"], origins=candidate_origins,
                                 constants=self._review_constants())
        open_continuation = not expected and (after_review or self._open_review)
        mapping = None if open_continuation else _matches(expected, candidate)
        if not open_continuation and mapping is None:
            raise TaskContractError("contract_obligation_changed",
                                    "The remaining program must preserve every required action, order, repetition, condition, path and goal state. Use contract.remaining_steps; IDs may change and auxiliary actions may be inserted before requirements.")
        bindings = {}
        for required, proposed in (mapping or {}).items():
            candidate_id = candidate_origins[proposed["step_id"]]
            bindings.setdefault(candidate_id, set()).add(expected_origins[required])
        return {"accepted": True, "contract_id": self.contract_id,
                "open_continuation": open_continuation, "remaining_steps": expected,
                "bindings": {key: sorted(value) for key, value in bindings.items()}}

    def accept_revision(self, plan, revision, telemetry=None):
        data = _program(plan)
        checked = self.validate_revision(data, telemetry, after_review=False)
        if revision <= self.revision:
            raise TaskContractError("contract_revision_invalid", "An accepted autonomous revision must advance its revision number.")
        self._cache_geometry(data["steps"])
        if checked["open_continuation"]:
            self._stack.append({"kind": "sequence", "nodes": deepcopy(data["steps"]), "index": 0})
            self._open_review = False
        self.revision = revision
        self._binding = {node["id"]: deepcopy(node) for node in _walk(data["steps"])}
        self._obligation_binding = ({node_id: {node_id} for node_id in self._binding}
                                    if checked["open_continuation"] else
                                    {key: set(value) for key, value in checked["bindings"].items()})
        self._advance(telemetry)

    def _cache_geometry(self, nodes):
        if self._scenario is None:
            return
        from .spatial_motion import compile_motion

        for node in _walk(nodes):
            if node["action"] != "motion":
                continue
            key = json.dumps(_action_value(node), sort_keys=True)
            if key not in self._motion_geometry:
                geometry = compile_motion(node, self._scenario, (0, 0, 0), self._constraints)
                self._motion_geometry[key] = (geometry.length_m, geometry.points[-1])

    def _verify_evidence(self, node, evidence):
        required = {"mission_id": self.mission_id, "vehicle_id": self.vehicle_id,
                    "revision": self.revision, "step_id": node.get("id"),
                    "execution_id": node.get("execution_id"), "action": node.get("action"),
                    "target_id": node.get("target_id")}
        if not required["execution_id"] or any(evidence.get(key) != value for key, value in required.items()):
            raise TaskContractError("contract_evidence_binding", "Completion evidence is not bound to this mission, revision and execution.")
        evidence_id = evidence.get("evidence_id")
        if (not isinstance(evidence_id, str) or not evidence_id or evidence_id in self._seen_evidence_ids
                or evidence["execution_id"] in self._seen_execution_ids):
            raise TaskContractError("contract_evidence_reused", "Each required occurrence needs distinct completion evidence.")
        bound = self._binding.get(node.get("id"))
        if bound is None or _action_value(bound) != _action_value(node):
            raise TaskContractError("contract_execution_changed", "The executed action differs from its accepted program.")
        action, result = node["action"], evidence.get("result")
        if evidence.get("source") != _SOURCES.get(action) or not isinstance(result, dict):
            raise TaskContractError("contract_evidence_source", "Completion requires trusted runtime or sensor evidence.")
        duration = evidence.get("duration_seconds")
        required_duration = node.get("duration_seconds", 0)
        if action == "control":
            required_duration = node["control"]["duration_seconds"]
        if not _finite(duration) or duration + 1e-8 < required_duration:
            raise TaskContractError("contract_dwell_incomplete", "The measured dwell does not satisfy the required duration.")
        flags = {"visit": "visited", "inspect": "inspected", "capture": "captured",
                 "return_home": "returned", "motion": "completed", "control": "completed", "review": "reviewed"}
        if action in flags and result.get(flags[action]) is not True:
            raise TaskContractError("contract_action_incomplete", "Runtime evidence does not prove the required action completed.")
        if action == "thermal_scan" and not _finite(result.get("temperature_c"), -math.inf):
            raise TaskContractError("contract_observation_invalid", "A thermal observation requires a finite measured result.")
        if action == "motion":
            total, covered = result.get("total_length_m"), result.get("distance_m")
            if (not _finite(total, .1) or not _finite(covered) or covered + 1e-6 < total
                    or result.get("progress") != 1.0):
                raise TaskContractError("contract_path_incomplete", "An endpoint is not evidence of complete ordered trajectory traversal.")
            if node["motion"]["heading"]["mode"] in ("fixed", "face_target"):
                samples = result.get("heading_samples")
                heading_error = result.get("max_heading_error_deg")
                if (isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0
                        or not _finite(heading_error)
                        or heading_error > _MOTION_HEADING_TOLERANCE_DEG):
                    raise TaskContractError("contract_motion_heading_incomplete",
                                            "Continuous heading requires measured tracking samples within the allowed angular error, including path entry.")
            geometry = self._motion_geometry.get(json.dumps(_action_value(node), sort_keys=True))
            if geometry is not None:
                from .safety import _point, metric_distance

                try:
                    valid_geometry = (abs(total - geometry[0]) <= 1e-5
                                      and metric_distance(_point(evidence.get("position")), geometry[1]) <= .18 + 1e-8)
                except (ValueError, TypeError):
                    valid_geometry = False
                if not valid_geometry:
                    raise TaskContractError("contract_path_geometry_changed", "Reported trajectory length and endpoint must match the frozen geometry.")
        if action == "return_home" and result.get("landed") is not True:
            raise TaskContractError("contract_landing_incomplete", "Return-home completion requires actual landing.")
        if action == "control":
            desired = node["control"]
            if result.get("state") != desired["state"] or result.get("landed") != (desired["state"] == "landed"):
                raise TaskContractError("contract_goal_state_incomplete", "The measured control state differs from the requested state.")
            if desired.get("yaw_deg") is not None:
                measured = evidence.get("measured_yaw_deg")
                if not _finite(measured, -math.inf) or abs((measured - desired["yaw_deg"] + 180) % 360 - 180) > 3:
                    raise TaskContractError("contract_heading_incomplete", "The measured heading does not satisfy the requested yaw.")
        if action not in ("motion", "review") and self._scenario is not None:
            self._verify_position(node, evidence)

    def _verify_position(self, node, evidence):
        from .safety import _point, metric_distance
        from .spatial_control import resolve_control_goal

        try:
            position = _point(evidence.get("position"))
            if node["action"] == "control":
                goal = resolve_control_goal(node, self._scenario, self._constraints)["position"]
            else:
                point = next(item for item in self._scenario.points_of_interest if item["id"] == node["target_id"])
                x, y, z = _point(point["position"])
                ceiling = self._constraints.get("max_altitude_m")
                goal = (x, y, min(z, ceiling / 5) if point["id"] != "home" and ceiling is not None else z)
            valid = metric_distance(position, goal) <= .18 + 1e-8
            if (node["action"] == "return_home"
                    or node["action"] == "control" and node["control"]["state"] == "landed"):
                valid = valid and abs(position[2] - goal[2]) * 5 <= .05 + 1e-8
        except (ValueError, TypeError, StopIteration):
            valid = False
        if not valid:
            raise TaskContractError("contract_position_incomplete", "Completion evidence is outside the required goal position.")

    def record(self, node, evidence, telemetry=None):
        """Consume one bound completion. Auxiliary actions never satisfy obligations."""
        try:
            self._advance(telemetry)
            self._verify_evidence(node, evidence)
            # Review is a reasoning opportunity, not a physical task obligation.
            # A corrected program may remove future reviews from a finite loop.
            while self._next is not None and self._next["action"] == "review" and node["action"] != "review":
                self._next = None
                self._advance(telemetry)
            if self._next is None:
                raise TaskContractError("contract_unexpected_completion", "An action completed after all accepted terminal requirements.")
            self._seen_execution_ids.add(evidence["execution_id"])
            self._seen_evidence_ids.add(evidence["evidence_id"])
            self._evidence[evidence["evidence_id"]] = deepcopy(evidence)
            matched = (self._next["id"] in self._obligation_binding.get(node["id"], set())
                       and _action_value(self._next) == _action_value(node)
                       or self._next["action"] == node["action"] == "review")
            if matched:
                expected = self._next
                self._step_evidence[expected["id"]] = deepcopy(evidence)
                self._matched_count += 1
                self._next = None
                self._advance(telemetry)
                self._open_review = expected["action"] == "review" and self._next is None and not self._stack
            return matched
        except TaskContractError as error:
            self._failures = [*self._failures[-7:], deepcopy(error.diagnostic)]
            raise

    def verdict(self, telemetry=None):
        try:
            self._advance(telemetry)
        except TaskContractError as error:
            self._failures = [*self._failures[-7:], deepcopy(error.diagnostic)]
        return self.snapshot()

    def snapshot(self):
        """Read status without advancing conditions, loops or execution."""
        diagnostics = deepcopy(self._failures)
        pending = []
        if self._next is not None:
            pending.append(self._next)
        for frame in reversed(self._stack):
            if len(pending) >= 24:
                break
            if frame["kind"] == "sequence":
                pending.extend(frame["nodes"][frame["index"]:frame["index"] + 24 - len(pending)])
            elif frame["node"].get("count") is None or frame["iteration"] < frame["node"]["count"]:
                pending.append(frame["node"])
        satisfied = not diagnostics and not self._next and not self._stack and not self._open_review
        value = {"contract_id": self.contract_id, "satisfied": satisfied,
                "matched_occurrences": self._matched_count, "skipped_occurrences": self._skipped_count,
                "pending": [{"id": node["id"], "action": node["action"], "target_id": node.get("target_id")}
                            for node in pending[:24]],
                "open_continuation": self._open_review, "diagnostics": diagnostics,
                "interpretation_boundary": "accepted_normalized_program"}
        value["status"] = ("invalid" if value["diagnostics"] else "satisfied" if value["satisfied"]
                           else "open" if value["open_continuation"] else "pending")
        value["error"] = deepcopy(value["diagnostics"][-1]) if value["diagnostics"] else None
        return value

    def context(self, telemetry=None):
        """Bounded public summary plus the required residual DSL for model repair."""
        value = self.snapshot()
        try:
            value["remaining_steps"] = self.remaining_steps(telemetry, after_review=self._next is not None and self._next["action"] == "review")
        except TaskContractError:
            value["remaining_steps"] = []
        value["constraints"] = deepcopy(self._constraints)
        return value
