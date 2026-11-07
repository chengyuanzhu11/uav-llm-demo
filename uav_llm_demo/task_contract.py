"""Preserve accepted task requirements across reviews and measured execution."""
from __future__ import annotations

from copy import deepcopy
from .task_observation import observation_validity, OBSERVATION_ACTIONS
import ast
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
    "motion": ("target_id", "target_ref", "motion"),
    "control": ("target_id", "target_ref", "control"),
    "review": ("instruction",),
    "visual": ("target_id", "visual"),
}
_SOURCES = {"visit": "runtime_telemetry", "return_home": "runtime_telemetry",
            "control": "runtime_telemetry", "inspect": "simulated_inspection",
            "thermal_scan": "simulated_thermal_sensor", "capture": "simulated_capture",
            "motion": "runtime_trajectory_tracking", "review": "validated_execution_review",
            "visual": "onboard_rgbd_model"}


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


def _same_value(required, actual):
    """Sparse predicates never acquire defaults from an execution program."""
    if isinstance(required, dict) and set(required) == {"x", "y", "z"} and isinstance(actual, (list, tuple)):
        required = [required[key] for key in "xyz"]
    if isinstance(actual, dict) and set(actual) == {"x", "y", "z"} and isinstance(required, (list, tuple)):
        actual = [actual[key] for key in "xyz"]
    if isinstance(required, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in required.items():
            if key not in actual:
                return False
            if key == "yaw_deg" and _finite(value, -math.inf) and _finite(actual[key], -math.inf):
                if abs((actual[key] - value + 180) % 360 - 180) > 1e-7:
                    return False
            elif not _same_value(value, actual[key]):
                return False
        return True
    if isinstance(required, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and len(required) == len(actual)
                and all(_same_value(left, right) for left, right in zip(required, actual)))
    if _finite(required, -math.inf) and _finite(actual, -math.inf):
        return math.isclose(required, actual, rel_tol=1e-9, abs_tol=1e-7)
    return type(required) is type(actual) and required == actual


def _expression_form(expression):
    """Conservative arithmetic equivalence, without sampling a curve as proof.

    Normalize constants, neutral operands and associative/commutative sums and
    products. Other identities remain explicit: unsupported equivalence is a
    rejected revision, never an excuse to weaken path verification.
    """
    from .spatial_motion import _Expression

    def fold(node):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return math.pi if node.id == "pi" else ("variable", node.id)
        if isinstance(node, ast.Call):
            value = fold(node.args[0])
            if isinstance(value, float):
                return (math.sin if node.func.id == "sin" else math.cos)(value)
            return (node.func.id, value)
        if isinstance(node, ast.UnaryOp):
            value = fold(node.operand)
            return value if isinstance(node.op, ast.UAdd) else combine("multiply", [-1.0, value])
        left, right = fold(node.left), fold(node.right)
        if isinstance(node.op, ast.Add):
            return combine("add", [left, right])
        if isinstance(node.op, ast.Sub):
            return combine("add", [left, combine("multiply", [-1.0, right])])
        if isinstance(node.op, ast.Mult):
            return combine("multiply", [left, right])
        if isinstance(node.op, ast.Div):
            return combine("multiply", [left, 1.0 / right]) if isinstance(right, float) else ("divide", left, right)
        if right == 0:
            return 1.0
        if right == 1:
            return left
        return left ** right if isinstance(left, float) else ("power", left, right)

    def combine(kind, values):
        flat = []
        for value in values:
            flat.extend(value[1:] if isinstance(value, tuple) and value[0] == kind else [value])
        numbers = [value for value in flat if isinstance(value, float)]
        terms = [value for value in flat if not isinstance(value, float)]
        constant = sum(numbers) if kind == "add" else math.prod(numbers)
        if kind == "multiply" and constant == 0:
            return 0.0
        if constant != (0 if kind == "add" else 1) or not terms:
            terms.append(constant)
        terms.sort(key=repr)
        return terms[0] if len(terms) == 1 else (kind, *terms)

    return fold(_Expression(expression).root)


def _polyline_form(points):
    """Discard only forward collinear subdivision, retaining order and turns."""
    from .safety import _point

    result = []
    for point in points:
        point = _point(point)
        if result and _same_value(result[-1], point):
            continue
        while len(result) >= 2:
            a, b = result[-2:]
            u, v = [y - x for x, y in zip(a, b)], [y - x for x, y in zip(b, point)]
            cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            if math.hypot(*cross) > 1e-8 or sum(x * y for x, y in zip(u, v)) < 0:
                break
            result.pop()
        result.append(point)
    return result


def _expression_terms(form):
    """Extract affine coefficients while preserving nonlinear terms as opaque.

    Numeric factors distribute over sums, but we do not infer trigonometric
    identities, parameter changes or equality from sampled coordinates.
    """
    if isinstance(form, float):
        return {(): form}
    if form[0] == "add":
        terms = {}
        for part in form[1:]:
            for basis, coefficient in _expression_terms(part).items():
                terms[basis] = terms.get(basis, 0.0) + coefficient
        return {basis: value for basis, value in terms.items() if value != 0}
    if form[0] == "multiply":
        coefficient = math.prod(part for part in form[1:] if isinstance(part, float))
        factors = [part for part in form[1:] if not isinstance(part, float)]
        if len(factors) == 1:
            return {basis: coefficient * value for basis, value in _expression_terms(factors[0]).items()}
        return {("multiply", *factors): coefficient}
    return {form: 1.0}


def _same_expression(left, right):
    # A small coefficient may multiply a large t-dependent term. An absolute
    # coordinate tolerance must not erase it while proving symbolic equality.
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=0)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(_same_expression(a, b) for a, b in zip(left, right))
    return left == right


def _affine_geometry_matches(reference, actual, flexible):
    """Prove actual = positive group scale * reference + permitted axis offset."""
    groups = {axis: index for index, group in enumerate(flexible.get("scale_groups", ())) for axis in group}
    offset_axes = set(flexible.get("offset_axes", ()))
    scales = {}
    for axis, expected in reference.items():
        selected = actual[axis]
        if not all(math.isfinite(value) for value in (*expected.values(), *selected.values())):
            return False
        for basis in expected.keys() | selected.keys():
            if not basis and axis in offset_axes:
                continue
            left, right = expected.get(basis, 0.0), selected.get(basis, 0.0)
            if left == 0:
                if (basis and right != 0) or not _same_value(right, 0.0):
                    return False
                continue
            if axis not in groups:
                equal = (_same_expression(left, right) if basis else _same_value(left, right))
                if not equal:
                    return False
                continue
            scale = right / left
            if not math.isfinite(scale) or scale <= 0:
                return False
            group = groups[axis]
            if group in scales and not math.isclose(scales[group], scale, rel_tol=1e-9, abs_tol=0):
                return False
            scales[group] = scale
    return True


def _flexible_path_matches(required, actual):
    flexible = required["flexible"]
    if (not isinstance(flexible, dict) or required.get("kind") != actual.get("kind")
            or actual.get("kind") not in ("curve", "polyline")):
        return False
    try:
        if actual["kind"] == "curve":
            axes = [axis for axis in "xyz" if axis in required]
            reference = {axis: _expression_terms(_expression_form(required[axis])) for axis in axes}
            selected = {axis: _expression_terms(_expression_form(actual[axis])) for axis in axes}
        else:
            left, right = _polyline_form(required["points_m"]), _polyline_form(actual["points_m"])
            if not left or len(left) != len(right):
                return False

            def coordinates(points):
                return {axis: {(): points[0][index], **{
                    (point_index,): point[index] - points[0][index]
                    for point_index, point in enumerate(points[1:], 1)}}
                        for index, axis in enumerate("xyz")}

            reference, selected = coordinates(left), coordinates(right)
        return _affine_geometry_matches(reference, selected, flexible)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError):
        return False


def path_height_delta(path):
    if path["kind"] == "boundary":
        return path["end_altitude_m"] - path["altitude_m"]
    if path["kind"] == "polyline":
        from .safety import _point
        return _point(path["points_m"][-1])[2] - _point(path["points_m"][0])[2]
    from .spatial_motion import _Expression
    expression = _Expression(path["z"])
    return expression.at(path["t_end"]) - expression.at(path["t_start"])


def _path_matches(required, actual):
    if not isinstance(actual, dict):
        return False
    if "flexible" in required and not _flexible_path_matches(required, actual):
        return False
    for key, value in required.items():
        if key == "flexible" or ("flexible" in required and key in ("x", "y", "z", "points_m")):
            continue
        if key == "altitude_trend":
            try:
                from .spatial_motion import curve_points
                from .safety import _point
                heights = ([actual["altitude_m"], actual["end_altitude_m"]] if actual["kind"] == "boundary" else
                           [_point(p)[2] for p in actual["points_m"]] if actual["kind"] == "polyline" else
                           [p[2] for p in curve_points(actual)])
                changes = [b-a for a, b in zip(heights, heights[1:])]
                if ((value == "nondecreasing" and any(d < -1e-8 for d in changes))
                        or (value == "nonincreasing" and any(d > 1e-8 for d in changes))
                        or (value == "level" and any(abs(d) > 1e-8 for d in changes))):
                    return False
            except (ValueError, KeyError, TypeError, ArithmeticError):
                return False
        elif key == "altitude_delta_m":
            try:
                delta = path_height_delta(actual)
            except (KeyError, IndexError, TypeError, ValueError, ArithmeticError):
                return False
            if not _same_value(value, delta):
                return False
        elif key not in actual:
            return False
        elif key in ("x", "y", "z") and required.get("kind", actual.get("kind")) == "curve":
            try:
                if not _same_expression(_expression_form(value), _expression_form(actual[key])):
                    return False
            except (ValueError, ArithmeticError):
                return False
        elif key == "points_m":
            try:
                same = _same_value(_polyline_form(value), _polyline_form(actual[key]))
            except (ValueError, TypeError, KeyError, IndexError):
                same = False
            if not same:
                return False
        elif not _same_value(value, actual[key]):
            return False
    return True


def _mismatch_details(expected, candidate, origins):
    """Explain unique action/target matches with the same acceptance predicates.

    Ambiguous occurrences and structural/order failures retain the global
    diagnostic; this helper never creates bindings or relaxes the contract.
    """
    details = []
    choices = {}
    for node in _walk(candidate):
        choices.setdefault((node.get("action"), node.get("target_id")), []).append(node)
    def bounded(value):
        text = json.dumps(value, ensure_ascii=False)
        return value if len(text) <= 500 else text[:500] + "..."

    for required in _walk(expected):
        options = choices.get((required.get("action"), required.get("target_id")), [])
        if len(options) != 1 or required.get("action") != "motion":
            continue
        actual = options[0]
        needed, selected = required.get("motion", {}), actual.get("motion", {})
        prefix = f"node:{origins.get(actual['id'], actual['id'])}.motion"
        for field, value in needed.items():
            if field != "path":
                if not _same_value(value, selected.get(field)):
                    details.append(dict(path=f"{prefix}.{field}", expected=bounded(value), actual=bounded(selected.get(field))))
                continue
            chosen = selected.get("path", {})
            for key, requirement in value.items():
                if "flexible" in value and key in ("x", "y", "z", "points_m"):
                    continue
                matched = (_flexible_path_matches(value, chosen) if key == "flexible"
                           else _path_matches({key: requirement}, chosen))
                if matched:
                    continue
                measured = chosen.get(key)
                if key == "altitude_delta_m":
                    try:
                        measured = path_height_delta(chosen)
                    except (ValueError, KeyError, TypeError, ArithmeticError):
                        measured = "unavailable"
                elif key == "altitude_trend":
                    measured = "compiled vertical profile violates this trend"
                elif key == "flexible":
                    measured = "reference geometry is not preserved by the permitted transforms"
                details.append(dict(path=f"{prefix}.path.{key}", expected=bounded(requirement), actual=bounded(measured)))
        if len(details) >= 8:
            break
    return details[:8]


def _requirement_matches(required, actual):
    if required.get("action") != actual.get("action"):
        return False
    for key in _ACTION_FIELDS.get(required["action"], ()):
        if key not in required or key == "instruction":
            continue
        if key == "motion":
            if not isinstance(actual.get(key), dict):
                return False
            for field, value in required[key].items():
                if field not in actual[key]:
                    return False
                if field == "path":
                    if not _path_matches(value, actual[key][field]):
                        return False
                elif not _same_value(value, actual[key][field]):
                    return False
        elif key == "visual":
            expected, selected = deepcopy(required[key]), actual.get(key, {})
            maximum = expected.pop("max_distance_m", None)
            if (not _same_value(expected, selected) or maximum is not None
                    and (not _finite(selected.get("max_distance_m"), 2) or selected["max_distance_m"] > maximum)):
                return False
        elif key == "control" and "altitude_m" in required[key] and required[key]["altitude_m"] is not None:
            from .safety import _point

            expected, selected = deepcopy(required[key]), actual.get(key, {})
            altitude = expected.pop("altitude_m") + _point(expected.get("offset_m", [0, 0, 0]))[2]
            selected_altitude = selected.get("altitude_m")
            if (selected_altitude is None
                    or not _same_value(altitude, selected_altitude + _point(selected.get("offset_m", [0, 0, 0]))[2])
                    or not _same_value(expected, selected)):
                return False
        elif not _same_value(required[key], actual.get(key)):
            return False
    return True


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
                if "target_ref" in node:
                    node["target_ref"] = _ref_expression(node["target_ref"], env)
                node["when"] = _ref_expression(node.get("when"), env)
                if node.get("when") is None:
                    env[old_id] = {"step_id": new_id}
            output.append(node)
        return output, env

    return walk(_fold_review_conditions(nodes, constants or {}), dict(references or {}))[0]


def _matches(expected, candidate, *, allow_observations=False, bounded_obligations=()):
    """Order-preserving structural inclusion with shared alpha-renamed references.

    Movement may support required work, but extra business actions and repeated
    obligations cannot be hidden as auxiliary steps. Preserve the existing end
    boundary: only an open review authorizes continuation after the last requirement.
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

    obligations = [node for node in _walk(expected) if node["action"] not in ("repeat", "branch", "review")]

    def auxiliary(node):
        if node["action"] == "review":
            return True
        if any(required["action"] == node["action"] and required.get("target_id") == node.get("target_id")
               for required in bounded_obligations):
            return False
        if node["action"] in ("inspect", "capture", "thermal_scan"):
            return allow_observations
        return node["action"] in ("visit", "motion", "control") and not any(
            required["action"] == node["action"] and required.get("target_id") == node.get("target_id")
            and _requirement_matches(required, node) for required in obligations)

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
        compared = deepcopy(left)
        if "target_ref" in compared:
            compared["target_ref"] = _ref_expression(compared["target_ref"], mapping)
        if not _requirement_matches(compared, right):
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
                    if auxiliary(right[ci]):
                        pending.append((li, ci + 1, mapping))
                    pending.append((li + 1, ci + 1, updated))
                    break
                if not auxiliary(right[ci]):
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

    def __init__(self, requirements, mission_id, vehicle_id="uav-1", revision=1, *,
                 plan, telemetry=None, evidence=(), scenario=None, cache_geometry=True):
        required, data = _program(requirements), _program(plan)
        self.contract_id = uuid.uuid4().hex
        self.mission_id, self.vehicle_id, self.revision = mission_id, vehicle_id, revision
        self._constraints = deepcopy(required.get("constraints", {}))
        self._plan_constraints = deepcopy(data["constraints"])
        self._scenario = scenario
        self._stack = [{"kind": "sequence", "nodes": deepcopy(required["steps"]), "index": 0}]
        self._next = None
        self._telemetry = deepcopy(telemetry or {})
        self._step_evidence, self._evidence = {}, {}
        self._observation_aliases = {}
        self._refresh_bindings = {}
        self._simulation_time = 0.0
        self._projected_review_results = {}
        self._seen_execution_ids, self._seen_evidence_ids = set(), set()
        self._matched_count = self._skipped_count = 0
        self._failures = []
        self._open_review = False
        self._delegated_continuation = False
        self._review_authorized = False
        self._bounded_obligations = []
        for node in _walk(required["steps"]):
            if node["action"] == "repeat" and node.get("count") is not None:
                self._bounded_obligations.extend(deepcopy(list(_walk(node["steps"]))))
            elif node["action"] == "motion" and "laps" in node.get("motion", {}).get("path", {}):
                self._bounded_obligations.append(deepcopy(node))
        self._binding = {node["id"]: deepcopy(node) for node in _walk(data["steps"])}
        checked = self._check_plan(data, required["steps"])
        self._obligation_binding = {key: set(value) for key, value in checked["bindings"].items()}
        self._motion_geometry = {}
        if cache_geometry:
            self._cache_geometry(data["steps"])
        for item in evidence:
            if item.get("mission_id") == mission_id and item.get("vehicle_id") == vehicle_id:
                self._evidence[item["evidence_id"]] = deepcopy(item)
                original_id = item.get("refresh_of")
                original = self._evidence.get(original_id)
                if (original is not None and item.get("action") in OBSERVATION_ACTIONS
                        and all(item.get(key) == original.get(key) for key in ("action", "target_id", "source"))
                        and item.get("simulation_time", -1) >= original.get("simulation_time", 0)):
                    self._alias_observation(original_id, item)

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
                if item is not None:
                    item = self._observation_aliases.get(item["evidence_id"], item)
                    if observation_validity(item, self._simulation_time) is not None:
                        error = TaskContractError("contract_observation_unavailable", "A condition requires a fresh, consistent observation.")
                        error.observation_evidence_id = item["evidence_id"]
                        raise error
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
            self._simulation_time = telemetry.get("simulation_time", self._simulation_time)
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
                exhausted = node.get("count") is not None and frame["iteration"] >= node["count"]
                until = not exhausted and node.get("until") is not None and self._evaluate(node["until"])
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
            action = node["action"]
            if action == "branch":
                selected = self._evaluate(node["condition"])
                frame["index"] += 1
                self._stack.append({"kind": "sequence", "nodes": node["then" if selected else "else"], "index": 0})
                continue
            if node.get("when") is not None:
                selected = self._evaluate(node["when"])
                if not selected:
                    frame["index"] += 1
                    self._skipped_count += 1
                    continue
            frame["index"] += 1
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
        clone._projection_wait = clone._advance_available(telemetry)
        if after_review:
            # A locally accepted auxiliary review may precede a required action.
            # It grants no authority to discard that pending action.
            if clone._next is not None and clone._next["action"] == "review":
                # This projection is conditional on a successful validated
                # review. It is not an evidence record and cannot satisfy a
                # physical obligation or leak an invented evidence identifier.
                clone._projected_review_results[clone._next["id"]] = {"reviewed": True}
                clone._next = None
                clone._projection_wait = clone._advance_available(telemetry)
                clone._open_review = clone._next is None and not clone._stack
        return clone

    def _advance_available(self, telemetry=None):
        """Retain an unresolved predicate for planning instead of inventing its arm.

        A missing fresh observation is an execution dependency. The remaining
        requirement graph still contains its full predicate and both branches,
        so validating a replacement plan cannot silently discard either arm.
        """
        try:
            self._advance(telemetry)
        except TaskContractError as error:
            if error.diagnostic["code"] != "contract_observation_unavailable":
                raise
            return deepcopy(error.diagnostic)
        return None

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

    def _check_plan(self, data, remaining, references=None, constants=None, *, open_continuation=False,
                    allow_observations=False):
        ceiling = self._constraints.get("max_altitude_m")
        actual_ceiling = data["constraints"].get("max_altitude_m")
        if (ceiling is not None and (actual_ceiling is None or actual_ceiling > ceiling)
                or not set(self._constraints.get("avoid_regions", ())).issubset(data["constraints"].get("avoid_regions", ()))):
            raise TaskContractError("contract_constraints_changed", "Autonomous revision must preserve the accepted hard constraints.", "$.constraints")
        expected_origins, candidate_origins = {}, {}
        expected = _normal_form(remaining, references, expected_origins, constants=constants)
        candidate = _normal_form(data["steps"], origins=candidate_origins,
                                 constants=self._review_constants())
        open_continuation = not expected and open_continuation
        mapping = None if open_continuation else _matches(expected, candidate, allow_observations=allow_observations,
                                                          bounded_obligations=self._bounded_obligations)
        if not open_continuation and mapping is None:
            error = TaskContractError("contract_obligation_changed",
                                    "The program must satisfy every independent requirement, including order, repetition, condition, explicit path parameters and goal state. Unspecified execution parameters may change; additional observations require review authorization and cannot alter explicit counts.")
            error.diagnostic["mismatches"] = _mismatch_details(expected, candidate, candidate_origins)
            raise error
        bindings = {}
        for required, proposed in (mapping or {}).items():
            candidate_id = candidate_origins[proposed["step_id"]]
            bindings.setdefault(candidate_id, set()).add(expected_origins[required])
        return {"accepted": True, "contract_id": self.contract_id,
                "open_continuation": open_continuation, "remaining_steps": expected,
                "bindings": {key: sorted(value) for key, value in bindings.items()}}

    def validate_revision(self, plan, telemetry=None, *, after_review=True):
        data = _program(plan)
        clone = self._projection(telemetry, after_review)
        references = {step_id: {"evidence_id": item["evidence_id"]}
                      for step_id, item in clone._step_evidence.items()}
        return self._check_plan(data, clone._raw_remaining(), references, clone._review_constants(),
                                open_continuation=clone._open_review or self._delegated_continuation,
                                allow_observations=after_review or self._review_authorized)

    def accept_revision(self, plan, revision, telemetry=None):
        data = _program(plan)
        checked = self.validate_revision(data, telemetry, after_review=False)
        if revision <= self.revision:
            raise TaskContractError("contract_revision_invalid", "An accepted autonomous revision must advance its revision number.")
        self._plan_constraints = deepcopy(data["constraints"])
        self._cache_geometry(data["steps"])
        if checked["open_continuation"]:
            self._delegated_continuation = True
            self._open_review = False
        self._review_authorized = False
        self.revision = revision
        self._binding = {node["id"]: deepcopy(node) for node in _walk(data["steps"])}
        self._obligation_binding = {key: set(value) for key, value in checked["bindings"].items()}
        self._advance_available(telemetry)

    def authorize_event_review(self):
        """Allow measured auxiliary recovery work without relaxing any obligation."""
        self._review_authorized = True

    def set_simulation_time(self, now):
        self._simulation_time = now

    def _cache_geometry(self, nodes):
        if self._scenario is None:
            return
        from .spatial_motion import compile_motion

        for node in _walk(nodes):
            if node["action"] != "motion" or node.get("target_ref"):
                continue
            key = json.dumps(_action_value(node), sort_keys=True)
            if key not in self._motion_geometry:
                geometry = compile_motion(node, self._scenario, (0, 0, 0), self._plan_constraints)
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
        if action in OBSERVATION_ACTIONS and observation_validity(evidence, evidence.get("simulation_time", self._simulation_time)) is not None:
            raise TaskContractError("contract_observation_invalid", "An unavailable observation cannot complete a task.")
        if evidence.get("source") != _SOURCES.get(action) or not isinstance(result, dict):
            raise TaskContractError("contract_evidence_source", "Completion requires trusted runtime or sensor evidence.")
        if node.get("target_ref"):
            from .observed_targets import resolve_target_reference
            try:
                expected_binding = resolve_target_reference(node, self._evidence.values())
            except ValueError as exc:
                raise TaskContractError("contract_target_unresolved", str(exc)) from None
            if result.get("target_binding") != expected_binding:
                raise TaskContractError("contract_target_binding_changed", "Execution must use the confirmed object, not its search region or another observation.")
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
            if node["motion"]["heading"]["mode"] in ("fixed", "face_target", "face_reference"):
                samples = result.get("heading_samples")
                heading_error = result.get("max_heading_error_deg")
                if (isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0
                        or not _finite(heading_error)
                        or heading_error > _MOTION_HEADING_TOLERANCE_DEG):
                    raise TaskContractError("contract_motion_heading_incomplete",
                                            "Continuous heading requires measured tracking samples within the allowed angular error, including path entry.")
            geometry = self._motion_geometry.get(json.dumps(_action_value(node), sort_keys=True))
            if node.get("target_ref") and self._scenario is not None:
                from .spatial_motion import compile_motion
                compiled = compile_motion(node, self._scenario, (0, 0, 0), self._plan_constraints,
                                          evidence=self._evidence.values())
                geometry = (compiled.length_m, compiled.points[-1])
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
        if action == "visual":
            from .onboard_observation import validate_visual_evidence
            try:
                validate_visual_evidence(node, evidence)
            except ValueError as exc:
                raise TaskContractError("contract_visual_incomplete", str(exc)) from None
        if action not in ("motion", "review", "visual") and self._scenario is not None:
            self._verify_position(node, evidence)

    def _verify_position(self, node, evidence):
        from .safety import _point, metric_distance
        from .spatial_control import resolve_control_goal

        try:
            position = _point(evidence.get("position"))
            if node["action"] == "control":
                goal = resolve_control_goal(node, self._scenario, self._plan_constraints,
                                            evidence=self._evidence.values())["position"]
            else:
                point = next(item for item in self._scenario.points_of_interest if item["id"] == node["target_id"])
                x, y, z = _point(point["position"])
                ceiling = self._plan_constraints.get("max_altitude_m")
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
            if self._next is None and node["id"] in self._obligation_binding:
                raise TaskContractError("contract_unexpected_completion", "An action completed after all accepted terminal requirements.")
            self._seen_execution_ids.add(evidence["execution_id"])
            self._seen_evidence_ids.add(evidence["evidence_id"])
            self._evidence[evidence["evidence_id"]] = deepcopy(evidence)
            if node["action"] == "review":
                self._review_authorized = True
                if self._delegated_continuation and self._next is None:
                    self._open_review = True
            matched = self._next is not None and (
                self._next["id"] in self._obligation_binding.get(node["id"], set())
                and self._matches_occurrence(self._next, node)
                or self._next["action"] == node["action"] == "review")
            if matched:
                expected = self._next
                self._step_evidence[expected["id"]] = deepcopy(evidence)
                self._matched_count += 1
                self._next = None
                try:
                    self._advance(telemetry)
                except TaskContractError as error:
                    if error.diagnostic["code"] != "contract_observation_unavailable":
                        raise
                self._open_review = expected["action"] == "review" and self._next is None and not self._stack
            return matched
        except TaskContractError as error:
            self._failures = [*self._failures[-7:], deepcopy(error.diagnostic)]
            raise

    def _matches_occurrence(self, required, node):
        required, node = deepcopy(required), deepcopy(node)
        if required.get("target_ref"):
            from .observed_targets import resolve_target_reference
            ref = required["target_ref"]
            if "step_id" in ref:
                source = self._step_evidence.get(ref["step_id"])
                if source is None:
                    return False
                required["target_ref"] = {"evidence_id": source["evidence_id"]}
            try:
                binding = resolve_target_reference(node, self._evidence.values())
            except ValueError:
                return False
            node["target_ref"] = {"evidence_id": binding["evidence_id"]} if binding else None
        return _requirement_matches(required, node)

    def bind_observation_refresh(self, original_evidence_id, node):
        original = self._evidence.get(original_evidence_id)
        if (original is None or original.get("action") not in OBSERVATION_ACTIONS
                or any(node.get(key) != original.get(key) for key in ("action", "target_id"))
                or node.get("id") in self._binding):
            raise TaskContractError("contract_refresh_binding", "A refresh must measure the same bound action and target.")
        self._binding[node["id"]] = deepcopy(node)
        self._refresh_bindings[node["id"]] = original_evidence_id

    def has_pending_observation_refresh(self, evidence_id):
        """An already bound measurement may resume after a reviewed revision."""
        return evidence_id in self._refresh_bindings.values()

    def record_observation_refresh(self, node, evidence, telemetry=None):
        original_id = self._refresh_bindings.get(node.get("id"))
        if original_id is None:
            raise TaskContractError("contract_refresh_binding", "Refresh evidence has no authorized source.")
        self._verify_evidence(node, evidence)
        self._seen_execution_ids.add(evidence["execution_id"])
        self._seen_evidence_ids.add(evidence["evidence_id"])
        self._evidence[evidence["evidence_id"]] = deepcopy(evidence)
        # Evidence history is immutable. Only decision-time lookup follows the
        # explicitly authorized same-target refresh; no obligation is counted twice.
        self._alias_observation(original_id, evidence)
        for node_id, source_id in list(self._refresh_bindings.items()):
            resolved = self._observation_aliases.get(source_id, self._evidence.get(source_id, {}))
            if resolved.get("evidence_id") == evidence["evidence_id"]:
                self._refresh_bindings.pop(node_id)
        try:
            self._advance(telemetry)
        except TaskContractError as error:
            if error.diagnostic["code"] != "contract_observation_unavailable":
                raise

    def _alias_observation(self, original_id, evidence):
        for root, record in list(self._observation_aliases.items()):
            if record["evidence_id"] == original_id:
                self._observation_aliases[root] = deepcopy(evidence)
        self._observation_aliases[original_id] = deepcopy(evidence)

    def verdict(self, telemetry=None):
        try:
            self._advance(telemetry)
        except TaskContractError as error:
            if error.diagnostic["code"] == "contract_observation_unavailable":
                return {**self.snapshot(), "satisfied": False, "status": "pending",
                        "diagnostics": [deepcopy(error.diagnostic)]}
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
                "delegated_continuation": self._delegated_continuation,
                "interpretation_boundary": "independent_user_requirements"}
        value["status"] = ("invalid" if value["diagnostics"] else "satisfied" if value["satisfied"]
                           else "open" if value["open_continuation"] else "pending")
        value["error"] = deepcopy(value["diagnostics"][-1]) if value["diagnostics"] else None
        return value

    def context(self, telemetry=None):
        """Bounded public summary plus the required residual DSL for model repair."""
        value = self.snapshot()
        try:
            clone = self._projection(telemetry, after_review=self._next is not None and self._next["action"] == "review")
            references = {step_id: {"evidence_id": item["evidence_id"]}
                          for step_id, item in clone._step_evidence.items()}
            value["remaining_steps"] = _normal_form(clone._raw_remaining(), references, expand_counts=False,
                                                     constants=clone._review_constants())
            if clone._projection_wait is not None:
                value["condition_wait"] = clone._projection_wait
        except TaskContractError as error:
            # A temporarily unavailable condition is unresolved work, never an
            # empty task. Keep the original branch and its bound evidence visible
            # to user updates or event recovery until a fresh reading decides it.
            references = {step_id: {"evidence_id": item["evidence_id"]}
                          for step_id, item in self._step_evidence.items()}
            value["remaining_steps"] = _normal_form(self._raw_remaining(), references, expand_counts=False,
                                                     constants=self._review_constants())
            value["condition_wait"] = deepcopy(error.diagnostic)
        value["constraints"] = deepcopy(self._constraints)
        return value
