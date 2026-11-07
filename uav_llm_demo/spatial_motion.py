"""Bounded geometric motion programs; expressions are interpreted, never evaluated as code."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import math

from .safety import SCALE, _point, metric_distance
from .path_tracking import PathFollower
from .task_protocol import TaskProtocolError, motion_schema, validate_structure


MOTION_LIMITS = {"max_points": 2048, "max_length_m": 3000.0, "max_laps": 20,
                 "max_speed_mps": 4.0, "max_expression_nodes": 64,
                 "max_expression_depth": 12, "max_expression_characters": 256}
MOTION_DEFAULTS = {"speed_mps": 2.5, "heading": {"mode": "tangent"},
                   "direction": "clockwise", "laps": 1}
_SAMPLE_DISTANCE_M = .75
_MAX_VALUE = 100000.0


class MotionConstraintError(ValueError):
    """A valid movement requirement conflicts with a known scene constraint."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.diagnostic = dict(code="motion_path_unsafe", path="$.steps", message=message, **details)


def motion_flight_envelope(lower, upper, *, anchor=None):
    """The actual path-reference bounds, including the reserved tracking corridor."""
    bounds = [[value + PathFollower.CORRIDOR_M / SCALE[i] for i, value in enumerate(lower)],
              [value - PathFollower.CORRIDOR_M / SCALE[i] for i, value in enumerate(upper)]]
    metric = [[value * SCALE[i] for i, value in enumerate(row)] for row in bounds]
    result = dict(flight_bounds_grid=[list(lower), list(upper)], trajectory_bounds_grid=bounds,
                  trajectory_bounds_m=metric, tracking_corridor_m=PathFollower.CORRIDOR_M)
    if anchor is not None:
        result.update(anchor_grid=list(anchor), anchor_m=[value * SCALE[i] for i, value in enumerate(anchor)],
                      relative_path_offset_bounds_m=[[(row[i] - anchor[i]) * SCALE[i] for i in range(3)] for row in bounds])
    return result


def motion_bound_violations(point, envelope):
    low, high = envelope["trajectory_bounds_m"]
    violations = []
    for i, value in enumerate(point):
        actual = value * SCALE[i]
        if actual < low[i] - 1e-8 * SCALE[i] or actual > high[i] + 1e-8 * SCALE[i]:
            violations.append(dict(axis="xyz"[i], actual_m=actual, minimum_m=low[i], maximum_m=high[i],
                                   required_shift_m=(low[i] - actual if actual < low[i] else high[i] - actual)))
    return violations


@dataclass(frozen=True)
class CompiledMotion:
    points: tuple[tuple[float, float, float], ...]
    speed_mps: float
    heading: dict
    length_m: float
    footprint_grid: tuple | None = None
    target_binding: dict | None = None


def _finite(value, lower=-_MAX_VALUE, upper=_MAX_VALUE):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Motion parameters must be finite numbers.")
    try:
        value = float(value)
    except (ValueError, OverflowError):
        raise ValueError("Motion parameters must be finite numbers.") from None
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("Motion parameter is outside its supported range.")
    return value


def _integer(value, lower, upper):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError("Motion count is outside its supported range.")
    return value


class _Expression:
    """A small arithmetic AST with interval derivative bounds for sampling density."""
    def __init__(self, expression):
        if isinstance(expression, (float, int)) and not isinstance(expression, bool):
            expression = str(_finite(expression))
        if not isinstance(expression, str) or not 1 <= len(expression) <= MOTION_LIMITS["max_expression_characters"]:
            raise ValueError("Motion expression must be short arithmetic.")
        try:
            self.root = ast.parse(expression, mode="eval").body
        except (SyntaxError, RecursionError, ValueError):
            raise ValueError("Invalid motion expression.") from None
        if sum(1 for _ in ast.walk(self.root)) > MOTION_LIMITS["max_expression_nodes"]:
            raise ValueError("Motion expression node budget exceeded.")
        self._validate(self.root, 0)

    def _validate(self, node, depth):
        if depth >= MOTION_LIMITS["max_expression_depth"]:
            raise ValueError("Motion expression nesting budget exceeded.")
        if isinstance(node, ast.Constant):
            _finite(node.value)
        elif isinstance(node, ast.Name) and node.id in ("t", "pi"):
            pass
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._validate(node.operand, depth + 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            self._validate(node.left, depth + 1)
            self._validate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant):
                    raise ValueError("Motion exponents must be integer constants from 0 to 4.")
                _integer(node.right.value, 0, 4)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in ("sin", "cos") and len(node.args) == 1 and not node.keywords):
            self._validate(node.args[0], depth + 1)
        else:
            raise ValueError("Only t, pi, sin, cos and bounded arithmetic are available.")

    def range_and_derivative(self, start, end):
        try:
            return self._interval(self.root, start, end)
        except ArithmeticError:
            raise ValueError("Motion expression exceeds supported numeric precision.") from None

    def _interval(self, node, start, end):
        # Return interval lower/upper and an absolute first derivative bound.
        if isinstance(node, ast.Constant):
            value = float(node.value)
            return value, value, 0.0
        if isinstance(node, ast.Name):
            return (start, end, 1.0) if node.id == "t" else (math.pi, math.pi, 0.0)
        if isinstance(node, ast.UnaryOp):
            lo, hi, derivative = self._interval(node.operand, start, end)
            return (-hi, -lo, derivative) if isinstance(node.op, ast.USub) else (lo, hi, derivative)
        if isinstance(node, ast.Call):
            _, _, derivative = self._interval(node.args[0], start, end)
            return -1.0, 1.0, derivative
        a, b, da = self._interval(node.left, start, end)
        c, d, db = self._interval(node.right, start, end)
        if isinstance(node.op, ast.Add):
            result = a + c, b + d, da + db
        elif isinstance(node.op, ast.Sub):
            result = a - d, b - c, da + db
        elif isinstance(node.op, ast.Mult):
            values = (a * c, a * d, b * c, b * d)
            result = min(values), max(values), da * max(abs(c), abs(d)) + db * max(abs(a), abs(b))
        elif isinstance(node.op, ast.Div):
            if c <= 0 <= d:
                raise ValueError("Motion expression denominator may reach zero in the parameter range.")
            values = (a / c, a / d, b / c, b / d)
            denominator = min(abs(c), abs(d))
            result = min(values), max(values), da / denominator + max(abs(a), abs(b)) * db / denominator ** 2
        else:
            exponent = int(node.right.value)
            values = [a ** exponent, b ** exponent]
            if a <= 0 <= b and exponent > 0:
                values.append(0.0)
            result = min(values), max(values), (exponent * max(abs(a), abs(b)) ** (exponent - 1) * da if exponent else 0.0)
        for value in result:
            _finite(value)
        return result

    def at(self, t):
        def run(node):
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Name):
                return t if node.id == "t" else math.pi
            if isinstance(node, ast.UnaryOp):
                value = run(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.Call):
                return (math.sin if node.func.id == "sin" else math.cos)(run(node.args[0]))
            left, right = run(node.left), run(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left ** right
        try:
            return _finite(run(self.root))
        except (ZeroDivisionError, OverflowError, ValueError):
            raise ValueError("Motion expression is not finite over its parameter range.") from None


def spatial_catalog(scenario):
    """Expose spatial anchors and shared scene boundaries, without sensor truth."""
    layout = getattr(scenario, "campus_layout", None) or {}
    geometries = layout.get("spatial_objects", {})
    objects = []
    for point in scenario.points_of_interest:
        if point["id"] == "home":
            continue
        center = list(_point(point["position"]))
        item = {"id": point["id"], "label": point["label"], "center_grid": center,
                "default_altitude_m": center[2] * SCALE[2]}
        geometry = geometries.get(point["id"])
        if geometry:
            item["center_grid"][:2] = list(geometry["center_grid"][:2])
            item["boundary_grid"] = deepcopy(geometry["boundary_grid"])
            if "footprint_grid" in geometry:
                item["footprint_grid"] = list(geometry["footprint_grid"])
            item["boundary_direction"] = "clockwise"
            item["boundary_min_altitude_m"] = max(
                geometry["boundary_min_altitude_m"],
                scenario.min_altitude * SCALE[2] + PathFollower.CORRIDOR_M + .1)
            item["boundary_default_altitude_m"] = max(item["default_altitude_m"],
                                                       item["boundary_min_altitude_m"])
        objects.append(item)
    return {"frame": "x east, y south, z up; directions viewed from above",
            "relative_coordinates": "curve and polyline x/y/z are metres relative to target center_grid, including z",
            "objects": objects, "limits": deepcopy(MOTION_LIMITS), "defaults": deepcopy(MOTION_DEFAULTS),
            "expressions": "numbers, t, pi, + - * / ** (constant integer 0..4), sin, cos",
            "path_kinds": {
                "boundary": {"direction": "clockwise or counterclockwise", "laps": "number 0 < laps <= 20; fractional traversal measured by boundary arc length; partial paths stay open",
                             "altitude_m": "absolute metres; null uses boundary_default_altitude_m; ceiling below boundary_min_altitude_m is unavailable",
                             "end_altitude_m": "absolute metres; null holds altitude, otherwise interpolate through path"},
                "curve": {"x": "relative metre expression", "y": "relative metre expression",
                          "z": "relative metre expression", "t_start": "finite number", "t_end": "greater than t_start",
                          "samples": "optional minimum sample count; actual density also enforces derivative bound"},
                "polyline": {"points_m": "2..2048 relative metre [east,south,up] vectors in required order"}},
            "heading_modes": {"tangent": "face travel direction",
                              "face_reference": "face this action's confirmed target_ref point",
                              "fixed": "yaw_deg: zero north, 90 east, 180 south, -90 west; clockwise viewed from above",
                              "face_target": "target_id of a published spatial object"}}


def normalize_motion(motion, target_id, catalog, constraints, *, path="$.motion"):
    """Validate and normalize defaults independently of runtime motion state."""
    validate_structure(motion, motion_schema(MOTION_LIMITS, MOTION_DEFAULTS), path)
    field = path
    spatial = catalog.get("spatial", catalog)
    objects = {item["id"]: item for item in spatial["objects"]}
    if not isinstance(target_id, str) or target_id not in objects:
        raise TaskProtocolError(field, "invalid_reference", "Motion target is not an available spatial object.", allowed=list(objects))
    obj = objects[target_id]
    speed = _finite(motion.get("speed_mps") if motion.get("speed_mps") is not None else MOTION_DEFAULTS["speed_mps"],
                    .1, MOTION_LIMITS["max_speed_mps"])
    heading = deepcopy(MOTION_DEFAULTS["heading"] if motion.get("heading") is None else motion["heading"])
    mode = heading.get("mode")
    if mode == "fixed":
        heading["yaw_deg"] = _finite(heading["yaw_deg"], -180, 180)
    elif mode == "face_target":
        if not isinstance(heading["target_id"], str) or heading["target_id"] not in objects:
            raise TaskProtocolError(f"{field}.heading.target_id", "invalid_reference", "Heading target is unavailable.", allowed=list(objects))
    path = deepcopy(motion["path"])
    kind = path.get("kind")
    if kind == "boundary":
        if not obj.get("boundary_grid"):
            raise TaskProtocolError(f"{field}.path.kind", "unavailable_geometry", "This object has no published boundary; use curve or polyline with its spatial anchor.", allowed=["curve", "polyline"])
        path["direction"] = path.get("direction", MOTION_DEFAULTS["direction"])
        path["laps"] = _finite(path.get("laps", MOTION_DEFAULTS["laps"]), 0, MOTION_LIMITS["max_laps"])
        default = obj["boundary_default_altitude_m"]
        ceiling = constraints.get("max_altitude_m")
        if ceiling is not None:
            if path.get("altitude_m") is None and ceiling < obj["boundary_min_altitude_m"]:
                raise MotionConstraintError("Altitude ceiling excludes the boundary's minimum safe altitude.")
            default = min(default, ceiling)
        for key in ("altitude_m", "end_altitude_m"):
            altitude = default if key == "altitude_m" else path["altitude_m"]
            try:
                path[key] = _finite(altitude if path.get(key) is None else path[key], 0)
            except ValueError as error:
                raise TaskProtocolError(f"{field}.path.{key}", "number_range", str(error)) from None
    elif kind == "curve":
        path["t_start"] = _finite(path["t_start"], -1000, 1000)
        path["t_end"] = _finite(path["t_end"], -1000, 1000)
        if not 0 < path["t_end"] - path["t_start"] <= 1000:
            raise TaskProtocolError(f"{field}.path.t_end", "parameter_range", "t_end must be greater than t_start, by at most 1000.")
        path["samples"] = _integer(path.get("samples", 256), 8, MOTION_LIMITS["max_points"])
        for axis in "xyz":
            try:
                expression = _Expression(path[axis])
                expression.range_and_derivative(path["t_start"], path["t_end"])
            except ValueError as error:
                raise TaskProtocolError(f"{field}.path.{axis}", "invalid_expression", str(error)) from None
    elif kind == "polyline":
        points = path["points_m"]
        path["points_m"] = [list(_point(point)) for point in points]
    result = {"path": path, "speed_mps": speed, "heading": heading}
    if "within_target" in motion:
        if motion["within_target"] and not obj.get("footprint_grid"):
            raise TaskProtocolError(f"{field}.within_target", "unavailable_geometry", "Target has no published footprint.")
        result["within_target"] = motion["within_target"]
    return result


def _resample(points):
    output = [points[0]]
    for start, end in zip(points, points[1:]):
        length = metric_distance(start, end)
        if length <= 1e-8:
            continue
        count = max(1, math.ceil(length / _SAMPLE_DISTANCE_M))
        if len(output) + count > MOTION_LIMITS["max_points"]:
            raise ValueError("Motion path point budget exceeded.")
        output.extend(tuple(a + (b - a) * index / count for a, b in zip(start, end))
                      for index in range(1, count + 1))
    return output


def curve_points(path):
    """The exact relative samples used by compilation and requirement checks."""
    expressions = [_Expression(path[axis]) for axis in "xyz"]
    derivative = math.hypot(*(item.range_and_derivative(path["t_start"], path["t_end"])[2] for item in expressions))
    intervals = max(path.get("samples", 256) - 1,
                    math.ceil(derivative * (path["t_end"] - path["t_start"]) / _SAMPLE_DISTANCE_M))
    if intervals + 1 > MOTION_LIMITS["max_points"]:
        raise ValueError("Curve requires too many samples for its variation; reduce frequency, extent or repetitions.")
    return [tuple(expression.at(path["t_start"] + (path["t_end"] - path["t_start"]) * i / intervals)
                  for expression in expressions) for i in range(intervals + 1)]


def compile_motion(node, scenario, start, constraints, *, evidence=()):
    """Generate bounded geometry in a worker; caller validates static reachability."""
    _point(start)  # Validate context without adding a shortcut from it to the task.
    spatial = spatial_catalog(scenario)
    motion = normalize_motion(node["motion"], node["target_id"], spatial, constraints)
    objects = {item["id"]: item for item in spatial["objects"]}
    from .observed_targets import bind_spatial_object
    obj, binding = bind_spatial_object(node, objects[node["target_id"]], evidence)
    origin = list(obj["center_grid"])
    ceiling = constraints.get("max_altitude_m")
    path = motion["path"]
    if binding is not None and path["kind"] == "boundary":
        raise ValueError("An observed target point has no measured boundary.")
    if path["kind"] == "boundary":
        boundary = [tuple(p) for p in obj["boundary_grid"]]
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])
        if path["direction"] == "counterclockwise":
            boundary.reverse()
        circuit = _resample([(x, y, path["altitude_m"] / SCALE[2]) for x, y in boundary])
        perimeter = sum(metric_distance(a, b) for a, b in zip(circuit, circuit[1:]))
        remaining = perimeter * path["laps"]
        points = [circuit[0]]
        # Traverse the measured boundary, retaining the exact fractional endpoint.
        # Never close the cropped portion or connect it back to the start.
        for _ in range(math.ceil(path["laps"])):
            for a, b in zip(circuit, circuit[1:]):
                length = metric_distance(a, b)
                if remaining <= 1e-8:
                    break
                fraction = 1. if remaining >= length - 1e-8 else remaining / length
                points.append(b if fraction == 1. else tuple(x + (y - x) * fraction for x, y in zip(a, b)))
                remaining -= length * fraction
            if remaining <= 1e-8:
                break
        if len(points) > MOTION_LIMITS["max_points"]:
            raise ValueError("Motion path point budget exceeded.")
        distance = [0.0]
        for a, b in zip(points, points[1:]):
            distance.append(distance[-1] + metric_distance(a, b))
        total = distance[-1]
        if total <= .1:
            raise ValueError("Motion path must have nonzero length.")
        points = [(point[0], point[1], (path["altitude_m"] +
                   (path["end_altitude_m"] - path["altitude_m"]) * travelled / total) / SCALE[2])
                  for point, travelled in zip(points, distance)]
    elif path["kind"] == "polyline":
        points = _resample([tuple(base + delta / scale for base, delta, scale in zip(origin, offset, SCALE))
                            for offset in path["points_m"]])
    else:
        points = [tuple(base + value / scale for base, value, scale in zip(origin, offset, SCALE))
                  for offset in curve_points(path)]
    points = _resample([_point(point) for point in points])
    if not 2 <= len(points) <= MOTION_LIMITS["max_points"]:
        raise ValueError("Motion path requires at least two distinct points.")
    total = sum(metric_distance(a, b) for a, b in zip(points, points[1:]))
    if not .1 <= total <= MOTION_LIMITS["max_length_m"]:
        raise ValueError("Motion path length is outside its execution budget.")
    upper_altitude = min(scenario.max_altitude * SCALE[2],
                         ceiling if ceiling is not None else scenario.max_altitude * SCALE[2])
    lower_altitude = scenario.min_altitude * SCALE[2]
    avoided = {region["id"]: region for region in scenario.semantic_regions}
    for point in points:
        if not (0 <= point[0] <= scenario.grid_size - 1 and 0 <= point[1] <= scenario.grid_size - 1
                and lower_altitude - 1e-8 <= point[2] * SCALE[2] <= upper_altitude + 1e-8):
            raise MotionConstraintError("Motion path leaves the allowed map or altitude range.",
                point_grid=list(point),
                **motion_flight_envelope((0, 0, lower_altitude / SCALE[2]),
                    (scenario.grid_size - 1, scenario.grid_size - 1, upper_altitude / SCALE[2]), anchor=origin),
                scale_m_per_grid=list(SCALE))
        for region_id in constraints.get("avoid_regions", []):
            region = avoided.get(region_id)
            if region is None:
                raise ValueError("Unknown avoided region.")
            left, right, north, south = region["bounds"]
            if left - .2 <= point[0] <= right + .2 and north - .2 <= point[1] <= south + .2:
                raise MotionConstraintError("Motion path enters an avoided region.")
    heading = deepcopy(motion["heading"])
    if heading["mode"] == "face_target":
        heading = {"mode": "face_point", "point": tuple(objects[heading["target_id"]]["center_grid"])}
    elif heading["mode"] == "face_reference":
        if binding is None:
            raise ValueError("face_reference requires an observed target.")
        heading = {"mode": "face_point", "point": tuple(binding["point_grid"])}
    footprint = tuple(obj["footprint_grid"]) if motion.get("within_target") else None
    if footprint:
        left, right, north, south = footprint
        margin = PathFollower.CORRIDOR_M / SCALE[0]
        if any(not (left + margin <= p[0] <= right - margin and north + margin <= p[1] <= south - margin)
               for p in points):
            raise MotionConstraintError("Motion leaves the target footprint or its reserved tracking corridor; reduce or reposition free geometry.")
    return CompiledMotion(tuple(points), motion["speed_mps"], heading, total, footprint, binding)
