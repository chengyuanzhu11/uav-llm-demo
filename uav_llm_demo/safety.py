"""Continuous safety checks in GRID coordinates; distances are metres."""
from __future__ import annotations

import math
from collections.abc import Mapping

Point = tuple[float, float, float]
SCALE = (3.0, 3.0, 5.0)
SAFETY_MARGIN = 0.6
MAX_WAYPOINTS = 64


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Coordinates must be finite numbers.")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError("Coordinates must be finite numbers.") from None
    if not math.isfinite(result):
        raise ValueError("Coordinates must be finite numbers.")
    return result


def _point(value) -> Point:
    if isinstance(value, Mapping):
        if set(value) != {"x", "y", "z"}:
            raise ValueError("Point requires exactly x, y, z.")
        value = [value[key] for key in ("x", "y", "z")]
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("Point requires three coordinates.")
    return tuple(_number(item) for item in value)


def metric_distance(a, b) -> float:
    a, b = _point(a), _point(b)
    return math.hypot(*((x - y) * scale for x, y, scale in zip(a, b, SCALE)))


def _intersects(a, b, lower, upper) -> bool:
    """Closed slab intersection: touching the safety envelope is unsafe."""
    enter, leave = 0.0, 1.0
    for start, end, lo, hi in zip(a, b, lower, upper):
        delta = end - start
        if delta == 0:
            if start < lo or start > hi:
                return False
        else:
            t0, t1 = sorted(((lo - start) / delta, (hi - start) / delta))
            enter, leave = max(enter, t0), min(leave, t1)
            if enter > leave:
                return False
    return True


class SafetyChecker:
    def __init__(self, context: dict, *, clearance=SAFETY_MARGIN):
        self.context = context
        self.clearance = _number(clearance)
        if self.clearance < 0:
            raise ValueError("Invalid clearance.")
        self.grid_size = _number(context["grid_size"])
        altitude = context["altitude_range"]
        self.lower = (0.0, 0.0, _number(altitude["min"]))
        self.upper = (self.grid_size - 1, self.grid_size - 1, _number(altitude["max"]))
        if self.grid_size < 1 or self.lower[2] > self.upper[2]:
            raise ValueError("Invalid flight bounds.")
        base = context.get("base")
        self.ground_base = (_point(base) if base is not None and _point(base)[2] == 0 else None)
        self.pad_radius_m = 0.75
        self.boxes = []
        for item in context.get("buildings", []):
            x, y, height = (_number(item[key]) for key in ("x", "y", "height"))
            if height < 0:
                raise ValueError("Invalid building height.")
            self._add_box((x - .39, y - .39, 0), (x + .39, y + .39, height), "building")
        for zone in context.get("no_fly_zones", []):
            self._add_box(_point(zone["min"]), _point(zone["max"]), "no-fly zone")

    def _add_box(self, lower, upper, label):
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError("Invalid obstacle bounds.")
        self.boxes.append((
            tuple(lo - self.clearance / scale for lo, scale in zip(lower, SCALE)),
            tuple(hi + self.clearance / scale for hi, scale in zip(upper, SCALE)), label))

    def _inside_bounds(self, point):
        x, y, z = point
        if not (self.lower[0] <= x <= self.upper[0] and self.lower[1] <= y <= self.upper[1] and z <= self.upper[2]):
            return False
        if z >= self.lower[2]:
            return True
        return z >= 0 and self._over_pad(point)

    def _over_pad(self, point):
        return self.ground_base is not None and math.hypot(
            (point[0] - self.ground_base[0]) * SCALE[0],
            (point[1] - self.ground_base[1]) * SCALE[1]) <= self.pad_radius_m

    def validate_waypoint(self, point) -> list[str]:
        try:
            point = _point(point)
        except ValueError:
            return ["Invalid waypoint: expected three finite numeric coordinates."]
        errors = []
        if not self._inside_bounds(point):
            errors.append("Waypoint outside flight bounds.")
        for lower, upper, label in self.boxes:
            if _intersects(point, point, lower, upper):
                errors.append(f"Waypoint intersects {label} safety envelope.")
        return errors

    def segment_safe(self, a, b) -> bool:
        try:
            a, b = _point(a), _point(b)
        except ValueError:
            return False
        # Check cruise bounds and the separate ground column before obstacles.
        # Closed segment tests below include both endpoints.
        if not all(self._inside_bounds(point) for point in (a, b)):
            return False
        # Low flight is confined to the base column; do not cut diagonally out of it.
        if min(a[2], b[2]) < self.lower[2] and not all(self._over_pad(p) for p in (a, b)):
            return False
        lower = tuple(min(x, y) for x, y in zip(a, b))
        upper = tuple(max(x, y) for x, y in zip(a, b))
        for lo, hi, _ in self.boxes:
            # Conservative broad phase: only disjoint boxes can be skipped.
            if (upper[0] < lo[0] or lower[0] > hi[0]
                    or upper[1] < lo[1] or lower[1] > hi[1]
                    or upper[2] < lo[2] or lower[2] > hi[2]):
                continue
            if _intersects(a, b, lo, hi):
                return False
        return True

    def validate_path(self, start, waypoints, require_goal=True, check_dynamic=False) -> list[str]:
        errors = [f"Start: {error}" for error in self.validate_waypoint(start)]
        if not isinstance(waypoints, (tuple, list)) or not 1 <= len(waypoints) <= MAX_WAYPOINTS:
            return errors + ["Path requires 1 to 64 waypoints."]
        previous = start
        for index, point in enumerate(waypoints):
            errors.extend(f"Waypoint {index}: {error}" for error in self.validate_waypoint(point))
            if not self.segment_safe(previous, point):
                errors.append(f"Segment {index} is statically unsafe.")
            elif check_dynamic and self._dynamic_conflict(_point(previous), _point(point)):
                errors.append(f"Segment {index} intersects a dynamic safety envelope.")
            previous = point
        if require_goal:
            try:
                reached = metric_distance(waypoints[-1], self.context["goal"]) <= 0.2
            except (KeyError, ValueError):
                reached = False
            if not reached:
                errors.append("Final waypoint must be within 0.2 metres of goal.")
        return errors

    def _dynamic_conflict(self, a, b) -> bool:
        # Snapshot checks only; no traversal timing is available to predict motion.
        delta = tuple((y - x) * s for x, y, s in zip(a, b, SCALE))
        length2 = sum(d * d for d in delta)
        for item in self.context.get("dynamic_objects", []):
            try:
                radius = _number(item.get("radius", 0.7))
                if radius < 0:
                    return True
                for key in ("position", "predicted_position"):
                    if key not in item and key == "predicted_position":
                        continue
                    center = _point(item[key])
                    offset = tuple((c - x) * s for c, x, s in zip(center, a, SCALE))
                    t = max(0.0, min(1.0, sum(o * d for o, d in zip(offset, delta)) / length2)) if length2 else 0.0
                    if math.hypot(*(o - t * d for o, d in zip(offset, delta))) <= radius * SCALE[0] + SAFETY_MARGIN:
                        return True
            except (KeyError, TypeError, ValueError):
                return True
        return False
