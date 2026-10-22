"""Continuous safety checks in GRID coordinates; distances are metres."""
from __future__ import annotations

import math
from collections.abc import Mapping
from .dynamic_risk import separation_distance_m

Point = tuple[float, float, float]
SCALE = (3.0, 3.0, 5.0)
SAFETY_MARGIN = 0.6
GROUND_COLUMN_RADIUS_M = 0.75
MAX_WAYPOINTS = 64
# Numeric boundary tolerance, distinct from obstacle clearance and arrival tolerance.
# The controller stops corrections below 1e-9 m; exact cruise comparisons can deadlock.
ALTITUDE_EPSILON = 1e-8 / SCALE[2]


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
            # A tangent at a box corner is still a collision. Division can put
            # equal slab times on opposite sides of a floating-point ulp.
            if enter > leave + 1e-12:
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
        self.pad_radius_m = GROUND_COLUMN_RADIUS_M
        # These sites come from local surface validation, never from model output.
        # The original ground base uses the same narrow access-column mechanism.
        sites = context.get("ground_sites", [])
        if not isinstance(sites, (tuple, list)):
            raise ValueError("Ground sites must be a list of validated ground points.")
        explicit_sites = tuple(_point(site) for site in sites)
        if any(site[2] != 0 or not (0 <= site[0] <= self.upper[0]
                                   and 0 <= site[1] <= self.upper[1]) for site in explicit_sites):
            raise ValueError("Ground sites must be inside the map at ground height zero.")
        base = _point(context["base"]) if context.get("base") is not None else None
        self.ground_sites = tuple(dict.fromkeys(
            ((base,) if base is not None and base[2] == 0 else ()) + explicit_sites))
        self.boxes = []
        for item in context.get("buildings", []):
            x, y, height = (_number(item[key]) for key in ("x", "y", "height"))
            if height < 0:
                raise ValueError("Invalid building height.")
            self._add_box((x - .39, y - .39, 0), (x + .39, y + .39, height), "building")
        for zone in context.get("no_fly_zones", []):
            self._add_box(_point(zone["min"]), _point(zone["max"]), "no-fly zone")
        for site in explicit_sites:
            entry = (*site[:2], min(self.upper[2], self.lower[2] + .2))
            if not self.segment_safe(site, entry):
                raise ValueError("Ground access column intersects flight bounds or an obstacle.")

    def _add_box(self, lower, upper, label):
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError("Invalid obstacle bounds.")
        self.boxes.append((
            tuple(lo - self.clearance / scale for lo, scale in zip(lower, SCALE)),
            tuple(hi + self.clearance / scale for hi, scale in zip(upper, SCALE)), label))

    def _inside_bounds(self, point):
        x, y, z = point
        if not (self.lower[0] <= x <= self.upper[0] and self.lower[1] <= y <= self.upper[1]
                and z <= self.upper[2] + ALTITUDE_EPSILON):
            return False
        if z >= self.lower[2] - ALTITUDE_EPSILON:
            return True
        return z >= 0 and self.ground_site_for(point) is not None

    def _over_site(self, point, site):
        return math.hypot((point[0] - site[0]) * SCALE[0],
                          (point[1] - site[1]) * SCALE[1]) <= self.pad_radius_m

    def ground_site_for(self, point):
        """Return a validated ground site beneath a point, if any."""
        return next((site for site in self.ground_sites if self._over_site(point, site)), None)

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
        # Both endpoints must share one convex column. Merely being over two
        # different valid sites must never permit low flight between them.
        if min(a[2], b[2]) < self.lower[2] - ALTITUDE_EPSILON and not any(
                self._over_site(a, site) and self._over_site(b, site) for site in self.ground_sites):
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
                radius = (_number(item["radius_m"]) if "radius_m" in item
                          else _number(item.get("radius", 0.7)) * SCALE[0])
                if radius < 0:
                    return True
                for key in ("position", "predicted_position"):
                    if key not in item and key == "predicted_position":
                        continue
                    center = _point(item[key])
                    offset = tuple((c - x) * s for c, x, s in zip(center, a, SCALE))
                    t = max(0.0, min(1.0, sum(o * d for o, d in zip(offset, delta)) / length2)) if length2 else 0.0
                    if math.hypot(*(o - t * d for o, d in zip(offset, delta))) <= separation_distance_m(radius):
                        return True
            except (KeyError, TypeError, ValueError):
                return True
        return False
