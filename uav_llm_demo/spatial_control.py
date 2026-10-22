"""Scene-derived goal positions and ground transitions for generic vehicle control.

GRID coordinates are kept at the execution boundary; input offsets and altitudes
are metres. A place is an anchor, not a landing permission. Landing support comes
from the rendered scene's flat ground surfaces plus actual local clearances.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .safety import GROUND_COLUMN_RADIUS_M, SAFETY_MARGIN, SCALE, SafetyChecker, _intersects, _point
from .task_protocol import TaskProtocolError, control_schema, validate_structure


CONTROL_DEFAULTS = {"anchor": "center", "offset_m": [0.0, 0.0, 0.0],
                    "altitude_m": None, "state": "airborne", "duration_seconds": 2.0,
                    "yaw_deg": None}
LANDING_CLEARANCE_M = SAFETY_MARGIN + GROUND_COLUMN_RADIUS_M
_MAX_DURATION_SECONDS = 600.0


def _number(value, field, lower=-100000.0, upper=100000.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Control {field} must be a finite number.")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"Control {field} must be a finite number.") from None
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"Control {field} is outside its supported range.")
    return value


def _layout(scenario):
    return getattr(scenario, "campus_layout", None) or {}


def _geometry(scenario, point):
    layout = _layout(scenario)
    place = next((place for place in layout.get("places", []) if place["id"] == point["id"]), {})
    geometry = layout.get("spatial_objects", {}).get(point["id"], {})
    arrival = _point(point["position"])
    parts = [item for collection, ids in (("buildings", "building_ids"), ("surfaces", "surface_ids"))
             for item in layout.get(collection, []) if item["id"] in place.get(ids, [])]
    if geometry.get("center_grid") is not None:
        center = tuple(geometry["center_grid"][:2])
    elif parts:
        center = ((min(item["footprint"][0] for item in parts) + max(item["footprint"][1] for item in parts)) / 2,
                  (min(item["footprint"][2] for item in parts) + max(item["footprint"][3] for item in parts)) / 2)
    else:
        center = arrival[:2]
    return dict(id=point["id"], label=point.get("label", point["id"]),
                center_grid=[*center, 0.0], arrival_grid=list(arrival),
                surface_ids=list(place.get("surface_ids", [])),
                default_altitude_m=max(scenario.min_altitude * SCALE[2], arrival[2] * SCALE[2]))


def control_catalog(scenario):
    """Publish all place anchors and known ground geometry, without sensor truth."""
    objects = [_geometry(scenario, point) for point in scenario.points_of_interest]
    if not any(item["id"] == "home" for item in objects) and getattr(scenario, "base", None) is not None:
        objects.append(_geometry(scenario, {"id": "home", "label": "基地", "position": scenario.base}))
    surfaces = [{"id": surface["id"], "kind": surface["kind"],
                 "footprint_grid": list(surface["footprint"]), "ground_altitude_m": 0.0}
                for surface in _layout(scenario).get("surfaces", [])]
    return {
        "frame": "x east, y south, z up; GRID scale is [3,3,5] metres",
        "objects": objects, "surfaces": surfaces,
        "ground_obstacles": deepcopy(_layout(scenario).get("ground_obstacles", [])),
        "altitude_range_m": {"min": scenario.min_altitude * SCALE[2], "max": scenario.max_altitude * SCALE[2]},
        "defaults": deepcopy(CONTROL_DEFAULTS),
        "fields": {
            "anchor": "center selects actual model/ground center; arrival selects the existing outdoor observation point",
            "offset_m": "[east,south,up] metres relative to the selected anchor; no automatic lateral relocation",
            "altitude_m": "absolute metres before offset_m[2]; null selects a collision-clear airborne altitude or ground for landed",
            "state": "airborne or landed; landed requires known flat ground support, clearance and safe vertical approach",
            "duration_seconds": "stable dwell from 0 to 600 seconds; null uses 2 seconds",
            "yaw_deg": "null preserves heading; angle in degrees, 0 north and 90 east"},
        "ground_policy": "Only flat ground surfaces and the established home pad are known supports. Buildings, roofs, roads, water and trees are excluded. A surface is not blanket landing permission; exact position and approach are checked. Positive airborne heights below the transit floor use the same verified vertical ground column; transfers remain above that floor.",
        "landing_clearance_m": LANDING_CLEARANCE_M,
    }


def normalize_control(value, target_id, catalog, constraints, *, path="$.control"):
    """Normalize one generic control request without inventing missing intent."""
    validate_structure(value, control_schema(CONTROL_DEFAULTS, _MAX_DURATION_SECONDS), path)
    if value is None:
        value = {}
    control = catalog.get("control", catalog)
    objects = {item["id"]: item for item in control["objects"]}
    if not isinstance(target_id, str) or target_id not in objects:
        raise TaskProtocolError(path, "invalid_reference", "Control target is not an available spatial object.", allowed=list(objects))
    result = deepcopy(CONTROL_DEFAULTS)
    result.update(value)
    offset = result["offset_m"]
    result["offset_m"] = [_number(value, "offset_m") for value in offset]
    if result["altitude_m"] is not None:
        result["altitude_m"] = _number(result["altitude_m"], "altitude_m", 0)
    duration = result["duration_seconds"]
    result["duration_seconds"] = _number(CONTROL_DEFAULTS["duration_seconds"] if duration is None else duration,
                                         "duration_seconds", 0, _MAX_DURATION_SECONDS)
    if result["yaw_deg"] is not None:
        result["yaw_deg"] = (_number(result["yaw_deg"], "yaw_deg", -math.inf, math.inf) + 180) % 360 - 180
    if result["state"] == "landed":
        if result["offset_m"][2] != 0 or result["altitude_m"] not in (None, 0):
            location = f"{path}.offset_m[2]" if result["offset_m"][2] != 0 else f"{path}.altitude_m"
            raise TaskProtocolError(location, "goal_state_conflict", "Landed control must use ground altitude: altitude_m null/0 and vertical offset 0.")
    elif result["altitude_m"] is not None:
        altitude = result["altitude_m"] + result["offset_m"][2]
        floor, ceiling = _altitude_bounds(control["altitude_range_m"], constraints)
        if not 0 < altitude <= ceiling:
            raise TaskProtocolError(f"{path}.altitude_m", "goal_state_conflict", "Requested airborne altitude conflicts with the ground floor or altitude ceiling.")
    return result


def _altitude_bounds(altitude_range, constraints):
    floor, ceiling = altitude_range["min"], altitude_range["max"]
    if constraints.get("max_altitude_m") is not None:
        ceiling = min(ceiling, _number(constraints["max_altitude_m"], "max_altitude_m", 0))
    if ceiling < floor:
        raise ValueError("Altitude ceiling is below the minimum transit altitude.")
    return floor, ceiling


def _context(scenario, constraints, *, ground=False):
    context = scenario.to_dict()
    _, ceiling = _altitude_bounds({"min": scenario.min_altitude * SCALE[2],
                                  "max": scenario.max_altitude * SCALE[2]}, constraints)
    context["altitude_range"]["max"] = ceiling / SCALE[2]
    if ground:
        context["altitude_range"]["min"] = 0.0
    regions = {item["id"]: item for item in scenario.semantic_regions}
    for region_id in constraints.get("avoid_regions", []):
        if region_id not in regions:
            raise ValueError("Control avoidance region is not available in this scene.")
        left, right, north, south = regions[region_id]["bounds"]
        context["no_fly_zones"].append(dict(min=dict(x=left, y=north, z=0),
                                           max=dict(x=right, y=south, z=scenario.max_altitude)))
    return context


def _contains(footprint, x, y, margin=0.0):
    left, right, north, south = footprint
    return (left - .39 + margin / SCALE[0] < x < right + .39 - margin / SCALE[0]
            and north - .39 + margin / SCALE[1] < y < south + .39 - margin / SCALE[1])


def _distance_to_segment_m(x, y, a, b):
    dx, dy = (b[0] - a[0]) * SCALE[0], (b[1] - a[1]) * SCALE[1]
    px, py = (x - a[0]) * SCALE[0], (y - a[1]) * SCALE[1]
    length2 = dx * dx + dy * dy
    t = max(0, min(1, (px * dx + py * dy) / length2)) if length2 else 0
    return math.hypot(px - t * dx, py - t * dy)


def _touches_line(x, y, line, margin):
    return any(_distance_to_segment_m(x, y, a, b) <= line["width"] * SCALE[0] / 2 + margin
               for a, b in zip(line["points"], line["points"][1:]))


def _trees(scenario):
    # The scene intentionally omits ornamental trees inside the launch apron.
    base = _point(scenario.base)
    for tree in _layout(scenario).get("trees", []):
        if math.hypot((tree["x"] - base[0]) * SCALE[0], (tree["y"] - base[1]) * SCALE[1]) >= 5.2:
            yield tree


def _landing_site(scenario, x, y, constraints):
    layout = _layout(scenario)
    base = _point(scenario.base)
    at_home = base[2] == 0 and math.hypot((x - base[0]) * SCALE[0], (y - base[1]) * SCALE[1]) < 1e-8
    supported = any(_contains(surface["footprint"], x, y, LANDING_CLEARANCE_M)
                    for surface in layout.get("surfaces", [])
                    if surface.get("kind") not in ("water", "road", "roof"))
    if not supported and not at_home:
        raise ValueError("Landing point lacks a known ground surface with sufficient support margin; choose a position inside a published ground surface.")
    if any(_contains(item["footprint"], x, y, -LANDING_CLEARANCE_M)
           for item in [*layout.get("buildings", []), *layout.get("roofs", [])]):
        raise ValueError("Landing point or vertical approach intersects a building or roof; rooftop landing is unavailable.")
    for obstacle in layout.get("ground_obstacles", []):
        lower = tuple(value - LANDING_CLEARANCE_M / scale
                      for value, scale in zip(obstacle["lower_grid"], SCALE))
        upper = tuple(value + LANDING_CLEARANCE_M / scale
                      for value, scale in zip(obstacle["upper_grid"], SCALE))
        if _intersects((x, y, 0.0), (x, y, scenario.min_altitude + .2), lower, upper):
            raise ValueError("Landing column intersects a low ground structure or its safety margin.")
    if any(_touches_line(x, y, item, LANDING_CLEARANCE_M) for item in layout.get("waterways", [])):
        raise ValueError("Landing point is within the water safety margin.")
    # Designated launch surfaces override decorative access roads sharing their
    # footprint; generic paving and sports surfaces receive no such exemption.
    launch_support = any(surface.get("kind") == "launch"
                         and _contains(surface["footprint"], x, y, LANDING_CLEARANCE_M)
                         for surface in layout.get("surfaces", []))
    if not launch_support and any(_touches_line(x, y, item, LANDING_CLEARANCE_M) for item in layout.get("roads", [])):
        raise ValueError("Landing point is within a road safety margin.")
    for tree in _trees(scenario):
        radius = min(1.65, tree["radius"] * SCALE[0])
        if math.hypot((x - tree["x"]) * SCALE[0], (y - tree["y"]) * SCALE[1]) <= radius + LANDING_CLEARANCE_M:
            raise ValueError("Landing point or vertical approach intersects a tree safety envelope.")
    point = (x, y, 0.0)
    checker = SafetyChecker(_context(scenario, constraints, ground=True), clearance=LANDING_CLEARANCE_M)
    if not checker.segment_safe(point, (x, y, float(scenario.min_altitude))):
        raise ValueError("Landing point or vertical approach conflicts with flight bounds, an obstacle or an avoided region.")
    return point


def _airborne_altitude(scenario, x, y, obj, constraints):
    floor, ceiling = _altitude_bounds({"min": scenario.min_altitude * SCALE[2],
                                      "max": scenario.max_altitude * SCALE[2]}, constraints)
    altitude = max(floor, min(obj["default_altitude_m"], ceiling))
    for item in [*_layout(scenario).get("buildings", []), *_layout(scenario).get("roofs", [])]:
        if _contains(item["footprint"], x, y, -LANDING_CLEARANCE_M):
            altitude = max(altitude, item["height"] * SCALE[2] + LANDING_CLEARANCE_M + .1)
    # Also covers generic scenes that have only collision columns, no campus layout.
    for bx, by, height in scenario.buildings:
        if _contains((bx, bx, by, by), x, y, -LANDING_CLEARANCE_M):
            altitude = max(altitude, height * SCALE[2] + LANDING_CLEARANCE_M + .1)
    return altitude


def resolve_control_goal(step, scenario, constraints):
    """Resolve an exact requested position; never replace an unsafe goal nearby."""
    catalog = control_catalog(scenario)
    target_id = step["target_id"]
    control = normalize_control(step.get("control"), target_id, catalog, constraints)
    obj = next(item for item in catalog["objects"] if item["id"] == target_id)
    anchor = obj["center_grid"] if control["anchor"] == "center" else obj["arrival_grid"]
    x, y = (anchor[i] + control["offset_m"][i] / SCALE[i] for i in (0, 1))
    if control["state"] == "landed":
        position = _landing_site(scenario, x, y, constraints)
        ground_site = position
    else:
        altitude = control["altitude_m"]
        if altitude is None:
            altitude = _airborne_altitude(scenario, x, y, obj, constraints)
        altitude += control["offset_m"][2]
        floor, ceiling = _altitude_bounds(catalog["altitude_range_m"], constraints)
        if not 0 < altitude <= ceiling:
            raise ValueError("Requested airborne altitude conflicts with the ground floor or altitude ceiling.")
        position = (x, y, altitude / SCALE[2])
        # Low hovering uses the same verified vertical access column as landing.
        # Transfer between columns still occurs above the normal transit floor.
        ground_site = _landing_site(scenario, x, y, constraints) if altitude < floor else None
        context = _context(scenario, constraints)
        if ground_site is not None:
            context["ground_sites"] = [dict(zip("xyz", ground_site))]
        errors = SafetyChecker(context).validate_waypoint(position)
        if errors:
            raise ValueError("Airborne control goal is unsafe: " + " ".join(dict.fromkeys(errors)))
        for tree in _trees(scenario):
            if (altitude <= min(2.7, tree["height"] * SCALE[2]) + SAFETY_MARGIN
                    and math.hypot((x - tree["x"]) * SCALE[0], (y - tree["y"]) * SCALE[1])
                    <= min(1.65, tree["radius"] * SCALE[0]) + SAFETY_MARGIN):
                raise ValueError("Airborne control goal intersects a tree safety envelope.")
    return {"position": position, "state": control["state"],
            "duration_seconds": control["duration_seconds"], "ground_site": ground_site}
