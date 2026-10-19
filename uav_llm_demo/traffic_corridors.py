"""Map-space bird corridors, constructed without mission or vehicle inputs.

Coordinates use the simulator grid (3, 3, 5 metres). Each corridor is a
fixed out-and-back centerline; altitude variation is part of that centerline.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .safety import SafetyChecker, metric_distance


@dataclass(frozen=True)
class TrafficCorridor:
    corridor_id: str
    label: str
    polyline: tuple[tuple[float, float, float], ...]

    def to_dict(self):
        return dict(id=self.corridor_id, label=self.label, polyline=self.polyline,
                    altitude_range_m=dict(min=min(p[2] for p in self.polyline) * 5,
                                          max=max(p[2] for p in self.polyline) * 5),
                    coordinate_unit="grid", motion="out_and_back")


_URBAN = (
    TrafficCorridor("urban-north-low", "北侧道路低空走廊", ((2., 9.5, 2.), (37., 9.5, 2.8))),
    TrafficCorridor("urban-west-low", "西侧道路低空走廊", ((9.5, 2., 2.4), (9.5, 37., 2.))),
    TrafficCorridor("urban-south-mid", "南侧道路中空走廊", ((2., 29.5, 3.2), (37., 29.5, 4.))),
    TrafficCorridor("urban-east-mid", "东侧道路中空走廊", ((29.5, 2., 4.), (29.5, 37., 3.2))),
    TrafficCorridor("urban-north-high", "北侧道路高空走廊", ((3., 9.5, 5.2), (36., 9.5, 6.))),
    TrafficCorridor("urban-west-high", "西侧道路高空走廊", ((9.5, 3., 5.6), (9.5, 36., 4.8))),
    TrafficCorridor("urban-south-low", "南侧道路低空走廊", ((3., 29.5, 2.), (36., 29.5, 2.8))),
    TrafficCorridor("urban-east-low", "东侧道路低空走廊", ((29.5, 3., 2.8), (29.5, 36., 2.))),
)
_CAMPUS = (
    TrafficCorridor("campus-west", "校园西侧道路走廊", ((38., 10., 2.4), (38., 88., 3.2))),
    TrafficCorridor("campus-north", "校园北侧道路走廊", ((20., 27., 3.6), (78., 27., 4.8))),
    TrafficCorridor("campus-south", "校园南侧道路走廊", ((20., 76., 2.), (78., 76., 2.8))),
)


def traffic_checker(scenario, radius):
    """Use physical map geometry and bird size; never the landing exception."""
    context = dict(grid_size=scenario.grid_size, altitude_range=dict(min=1.6, max=6.4),
                   buildings=[dict(x=x, y=y, height=height) for x, y, height in scenario.buildings],
                   no_fly_zones=list(scenario.no_fly_zones))
    checker = SafetyChecker(context, clearance=radius * 3 + .2)
    edge = max(1., radius + .1)
    checker.lower = (edge, edge, max(1.6, checker.clearance / 5))
    checker.upper = (scenario.grid_size - 1 - edge, scenario.grid_size - 1 - edge, 6.4)
    return checker


def _safe(corridor, checker):
    return all(checker.segment_safe(a, b) for a, b in zip(corridor.polyline, corridor.polyline[1:]))


def _geometry_corridors(checker):
    """Deterministic straight clear-space runs for custom or small test maps."""
    corridors, isolated = [], []
    lower, upper = checker.lower, checker.upper
    for height in (2., 3.2, 4.4, 5.6):
        if not lower[2] <= height <= upper[2]:
            continue
        for axis in (0, 1):
            for cross in range(int(lower[1-axis]), int(upper[1-axis]) + 1):
                run = []
                for along in range(int(lower[axis]), int(upper[axis]) + 2):
                    point = ((float(along), float(cross), height) if axis == 0
                             else (float(cross), float(along), height))
                    valid = not checker.validate_waypoint(point)
                    if run and (not valid or not checker.segment_safe(run[-1], point)):
                        if len(run) > 1:
                            corridors.append(TrafficCorridor(
                                f"geometry-{axis}-{cross}-{height}-{run[0][axis]}",
                                "地图空隙走廊", (run[0], run[-1])))
                        else:
                            isolated.append(run[0])
                        run = []
                    if valid:
                        run.append(point)
    # Prefer useful multi-metre corridors, then stable coordinate order.
    corridors.sort(key=lambda item: (-metric_distance(*item.polyline), item.corridor_id))
    if corridors:
        return tuple(corridors)
    if isolated:
        point = sorted(set(isolated))[0]
        return (TrafficCorridor("geometry-isolated", "局部安全空域", (point, point)),)
    raise ValueError("No statically safe bird corridor exists in this map.")


def corridor_catalog(scenario, checker):
    presets = {"urban_crossing": _URBAN, "semantic_campus": _CAMPUS}.get(scenario.scenario_id, ())
    safe = tuple(corridor for corridor in presets if _safe(corridor, checker))
    return safe or _geometry_corridors(checker)


def select_corridor(catalog, object_id):
    # Existing IDs have stable numbered slots. Arbitrary fixture IDs use a stable
    # digest rather than process-randomized hash() or collection iteration order.
    suffix = object_id.rsplit("-", 1)[-1]
    slot = int(suffix) - 1 if suffix.isdecimal() and int(suffix) > 0 else int.from_bytes(
        hashlib.sha256(object_id.encode("utf-8")).digest()[:8], "big")
    return catalog[slot % len(catalog)]
