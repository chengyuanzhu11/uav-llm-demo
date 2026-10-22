from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy

from .campus_layout import CAMPUS_LAYOUT, campus_building_cells


@dataclass(frozen=True)
class DynamicObjectSpec:
    object_id: str
    kind: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    radius: float = 0.7

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.object_id,
            "kind": self.kind,
            "position": _point_to_dict(self.position),
            "velocity": _point_to_dict(self.velocity),
            "radius": self.radius,
        }


@dataclass(frozen=True)
class UrbanParcel:
    parcel_id: str
    use: str
    footprint: tuple[int, int, int, int]
    podium_height: int
    facade: str
    tower: tuple[int, int, int, int, int] | None = None
    profile: str = "slab"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.parcel_id,
            "use": self.use,
            "footprint": self.footprint,
            "podium_height": self.podium_height,
            "facade": self.facade,
            "tower": self.tower,
            "profile": self.profile,
        }


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    name: str
    grid_size: int
    min_altitude: int
    max_altitude: int
    base: tuple[int, int, int]
    target: tuple[float, float, float]
    buildings: tuple[tuple[int, int, int], ...]
    dynamic_objects: tuple[DynamicObjectSpec, ...]
    seed: int = 0
    no_fly_zones: tuple[dict, ...] = ()
    preferred_altitude: tuple[float, float] | None = None
    urban_parcels: tuple[UrbanParcel, ...] = ()
    urban_street_centers: tuple[float, ...] = ()
    mission_mode: str = "navigation"
    points_of_interest: tuple[dict, ...] = ()
    recommended_tasks: tuple[str, ...] = ()
    semantic_regions: tuple[dict, ...] = ()
    campus_layout: dict | None = None

    @property
    def building_map(self) -> dict[tuple[int, int], int]:
        return {(x, y): height for x, y, height in self.buildings}

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.scenario_id,
            "name": self.name,
            "grid_size": self.grid_size,
            "altitude_range": {
                "min": self.min_altitude,
                "max": self.max_altitude,
            },
            "base": _point_to_dict(self.base),
            "target": _point_to_dict(self.target),
            "buildings": [
                {"x": x, "y": y, "height": height}
                for x, y, height in self.buildings
            ],
            "dynamic_objects": [item.to_dict() for item in self.dynamic_objects],
            "seed": self.seed,
            "no_fly_zones": list(self.no_fly_zones),
            "preferred_altitude": (dict(min=self.preferred_altitude[0], max=self.preferred_altitude[1])
                                   if self.preferred_altitude else None),
            "urban_parcels": [parcel.to_dict() for parcel in self.urban_parcels],
            "urban_street_centers": list(self.urban_street_centers),
            "mission_mode": self.mission_mode,
            "points_of_interest": deepcopy(list(self.points_of_interest)),
            "recommended_tasks": list(self.recommended_tasks),
            "semantic_regions": deepcopy(list(self.semantic_regions)),
            "campus_layout": deepcopy(self.campus_layout),
        }


def _point_to_dict(point: tuple[float, float, float]) -> dict[str, float]:
    return {"x": point[0], "y": point[1], "z": point[2]}


_URBAN_DISTRICT_PARCELS = (
    UrbanParcel("northwest-office", "office", (2, 5, 2, 6), 3, "glass", (3, 4, 3, 5, 8)),
    UrbanParcel("northwest-residential", "residential", (6, 7, 3, 6), 4, "brick"),
    UrbanParcel("northeast-mixed", "mixed", (12, 15, 2, 6), 3, "bronze", (13, 14, 3, 5, 7)),
    UrbanParcel("northeast-civic", "civic", (17, 18, 4, 7), 3, "stone"),
    UrbanParcel("southwest-residential", "residential", (2, 5, 13, 17), 4, "terracotta"),
    UrbanParcel("southwest-retail", "retail", (6, 7, 14, 17), 2, "stone"),
    UrbanParcel("southeast-office", "office", (12, 15, 13, 17), 3, "silver", (13, 14, 14, 16, 8)),
    UrbanParcel("southeast-residential", "residential", (17, 18, 12, 15), 4, "brick"),
)


_URBAN_PARCELS = tuple(
    replace(parcel,
            parcel_id=prefix + parcel.parcel_id,
            footprint=tuple(value + (dx if index < 2 else dy)
                            for index, value in enumerate(parcel.footprint)),
            podium_height=podium, facade=facade, profile=profile,
            tower=(tuple(value + (dx if index < 2 else dy)
                         for index, value in enumerate(parcel.tower[:4])) + (height,)
                   if parcel.tower else None))
    for prefix, dx, dy, styles in (
        ("", 0, 0, (
            (3, "stone", "slab", 12), (5, "brick", "slab", 0),
            (2, "bronze", "braced", 11), (3, "stone", "slab", 0),
            (4, "terracotta", "slab", 0), (2, "stone", "slab", 0),
            (3, "glass", "round", 10), (6, "silver", "slab", 0))),
        ("east-", 20, 0, (
            (2, "silver", "slab", 9), (3, "stone", "slab", 0),
            (3, "stone", "slab", 13), (5, "brick", "slab", 0),
            (3, "bronze", "slab", 0), (2, "terracotta", "slab", 0),
            (4, "bronze", "braced", 12), (4, "stone", "slab", 0))),
        ("south-", 0, 20, (
            (3, "glass", "round", 11), (4, "terracotta", "slab", 0),
            (2, "silver", "slab", 8), (2, "stone", "slab", 0),
            (5, "stone", "slab", 0), (3, "brick", "slab", 0),
            (2, "stone", "slab", 10), (3, "bronze", "slab", 0))),
        ("southeast-", 20, 20, (
            (2, "bronze", "braced", 13), (6, "silver", "slab", 0),
            (3, "glass", "round", 9), (4, "stone", "slab", 0),
            (3, "brick", "slab", 0), (2, "terracotta", "slab", 0),
            (3, "silver", "slab", 11), (5, "stone", "slab", 0))),
    )
    for parcel, (podium, facade, profile, height) in zip(_URBAN_DISTRICT_PARCELS, styles)
)


def _parcel_buildings(parcels: tuple[UrbanParcel, ...]) -> tuple[tuple[int, int, int], ...]:
    cells: dict[tuple[int, int], int] = {}
    for parcel in parcels:
        left, right, top, bottom = parcel.footprint
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                cells[x, y] = max(cells.get((x, y), 0), parcel.podium_height)
        if parcel.tower:
            left, right, top, bottom, height = parcel.tower
            for x in range(left, right + 1):
                for y in range(top, bottom + 1):
                    cells[x, y] = max(cells.get((x, y), 0), height)
    return tuple((x, y, height) for (x, y), height in sorted(cells.items()))


_CAMPUS_REGIONS = (
    dict(id="south_plaza", label="南广场区域", bounds=[40, 57, 78, 86]),
    dict(id="south_residential", label="南区生活区", bounds=[61, 78, 56, 74]),
    dict(id="north_residential", label="北区生活区", bounds=[50, 79, 7, 25]),
)


_URBAN_TRAFFIC = (
    DynamicObjectSpec("bird-1", "bird", (2.0, 9.5, 2.0), (0.0, 0.0, 0.0)),
    DynamicObjectSpec("bird-2", "bird", (9.5, 2.0, 2.4), (0.0, 0.0, 0.0)),
    DynamicObjectSpec("bird-3", "bird", (2.0, 29.5, 3.2), (0.0, 0.0, 0.0)),
    DynamicObjectSpec("bird-4", "bird", (29.5, 2.0, 4.0), (0.0, 0.0, 0.0)),
    DynamicObjectSpec("bird-5", "bird", (3.0, 9.5, 5.2), (0.0, 0.0, 0.0), radius=0.45),
    DynamicObjectSpec("bird-6", "bird", (9.5, 3.0, 5.6), (0.0, 0.0, 0.0), radius=0.45),
    DynamicObjectSpec("bird-7", "bird", (3.0, 29.5, 2.0), (0.0, 0.0, 0.0), radius=0.45),
    DynamicObjectSpec("bird-8", "bird", (29.5, 3.0, 2.8), (0.0, 0.0, 0.0), radius=0.45),
)


SCENARIOS = {
    "urban_crossing": ScenarioDefinition(
        scenario_id="urban_crossing",
        name="Urban Crossing",
        grid_size=40,
        min_altitude=2,
        max_altitude=10,
        base=(1, 18, 0),
        target=(38, 2, 5),
        buildings=_parcel_buildings(_URBAN_PARCELS),
        dynamic_objects=_URBAN_TRAFFIC,
        seed=7,
        urban_parcels=_URBAN_PARCELS,
        urban_street_centers=(9.5, 29.5),
    ),
    "semantic_campus": ScenarioDefinition(
        scenario_id="semantic_campus",
        name="北航杭州国际校园",
        grid_size=96,
        min_altitude=2,
        max_altitude=10,
        base=(48, 88, 0),
        target=(53, 74, 3),
        buildings=campus_building_cells(),
        dynamic_objects=(
            DynamicObjectSpec("campus-bird-1", "bird", (38, 10, 2.4), (0.0, 0.0, 0.0)),
            DynamicObjectSpec("campus-bird-2", "bird", (20, 27, 3.6), (0.0, 0.0, 0.0)),
            DynamicObjectSpec("campus-bird-3", "bird", (20, 76, 2.0), (0.0, 0.0, 0.0)),
        ),
        seed=31,
        mission_mode="semantic",
        points_of_interest=tuple({key: deepcopy(place[key])
                                  for key in ("id", "label", "code", "kind", "position", "allowed_actions")}
                                 for place in CAMPUS_LAYOUT["places"]),
        recommended_tasks=(
            "沿田径场顺时针飞行两圈，机头始终朝向场地中心，最后返航。",
            "在田径场上空飞一个八字，飞行过程中逐渐升高4米，最后返航。",
            "检查图书馆温度，超过25摄氏度就模拟采样，否则巡检会议中心，最后返航。",
            "重复巡检图书馆和教学一号楼3次，最后返航。",
            "反复检查实验一号楼，温度低于40摄氏度或检查满5次后返航。",
            "去北区足球场中央降落停留，不用返航。",
        ),
        semantic_regions=_CAMPUS_REGIONS,
        campus_layout=CAMPUS_LAYOUT,
    ),
}

DEFAULT_SCENARIO = SCENARIOS["urban_crossing"]


def load_scenario(scenario_id: str = "urban_crossing") -> ScenarioDefinition:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{scenario_id}'. Choose one of: {available}.") from exc


def list_scenarios() -> list[dict[str, object]]:
    return [scenario.to_dict() for scenario in SCENARIOS.values()]
