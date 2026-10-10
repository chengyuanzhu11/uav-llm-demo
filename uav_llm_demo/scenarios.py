from __future__ import annotations

from dataclasses import dataclass, replace


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


_NARROW_BUILDINGS = tuple((x, y, 4) for x in (6, 13) for y in range(5, 11))


_URBAN_TRAFFIC = (
    DynamicObjectSpec("bird-1", "bird", (7.5, 15.5, 5.0), (0.7, -0.55, 0.0)),
    DynamicObjectSpec("bird-2", "bird", (16.5, 8.0, 6.5), (-0.6, 0.35, -0.08)),
    DynamicObjectSpec("bird-3", "bird", (4.0, 5.5, 4.0), (0.45, 0.65, 0.1)),
)


SCENARIOS = {
    "urban_crossing": ScenarioDefinition(
        scenario_id="urban_crossing",
        name="Urban Crossing",
        grid_size=40,
        min_altitude=2,
        max_altitude=10,
        base=(1, 18, 2),
        target=(38, 2, 2),
        buildings=_parcel_buildings(_URBAN_PARCELS),
        dynamic_objects=_URBAN_TRAFFIC,
        seed=7,
        preferred_altitude=(2.0, 2.4),
        urban_parcels=_URBAN_PARCELS,
        urban_street_centers=(9.5, 29.5),
    ),
    "open_field": ScenarioDefinition(
        scenario_id="open_field",
        name="Open Field",
        grid_size=20,
        min_altitude=2,
        max_altitude=10,
        base=(1, 18, 2),
        target=(18, 2, 5),
        buildings=(),
        dynamic_objects=(
            DynamicObjectSpec("bird-1", "bird", (9.5, 10.5, 5.0), (0.35, -0.2, 0.0)),
        ),
        seed=11,
    ),
    "narrow_corridor": ScenarioDefinition(
        scenario_id="narrow_corridor",
        name="Narrow Corridor",
        grid_size=20,
        min_altitude=2,
        max_altitude=10,
        base=(1, 18, 2),
        target=(18, 2, 2),
        buildings=_NARROW_BUILDINGS,
        dynamic_objects=(
            DynamicObjectSpec("bird-1", "bird", (10.0, 12.0, 5.0), (-0.25, -0.35, 0.0)),
            DynamicObjectSpec("bird-2", "bird", (15.0, 6.0, 6.0), (0.2, 0.25, 0.0)),
        ),
        seed=23,
        preferred_altitude=(2.0, 2.4),
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
