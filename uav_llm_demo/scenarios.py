from __future__ import annotations

from dataclasses import dataclass


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
class ScenarioDefinition:
    scenario_id: str
    name: str
    grid_size: int
    min_altitude: int
    max_altitude: int
    base: tuple[int, int, int]
    target: tuple[int, int, int]
    buildings: tuple[tuple[int, int, int], ...]
    dynamic_objects: tuple[DynamicObjectSpec, ...]
    seed: int = 0

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
        }


def _point_to_dict(point: tuple[float, float, float]) -> dict[str, float]:
    return {"x": point[0], "y": point[1], "z": point[2]}


_URBAN_BUILDINGS = (
    (6, 5, 4),
    (6, 6, 4),
    (6, 7, 4),
    (6, 8, 4),
    (6, 9, 4),
    (6, 10, 4),
    (7, 10, 4),
    (8, 10, 5),
    (9, 10, 5),
    (10, 10, 3),
    (11, 10, 5),
    (12, 10, 5),
    (13, 10, 5),
    (13, 9, 4),
    (13, 8, 3),
    (13, 7, 5),
    (13, 6, 4),
    (13, 5, 3),
    (9, 4, 4),
    (10, 4, 5),
    (11, 4, 3),
    (12, 4, 5),
    (3, 14, 4),
    (4, 14, 4),
    (5, 14, 5),
    (15, 14, 4),
    (16, 14, 4),
    (17, 14, 3),
)


_URBAN_TRAFFIC = (
    DynamicObjectSpec("bird-1", "bird", (7.5, 15.5, 5.0), (0.7, -0.55, 0.0)),
    DynamicObjectSpec("bird-2", "bird", (16.5, 8.0, 6.5), (-0.6, 0.35, -0.08)),
    DynamicObjectSpec("bird-3", "bird", (4.0, 5.5, 4.0), (0.45, 0.65, 0.1)),
)


SCENARIOS = {
    "urban_crossing": ScenarioDefinition(
        scenario_id="urban_crossing",
        name="Urban Crossing",
        grid_size=20,
        min_altitude=2,
        max_altitude=10,
        base=(1, 18, 2),
        target=(18, 2, 5),
        buildings=_URBAN_BUILDINGS,
        dynamic_objects=_URBAN_TRAFFIC,
        seed=7,
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
        target=(18, 2, 5),
        buildings=tuple(item for item in _URBAN_BUILDINGS if item[0] in {6, 13}),
        dynamic_objects=(
            DynamicObjectSpec("bird-1", "bird", (10.0, 12.0, 5.0), (-0.25, -0.35, 0.0)),
            DynamicObjectSpec("bird-2", "bird", (15.0, 6.0, 6.0), (0.2, 0.25, 0.0)),
        ),
        seed=23,
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
