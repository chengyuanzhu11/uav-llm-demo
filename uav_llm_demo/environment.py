from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import dist
from typing import Iterable, Mapping

from .scenarios import DEFAULT_SCENARIO, ScenarioDefinition, load_scenario

GRID_SIZE = DEFAULT_SCENARIO.grid_size
MIN_ALTITUDE = DEFAULT_SCENARIO.min_altitude
MAX_ALTITUDE = DEFAULT_SCENARIO.max_altitude
BASE = DEFAULT_SCENARIO.base
TARGET = DEFAULT_SCENARIO.target
DYNAMIC_SAFETY_RADIUS = 1.6
METERS_PER_CELL = 3.0
METERS_PER_ALTITUDE_LEVEL = 5.0

BUILDINGS = DEFAULT_SCENARIO.building_map


@dataclass
class DynamicFlyingObject:
    object_id: str
    kind: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    radius: float = 0.7
    grid_size: int = GRID_SIZE
    min_altitude: float = 3.0
    max_altitude: float = 8.5

    def predicted_position(self) -> tuple[float, float, float]:
        if hasattr(self, "motion_prediction"):
            return self.motion_prediction
        return _bounce_position(
            self.position,
            self.velocity,
            self.grid_size,
            self.min_altitude,
            self.max_altitude,
        )[0]

    def advance(self) -> None:
        self.position, self.velocity = _bounce_position(
            self.position,
            self.velocity,
            self.grid_size,
            self.min_altitude,
            self.max_altitude,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.object_id,
            "kind": self.kind,
            "position": point_to_dict(self.position),
            "velocity": vector_to_dict(self.velocity),
            "predicted_position": point_to_dict(self.predicted_position()),
            "radius": self.radius,
        }


@dataclass(frozen=True)
class Observation:
    scenario_id: str
    step: int
    position: tuple[float, float, float]
    battery: int
    phase: str
    status: str
    goal: tuple[int, int, int]
    goal_label: str
    goal_distance: int
    nearest_static_obstacle: dict[str, object] | None
    dynamic_objects: tuple[dict[str, object], ...]
    nearest_dynamic_obstacle: dict[str, object] | None
    grid_size: int
    min_altitude: int
    max_altitude: int
    base: tuple[int, int, int]
    target: tuple[int, int, int]
    buildings: Mapping[tuple[int, int], int]

    def to_text(self) -> str:
        return (
            f"UAV position {format_point(self.position)}; goal {format_point(self.goal)}. "
            f"Grid bounds 0..{self.grid_size - 1}; altitude {self.min_altitude}..{self.max_altitude}. "
            "x increases east, y increases south, z increases up. "
            "Plan key waypoints for the whole route; never output flight-control actions."
        )

    def to_dict(self) -> dict[str, object]:
        dx, dy, dz = goal_delta(self.position, self.goal)
        return {
            "scenario_id": self.scenario_id,
            "step": self.step,
            "position": point_to_dict(self.position),
            "battery": self.battery,
            "phase": self.phase,
            "status": self.status,
            "goal": point_to_dict(self.goal),
            "goal_label": self.goal_label,
            "goal_distance": self.goal_distance,
            "goal_delta": {"dx": dx, "dy": dy, "dz": dz},
            "goal_direction": relative_direction_text(dx, dy, dz),
            "nearest_static_obstacle": self.nearest_static_obstacle,
            "dynamic_objects": list(self.dynamic_objects),
            "nearest_dynamic_obstacle": self.nearest_dynamic_obstacle,
            "altitude_range": {
                "min": self.min_altitude,
                "max": self.max_altitude,
            },
            "text": self.to_text(),
        }


class UAVEnvironment:
    def __init__(self, scenario: ScenarioDefinition | str | None = None) -> None:
        if scenario is None:
            scenario = DEFAULT_SCENARIO
        if isinstance(scenario, str):
            scenario = load_scenario(scenario)
        self.scenario = scenario
        self.scenario_id = scenario.scenario_id
        self.grid_size = scenario.grid_size
        self.min_altitude = scenario.min_altitude
        self.max_altitude = scenario.max_altitude
        self.base = scenario.base
        self.target = scenario.target
        self.buildings = scenario.building_map
        self.reset()

    def reset(self) -> None:
        self.position = self.base
        self.battery = 100
        self.phase = "fly_to_target"
        self.status = "active"
        self.step_count = 0
        self.path = [self.base]
        self.last_trajectory = [point_to_dict(self.base)]
        self.continuous_path = [point_to_dict(self.base)]
        self.last_trajectory_duration = 0.0
        self.kinematics = default_kinematics(self.base)
        self.history: list[dict[str, object]] = []
        self.dynamic_objects = [
            DynamicFlyingObject(
                item.object_id,
                item.kind,
                item.position,
                item.velocity,
                item.radius,
                self.grid_size,
                max(3.0, float(self.min_altitude)),
                min(8.5, float(self.max_altitude)),
            )
            for item in self.scenario.dynamic_objects
        ]
        self.last_feedback = (
            "Mission initialized in 3D airspace. Dynamic flying objects are active."
        )

    def get_observation(self) -> Observation:
        goal, goal_label = self._current_goal()
        dynamic_data = tuple(self._dynamic_observation())
        nearest_dynamic = min(
            dynamic_data,
            key=lambda item: item["predicted_distance"],
            default=None,
        )
        return Observation(
            scenario_id=self.scenario_id,
            step=self.step_count,
            position=self.position,
            battery=self.battery,
            phase=self.phase,
            status=self.status,
            goal=goal,
            goal_label=goal_label,
            goal_distance=manhattan_3d(self.position, goal),
            nearest_static_obstacle=nearest_building(self.position, self.buildings),
            dynamic_objects=dynamic_data,
            nearest_dynamic_obstacle=nearest_dynamic,
            grid_size=self.grid_size,
            min_altitude=self.min_altitude,
            max_altitude=self.max_altitude,
            base=self.base,
            target=self.target,
            buildings=self.buildings,
        )

    def snapshot(self, feedback: str | None = None) -> dict[str, object]:
        observation = self.get_observation()
        return {
            "scenario_id": self.scenario_id,
            "scenario": self.scenario.to_dict(),
            "grid_size": self.grid_size,
            "meters_per_cell": METERS_PER_CELL,
            "meters_per_altitude_level": METERS_PER_ALTITUDE_LEVEL,
            "altitude_range": {"min": self.min_altitude, "max": self.max_altitude},
            "base": point_to_dict(self.base),
            "target": point_to_dict(self.target),
            "buildings": [
                {"x": cell[0], "y": cell[1], "height": height}
                for cell, height in sorted(self.buildings.items())
            ],
            "dynamic_objects": [item.to_dict() for item in self.dynamic_objects],
            "uav": point_to_dict(self.position),
            "battery": self.battery,
            "phase": self.phase,
            "status": self.status,
            "step_count": self.step_count,
            "path": [point_to_dict(point) for point in self.path],
            "continuous_path": self.continuous_path,
            "trajectory": self.last_trajectory,
            "trajectory_duration": self.last_trajectory_duration,
            "kinematics": self.kinematics,
            "observation": observation.to_dict(),
            "feedback": feedback or self.last_feedback,
            "history": list(reversed(self.history[-12:])),
        }

    def _dynamic_observation(self) -> list[dict[str, object]]:
        output = []
        for item in self.dynamic_objects:
            predicted = item.predicted_position()
            predicted_distance = dist(self.position, predicted)
            output.append(
                {
                    **item.to_dict(),
                    "distance": dist(self.position, item.position),
                    "predicted_distance": predicted_distance,
                    "threat_level": threat_level(predicted_distance),
                }
            )
        return output

    def _current_goal(self) -> tuple[tuple[int, int, int], str]:
        if self.phase in {"returning_home", "completed"}:
            return self.base, "base"
        return self.target, "target"


def shortest_next_step(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    buildings: Mapping[tuple[int, int], int],
    dynamic_positions: Iterable[tuple[float, float, float]] = (),
    grid_size: int = GRID_SIZE,
    min_altitude: int = MIN_ALTITUDE,
    max_altitude: int = MAX_ALTITUDE,
) -> tuple[int, int, int] | None:
    if start == goal:
        return start

    queue: deque[tuple[int, int, int]] = deque([start])
    parents: dict[
        tuple[int, int, int],
        tuple[int, int, int] | None,
    ] = {start: None}

    while queue:
        cell = queue.popleft()
        for neighbor in neighbors(cell):
            if neighbor in parents:
                continue
            if not is_safe_cell(
                neighbor,
                buildings,
                grid_size,
                min_altitude,
                max_altitude,
            ):
                continue
            if conflicts_with_dynamic(neighbor, dynamic_positions):
                continue
            parents[neighbor] = cell
            if neighbor == goal:
                return first_step_from_parents(start, goal, parents)
            queue.append(neighbor)
    return None


def shortest_path_distance(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    buildings: Mapping[tuple[int, int], int],
    dynamic_positions: Iterable[tuple[float, float, float]] = (),
    grid_size: int = GRID_SIZE,
    min_altitude: int = MIN_ALTITUDE,
    max_altitude: int = MAX_ALTITUDE,
) -> int | None:
    if start == goal:
        return 0

    queue: deque[tuple[tuple[int, int, int], int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, distance = queue.popleft()
        for neighbor in neighbors(cell):
            if neighbor in seen:
                continue
            if not is_safe_cell(
                neighbor,
                buildings,
                grid_size,
                min_altitude,
                max_altitude,
            ):
                continue
            if conflicts_with_dynamic(neighbor, dynamic_positions):
                continue
            if neighbor == goal:
                return distance + 1
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def first_step_from_parents(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    parents: dict[
        tuple[int, int, int],
        tuple[int, int, int] | None,
    ],
) -> tuple[int, int, int]:
    current = goal
    while parents[current] != start:
        parent = parents[current]
        if parent is None:
            return start
        current = parent
    return current


def neighbors(point: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    return tuple(add_points(point, delta) for delta in ((0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)))


def is_safe_cell(
    point: tuple[int, int, int],
    buildings: Mapping[tuple[int, int], int],
    grid_size: int = GRID_SIZE,
    min_altitude: int = MIN_ALTITUDE,
    max_altitude: int = MAX_ALTITUDE,
) -> bool:
    x, y, z = point
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        return False
    if not (min_altitude <= z <= max_altitude):
        return False
    return z > buildings.get((x, y), 0)


def conflicts_with_dynamic(
    point: tuple[int, int, int],
    dynamic_positions: Iterable[tuple[float, float, float]],
) -> bool:
    return any(dist(point, dynamic) < DYNAMIC_SAFETY_RADIUS for dynamic in dynamic_positions)


def nearest_building(
    position: tuple[int, int, int],
    buildings: Mapping[tuple[int, int], int],
) -> dict[str, object] | None:
    if not buildings:
        return None
    horizontal = (position[0], position[1])
    cell = min(buildings, key=lambda item: manhattan_2d(horizontal, item))
    return {
        "position": {"x": cell[0], "y": cell[1]},
        "height": buildings[cell],
        "distance": manhattan_2d(horizontal, cell),
        "direction": compass_direction(horizontal, cell),
    }


def _bounce_position(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    grid_size: int = GRID_SIZE,
    min_altitude: float = 3.0,
    max_altitude: float = 8.5,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    values = list(position)
    speeds = list(velocity)
    bounds = (
        (1.0, grid_size - 2.0),
        (1.0, grid_size - 2.0),
        (float(min_altitude), float(max_altitude)),
    )
    for axis, (lower, upper) in enumerate(bounds):
        candidate = values[axis] + speeds[axis]
        if candidate < lower or candidate > upper:
            speeds[axis] *= -1
            candidate = values[axis] + speeds[axis]
        values[axis] = candidate
    return tuple(values), tuple(speeds)


def obstacle_to_dict(obstacle: dict[str, object] | None) -> dict[str, object] | None:
    return obstacle


def compass_direction(
    origin: tuple[int, int],
    target: tuple[int, int],
) -> str:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    horizontal = "east" if dx > 0 else "west" if dx < 0 else ""
    vertical = "south" if dy > 0 else "north" if dy < 0 else ""
    return "-".join(part for part in (vertical, horizontal) if part) or "same cell"


def threat_level(distance: float) -> str:
    if distance < DYNAMIC_SAFETY_RADIUS:
        return "critical"
    if distance < 3.0:
        return "high"
    if distance < 5.0:
        return "medium"
    return "low"


def manhattan_2d(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def manhattan_3d(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def goal_delta(
    position: tuple[int, int, int],
    goal: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        goal[0] - position[0],
        goal[1] - position[1],
        goal[2] - position[2],
    )


def relative_direction_text(dx: int, dy: int, dz: int) -> str:
    parts = []
    if dx > 0:
        parts.append("east/right")
    elif dx < 0:
        parts.append("west/left")
    if dy > 0:
        parts.append("south/down-map")
    elif dy < 0:
        parts.append("north/up-map")
    if dz > 0:
        parts.append("above")
    elif dz < 0:
        parts.append("below")
    return " and ".join(parts) if parts else "at the same 3D point"


def add_points(
    point: tuple[int, int, int],
    delta: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        point[0] + delta[0],
        point[1] + delta[1],
        point[2] + delta[2],
    )


def default_kinematics(
    position: tuple[int | float, int | float, int | float],
) -> dict[str, object]:
    return {
        "position": point_to_dict(position),
        "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "linear_velocity": vector_to_dict((0.0, 0.0, 0.0)),
        "angular_velocity": vector_to_dict((0.0, 0.0, 0.0)),
        "linear_acceleration": vector_to_dict((0.0, 0.0, 0.0)),
        "angular_acceleration": vector_to_dict((0.0, 0.0, 0.0)),
        "source": "planned",
    }


def point_to_dict(
    point: tuple[int | float, int | float, int | float],
) -> dict[str, int | float]:
    return {"x": point[0], "y": point[1], "z": point[2]}


def vector_to_dict(
    vector: tuple[float, float, float],
) -> dict[str, float]:
    return {"x": vector[0], "y": vector[1], "z": vector[2]}


def format_point(point: object) -> str:
    if isinstance(point, dict):
        return f"({point['x']}, {point['y']}, {point['z']})"
    values = tuple(point)
    return f"({values[0]}, {values[1]}, {values[2]})"


def format_horizontal_point(point: object) -> str:
    if isinstance(point, dict):
        return f"({point['x']}, {point['y']})"
    values = tuple(point)
    return f"({values[0]}, {values[1]})"


def format_vector(vector: object) -> str:
    if isinstance(vector, dict):
        return f"({vector['x']}, {vector['y']}, {vector['z']})"
    values = tuple(vector)
    return f"({values[0]}, {values[1]}, {values[2]})"
