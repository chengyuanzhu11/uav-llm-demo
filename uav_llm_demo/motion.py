"""Metric velocity tracking and waypoint progression, independent of any planner."""
from dataclasses import dataclass
from math import sqrt, isfinite

SCALE = (3.0, 3.0, 5.0)


def norm(vector):
    return sqrt(sum(v * v for v in vector))


def distance(a, b):
    return norm(tuple((a[i] - b[i]) * SCALE[i] for i in range(3)))


@dataclass(frozen=True)
class VehicleState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0


class WaypointManager:
    def __init__(self):
        self.points = ()
        self.index = 0
        self.segment_start = None
        self.plan_start = None
        self.skipped = 0

    @property
    def current(self):
        return self.points[self.index] if self.index < len(self.points) else None

    def load(self, points, start):
        self.points = tuple(tuple(p) for p in points)
        self.index = 0
        self.segment_start = tuple(start)
        self.plan_start = tuple(start)
        self.skipped = 0

    def remaining_distance(self, position):
        points = (position,) + self.points[self.index:]
        return sum(distance(a, b) for a, b in zip(points, points[1:]))

    def reconnect(self, state, checker, reachable):
        """Only ordinary transit waypoints may be skipped; always retain the goal."""
        if self.current is None:
            return False
        remaining = self.remaining_distance(state.position)
        for index in range(len(self.points) - 1, self.index, -1):
            point = self.points[index]
            tail = self.points[index:]
            cost = distance(state.position, point) + sum(distance(a, b) for a, b in zip(tail, tail[1:]))
            passing = distance(state.position, self.current) <= 0.8
            if (passing or cost < remaining - 0.5) and checker.segment_safe(state.position, point) and reachable(point, index == len(self.points) - 1):
                self.skipped += index - self.index
                self.index, self.segment_start = index, state.position
                return True
        return False

    def update(self, state):
        speed_limit = 0.02 if self.index == len(self.points) - 1 else 0.25
        if self.current is not None and distance(state.position, self.current) <= 0.18 and norm(state.velocity) <= speed_limit:
            self.segment_start = self.current
            self.index += 1
            return True
        return False


class TrajectoryController:
    def __init__(self, max_speed=4.0, max_acceleration=2.0, max_climb_speed=2.0, max_descent_speed=1.0):
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_climb_speed = max_climb_speed
        self.max_descent_speed = max_descent_speed

    def velocity_command(self, state, target, dt):
        if not isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("Controller dt must be in (0, 0.2] seconds.")
        desired = (0.0, 0.0, 0.0)
        if target is not None:
            delta = tuple((target[i] - state.position[i]) * SCALE[i] for i in range(3))
            length = norm(delta)
            if length > 1e-9:
                braking_acceleration = self.max_acceleration * 0.5
                speed = min(self.max_speed, sqrt(2 * braking_acceleration * length), length * 1.4)
                desired = tuple(v * speed / length for v in delta)
                vertical_limit = self.max_climb_speed if desired[2] >= 0 else self.max_descent_speed
                factor = min(1.0, vertical_limit / max(abs(desired[2]), 1e-12))
                desired = tuple(v * factor for v in desired)
        change = tuple(desired[i] - state.velocity[i] for i in range(3))
        factor = min(1.0, self.max_acceleration * dt / max(norm(change), 1e-12))
        return tuple(state.velocity[i] + change[i] * factor for i in range(3))


class ReplanningManager:
    def __init__(self):
        self.goal = None
        self.map_signature = None
        self.best_distance = float("inf")
        self.last_progress = 0.0
        self.waypoint_index = -1

    def accepted(self, context, now):
        self.goal = context["goal"]
        self.map_signature = repr((context["buildings"], context["no_fly_zones"]))
        self.best_distance = float("inf")
        self.last_progress = now
        self.waypoint_index = -1

    def should_replan(self, context, state, manager, checker, now):
        if self.goal != context["goal"]:
            return "mission_goal_changed"
        if self.map_signature != repr((context["buildings"], context["no_fly_zones"])):
            return "obstacles_or_restricted_zones_changed"
        target = manager.current
        if target is None:
            return None
        if checker.validate_waypoint(target):
            return "waypoint_unreachable"
        if not checker.segment_safe(state.position, target):
            return "route_blocked"
        start = manager.segment_start
        a = tuple((state.position[i] - start[i]) * SCALE[i] for i in range(3))
        b = tuple((target[i] - start[i]) * SCALE[i] for i in range(3))
        t = max(0, min(1, sum(a[i] * b[i] for i in range(3)) / max(norm(b) ** 2, 1e-12)))
        if norm(tuple(a[i] - b[i] * t for i in range(3))) > 3.0:
            return "tracking_deviation_exceeded_3m"
        remaining = distance(state.position, target)
        if manager.index != self.waypoint_index or remaining < self.best_distance - 0.15:
            self.best_distance = remaining
            self.last_progress = now
            self.waypoint_index = manager.index
        if now - self.last_progress > 20:
            return "waypoint_stalled_20s"
        return None
