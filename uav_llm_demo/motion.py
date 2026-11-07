"""Metric velocity tracking and waypoint progression, independent of any planner."""
from dataclasses import dataclass
from math import sqrt, isfinite

SCALE = (3.0, 3.0, 5.0)
MAX_TRACKING_DEVIATION_M = 3.0


def norm(vector):
    return sqrt(sum(v * v for v in vector))


def distance(a, b):
    return norm(tuple((a[i] - b[i]) * SCALE[i] for i in range(3)))


def segment_deviation(position, start, end):
    """Distance in metres to a closed 3D reference segment."""
    offset = tuple((position[i] - start[i]) * SCALE[i] for i in range(3))
    direction = tuple((end[i] - start[i]) * SCALE[i] for i in range(3))
    fraction = max(0, min(1, sum(a * b for a, b in zip(offset, direction))
                          / max(norm(direction) ** 2, 1e-12)))
    return norm(tuple(a - b * fraction for a, b in zip(offset, direction)))


@dataclass(frozen=True)
class VehicleState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    sample_time: float | None = None  # Absolute simulator time for native feedback.


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

    def tracking_target(self, state, lookahead_m=8.0):
        """A bounded reference on the accepted segment, preserving its slope.

        Pointing at a distant endpoint dilutes perpendicular/height correction.
        Project onto the segment and look ahead, without skipping a waypoint or
        changing the validated route. Near its end the reference is the endpoint.
        """
        end = self.current
        if end is None:
            return None
        start = self.segment_start
        vector = tuple((end[i]-start[i])*SCALE[i] for i in range(3))
        length = norm(vector)
        if length < 1e-9:
            return end
        offset = tuple((state.position[i]-start[i])*SCALE[i] for i in range(3))
        along = sum(a*b for a,b in zip(offset,vector))/length
        fraction = min(1., (max(0.,along)+lookahead_m)/length)
        return tuple(start[i]+(end[i]-start[i])*fraction for i in range(3))

    def reconnect(self, state, checker, reachable):
        """Only ordinary transit waypoints may be skipped; always retain the goal."""
        if self.current is None:
            return False
        remaining = self.remaining_distance(state.position)
        furthest = len(self.points) - 1
        if self.plan_start[2] < checker.lower[2]:
            departure = next((i for i, p in enumerate(self.points) if p[2] >= checker.lower[2]), None)
            if departure is not None and self.index <= departure:
                # Complete the takeoff column before a horizontal shortcut.
                # Being close to the cruise floor does not prove stable ascent.
                furthest = min(furthest, departure)
        landing_site = checker.ground_site_for(self.points[-1])
        if landing_site is not None and self.points[-1][2] < checker.lower[2]:
            # Reach and decelerate at the entrance to the landing column before
            # selecting a lower point. Intermediate collinear descents may still
            # be skipped once that approach waypoint has actually been reached.
            approach = furthest
            for index in range(furthest - 1, -1, -1):
                point, following = self.points[index:index + 2]
                horizontal = norm(tuple((point[i] - landing_site[i]) * SCALE[i] for i in range(2)))
                if horizontal > checker.pad_radius_m or point[2] < following[2]:
                    break
                approach = index
            if self.index <= approach:
                furthest = min(furthest, approach)
        for index in range(furthest, self.index, -1):
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
    def __init__(self, max_speed=4.0, max_acceleration=2.0, max_climb_speed=2.0, max_descent_speed=1.0,
                 response_seconds=0.0):
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_climb_speed = max_climb_speed
        self.max_descent_speed = max_descent_speed
        self.response_seconds = response_seconds

    def velocity_command(self, state, target, dt, *, landing_radius=None):
        if not isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("Controller dt must be in (0, 0.2] seconds.")
        desired = (0.0, 0.0, 0.0)
        if target is not None:
            delta = tuple((target[i] - state.position[i]) * SCALE[i] for i in range(3))
            length = norm(delta)
            if length > 1e-9:
                braking_acceleration = self.max_acceleration * 0.5
                # Reserve both response distance v*tau and braking distance
                # v²/(2a). Smoothing acceleration without this reserve makes
                # a physical follower brake too late at corners/endpoints.
                lag_speed = braking_acceleration * self.response_seconds
                braking_speed = sqrt(lag_speed**2 + 2*braking_acceleration*length) - lag_speed
                speed = min(self.max_speed, braking_speed, length * 1.4)
                if self.response_seconds:
                    # If actual closing speed already exceeds the envelope,
                    # reserve the distance the aircraft will travel before it responds.
                    closing = max(0., sum(v*d for v, d in zip(state.velocity, delta)) / length)
                    available = max(0., length - closing*self.response_seconds)
                    speed = min(speed, sqrt(2*braking_acceleration*available), 1.4*available)
                desired = tuple(v * speed / length for v in delta)
                vertical_limit = self.max_climb_speed if desired[2] >= 0 else self.max_descent_speed
                factor = min(1.0, vertical_limit / max(abs(desired[2]), 1e-12))
                desired = tuple(v * factor for v in desired)
                if landing_radius is not None and delta[2] < 0:
                    # A vertical approach needs horizontal position correction
                    # independent of its remaining height and descent speed.
                    horizontal = norm(delta[:2])
                    horizontal_speed = min(self.max_speed, horizontal*1.4,
                        sqrt(lag_speed**2 + 2*braking_acceleration*horizontal)-lag_speed)
                    lateral_speed = norm(state.velocity[:2])
                    stopping = lateral_speed*self.response_seconds + lateral_speed**2/(2*braking_acceleration)
                    # Reserve margin before the hard column boundary. Stop
                    # descending while lateral motion is braking/recentring.
                    descent_fraction = max(0., min(1.,
                        (.7*landing_radius-horizontal-stopping)/max(.3*landing_radius, 1e-9)))
                    down = min(self.max_descent_speed, -delta[2]*1.4,
                        sqrt(lag_speed**2 - 2*braking_acceleration*delta[2])-lag_speed)*descent_fraction
                    horizontal_speed = min(horizontal_speed, sqrt(max(0., self.max_speed**2-down**2)))
                    desired = tuple(v*horizontal_speed/max(horizontal, 1e-12) for v in delta[:2]) + (-down,)
        change = tuple(desired[i] - state.velocity[i] for i in range(3))
        # A physical attitude/thrust loop cannot track bang-bang acceleration
        # around zero velocity. Keep the ideal-runtime default, but give native
        # flight the same smooth response in both execution and safety rollouts.
        factor = min(1.0, dt / max(dt, self.response_seconds),
                     self.max_acceleration * dt / max(norm(change), 1e-12))
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
        if segment_deviation(state.position, manager.segment_start, target) > MAX_TRACKING_DEVIATION_M:
            return "tracking_deviation_exceeded_3m"
        remaining = distance(state.position, target)
        if manager.index != self.waypoint_index or remaining < self.best_distance - 0.15:
            self.best_distance = remaining
            self.last_progress = now
            self.waypoint_index = manager.index
        if now - self.last_progress > 20:
            return "waypoint_stalled_20s"
        return None
