"""Short-horizon local trajectory replanning, independent of the remote planner."""
from dataclasses import dataclass
from math import cos, sin, pi

from .motion import SCALE, VehicleState, distance, norm
from .dynamic_risk import AVOIDANCE_BUFFER_M, separation_distance_m
from .traffic_prediction import TrafficPredictor
from .avoidance_evidence import AvoidanceEvidence


@dataclass(frozen=True)
class PredictedTraffic:
    radius_m: float
    positions: tuple
    uncertainty_horizontal_m: tuple
    uncertainty_vertical_m: tuple
    object_id: str | None = None


@dataclass(frozen=True)
class Rollout:
    clearance: float
    exposure: float
    end: tuple
    path: tuple = ()
    object_clearances: tuple = ()


class LocalAvoidance:
    HORIZON = 3.0
    DT = 0.15
    BUFFER = AVOIDANCE_BUFFER_M
    TERMINAL_HORIZON = 6.0

    def __init__(self, *, calibration_profile="combined"):
        self.active = False
        self.target = None
        self.since = None
        self.clear_since = None
        self.replans = 0
        self.message = ""
        self.limited = False
        self.next_search = 0.0
        self.progress_goal = None
        self.best_remaining = float("inf")
        self.last_progress = 0.0
        self.predictor = TrafficPredictor(calibration_profile=calibration_profile)
        self._radii = {}
        self.evidence = AvoidanceEvidence()

    def snapshot(self, *, include_visualization=False):
        result = dict(active=self.active, target=(dict(zip("xyz", self.target)) if self.target else None),
                    replans=self.replans, limited=self.limited, message=self.message,
                    prediction="history_damped_acceleration", horizon_seconds=self.HORIZON,
                    observed_tracks=len(self.predictor.object_ids))
        if include_visualization:
            result["visualization"] = self.evidence.snapshot()
        return result

    def observe(self, observations, now):
        self.predictor.update(observations, now)
        for item in observations:
            if item["id"] in self.predictor.object_ids:
                self._radii[item["id"]] = separation_distance_m(item["radius_m"])
        self._radii = {key: value for key, value in self._radii.items()
                       if key in self.predictor.object_ids}

    def predictions(self, state, now, max_speed=4.0):
        tracks = []
        for object_id in self.predictor.object_ids:
            forecast = self.predictor.forecast(object_id, now, self.TERMINAL_HORIZON, self.DT)
            if forecast is None:
                continue
            radius = self._radii[object_id]
            uncertainty = max(norm((horizontal, vertical)) for horizontal, vertical in zip(
                forecast.uncertainty_horizontal_m, forecast.uncertainty_vertical_m))
            if min(distance(state.position, p) for p in forecast.positions) <= max(max_speed, norm(state.velocity)) * self.TERMINAL_HORIZON + radius + uncertainty:
                tracks.append(PredictedTraffic(radius, forecast.positions,
                                               forecast.uncertainty_horizontal_m,
                                               forecast.uncertainty_vertical_m, object_id))
        return tracks

    def stalled(self, goal, remaining, now):
        """Measure goal/path progress without assuming a skippable waypoint queue."""
        if goal is None:
            return False
        if goal != self.progress_goal:
            self.progress_goal, self.best_remaining, self.last_progress = goal, remaining, now
        elif remaining < self.best_remaining - 0.3:
            self.best_remaining, self.last_progress = remaining, now
        return self.active and now - self.last_progress >= 8.0

    @staticmethod
    def _swept_distance(a, b):
        delta = tuple(y - x for x, y in zip(a, b))
        t = max(0, min(1, -sum(x * d for x, d in zip(a, delta)) / max(norm(delta) ** 2, 1e-12)))
        return norm(tuple(x + t * d for x, d in zip(a, delta)))

    @classmethod
    def _uncertain_swept_distance(cls, a, b, horizontal, vertical):
        # Distance from the relative swept line to a cylindrical error set.
        # Independent minima may occur at different times: their combination is
        # a conservative lower bound, never an optimistic clearance estimate.
        xy = max(0.0, cls._swept_distance(a[:2], b[:2]) - horizontal)
        z = 0.0 if a[2] * b[2] <= 0 else min(abs(a[2]), abs(b[2]))
        return (xy * xy + max(0.0, z - vertical) ** 2) ** .5

    def rollout(self, state, target, controller, checker, tracks, *, terminal=False, capture=False):
        if target is not None and not checker.segment_safe(state.position, target):
            return None
        clearance, exposure = float("inf"), 0.0
        settled_at = None
        path = [state.position] if capture else []
        object_clearances = {}
        horizon = self.TERMINAL_HORIZON if terminal else self.HORIZON
        for index in range(1, round(horizon / self.DT) + 1):
            velocity = controller.velocity_command(state, target, self.DT)
            position = tuple(state.position[i] + (state.velocity[i] + velocity[i]) * self.DT / (2 * SCALE[i])
                             for i in range(3))
            if not checker.segment_safe(state.position, position):
                return None
            for track in tracks:
                positions = track.positions
                if index >= len(positions):
                    return None
                a = tuple((positions[index - 1][i] - state.position[i]) * SCALE[i] for i in range(3))
                b = tuple((positions[index][i] - position[i]) * SCALE[i] for i in range(3))
                gap = self._uncertain_swept_distance(
                    a, b, max(track.uncertainty_horizontal_m[index - 1:index + 1]),
                    max(track.uncertainty_vertical_m[index - 1:index + 1])) - track.radius_m
                clearance = min(clearance, gap)
                exposure += max(0, self.BUFFER - gap) * self.DT
                if capture and track.object_id is not None:
                    object_clearances[track.object_id] = min(object_clearances.get(track.object_id, float("inf")), gap)
            state = VehicleState(position, velocity)
            if capture:
                path.append(position)
            if terminal:
                if distance(position, target) <= 0.18 and norm(velocity) <= 0.25:
                    settled_at = index * self.DT if settled_at is None else settled_at
                else:
                    settled_at = None
        if terminal and (settled_at is None or horizon - settled_at < 0.75):
            return None
        return Rollout(clearance, exposure, state.position, tuple(path), tuple(object_clearances.items()))

    def choose(self, state, nominal, controller, checker, tracks, now, *, terminal=False):
        was_active = self.active
        target = self._choose(state, nominal, controller, checker, tracks, now, terminal=terminal)
        # Capture the same controller simulation used for decisions. Diagnostics
        # never feed back into candidate selection or consume world truth.
        if self.evidence.due(now, self.active, was_active):
            nominal_rollout = self.rollout(state, nominal, controller, checker, tracks, capture=True,
                terminal=terminal and nominal is not None and distance(state.position, nominal) <= 6)
            selected_rollout = self.rollout(state, target, controller, checker, tracks, capture=True) if self.active else None
            self.evidence.record(now, self.active, self.limited, nominal_rollout, selected_rollout, self.BUFFER)
        return target

    def _choose(self, state, nominal, controller, checker, tracks, now, *, terminal=False):
        """Return a temporary waypoint (or hold); never alter the mission queue."""
        starting = not self.active
        terminal = terminal and nominal is not None and distance(state.position, nominal) <= 6
        direct = self.rollout(state, nominal, controller, checker, tracks, terminal=terminal) if tracks else None
        safe_direct = not tracks or (direct is not None and direct.clearance >= self.BUFFER)
        if safe_direct:
            if not self.active:
                return nominal
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since >= 0.75:
                self.active, self.target, self.since = False, None, None
                self.limited = False
                self.message = "Traffic clear; resuming mission route." if nominal else "Traffic clear; holding."
                return nominal
        else:
            self.clear_since = None
        existing = None
        if self.active:
            existing = self.rollout(state, self.target, controller, checker, tracks)
            if existing is not None and existing.clearance >= self.BUFFER:
                if self.limited:
                    self.limited = False
                    self.message = "Local avoidance: current local trajectory is predicted-safe."
                if now < self.next_search:
                    return self.target
        if not self.active:
            self.since = now
        self.active = True
        self.next_search = now + 0.5
        candidates = []
        if existing is not None:
            candidates.append((self.target, existing))
        horizontal = [(cos(i * pi / 4), sin(i * pi / 4), 0) for i in range(8)]
        vertical = [(0, 0, 1), (0, 0, -1)] + [
            (x / 2 ** .5, y / 2 ** .5, z / 2 ** .5)
            for x, y, _ in horizontal for z in (-1, 1)]
        # Compare all three dimensions; height preference is a cost, not a veto.
        for length in (3.0, 6.0):
            for direction in horizontal + vertical:
                target = tuple(state.position[i] + length * direction[i] / SCALE[i] for i in range(3))
                result = self.rollout(state, target, controller, checker, tracks)
                if result is not None:
                    candidates.append((target, result))
        hold = self.rollout(state, None, controller, checker, tracks)
        if hold is not None:
            candidates.append((None, hold))
        safe = [(t, r) for t, r in candidates if r.clearance >= self.BUFFER]
        self.limited = not bool(safe)

        def cost(item):
            target, result = item
            progress = distance(result.end, nominal) if nominal else distance(result.end, state.position) * .1
            hysteresis = distance(target, self.target) * .2 if target and self.target else 0
            altitude = checker.context.get("preferred_altitude")
            height_cost = (max(0, altitude["min"] - result.end[2], result.end[2] - altitude["max"]) * SCALE[2] * .15
                           if altitude else abs(result.end[2] - state.position[2]) * SCALE[2] * .15)
            effort = distance(state.position, result.end) * .08
            return progress + hysteresis + height_cost + effort - min(result.clearance, 3) * .1

        if safe:
            target, _ = min(safe, key=cost)
            if existing is not None and existing.clearance >= self.BUFFER:
                if cost((self.target, existing)) <= min(cost(item) for item in safe) + 0.5:
                    target = self.target
        elif candidates:
            # No collision-free option: minimize predicted intrusion, without
            # crossing a building. This is mitigation, not a safety guarantee.
            target, _ = max(candidates, key=lambda item: (item[1].clearance - item[1].exposure, -cost(item)))
        else:
            target = None
        if target != self.target or starting:
            self.replans += 1
        self.target = target
        self.message = ("Local avoidance: no collision-free trajectory found; best-effort escape."
                        if self.limited else "Local avoidance: tracking a predicted-safe detour."
                        if target else "Local avoidance: braking/holding is the predicted-safe option.")
        return target
