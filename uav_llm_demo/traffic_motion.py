"""Collision-checked, time-parameterized bird routes shared with the renderer."""
import math
import random
import time

from .safety import metric_distance
from .traffic_corridors import corridor_catalog, select_corridor, traffic_checker


TRAFFIC_POLICIES = ("fixed",)
TRAFFIC_MODEL_VERSION = "map-corridors-v2"


def sample_motion(model, elapsed):
    segments = model["segments"]
    segment = next((item for item in segments if elapsed < item["start"] + item["duration"]), segments[-1])
    fraction = min(1.0, max(0.0, (elapsed - segment["start"]) / segment["duration"]))
    remaining = 1 - fraction
    controls = segment["controls"]
    position = tuple(remaining ** 3 * controls[0][axis] + 3 * remaining ** 2 * fraction * controls[1][axis]
                     + 3 * remaining * fraction ** 2 * controls[2][axis] + fraction ** 3 * controls[3][axis]
                     for axis in range(3))
    velocity = tuple(3 * (remaining ** 2 * (controls[1][axis] - controls[0][axis])
                         + 2 * remaining * fraction * (controls[2][axis] - controls[1][axis])
                         + fraction ** 2 * (controls[3][axis] - controls[2][axis])) / segment["duration"]
                     for axis in range(3))
    return position, velocity


class ContinuousTraffic:
    """Periodic map-space traffic independent of UAV goals, plans and position."""
    def __init__(self, scenario, *, seed=None, clock=time.monotonic, policy="fixed"):
        if policy not in TRAFFIC_POLICIES:
            raise ValueError("Traffic policy must be fixed.")
        self.policy = policy
        self.clock = clock
        self.started = clock()
        resolved_seed = random.SystemRandom().getrandbits(128) if seed is None else seed
        self.rngs = {bird.object_id: random.Random(f"{resolved_seed}:{bird.object_id}")
                     for bird in scenario.dynamic_objects}
        self.models = {}
        self.checkers = {}
        self.corridors = {}
        self.cycles = {}
        catalogs = {}
        for bird in scenario.dynamic_objects:
            if bird.radius not in catalogs:
                checker = traffic_checker(scenario, bird.radius)
                catalogs[bird.radius] = checker, corridor_catalog(scenario, checker)
            checker, catalog = catalogs[bird.radius]
            corridor = select_corridor(catalog, bird.object_id)
            start, end = corridor.polyline
            controls = (start, start, end, end)
            rng = self.rngs[bird.object_id]
            speed = rng.uniform(2.4, 3.6)
            # Cubic smoothstep reaches zero velocity at both reversals. Its
            # derivative is bounded analytically by 1.5 * chord / duration.
            duration = max(4.0, 1.5 * metric_distance(start, end) / speed)
            offset = rng.uniform(0.0, duration * 2)
            self.checkers[bird.object_id] = checker
            self.corridors[bird.object_id] = dict(corridor.to_dict(), object_id=bird.object_id,
                                                  period_seconds=duration * 2,
                                                  phase_seconds=offset,
                                                  peak_speed_mps=1.5 * metric_distance(start, end) / duration)
            self.cycles[bird.object_id] = (controls, duration, offset)
            self.models[bird.object_id] = dict(segments=[], phase=rng.uniform(0, math.tau),
                                               corridor_id=corridor.corridor_id)
            self._window(bird.object_id, 0.0)

    def _window(self, object_id, elapsed):
        """Reconstruct the same short curve window at any simulation cadence."""
        controls, duration, offset = self.cycles[object_id]
        first = math.floor((max(0.0, elapsed - 1.0) + offset) / duration)
        last = math.floor((elapsed + 8.0 + offset) / duration)
        self.models[object_id]["segments"] = [
            dict(start=index * duration - offset, duration=duration,
                 controls=controls if index % 2 == 0 else controls[::-1])
            for index in range(first, last + 1)]

    def sync(self, environment):
        elapsed = max(0.0, self.clock() - self.started)
        for bird in environment.dynamic_objects:
            self._window(bird.object_id, elapsed)
            bird.position, bird.velocity = sample_motion(self.models[bird.object_id], elapsed)
            bird.motion_prediction = tuple(p + v for p, v in zip(bird.position, bird.velocity))
        return {"elapsed": elapsed, "models": self.models, "corridors": self.corridors,
                "policy": self.policy, "model_version": TRAFFIC_MODEL_VERSION}
