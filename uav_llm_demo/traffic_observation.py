"""Causal simulated traffic observations, separate from rendered world truth."""
from collections import deque
from dataclasses import dataclass
import random

from .dynamic_risk import physical_radius_m
from .safety import SCALE, _intersects, _number, _point, metric_distance


@dataclass(frozen=True)
class TrafficObservationConfig:
    range_m: float = 60.0
    sample_period_seconds: float = 0.1
    latency_seconds: float = 0.1
    occlusion: bool = True
    position_noise_m: float = 0.0
    velocity_noise_mps: float = 0.0
    dropout_probability: float = 0.0

    def __post_init__(self):
        for field in ('range_m', 'sample_period_seconds', 'latency_seconds',
                      'position_noise_m', 'velocity_noise_mps', 'dropout_probability'):
            value = _number(getattr(self, field))
            if value < 0 or (field in {'range_m', 'sample_period_seconds'} and value == 0):
                requirement = 'positive' if field in {'range_m', 'sample_period_seconds'} else 'nonnegative'
                raise ValueError(f'{field} must be finite and {requirement}.')
        if self.dropout_probability > 1:
            raise ValueError('dropout_probability must be between zero and one.')
        if not isinstance(self.occlusion, bool):
            raise ValueError('occlusion must be boolean.')


class TrafficObserver:
    """Snapshot visible current states, then deliver them after configured delay.

    Positions use grid coordinates and velocities grid units/s; sensor ranges,
    per-axis Gaussian noise standard deviations and obstacle radii use SI units.
    ``sample`` releases observations exactly once. ``latest`` reads only already
    delivered snapshots. Neither method reads a traffic model or future samples.
    """
    STALE_SECONDS = 1.5

    def __init__(self, config=None, *, seed=None):
        if config is not None and not isinstance(config, TrafficObservationConfig):
            raise ValueError('config must be TrafficObservationConfig or None.')
        self.config = config if config is not None else TrafficObservationConfig()
        self._seed = random.SystemRandom().getrandbits(128) if seed is None else seed
        self.reset()

    def reset(self):
        self._now = None
        self._next_sample_at = None
        self._pending = deque()
        self._latest = {}
        self._rngs = {}

    def _time(self, now):
        now = _number(now)
        if self._now is not None and now < self._now:
            raise ValueError('world clock moved backwards; reset observer first')
        self._now = now
        return now

    @staticmethod
    def _physical_buildings(checker):
        # Navigation boxes include safety clearance; only physical buildings
        # obscure the sensor. Administrative no-fly zones are transparent.
        buildings = []
        for lower, upper, label in checker.boxes:
            if label != 'building':
                continue
            physical_lower = tuple(value + checker.clearance / scale for value, scale in zip(lower, SCALE))
            physical_upper = tuple(value - checker.clearance / scale for value, scale in zip(upper, SCALE))
            buildings.append((physical_lower, physical_upper))
        return buildings

    @staticmethod
    def _occluded(start, end, buildings):
        # Broad phase is conservative, including the slab test's parameter
        # tolerance at touching corners and segment endpoints.
        padding = tuple(1e-12 * max(1.0, abs(b - a)) for a, b in zip(start, end))
        lower = tuple(min(a, b) - pad for a, b, pad in zip(start, end, padding))
        upper = tuple(max(a, b) + pad for a, b, pad in zip(start, end, padding))
        for lo, hi in buildings:
            if (upper[0] < lo[0] or lower[0] > hi[0]
                    or upper[1] < lo[1] or lower[1] > hi[1]
                    or upper[2] < lo[2] or lower[2] > hi[2]):
                continue
            if _intersects(start, end, lo, hi):
                return True
        return False

    def _expire(self, now):
        for object_id in list(self._latest):
            if now - self._latest[object_id]['observed_at'] > self.STALE_SECONDS:
                del self._latest[object_id]

    def sample(self, birds, vehicle_position, checker, now):
        now = self._time(now)
        vehicle_position = _point(vehicle_position)
        if self.config.occlusion and checker is None:
            raise ValueError('Occlusion requires a SafetyChecker.')
        if self._next_sample_at is None or now + 1e-9 >= self._next_sample_at:
            self._next_sample_at = now + self.config.sample_period_seconds
            buildings = self._physical_buildings(checker) if self.config.occlusion else ()
            for bird in birds or ():
                try:
                    object_id = bird.object_id
                    if not isinstance(object_id, str) or not object_id:
                        continue
                    position, velocity = _point(bird.position), _point(bird.velocity)
                    radius_m = physical_radius_m(bird)
                    radius = _number(bird.radius)
                    if radius < 0:
                        continue
                except (AttributeError, TypeError, ValueError, OverflowError):
                    continue
                if metric_distance(position, vehicle_position) > self.config.range_m:
                    continue
                if self._occluded(vehicle_position, position, buildings):
                    continue
                rng = self._rngs.get(object_id)
                if rng is None:
                    rng = self._rngs[object_id] = random.Random(f'{self._seed}:sensor:{object_id}')
                if self.config.dropout_probability and rng.random() < self.config.dropout_probability:
                    continue
                if self.config.position_noise_m:
                    position = tuple(value + rng.gauss(0, self.config.position_noise_m) / scale
                                     for value, scale in zip(position, SCALE))
                if self.config.velocity_noise_mps:
                    velocity = tuple(value + rng.gauss(0, self.config.velocity_noise_mps) / scale
                                     for value, scale in zip(velocity, SCALE))
                observation = dict(id=object_id, kind=bird.kind, position=position, velocity=velocity,
                                   observed_at=now, radius=radius, radius_m=radius_m)
                self._pending.append((now + self.config.latency_seconds, observation))
        delivered = []
        while self._pending:
            delivery_at, observation = self._pending[0]
            expired = now - observation['observed_at'] > self.STALE_SECONDS
            if not expired and delivery_at > now + 1e-9:
                break
            self._pending.popleft()
            if not expired:
                self._latest[observation['id']] = observation.copy()
                delivered.append(observation.copy())
        self._expire(now)
        return delivered

    def latest(self, now):
        now = self._time(now)
        self._expire(now)
        return [self._latest[key].copy() for key in sorted(self._latest)]
