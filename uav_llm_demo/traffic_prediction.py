"""Causal metric motion extrapolation and empirically calibrated error envelopes.

This is a deterministic, exponentially damped acceleration model inspired by the
acceleration-decay assumption of Singer (1970), DOI 10.1109/TAES.1970.310128.
It is neither a Kalman/IMM implementation nor QCNet. No future traffic routes,
map intent, or privileged generator state are accepted by this interface.
Reference: https://stonesoup.readthedocs.io/en/latest/stonesoup.models.transition.html

"""
from collections import deque
from dataclasses import dataclass
import math

from .motion import SCALE


@dataclass(frozen=True)
class Forecast:
    positions: tuple
    uncertainty_m: tuple
    model: str
    observed_at: float
    age_seconds: float
    uncertainty_horizontal_m: tuple
    uncertainty_vertical_m: tuple


@dataclass(frozen=True)
class _Observation:
    time: float
    position: tuple
    velocity: tuple


class TrafficPredictor:
    """Keep up to 0.8 s of observations; forget tracks absent for 1.5 s.

    Input position and velocity use simulation grid units (and grid units/s).
    Estimation, acceleration limits and uncertainty are in SI units. Forecasts
    start at ``now`` including propagation through the observation age. Error
    envelopes are empirical simulator quantiles, not guaranteed safety bounds.
    Unknown map profiles use the component-wise maximum of both calibrated maps.
    Custom decay/speed settings require new calibration before interpreting coverage.
    """
    HISTORY_SECONDS = 0.8
    STALE_SECONDS = 1.5
    MAX_ACCELERATION_MPS2 = 4.0
    DECAY_SECONDS = 2.0
    # Frozen on map-corridors-v2: both maps, seeds 2/4/6/8, 120 s each,
    # 10 Hz observations delayed by 0.1 s. Radial q99 is diagnostic; horizontal
    # and vertical q99.5 are separate empirical bounds. Joint coverage must be
    # measured separately. Reproduce with scripts/verify_traffic_prediction.py.
    CALIBRATION_TRAFFIC_VERSION = "map-corridors-v2"
    CALIBRATION_PROFILES = {
        "urban_crossing": {
            "radial": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.07), (1.1, 0.15),
                (1.6, 0.25), (2.1, 0.4), (2.6, 0.6), (3.1, 0.85),
                (3.6, 1.17), (4.1, 1.54), (4.6, 1.97), (5.1, 2.48),
                (5.6, 3.04), (6.1, 3.69), (6.6, 4.4), (7.1, 5.16),
                (7.6, 6.01), (8.1, 6.91),
            ),
            "cold_radial": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.12), (1.1, 0.29),
                (1.6, 0.54), (2.1, 0.85), (2.6, 1.24), (3.1, 1.7),
                (3.6, 2.24), (4.1, 2.84), (4.6, 3.51), (5.1, 4.27),
                (5.6, 5.08), (6.1, 5.95), (6.6, 6.9), (7.1, 7.93),
                (7.6, 8.99), (8.1, 10.13),
            ),
            "horizontal": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.07), (1.1, 0.15),
                (1.6, 0.26), (2.1, 0.42), (2.6, 0.62), (3.1, 0.89),
                (3.6, 1.21), (4.1, 1.59), (4.6, 2.05), (5.1, 2.57),
                (5.6, 3.17), (6.1, 3.82), (6.6, 4.57), (7.1, 5.36),
                (7.6, 6.21), (8.1, 7.16),
            ),
            "vertical": (
                (0.0, 0.0), (0.1, 0.01), (0.6, 0.02), (1.1, 0.03),
                (1.6, 0.04), (2.1, 0.05), (2.6, 0.07), (3.1, 0.09),
                (3.6, 0.11), (4.1, 0.13), (4.6, 0.16), (5.1, 0.19),
                (5.6, 0.22), (6.1, 0.26), (6.6, 0.29), (7.1, 0.33),
                (7.6, 0.38), (8.1, 0.42),
            ),
            "cold_horizontal": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.12), (1.1, 0.3),
                (1.6, 0.55), (2.1, 0.88), (2.6, 1.29), (3.1, 1.77),
                (3.6, 2.32), (4.1, 2.95), (4.6, 3.64), (5.1, 4.42),
                (5.6, 5.26), (6.1, 6.17), (6.6, 7.14), (7.1, 8.19),
                (7.6, 9.28), (8.1, 10.47),
            ),
            "cold_vertical": (
                (0.0, 0.0), (0.1, 0.01), (0.6, 0.02), (1.1, 0.03),
                (1.6, 0.05), (2.1, 0.07), (2.6, 0.1), (3.1, 0.13),
                (3.6, 0.16), (4.1, 0.19), (4.6, 0.23), (5.1, 0.26),
                (5.6, 0.31), (6.1, 0.35), (6.6, 0.4), (7.1, 0.45),
                (7.6, 0.5), (8.1, 0.56),
            ),
        },
        "semantic_campus": {
            "radial": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.07), (1.1, 0.13),
                (1.6, 0.21), (2.1, 0.31), (2.6, 0.44), (3.1, 0.6),
                (3.6, 0.79), (4.1, 1.01), (4.6, 1.26), (5.1, 1.55),
                (5.6, 1.88), (6.1, 2.23), (6.6, 2.63), (7.1, 3.06),
                (7.6, 3.52), (8.1, 4.02),
            ),
            "cold_radial": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.09), (1.1, 0.21),
                (1.6, 0.37), (2.1, 0.56), (2.6, 0.79), (3.1, 1.06),
                (3.6, 1.38), (4.1, 1.72), (4.6, 2.11), (5.1, 2.53),
                (5.6, 2.99), (6.1, 3.48), (6.6, 4.01), (7.1, 4.58),
                (7.6, 5.18), (8.1, 5.82),
            ),
            "horizontal": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.07), (1.1, 0.13),
                (1.6, 0.21), (2.1, 0.32), (2.6, 0.45), (3.1, 0.61),
                (3.6, 0.81), (4.1, 1.04), (4.6, 1.31), (5.1, 1.61),
                (5.6, 1.96), (6.1, 2.35), (6.6, 2.79), (7.1, 3.27),
                (7.6, 3.77), (8.1, 4.31),
            ),
            "vertical": (
                (0.0, 0.0), (0.1, 0.01), (0.6, 0.02), (1.1, 0.03),
                (1.6, 0.04), (2.1, 0.05), (2.6, 0.06), (3.1, 0.08),
                (3.6, 0.09), (4.1, 0.1), (4.6, 0.12), (5.1, 0.14),
                (5.6, 0.16), (6.1, 0.17), (6.6, 0.19), (7.1, 0.22),
                (7.6, 0.24), (8.1, 0.26),
            ),
            "cold_horizontal": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.1), (1.1, 0.22),
                (1.6, 0.38), (2.1, 0.59), (2.6, 0.84), (3.1, 1.12),
                (3.6, 1.46), (4.1, 1.84), (4.6, 2.24), (5.1, 2.71),
                (5.6, 3.21), (6.1, 3.74), (6.6, 4.34), (7.1, 4.97),
                (7.6, 5.62), (8.1, 6.33),
            ),
            "cold_vertical": (
                (0.0, 0.0), (0.1, 0.01), (0.6, 0.02), (1.1, 0.03),
                (1.6, 0.04), (2.1, 0.06), (2.6, 0.07), (3.1, 0.09),
                (3.6, 0.11), (4.1, 0.13), (4.6, 0.14), (5.1, 0.17),
                (5.6, 0.19), (6.1, 0.21), (6.6, 0.23), (7.1, 0.26),
                (7.6, 0.29), (8.1, 0.31),
            ),
        },
        "combined": {
            "radial": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.07), (1.1, 0.15),
                (1.6, 0.25), (2.1, 0.4), (2.6, 0.6), (3.1, 0.85),
                (3.6, 1.17), (4.1, 1.54), (4.6, 1.97), (5.1, 2.48),
                (5.6, 3.04), (6.1, 3.69), (6.6, 4.4), (7.1, 5.16),
                (7.6, 6.01), (8.1, 6.91),
            ),
            "cold_radial": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.12), (1.1, 0.29),
                (1.6, 0.54), (2.1, 0.85), (2.6, 1.24), (3.1, 1.7),
                (3.6, 2.24), (4.1, 2.84), (4.6, 3.51), (5.1, 4.27),
                (5.6, 5.08), (6.1, 5.95), (6.6, 6.9), (7.1, 7.93),
                (7.6, 8.99), (8.1, 10.13),
            ),
            "horizontal": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.07), (1.1, 0.15),
                (1.6, 0.26), (2.1, 0.42), (2.6, 0.62), (3.1, 0.89),
                (3.6, 1.21), (4.1, 1.59), (4.6, 2.05), (5.1, 2.57),
                (5.6, 3.17), (6.1, 3.82), (6.6, 4.57), (7.1, 5.36),
                (7.6, 6.21), (8.1, 7.16),
            ),
            "vertical": (
                (0.0, 0.0), (0.1, 0.01), (0.6, 0.02), (1.1, 0.03),
                (1.6, 0.04), (2.1, 0.05), (2.6, 0.07), (3.1, 0.09),
                (3.6, 0.11), (4.1, 0.13), (4.6, 0.16), (5.1, 0.19),
                (5.6, 0.22), (6.1, 0.26), (6.6, 0.29), (7.1, 0.33),
                (7.6, 0.38), (8.1, 0.42),
            ),
            "cold_horizontal": (
                (0.0, 0.0), (0.1, 0.02), (0.6, 0.12), (1.1, 0.3),
                (1.6, 0.55), (2.1, 0.88), (2.6, 1.29), (3.1, 1.77),
                (3.6, 2.32), (4.1, 2.95), (4.6, 3.64), (5.1, 4.42),
                (5.6, 5.26), (6.1, 6.17), (6.6, 7.14), (7.1, 8.19),
                (7.6, 9.28), (8.1, 10.47),
            ),
            "cold_vertical": (
                (0.0, 0.0), (0.1, 0.01), (0.6, 0.02), (1.1, 0.03),
                (1.6, 0.05), (2.1, 0.07), (2.6, 0.1), (3.1, 0.13),
                (3.6, 0.16), (4.1, 0.19), (4.6, 0.23), (5.1, 0.26),
                (5.6, 0.31), (6.1, 0.35), (6.6, 0.4), (7.1, 0.45),
                (7.6, 0.5), (8.1, 0.56),
            ),
        },
    }

    def __init__(self, *, stale_seconds=STALE_SECONDS, decay_seconds=DECAY_SECONDS, max_speed_mps=4.0, calibration_profile="combined"):
        if not math.isfinite(stale_seconds) or stale_seconds <= 0:
            raise ValueError("stale_seconds must be finite and positive")
        if not math.isfinite(decay_seconds) or decay_seconds <= 0:
            raise ValueError("decay_seconds must be finite and positive")
        if not math.isfinite(max_speed_mps) or max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be finite and positive")
        self.calibration_profile = calibration_profile if calibration_profile in self.CALIBRATION_PROFILES else "combined"
        self.max_speed_mps = float(max_speed_mps)
        self.stale_seconds = float(stale_seconds)
        self.decay_seconds = float(decay_seconds)
        self._tracks = {}
        self._now = None

    def reset(self):
        self._tracks.clear()
        self._now = None

    @staticmethod
    def _vector(value):
        if isinstance(value, dict):
            value = tuple(value[key] for key in "xyz")
        values = tuple(float(item) for item in value)
        if len(values) != 3 or not all(math.isfinite(item) for item in values):
            raise ValueError("motion vectors must contain three finite numbers")
        return values

    @staticmethod
    def _time(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("time must be finite")
        return value

    def update(self, observations, now):
        """Ignore invalid, future, repeated and out-of-order observations.

        A backward world clock is rejected; callers must explicitly ``reset``
        when resetting the simulation. Missing objects retain their last causal
        state until stale; an empty observation list therefore still expires it.
        """
        now = self._time(now)
        if self._now is not None and now < self._now:
            raise ValueError("world clock moved backwards; reset predictor first")
        self._now = now
        for item in observations or ():
            try:
                object_id = item["id"]
                if not isinstance(object_id, str) or not object_id:
                    continue
                observed_at = self._time(item["observed_at"])
                if observed_at > now + 1e-9 or now - observed_at > self.stale_seconds:
                    continue
                position = self._vector(item["position"])
                velocity = self._vector(item["velocity"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            history = self._tracks.setdefault(object_id, deque(maxlen=64))
            if history and observed_at <= history[-1].time + 1e-9:
                continue
            if history and observed_at - history[-1].time > self.stale_seconds:
                history.clear()
            history.append(_Observation(observed_at, position, velocity))
            while len(history) > 1 and history[0].time < observed_at - self.HISTORY_SECONDS:
                history.popleft()
        self._expire(now)

    def _expire(self, now):
        for object_id in list(self._tracks):
            if now - self._tracks[object_id][-1].time > self.stale_seconds:
                del self._tracks[object_id]

    @property
    def object_ids(self):
        return tuple(self._tracks)

    def _acceleration(self, history):
        if len(history) < 3 or history[-1].time - history[0].time < 0.19:
            return (0.0, 0.0, 0.0), "observed_constant_velocity", (0.0, 0.0, 0.0)
        times = [item.time - history[-1].time for item in history]
        average = sum(times) / len(times)
        denominator = sum((value - average) ** 2 for value in times)
        acceleration = tuple(sum((time - average) * item.velocity[axis] * SCALE[axis]
                                 for time, item in zip(times, history)) / denominator
                             for axis in range(3))
        length = math.hypot(*acceleration)
        if length > self.MAX_ACCELERATION_MPS2:
            acceleration = tuple(value * self.MAX_ACCELERATION_MPS2 / length for value in acceleration)
        # Recent fit residual grows the envelope when history is not locally linear.
        means = [sum(item.velocity[axis] * SCALE[axis] for item in history) / len(history)
                 for axis in range(3)]
        residuals = [tuple(item.velocity[axis] * SCALE[axis] - means[axis]
                           - acceleration[axis] * (time - average) for axis in range(3))
                     for time, item in zip(times, history)]
        residual = (max(math.hypot(*value) for value in residuals),
                    max(math.hypot(*value[:2]) for value in residuals),
                    max(abs(value[2]) for value in residuals))
        return acceleration, "history_damped_acceleration", residual

    @staticmethod
    def _envelope(seconds, knots):
        if seconds <= 0:
            return 0.0
        for left, right in zip(knots, knots[1:]):
            if seconds <= right[0]:
                fraction = (seconds - left[0]) / (right[0] - left[0])
                return left[1] + fraction * (right[1] - left[1])
        left, right = knots[-2:]
        return right[1] + (seconds - right[0]) * (right[1] - left[1]) / (right[0] - left[0])

    def forecast(self, object_id, now, horizon=6.0, dt=0.15):
        now = self._time(now)
        horizon, dt = float(horizon), float(dt)
        if not math.isfinite(horizon) or not 0 <= horizon <= 12:
            raise ValueError("horizon must be between zero and twelve seconds")
        if not math.isfinite(dt) or not 0 < dt <= 1 or horizon / dt > 4000:
            raise ValueError("dt must be positive, at most one second and produce no more than 4001 samples")
        if self._now is not None and now < self._now:
            raise ValueError("forecast cannot inspect a past world state")
        self._expire(now)
        history = self._tracks.get(object_id)
        if not history:
            return None
        last = history[-1]
        age = max(0.0, now - last.time)
        acceleration, model, residual = self._acceleration(history)
        cold = model == "observed_constant_velocity"
        profile = self.CALIBRATION_PROFILES[self.calibration_profile]
        knots = profile["cold_radial" if cold else "radial"]
        horizontal_knots = profile["cold_horizontal" if cold else "horizontal"]
        vertical_knots = profile["cold_vertical" if cold else "vertical"]
        positions, uncertainties, horizontal_uncertainty, vertical_uncertainty = [], [], [], []
        times = [index * dt for index in range(math.floor(horizon / dt + 1e-9) + 1)]
        if times[-1] < horizon - 1e-9:
            times.append(horizon)
        tau = self.decay_seconds
        # The known type-level speed limit bounds extrapolation; never infer it
        # from hidden future segments. Preserve a faster observed velocity rather
        # than creating an artificial instantaneous slowdown.
        velocity = tuple(value * scale for value, scale in zip(last.velocity, SCALE))
        speed_limit = max(self.max_speed_mps, math.hypot(*velocity))
        metric_position = tuple(value * scale for value, scale in zip(last.position, SCALE))
        elapsed = 0.0
        for offset in times:
            seconds = age + offset
            if math.hypot(*acceleration) < 1e-10:
                position = tuple(last.position[axis] + last.velocity[axis] * seconds for axis in range(3))
            else:
                # Fixed bounded integration also supports aged observations; no
                # output sample assumes a fresh observation at the current time.
                while elapsed < seconds - 1e-10:
                    step = min(.05, seconds - elapsed)
                    gain = tau * math.exp(-elapsed / tau) * (-math.expm1(-step / tau))
                    next_velocity = tuple(velocity[axis] + acceleration[axis] * gain for axis in range(3))
                    speed = math.hypot(*next_velocity)
                    if speed > speed_limit:
                        next_velocity = tuple(value * speed_limit / speed for value in next_velocity)
                    metric_position = tuple(metric_position[axis] + (velocity[axis] + next_velocity[axis]) * step / 2
                                            for axis in range(3))
                    velocity = next_velocity
                    elapsed += step
                position = tuple(value / scale for value, scale in zip(metric_position, SCALE))
            positions.append(position)
            uncertainties.append(self._envelope(seconds, knots) + residual[0] * seconds)
            horizontal_uncertainty.append(self._envelope(seconds, horizontal_knots) + residual[1] * seconds)
            vertical_uncertainty.append(self._envelope(seconds, vertical_knots) + residual[2] * seconds)
        return Forecast(tuple(positions), tuple(uncertainties), model, last.time, age,
                        tuple(horizontal_uncertainty), tuple(vertical_uncertainty))
