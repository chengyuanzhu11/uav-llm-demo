"""Independent simulated task world, sensor readings and usable observations.

Only ``TaskObserver.observe`` crosses the world/sensor boundary. The mission
publishes delivered readings, never world values or future fault schedules.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import random


OBSERVATION_ACTIONS = frozenset({"thermal_scan", "inspect", "capture"})
SENSOR_SOURCES = {"thermal_scan": "simulated_thermal_sensor", "inspect": "simulated_inspection",
                  "capture": "simulated_capture"}


def _finite(value, name, lower=0.0, upper=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be a finite number in the supported range.")
    return float(value)


@dataclass(frozen=True)
class TaskObservationConfig:
    seed: int = 0
    noise_std_c: float = 0.0
    dropout_probability: float = 0.0
    latency_seconds: float = 0.0
    max_age_seconds: float = 120.0
    min_confidence: float = 0.6
    conflict_tolerance_c: float = 3.0
    faults: tuple[dict, ...] = ()

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("Sensor seed must be an integer.")
        for name in ("noise_std_c", "latency_seconds", "max_age_seconds", "conflict_tolerance_c"):
            _finite(getattr(self, name), name)
        for name in ("dropout_probability", "min_confidence"):
            _finite(getattr(self, name), name, upper=1)
        if self.max_age_seconds <= 0 or self.conflict_tolerance_c <= 0:
            raise ValueError("Observation lifetime and conflict tolerance must be positive.")
        if not isinstance(self.faults, (list, tuple)):
            raise ValueError("Sensor faults must be a sequence.")
        seen = set()
        for fault in self.faults:
            if (not isinstance(fault, dict) or set(fault) != {"action", "target_id", "attempt", "status"}
                    or not isinstance(fault["action"], str) or fault["action"] not in OBSERVATION_ACTIONS
                    or not isinstance(fault["target_id"], str) or not fault["target_id"]
                    or isinstance(fault["attempt"], bool) or not isinstance(fault["attempt"], int) or fault["attempt"] < 1
                    or not isinstance(fault["status"], str) or fault["status"] not in {"unknown", "conflict", "stale"}):
                raise ValueError("Invalid task sensor fault specification.")
            identity = (fault["action"], fault["target_id"], fault["attempt"])
            if identity in seen:
                raise ValueError("A sensor attempt may have only one specified fault.")
            seen.add(identity)
        object.__setattr__(self, "faults", tuple(deepcopy(self.faults)))


@dataclass(frozen=True)
class TaskObservationWorld:
    """Private environmental state; its schema is deliberately not the task schema."""
    _targets: dict = field(repr=False)

    def __post_init__(self):
        if not isinstance(self._targets, dict) or not self._targets:
            raise ValueError("Task world requires at least one target.")
        values = {}
        for target, value in self._targets.items():
            if not isinstance(target, str) or not target or not isinstance(value, dict) or set(value) != {"temperature_c"}:
                raise ValueError("A task world target requires temperature_c only.")
            values[target] = {"temperature_c": _finite(value["temperature_c"], "temperature_c", -100, 500)}
        object.__setattr__(self, "_targets", values)

    @classmethod
    def from_dict(cls, targets):
        return cls(deepcopy(targets))

    @classmethod
    def generate(cls, scenario, seed=0):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("World seed must be an integer.")
        rng = random.Random(seed)
        return cls({target: {"temperature_c": round(rng.uniform(60, 85) if rng.random() < .25 else rng.uniform(22, 35), 2)}
                    for target in sorted(point["id"] for point in scenario.points_of_interest)})

    @classmethod
    def demonstration(cls, scenario):
        """Explicit stable demo scene retained for existing task examples."""
        temperatures = {"laboratory": 68.0, "library": 29.0}
        return cls({point["id"]: {"temperature_c": temperatures.get(point["id"], 27.0)}
                    for point in scenario.points_of_interest})

    def _sample(self, target_id):
        if target_id not in self._targets:
            return None
        return deepcopy(self._targets[target_id])


def observation_validity(record, now):
    """Return None for usable evidence, otherwise its unavailability reason.

    Legacy persisted evidence without sensor metadata retains its original
    semantics. New sensor evidence always includes the observation envelope.
    """
    observation = record.get("observation") if isinstance(record, dict) else None
    if observation is None:
        return None
    if (not isinstance(now, (int, float)) or isinstance(now, bool) or not math.isfinite(now)
            or not isinstance(observation, dict) or not isinstance(observation.get("status"), str)
            or observation.get("status") not in {"valid", "unknown", "conflict", "stale"}):
        return "unknown"
    if observation["status"] != "valid":
        return observation["status"]
    if not all(isinstance(observation.get(key), (int, float)) and not isinstance(observation.get(key), bool)
               and math.isfinite(observation[key]) for key in ("observed_at", "valid_until", "confidence")):
        return "unknown"
    if observation["observed_at"] > now + 1e-9 or observation["valid_until"] < now - 1e-9:
        return "stale"
    if not 0 <= observation["confidence"] <= 1:
        return "unknown"
    return None


class TaskObserver:
    """Produce repeatable independent channels and fuse only consistent readings."""
    def __init__(self, world, config=None):
        if not isinstance(world, TaskObservationWorld):
            raise ValueError("Task observer requires an independent task world.")
        self._world = world
        self.config = config or TaskObservationConfig()
        if not isinstance(self.config, TaskObservationConfig):
            raise ValueError("Invalid task observation configuration.")
        self._rng = random.Random(self.config.seed)
        self._attempts = {}

    def observe(self, action, target_id, now):
        if action not in OBSERVATION_ACTIONS:
            raise ValueError("Unsupported observation action.")
        now = _finite(now, "simulation time")
        key = (action, target_id)
        attempt = self._attempts[key] = self._attempts.get(key, 0) + 1
        fault = next((item["status"] for item in self.config.faults
                      if (item["action"], item["target_id"], item["attempt"]) == (*key, attempt)), None)
        truth = self._world._sample(target_id)
        latency = self.config.latency_seconds
        if fault == "stale":
            latency = max(latency, self.config.max_age_seconds + 1)
        observed_at = now - latency
        readings = []
        for channel in range(2):
            dropped = truth is None or fault == "unknown" or self._rng.random() < self.config.dropout_probability
            confidence = 0.0 if dropped else 1.0 / (1.0 + self.config.noise_std_c / 10.0)
            value = None if dropped else (truth["temperature_c"] + self._rng.gauss(0, self.config.noise_std_c)
                                         if action == "thermal_scan" else True)
            if fault == "conflict" and channel == 1 and value is not None:
                value = value + self.config.conflict_tolerance_c * 2 if action == "thermal_scan" else False
            readings.append({"sensor_id": f"{action}:{channel}", "value": value,
                             "confidence": confidence, "observed_at": observed_at})
        usable = [item for item in readings if item["value"] is not None and item["confidence"] >= self.config.min_confidence]
        status = "valid"
        if len(usable) < 2:
            status = "unknown"
        elif (abs(usable[0]["value"] - usable[1]["value"]) > self.config.conflict_tolerance_c
              if action == "thermal_scan" else usable[0]["value"] != usable[1]["value"]):
            status = "conflict"
        elif observed_at < 0 or now - observed_at > self.config.max_age_seconds:
            status = "stale"
        confidence = min(item["confidence"] for item in readings)
        result = {"simulated": True}
        if status == "valid":
            if action == "thermal_scan":
                temperature = sum(item["value"] for item in usable) / len(usable)
                result.update(temperature_c=temperature, anomaly=temperature >= 60.0)
            elif action == "inspect":
                result["inspected"] = True
            else:
                result.update(captured=True, artifact_type="observation_record", image_generated=False)
        return {"action": action, "target_id": target_id, "attempt": attempt,
                "source": SENSOR_SOURCES[action], "status": status, "observed_at": observed_at,
                "delivered_at": now, "valid_until": observed_at + self.config.max_age_seconds,
                "confidence": confidence, "readings": readings, "result": result}
