"""Read-only controller evidence, separate from traffic prediction and decisions."""
from copy import deepcopy

from .dynamic_risk import physical_radius_m, separation_distance_m
from .motion import SCALE, norm


class AvoidanceEvidence:
    def __init__(self):
        self.sequence = 0
        self.time = 0.0
        self.active = self.limited = False
        self.nominal_path = self.selected_path = ()
        self.threats = []
        self.episode = None
        self._episode_id = 0
        self._previous = None
        self._events = []

    def due(self, now, active, was_active):
        return active != was_active or (active and now - self.time >= .1 - 1e-9)

    def record(self, now, active, limited, nominal, selected, buffer):
        self.sequence += 1
        self.time, self.active, self.limited = now, active, limited
        self.threats = [dict(id=object_id, predicted_clearance_m=gap)
                        for object_id, gap in (nominal.object_clearances if nominal else ()) if gap < buffer]
        if active:
            starting = self.episode is None or self.episode["status"] != "active"
            if starting:
                self._episode_id += 1
                self.episode = dict(id=self._episode_id, status="active", started_at=now, ended_at=None,
                                    min_separation_clearance_m=None, object_ids=[])
            self.episode["object_ids"] = sorted(set(self.episode["object_ids"]) | {t["id"] for t in self.threats})
            if starting:
                self._events.append(("avoidance_started", deepcopy(self.episode)))
            self.nominal_path = nominal.path if nominal else ()
            self.selected_path = selected.path if selected else ()
        elif self.episode is not None and self.episode["status"] == "active":
            self.episode.update(status="cleared", ended_at=now)
            self._events.append(("avoidance_cleared", deepcopy(self.episode)))

    def interrupt(self, now):
        if self.episode is not None and self.episode["status"] == "active":
            self.episode.update(status="interrupted", ended_at=now)
            self._events.append(("avoidance_interrupted", deepcopy(self.episode)))
            self.sequence += 1
            self.time, self.active, self.limited = now, False, False

    def measure(self, position, birds, now):
        """Swept, measured clearance for related objects; never used by choose()."""
        current = {b.object_id: (tuple((b.position[i] - position[i]) * SCALE[i] for i in range(3)),
                                 separation_distance_m(physical_radius_m(b))) for b in birds}
        episode = self.episode
        if episode and (episode["status"] == "active" or episode["ended_at"] == now):
            for object_id in episode["object_ids"]:
                if object_id not in current:
                    continue
                b, radius = current[object_id]
                a = self._previous.get(object_id, (b, radius))[0] if self._previous else b
                delta = tuple(y - x for x, y in zip(a, b))
                t = max(0.0, min(1.0, -sum(x * d for x, d in zip(a, delta)) / max(norm(delta) ** 2, 1e-12)))
                gap = norm(tuple(x + t * d for x, d in zip(a, delta))) - radius
                old = episode["min_separation_clearance_m"]
                episode["min_separation_clearance_m"] = gap if old is None else min(old, gap)
        self._previous = current

    def events(self):
        events, self._events = self._events, []
        for _, episode in events:
            if self.episode is not None and episode["id"] == self.episode["id"]:
                episode["min_separation_clearance_m"] = self.episode["min_separation_clearance_m"]
        return events

    def snapshot(self):
        return dict(sequence=self.sequence, simulation_time=self.time, active=self.active, limited=self.limited,
                    nominal_path=[dict(zip("xyz", p)) for p in self.nominal_path],
                    selected_path=[dict(zip("xyz", p)) for p in self.selected_path],
                    threats=deepcopy(self.threats), episode=deepcopy(self.episode))
