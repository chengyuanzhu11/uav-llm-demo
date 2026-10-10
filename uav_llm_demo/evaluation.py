from __future__ import annotations

from math import dist
from pathlib import Path
import json
from typing import Iterable, Mapping
from .motion import SCALE, norm


def path_length(points: Iterable[Mapping[str, float]]) -> float:
    values = list(points)
    return round(
        sum(
            dist(
                tuple(float(previous[key]) * scale for key, scale in zip("xyz", SCALE)),
                tuple(float(current[key]) * scale for key, scale in zip("xyz", SCALE)),
            )
            for previous, current in zip(values, values[1:])
        ),
        3,
    )


def metrics_for_events(
    events: Iterable[Mapping[str, object]],
    *,
    final_status: str,
    path_length: float,
    battery_used: int,
) -> dict[str, object]:
    event_list = list(events)
    interventions = sum(
        1
        for event in event_list
        if bool((event.get("decision") or {}).get("safety_intervened"))
    )
    agent_errors = sum(
        1
        for event in event_list
        if bool((event.get("decision") or {}).get("provider_error")) or str((event.get("decision") or {}).get("reason", "")).startswith(
            "Agent backend error:"
        )
    )
    steps = len(event_list)
    return {
        "success": final_status == "completed",
        "status": final_status,
        "steps": steps,
        "safety_interventions": interventions,
        "safety_intervention_rate": round(interventions / steps, 3) if steps else 0.0,
        "agent_errors": agent_errors,
        "path_length": float(path_length),
        "path_length_unit": "m",
        "battery_used": int(battery_used),
    }


class EvaluationLog:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.total_events = 0
        self.total_interventions = 0
        self.total_agent_errors = 0
        self.dynamic_incidents = []
        self.dynamic_event_count = 0
        self.minimum_dynamic_clearance = None
        self._previous_dynamic = None
        self._contacts = set()

    def reset(self) -> None:
        self.events.clear()
        self.total_events = 0
        self.total_interventions = 0
        self.total_agent_errors = 0
        self.dynamic_incidents.clear()
        self.dynamic_event_count = 0
        self.minimum_dynamic_clearance = None
        self._previous_dynamic = None
        self._contacts.clear()

    def observe_dynamic(self, position, birds, elapsed):
        """Measure sampled/swept envelope overlap; this is not mesh-contact telemetry."""
        current = {bird.object_id: (tuple(bird.position), bird.radius * 3 + 1.0) for bird in birds}
        previous_position, previous_birds = self._previous_dynamic or (position, current)
        contacts = set()
        for object_id, (point, radius) in current.items():
            before = previous_birds.get(object_id, (point, radius))[0]
            a = tuple((before[i] - previous_position[i]) * SCALE[i] for i in range(3))
            b = tuple((point[i] - position[i]) * SCALE[i] for i in range(3))
            delta = tuple(b[i] - a[i] for i in range(3))
            fraction = max(0, min(1, -sum(a[i] * delta[i] for i in range(3)) / max(norm(delta) ** 2, 1e-12)))
            clearance = norm(tuple(a[i] + fraction * delta[i] for i in range(3))) - radius
            self.minimum_dynamic_clearance = (clearance if self.minimum_dynamic_clearance is None else
                                               min(clearance, self.minimum_dynamic_clearance))
            if clearance < 0:
                contacts.add(object_id)
                if object_id not in self._contacts:
                    self.dynamic_event_count += 1
                    self.dynamic_incidents.append(dict(kind="dynamic_separation_violation", object_id=object_id,
                        elapsed_seconds=round(elapsed, 3), clearance_m=round(clearance, 3)))
                    self.dynamic_incidents = self.dynamic_incidents[-100:]
        self._contacts = contacts
        self._previous_dynamic = (tuple(position), current)

    def record(self, event: Mapping[str, object]) -> None:
        self.events.append(dict(event))
        self.events = self.events[-200:]
        counts = metrics_for_events([event], final_status="active", path_length=0, battery_used=0)
        self.total_events += 1
        self.total_interventions += counts["safety_interventions"]
        self.total_agent_errors += counts["agent_errors"]

    def snapshot(
        self,
        *,
        final_status: str,
        trajectory: Iterable[Mapping[str, float]],
        battery: int,
    ) -> dict[str, object]:
        metrics = metrics_for_events(
            self.events,
            final_status=final_status,
            path_length=path_length(trajectory),
            battery_used=max(0, 100 - int(battery)),
        )
        metrics.update(steps=self.total_events, safety_interventions=self.total_interventions,
                       agent_errors=self.total_agent_errors,
                       safety_intervention_rate=round(self.total_interventions/self.total_events, 3) if self.total_events else 0.0,
                       collision_or_separation_events=self.dynamic_event_count,
                       min_dynamic_clearance_m=(round(self.minimum_dynamic_clearance, 3)
                                               if self.minimum_dynamic_clearance is not None else None))
        return {"events": list(self.events[-50:]), "dynamic_incidents": list(self.dynamic_incidents), "metrics": metrics}

    def write_jsonl(self, destination: str | Path) -> None:
        path = Path(destination)
        with path.open("w", encoding="utf-8") as stream:
            for event in self.events:
                stream.write(json.dumps(event, ensure_ascii=True) + "\n")


class MissionEvidence:
    """Observed milestones, independent of the controller's completed flag."""
    def __init__(self):
        self.target_reached = None
        self.base_returned = None
        self.static_blocked = False
        self.latest_speed = None

    def observe(self, vehicle, target, base, now):
        from .motion import distance
        self.latest_speed = norm(vehicle.velocity)
        if self.latest_speed > 0.02:
            return
        if self.target_reached is None and distance(vehicle.position, target) <= 0.18:
            self.target_reached = now
        elif self.target_reached is not None and self.base_returned is None and distance(vehicle.position, base) <= 0.18:
            self.base_returned = now

    def snapshot(self, *, dynamic_events=0, dynamic_measured=True):
        reached = self.target_reached is not None
        returned = self.base_returned is not None
        dynamic_safe = dynamic_events == 0 if dynamic_measured else None
        completed = reached and returned
        return dict(target_reached=reached, base_returned=returned,
                    target_reached_at=self.target_reached, base_returned_at=self.base_returned,
                    stationary=self.latest_speed is not None and self.latest_speed <= 0.02,
                    task_completed=completed, static_path_unblocked=not self.static_blocked,
                    dynamic_separation_safe=dynamic_safe,
                    safe_success=(completed and not self.static_blocked and dynamic_safe)
                        if dynamic_measured else None,
                    remaining=[name for name, done in [('reach_target', reached), ('return_to_base', returned)] if not done])
