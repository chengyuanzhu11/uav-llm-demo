from __future__ import annotations

from math import dist, isfinite
from pathlib import Path
import json
from typing import Iterable, Mapping
from .motion import SCALE, norm
from .dynamic_risk import (NEAR_MISS_MARGIN_M, contact_distance_m,
                           physical_radius_m, separation_distance_m)


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
        self.dynamic_incidents: list[dict[str, object]] = []
        # Retained for mission evidence / historical output: separation violations.
        self.dynamic_event_count = 0
        self.contact_event_count = 0
        self.near_miss_event_count = 0
        self.minimum_dynamic_clearance = None
        self.minimum_contact_clearance = None
        self._previous_dynamic = None
        self._active_incidents = {}
        self._near_encounters = {}
        self._incident_updates = []

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def _interval(a, b, radius):
        """Open-sphere crossing interval along a metric relative-motion segment."""
        delta = tuple(b[i] - a[i] for i in range(3))
        squared_speed = sum(v * v for v in delta)
        c = sum(v * v for v in a) - radius * radius
        if squared_speed < 1e-18:
            return (0.0, 1.0) if c < 0 else None
        projection = sum(a[i] * delta[i] for i in range(3))
        discriminant = projection * projection - squared_speed * c
        if discriminant <= 0:
            return None
        root = discriminant ** 0.5
        entry, exit = max(0.0, (-projection - root) / squared_speed), min(1.0, (-projection + root) / squared_speed)
        return (entry, exit) if entry < exit else None

    @staticmethod
    def _minimum_distance(a, b, entry=0.0, exit=1.0):
        delta = tuple(b[i] - a[i] for i in range(3))
        fraction = max(entry, min(exit, -sum(a[i] * delta[i] for i in range(3)) /
                                  max(sum(v * v for v in delta), 1e-18)))
        return norm(tuple(a[i] + fraction * delta[i] for i in range(3)))

    def _finish(self, incident, elapsed):
        incident["end_elapsed_seconds"] = round(elapsed, 6)
        incident["duration_seconds"] = round(max(0.0, elapsed - incident["start_elapsed_seconds"]), 6)
        incident["active"] = False
        if incident["kind"] != "dynamic_near_miss":
            self._incident_updates.append(incident)

    @staticmethod
    def _update(incident, minimum_distance, contact_radius, separation_radius, relative_speed, elapsed):
        if minimum_distance < incident.get("minimum_center_distance_m", float("inf")):
            incident["relative_speed_at_minimum_mps"] = relative_speed
        incident["minimum_center_distance_m"] = min(incident.get("minimum_center_distance_m", float("inf")), minimum_distance)
        incident["minimum_contact_clearance_m"] = incident["minimum_center_distance_m"] - contact_radius
        incident["minimum_separation_clearance_m"] = incident["minimum_center_distance_m"] - separation_radius
        # Historical clearance_m means safety-separation clearance, not mesh penetration.
        incident["clearance_m"] = incident["minimum_separation_clearance_m"]
        if relative_speed is not None:
            incident["relative_speed_mps"] = relative_speed
            incident["max_relative_speed_mps"] = max(incident.get("max_relative_speed_mps") or 0.0, relative_speed)
        incident["duration_seconds"] = round(max(0.0, elapsed - incident["start_elapsed_seconds"]), 6)

    @staticmethod
    def _new_incident(object_id, kind, elapsed, contact_radius, separation_radius):
        return dict(kind=kind, object_id=object_id, elapsed_seconds=round(elapsed, 6),
                    start_elapsed_seconds=round(elapsed, 6), end_elapsed_seconds=None,
                    duration_seconds=0.0, active=True, distance_unit="m", time_unit="s",
                    measurement="swept_spherical_envelopes", contact_distance_m=contact_radius,
                    separation_distance_m=separation_radius, relative_speed_mps=None,
                    max_relative_speed_mps=None)

    def _append_incident(self, incident):
        self.dynamic_incidents.append(incident)
        self._incident_updates.append(incident)
        self.dynamic_incidents = self.dynamic_incidents[-100:]

    def _observe_interval(self, object_id, kind, a, b, radius, contact_radius,
                          separation_radius, before_time, elapsed, relative_speed):
        key = (object_id, kind)
        interval = self._interval(a, b, radius)
        active = self._active_incidents.get(key)
        if interval is None:
            if active is not None:
                self._finish(active, before_time)
                self._active_incidents.pop(key)
            return
        entry, exit = interval
        start = before_time + (elapsed - before_time) * entry
        end = before_time + (elapsed - before_time) * exit
        if active is None:
            active = self._new_incident(object_id, kind, start, contact_radius, separation_radius)
            self._append_incident(active)
            self._active_incidents[key] = active
            if kind == "dynamic_separation_violation":
                self.dynamic_event_count += 1
            else:
                self.contact_event_count += 1
        self._update(active, self._minimum_distance(a, b, entry, exit), contact_radius,
                     separation_radius, relative_speed, end)
        if exit < 1.0 or norm(b) >= radius:
            self._finish(active, end)
            self._active_incidents.pop(key)

    def _finish_near_encounter(self, object_id, elapsed):
        incident = self._near_encounters.pop(object_id)
        self._finish(incident, elapsed)
        # A near miss is an entire completed close encounter that never violated
        # the safety distance. A contact/violation must not also count as a near miss.
        if incident["minimum_separation_clearance_m"] >= 0:
            self.near_miss_event_count += 1
            self._append_incident(incident)

    def _observe_near_encounter(self, object_id, a, b, contact_radius, separation_radius,
                                before_time, elapsed, relative_speed):
        radius = separation_radius + NEAR_MISS_MARGIN_M
        interval = self._interval(a, b, radius)
        if interval is None:
            if object_id in self._near_encounters:
                self._finish_near_encounter(object_id, before_time)
            return
        entry, exit = interval
        start = before_time + (elapsed - before_time) * entry
        end = before_time + (elapsed - before_time) * exit
        incident = self._near_encounters.get(object_id)
        if incident is None:
            incident = self._new_incident(object_id, "dynamic_near_miss", start, contact_radius, separation_radius)
            self._near_encounters[object_id] = incident
        self._update(incident, self._minimum_distance(a, b, entry, exit), contact_radius,
                     separation_radius, relative_speed, end)
        if exit < 1.0 or norm(b) >= radius:
            self._finish_near_encounter(object_id, end)

    def observe_dynamic(self, position, birds, elapsed):
        """Measure both envelope contacts and safety intrusions along each step.

        Relative motion is linearly interpolated between observations. This catches
        crossings between clear endpoints; it is not mesh-contact telemetry.
        Return new/completed event copies for the run log, without per-frame spam.
        """
        elapsed = float(elapsed)
        if not isfinite(elapsed) or elapsed < 0:
            raise ValueError("Dynamic observation time must be finite and nonnegative.")
        current = {bird.object_id: (tuple(bird.position), physical_radius_m(bird)) for bird in birds}
        previous_position, previous_birds, before_time = self._previous_dynamic or (position, current, elapsed)
        if elapsed < before_time:
            raise ValueError("Dynamic observation time must be monotonic; reset between runs.")
        self._incident_updates = []
        for key, incident in list(self._active_incidents.items()):
            if key[0] not in current:
                self._finish(incident, before_time)
                incident["end_reason"] = "object_no_longer_observed"
                self._active_incidents.pop(key)
        for object_id in list(self._near_encounters):
            if object_id not in current:
                # Unknown continuation cannot establish a safely completed near miss.
                self._near_encounters.pop(object_id)
        for object_id, (point, physical_radius) in current.items():
            before = previous_birds.get(object_id, (point, physical_radius))[0]
            observed_before = object_id in previous_birds
            a = tuple((before[i] - previous_position[i]) * SCALE[i] for i in range(3))
            b = tuple((point[i] - position[i]) * SCALE[i] for i in range(3))
            start_time = before_time
            if not observed_before or elapsed == before_time:
                a, start_time = b, elapsed
            relative_speed = (norm(tuple(b[i] - a[i] for i in range(3))) / (elapsed - start_time)
                              if elapsed > start_time else None)
            contact_radius = contact_distance_m(physical_radius)
            separation_radius = separation_distance_m(physical_radius)
            minimum_distance = self._minimum_distance(a, b)
            clearance = minimum_distance - separation_radius
            contact_clearance = minimum_distance - contact_radius
            self.minimum_dynamic_clearance = (clearance if self.minimum_dynamic_clearance is None else
                                               min(clearance, self.minimum_dynamic_clearance))
            self.minimum_contact_clearance = (contact_clearance if self.minimum_contact_clearance is None else
                                               min(contact_clearance, self.minimum_contact_clearance))
            self._observe_interval(object_id, "dynamic_separation_violation", a, b, separation_radius,
                                   contact_radius, separation_radius, start_time, elapsed, relative_speed)
            self._observe_interval(object_id, "dynamic_envelope_contact", a, b, contact_radius,
                                   contact_radius, separation_radius, start_time, elapsed, relative_speed)
            self._observe_near_encounter(object_id, a, b, contact_radius, separation_radius,
                                         start_time, elapsed, relative_speed)
        self._previous_dynamic = (tuple(position), current, elapsed)
        # A within-step crossing starts and ends in this call: emit its final state once.
        return [dict(incident) for incident in {id(item): item for item in self._incident_updates}.values()]

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
                       separation_violation_events=self.dynamic_event_count,
                       contact_events=self.contact_event_count,
                       near_miss_events=self.near_miss_event_count,
                       dynamic_measurement="swept_spherical_envelopes",
                       near_miss_margin_m=NEAR_MISS_MARGIN_M,
                       min_contact_clearance_m=(round(self.minimum_contact_clearance, 3)
                                                if self.minimum_contact_clearance is not None else None),
                       min_dynamic_clearance_m=(round(self.minimum_dynamic_clearance, 3)
                                               if self.minimum_dynamic_clearance is not None else None))
        return {"events": list(self.events[-50:]), "dynamic_incidents": [dict(incident) for incident in self.dynamic_incidents], "metrics": metrics}

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
