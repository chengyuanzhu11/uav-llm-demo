from __future__ import annotations

from math import dist
from pathlib import Path
import json
from typing import Iterable, Mapping


def path_length(points: Iterable[Mapping[str, float]]) -> float:
    values = list(points)
    return round(
        sum(
            dist(
                (float(previous["x"]), float(previous["y"]), float(previous["z"])),
                (float(current["x"]), float(current["y"]), float(current["z"])),
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
        if str((event.get("decision") or {}).get("reason", "")).startswith(
            "Agent backend error:"
        )
    )
    collisions = sum(
        1
        for event in event_list
        if any(
            marker in str(event.get("feedback", "")).lower()
            for marker in ("collision", "separation", "failed")
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
        "collision_or_separation_events": collisions,
        "path_length": float(path_length),
        "battery_used": int(battery_used),
    }


class EvaluationLog:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def reset(self) -> None:
        self.events.clear()

    def record(self, event: Mapping[str, object]) -> None:
        self.events.append(dict(event))

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
        return {"events": list(self.events[-50:]), "metrics": metrics}

    def write_jsonl(self, destination: str | Path) -> None:
        path = Path(destination)
        with path.open("w", encoding="utf-8") as stream:
            for event in self.events:
                stream.write(json.dumps(event, ensure_ascii=True) + "\n")
