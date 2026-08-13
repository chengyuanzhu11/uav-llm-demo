from __future__ import annotations

from .agents import build_agent
from .mission_service import MissionService
from .scenarios import list_scenarios


def run_scenario(
    scenario_id: str,
    *,
    backend: str = "mock",
    max_steps: int = 200,
) -> dict[str, object]:
    service = MissionService(
        agent=build_agent(backend),
        scenario_id=scenario_id,
    )
    state = service.state()
    for _ in range(max(0, int(max_steps))):
        if state["status"] != "active":
            break
        state = service.step()
    state = service.state()
    return {
        "scenario_id": scenario_id,
        "agent_backend": state["agent_backend"],
        "status": state["status"],
        "metrics": state["metrics"],
    }


def run_benchmark(
    scenario_ids: list[str] | None = None,
    *,
    backend: str = "mock",
    max_steps: int = 200,
) -> list[dict[str, object]]:
    selected = scenario_ids or [item["id"] for item in list_scenarios()]
    return [
        run_scenario(scenario_id, backend=backend, max_steps=max_steps)
        for scenario_id in selected
    ]
