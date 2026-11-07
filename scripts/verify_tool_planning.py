"""Live language-to-flight acceptance with independent measured-motion checks.

Loads the ignored local .env. Uses fixed simulation steps, waiting for planning
workers; this checks task execution, not real-time scheduling or real flight.
Birds are disabled to isolate geometry and language. No keys/prompts are logged.
"""
import argparse
from concurrent.futures import TimeoutError
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from uav_llm_demo.agents import DeepSeekAgent, PlannerProviderError, _redact_task_data
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import distance, norm
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.spatial_motion import spatial_catalog


CASES = {
    "half_boundary": "去田径场顺时针飞半圈，始终面向场地中心，随后返航。",
    "crossing_curve": "在田径场上空飞一个八字，飞行过程中逐渐升高4米，最后返航。",
}


class AcceptanceAgent(DeepSeekAgent):
    def __init__(self, replay=None):
        super().__init__(timeout_seconds=40)
        self.responses, self.calls, self.replay = [], 0, replay

    def _request_json_content(self, messages, result_kind, **kwargs):
        if self.calls >= 16:
            raise PlannerProviderError("Acceptance request budget exhausted.", retryable=False)
        self.calls += 1
        if self.replay is None:
            content = super()._request_json_content(messages, result_kind, **kwargs)
        else:
            if self.calls > len(self.replay):
                raise PlannerProviderError("Acceptance replay exhausted.", retryable=False)
            content = json.dumps(self.replay[self.calls - 1])
        self.responses.append(_redact_task_data(json.loads(content), self.api_key))
        print(json.dumps(dict(stage=result_kind, call=self.calls), ensure_ascii=False), flush=True)
        return content


class Journal:
    dropped = errors = 0

    def __init__(self):
        self.events = []

    def emit(self, value):
        if value["event"].startswith("semantic_"):
            self.events.append(deepcopy(value))


def signed_area(points):
    return sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1])) / 2


def measure(case, samples, center, footprint):
    if len(samples) < 20:
        return dict(ok=False, reason="insufficient measured trajectory")
    points = [[(s["position"][i] - center[i]) * (3, 3, 5)[i] for i in range(3)] for s in samples]
    if case == "half_boundary":
        angles = [math.atan2(p[1], p[0]) for p in points]
        turns = sum((b-a+math.pi) % (2*math.pi)-math.pi for a, b in zip(angles, angles[1:])) / (2*math.pi)
        headings = [abs((s["yaw"]-math.atan2(-p[0], p[1])+math.pi) % (2*math.pi)-math.pi)
                    for s, p in zip(samples, points)]
        endpoint_separation = math.dist(points[0], points[-1])
        return dict(ok=abs(turns-.5) < .025 and max(headings) < .2 and endpoint_separation > 5,
                    measured_laps=turns, heading_error_max_deg=math.degrees(max(headings)),
                    endpoints_distance_m=endpoint_separation)
    # Find a nonlocal crossing in measured XY, then independently check that the
    # two closed lobes have opposite orientation and nontrivial area. The oracle
    # never reads the generated formula or its claimed completion flag.
    xy = [p[:2] for p in points[::max(1, len(points)//350)]]
    xy.append(points[-1][:2])
    arc = [0.]
    for a, b in zip(xy, xy[1:]):
        arc.append(arc[-1]+math.dist(a, b))
    candidates = []
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            if not .2*arc[-1] < arc[j]-arc[i] < .8*arc[-1]:
                continue
            gap = math.dist(xy[i], xy[j])
            if gap < .65:
                areas = [signed_area(xy[i:j+1]), signed_area(xy[j:]+xy[:i+1])]
                if areas[0]*areas[1] < 0 and min(map(abs, areas)) > .1:
                    candidates.append((gap, areas))
    crossing = min(candidates, key=lambda item: item[0]) if candidates else None
    delta = points[-1][2]-points[0][2]
    downward = sum(max(0, a[2]-b[2]) for a, b in zip(points, points[1:]))
    left, right, north, south = footprint
    inside = all(left <= s["position"][0] <= right and north <= s["position"][1] <= south for s in samples)
    closed = math.dist(xy[0], xy[-1]) < .35
    return dict(ok=bool(crossing) and closed and abs(delta-4) < .15 and downward < .15 and inside,
                crossing_gap_m=crossing[0] if crossing else None,
                lobe_signed_areas_m2=crossing[1] if crossing else None,
                xy_closed=closed, altitude_change_m=delta, downward_travel_m=downward, within_target=inside)


def run(case, replay=None):
    scene = replace(load_scenario("semantic_campus"), dynamic_objects=())
    agent, journal = AcceptanceAgent(replay), Journal()
    service = MissionService(scenario=scene, agent=agent, journal=journal, traffic_seed=17)
    obj = next(o for o in spatial_catalog(scene)["objects"] if o["id"] == "stadium")
    samples, started = [], time.monotonic()
    try:
        service.submit_task(CASES[case], background=False)
        while service._time < 600 and time.monotonic()-started < 300:
            if service._future is not None:
                try:
                    service._future.result(timeout=1)
                except TimeoutError:
                    continue
                except Exception:
                    pass  # The production collector owns failure and repair.
            service.advance(.05)
            if service._motion_phase == "tracking" and service._running:
                samples.append(dict(position=list(service._vehicle.position), yaw=service._vehicle.yaw))
            status = service._snapshot["status"]
            if status == "completed" or service._error or service.semantic.status in {"generation_failed", "needs_clarification"}:
                break
        state = service.state()
        measured = measure(case, samples, obj["center_grid"], obj["footprint_grid"])
        events = [e for e in journal.events if e["event"] == "semantic_planning_tool"]
        tools = [e["tool"] for e in events]
        home_distance = distance(service._vehicle.position, scene.base)
        evidence = state["semantic_mission"]["evidence"]
        expected_actions = [e["action"] for e in evidence] == ["motion", "return_home"]
        ok = (measured["ok"] and state["status"] == "completed" and state["milestones"]["safe_success"]
              and expected_actions and home_distance < .18 and norm(service._vehicle.velocity) < .03
              and all(tool in tools for tool in ("query_geometry", "preview_plan", "submit_plan")))
        return dict(case=case, ok=ok, backend="replay" if replay is not None else "live", runtime="html", model=agent.model,
                    status=state["status"], message=service.semantic.message, calls=agent.calls,
                    simulation_seconds=service._time, wall_seconds=time.monotonic()-started,
                    measured=measured, home_distance_m=home_distance, tool_events=events,
                    requirements=service.semantic.requirements.to_dict() if service.semantic.requirements else None,
                    plan=service.semantic.plan.to_dict() if service.semantic.plan else None,
                    evidence=evidence, responses=agent.responses)
    finally:
        service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append", dest="cases")
    parser.add_argument("--replay", type=Path, help="Replay a prior acceptance output without API calls")
    parser.add_argument("--output", type=Path, required=True, help="New local result file; never overwrite")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    load_dotenv(ROOT / ".env", override=False)
    prior = {r["case"]: r["responses"] for r in json.loads(args.replay.read_text())} if args.replay else None
    results = []
    for case in args.cases or CASES:
        results.append(run(case, prior[case] if prior is not None else None))
        print(json.dumps({k: results[-1][k] for k in ("case", "ok", "status", "calls", "measured")}, ensure_ascii=False), flush=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
