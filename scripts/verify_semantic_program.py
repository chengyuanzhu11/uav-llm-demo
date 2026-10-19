"""Offline campus flights with scripted provider replies and production execution.

Does not contact DeepSeek or the running Flask service. Birds are disabled to
isolate task-program acceptance; buildings, routes and control remain real.
"""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_llm_demo.agents import DeepSeekAgent
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import distance, norm
from uav_llm_demo.scenarios import load_scenario


def skill(name, action, target, when=None):
    return dict(id=name, action=action, target_id=target, duration_seconds=None, when=when)


def program(nodes):
    return dict(status="ready", summary="完成任务并返航。", steps=nodes,
                constraints=dict(max_altitude_m=None, avoid_regions=[]))


def repeat(nodes, count, until=None):
    return dict(id="patrol", action="repeat", steps=nodes, count=count, until=until, when=None)


class Flight:
    def __init__(self, *programs):
        self.replies, self.requests = deque(programs), []
        scenario = replace(load_scenario("semantic_campus"), dynamic_objects=())
        self.service = MissionService(scenario=scenario, traffic_seed=17,
                                     agent=DeepSeekAgent(api_key="offline-verification-key"))

    def respond(self, url, **kwargs):
        assert url.endswith("/chat/completions")
        request = kwargs["json"]
        assert request["max_tokens"] == 12000
        self.requests.append(deepcopy(request))
        assert self.replies, "Unexpected provider retry"
        content = json.dumps(self.replies.popleft(), ensure_ascii=False)
        return Mock(status_code=200, json=Mock(return_value={
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]}))

    def until(self, predicate, seconds=240):
        for _ in range(round(seconds / .1)):
            future = self.service._future
            if future is not None:
                future.result(timeout=30)
            self.service.advance(.1)
            state = self.service.state()
            if predicate(state):
                return state
            assert state["status"] != "error", state.get("error")
        raise AssertionError("Campus program did not reach the expected state")

    def completed(self):
        state = self.until(lambda s: s["status"] == "completed")
        assert state["milestones"]["safe_success"]
        assert distance(self.service._vehicle.position, self.service._environment.base) <= .18
        assert norm(self.service._vehicle.velocity) <= .02
        assert not self.replies
        assert self.service._thread is None
        return state

    def report(self, name, state):
        return dict(case=name, safe_success=True, simulation_seconds=round(self.service._time, 2),
                    provider_replies=len(self.requests), path_length_m=state["metrics"]["path_length"],
                    semantic_mission=state["semantic_mission"], final_uav=state["uav"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/semantic-program/campus-flights.json")
    args = parser.parse_args()
    reports = []
    home = skill("home", "return_home", "home")
    condition = dict(step_id="scan", field="temperature_c", op=">", value=25)
    branch = dict(id="decision", action="branch", condition=condition,
                  then=[skill("sample", "capture", "library")],
                  **{"else": [skill("normal", "inspect", "conference")]})
    cases = [
        ("counted_loop_and_numeric_branch", "检查图书馆3次，每次超过25度就采样，否则检查会议中心，然后返航。",
         program([repeat([skill("scan", "thermal_scan", "library"), branch], 3), home])),
        ("until_condition", "最多检查图书馆5次，温度低于30度就结束并返航。",
         program([repeat([skill("scan", "thermal_scan", "library")], 5,
                         dict(step_id="scan", field="temperature_c", op="<", value=30)), home])),
    ]
    for name, instruction, response in cases:
        flight = Flight(response)
        try:
            with patch("uav_llm_demo.agents._post_deepseek", side_effect=flight.respond):
                flight.service.submit_task(instruction, background=False)
                state = flight.completed()
            evidence = state["semantic_mission"]["evidence"]
            actions = [record["action"] for record in evidence]
            assert instruction in flight.requests[0]["messages"][1]["content"]
            assert all(record["result"]["temperature_c"] == 29 for record in evidence if record["action"] == "thermal_scan")
            if name == "counted_loop_and_numeric_branch":
                assert actions == ["thermal_scan", "capture"] * 3 + ["return_home"]
                assert all(record["result"]["anomaly"] is False for record in evidence if record["action"] == "thermal_scan")
            else:
                assert actions == ["thermal_scan", "return_home"]
                assert state["semantic_mission"]["loops"][0]["iteration"] == 1
            reports.append(flight.report(name, state))
            print(json.dumps(dict(case=name, ok=True), ensure_ascii=False), flush=True)
        finally:
            flight.service.close()

    flight = Flight(program([skill("library", "inspect", "library"),
                             skill("old_lab", "inspect", "laboratory"), home]),
                    program([skill("new_conference", "inspect", "conference"), home]))
    try:
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=flight.respond):
            flight.service.submit_task("先检查图书馆，再检查实验楼，然后返航。", background=False)
            before = flight.until(lambda s: len(s["semantic_mission"]["evidence"]) == 1
                                  and s["semantic_mission"]["current_step_id"] == "old_lab"
                                  and distance(flight.service._vehicle.position, tuple(
                                      s["semantic_mission"]["evidence"][0]["position"][axis] for axis in "xyz")) > 1)
            pose = flight.service._vehicle.position
            clock, run = flight.service._time, flight.service.trace.run_id
            path, traffic = list(flight.service._environment.path), flight.service._traffic
            flight.service.submit_task("取消剩余实验楼巡检，改去会议中心检查，然后返航。", background=False)
            assert distance(pose, flight.service._vehicle.position) < .3
            assert flight.service._time >= clock
            state = flight.completed()
        mission = state["semantic_mission"]
        assert [record["step_id"] for record in mission["evidence"]] == ["library", "new_conference", "home"]
        assert mission["evidence"][0] == before["semantic_mission"]["evidence"][0]
        assert mission["revision"] == 2 and flight.service.trace.run_id == run
        assert flight.service._traffic is traffic
        assert flight.service._environment.path[:len(path)] == path
        assert any(step["id"] == "old_lab" and step["status"] == "cancelled" for step in mission["steps"])
        reports.append(flight.report("in_flight_task_replacement", state))
        print(json.dumps(dict(case="in_flight_task_replacement", ok=True)), flush=True)
    finally:
        flight.service.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(ok=True, actual_campus_geometry=True, scripted_provider=True,
        external_api_calls=0, birds_enabled=False, live_service_touched=False, cases=reports),
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
