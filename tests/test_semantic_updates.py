"""Task-program integration with real geometry/control and no remote provider."""
from collections import deque
from copy import deepcopy
from dataclasses import replace
import json
from threading import Event
import unittest
from unittest.mock import Mock, patch

from tests.scenario_fixtures import empty_airspace
from uav_llm_demo.agents import DeepSeekAgent, PlannerProviderError
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import VehicleState, distance, norm
from uav_llm_demo.semantic_navigation import validate_task_routes
from uav_llm_demo.semantic_tasks import semantic_plan_from_json_text, task_catalog


def skill(name, action, target, duration=None, when=None):
    return dict(id=name, action=action, target_id=target, duration_seconds=duration, when=when)


def program(nodes, ceiling=None, avoid=()):
    return dict(status="ready", summary="执行任务程序后返航。", steps=nodes,
                constraints=dict(max_altitude_m=ceiling, avoid_regions=list(avoid)))


def home():
    return skill("home", "return_home", "home")


def repeat(nodes, count=3, until=None):
    return dict(id="patrol", action="repeat", count=count, until=until, steps=nodes, when=None)


def compact_campus():
    points = []
    for name, position in (("home", (2, 15, 0)), ("library", (4, 15, 2.2)),
                           ("laboratory", (7, 15, 2.2)), ("conference", (10, 15, 2.2))):
        actions = ["return_home"] if name == "home" else ["visit", "inspect", "capture"]
        if name in ("library", "laboratory"):
            actions.append("thermal_scan")
        points.append(dict(id=name, label=name, kind=name, position=dict(zip("xyz", position)),
                           allowed_actions=actions))
    return replace(empty_airspace(), scenario_id="semantic_program_fixture", base=(2, 15, 0),
                   target=(4, 15, 2.2), dynamic_objects=(), mission_mode="semantic",
                   points_of_interest=tuple(points), semantic_regions=(
                       dict(id="east_yard", label="东侧区域", bounds=[12, 15, 5, 8]),))


class OfflineSemanticAgent(DeepSeekAgent):
    """A test response source; all navigation and execution remain production code."""
    def __init__(self, replies):
        super().__init__(api_key="offline-program-test-key")
        self.replies = deque(replies)
        self.inputs = []

    def interpret_task(self, catalog, instruction, feedback="", *, repair=None):
        self.inputs.append((deepcopy(catalog), instruction, feedback))
        reply = self.replies.popleft()
        if callable(reply):
            reply = reply()
        if isinstance(reply, Exception):
            raise reply
        return semantic_plan_from_json_text(json.dumps(reply), catalog)


def pump(service, predicate, seconds=150):
    for _ in range(round(seconds / .1)):
        if service._future is not None:
            try:
                service._future.result(timeout=5)
            except Exception:
                pass
        service.advance(.1)
        state = service.state()
        if predicate(state):
            return state
    raise AssertionError("Task did not reach its expected state: " + str(service.semantic.to_dict()))


class SemanticUpdateTests(unittest.TestCase):
    def setUp(self):
        guard = patch("uav_llm_demo.agents._post_deepseek", side_effect=AssertionError("Remote API forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def service(self, *replies):
        service = MissionService(scenario=compact_campus(), agent=OfflineSemanticAgent(replies),
                                 planner_latency_seconds=.01, traffic_seed=17)
        self.addCleanup(service.close)
        return service

    def test_numeric_threshold_and_counted_loop_execute_then_really_land(self):
        condition = dict(step_id="scan", field="temperature_c", op=">", value=25)
        service = self.service(program([repeat([
            skill("scan", "thermal_scan", "library", .1),
            skill("sample", "capture", "library", .1, condition)]), home()]))
        service.submit_task("检查图书馆三次，每次超过25度就采样，然后返航。", background=False)
        final = pump(service, lambda s: s["status"] == "completed")
        evidence = final["semantic_mission"]["evidence"]
        self.assertEqual([e["action"] for e in evidence], ["thermal_scan", "capture"] * 3 + ["return_home"])
        self.assertEqual(len({e["execution_id"] for e in evidence}), 7)
        self.assertTrue(all(e["result"]["anomaly"] is False for e in evidence if e["action"] == "thermal_scan"))
        self.assertEqual(final["semantic_mission"]["loops"][0]["iteration"], 3)
        self.assertLessEqual(distance(service._vehicle.position, service._environment.base), .18)
        self.assertLessEqual(norm(service._vehicle.velocity), .02)
        self.assertTrue(final["milestones"]["safe_success"])

    def test_branch_uses_observed_temperature_and_does_not_claim_unselected_work(self):
        branch = dict(id="choice", action="branch",
                      condition=dict(step_id="scan", field="temperature_c", op=">", value=40),
                      then=[skill("hot", "capture", "laboratory", .1)],
                      **{"else": [skill("normal", "inspect", "conference", .1)]})
        service = self.service(program([skill("scan", "thermal_scan", "library", .1), branch, home()]))
        service.submit_task("检查图书馆温度，超过40度就去实验楼采样，否则巡检会议中心。", background=False)
        final = pump(service, lambda s: s["status"] == "completed")
        records = final["semantic_mission"]["evidence"]
        self.assertEqual([e["step_id"] for e in records], ["scan", "normal", "home"])
        self.assertEqual(records[0]["result"]["temperature_c"], 29)

    def test_in_flight_update_preserves_pose_clock_path_evidence_and_mission(self):
        first = program([skill("library", "inspect", "library", .1),
                         skill("old_lab", "inspect", "laboratory", 30), home()])
        second = program([skill("new_site", "capture", "conference", .1), home()])
        service = self.service(first, second)
        original_instruction = "先检查图书馆，再检查实验楼。"
        service.submit_task(original_instruction, background=False)
        pump(service, lambda s: len(s["semantic_mission"]["evidence"]) == 1
             and distance(service._vehicle.position, (4, 15, 2.2)) > 1)
        before = service.state()
        run_id, mission_id, traffic, seed = service.trace.run_id, service.semantic.mission_id, service._traffic, service._traffic_seed
        position, now, path = service._vehicle.position, service._time, list(service._environment.path)
        service.submit_task("取消实验楼，改去会议中心采样再回来。", background=False)
        self.assertGreater(service._time, now)
        self.assertLess(distance(service._vehicle.position, position), .3)
        self.assertEqual(service._environment.path[:len(path)], path)
        self.assertIs(service._traffic, traffic)
        self.assertEqual(service._traffic_seed, seed)
        final = pump(service, lambda s: s["status"] == "completed")
        self.assertEqual((service.trace.run_id, service.semantic.mission_id), (run_id, mission_id))
        records = final["semantic_mission"]["evidence"]
        self.assertEqual([e["step_id"] for e in records], ["library", "new_site", "home"])
        self.assertEqual(records[0], before["semantic_mission"]["evidence"][0])
        context = service.agent.inputs[1][0]["mission_context"]
        self.assertEqual(context["accepted_instruction"], original_instruction)
        self.assertEqual(context["current_step"]["id"], "old_lab")
        self.assertEqual(context["evidence"][0]["evidence_id"], records[0]["evidence_id"])
        self.assertEqual(final["semantic_mission"]["revision"], 2)

    def test_unavailable_update_holds_with_protection_and_can_be_changed_to_return(self):
        rejected = dict(status="needs_clarification", summary="缺少投递载荷能力，可以改为访问。", steps=[],
                        constraints=dict(max_altitude_m=None, avoid_regions=[]))
        service = self.service(program([skill("old", "inspect", "library", 60), home()]),
                               rejected, program([home()]))
        service.submit_task("巡检图书馆。", background=False)
        pump(service, lambda s: s["uav"]["z"] >= 2 and s["flight"]["speed_mps"] > .1)
        old_plan = service.semantic.plan
        service.submit_task("改为投递包裹。", background=False)
        waiting = pump(service, lambda s: s["semantic_mission"]["status"] == "needs_clarification")
        self.assertTrue(waiting["flight"]["safety_armed"])
        self.assertFalse(waiting["flight"]["running"])
        self.assertIs(service.semantic.plan, old_plan)
        self.assertFalse(waiting["milestones"]["task_completed"])
        with self.assertRaises(ValueError):
            service.start(background=False)
        service.submit_task("取消剩余任务，立即返航。", background=False)
        final = pump(service, lambda s: s["status"] == "completed")
        self.assertEqual([e["action"] for e in final["semantic_mission"]["evidence"]], ["return_home"])

    def test_update_cannot_apply_ceiling_that_excludes_current_position(self):
        service = self.service(program([skill("old", "inspect", "library", 60), home()]),
                               program([skill("next", "visit", "conference"), home()], ceiling=10))
        service.submit_task("检查图书馆。", background=False)
        pump(service, lambda s: s["uav"]["z"] > 2.1)
        service.submit_task("把高度上限改成10米。", background=False)
        waiting = pump(service, lambda s: s["semantic_mission"]["status"] == "needs_clarification")
        self.assertIn("当前位置", waiting["semantic_mission"]["message"])
        self.assertIsNone(service.semantic.plan.constraints["max_altitude_m"])
        self.assertEqual(service._context()["altitude_range"]["max"], 10)
        self.assertTrue(waiting["flight"]["safety_armed"])

    def test_relaxed_constraints_do_not_keep_old_ceiling_or_avoidance(self):
        service = self.service(program([skill("old", "inspect", "library", 60), home()], ceiling=10, avoid=["east_yard"]),
                               program([skill("new", "visit", "conference"), home()], ceiling=30))
        service.submit_task("限高10米，避开东侧区域并检查图书馆。", background=False)
        pump(service, lambda s: s["uav"]["z"] > 1.9)
        service.submit_task("取消区域避让，高度上限改成30米，先去会议中心。", background=False)
        ready = pump(service, lambda s: s["semantic_mission"]["revision"] == 2
                     and s["semantic_mission"]["status"] == "running")
        self.assertEqual(ready["semantic_mission"]["constraints"], dict(max_altitude_m=30, avoid_regions=[]))
        self.assertEqual(service._context()["altitude_range"]["max"], 6)
        self.assertEqual(service._context()["no_fly_zones"], [])

    def test_latest_pose_is_checked_after_asynchronous_interpretation(self):
        entered, release = Event(), Event()
        def delayed():
            entered.set()
            release.wait(5)
            return program([skill("new", "visit", "conference"), home()], ceiling=15)
        service = self.service(program([skill("old", "inspect", "library", 60), home()]), delayed)
        self.addCleanup(release.set)
        service.submit_task("先检查图书馆。", background=False)
        pump(service, lambda s: s["uav"]["z"] > 2.1)
        service.submit_task("改成限高15米的任务。", background=False)
        for _ in range(100):
            service.advance(.05)
            if entered.wait(.005):
                break
        self.assertTrue(entered.is_set())
        # Inject measured motion during interpretation; no new constraints are active yet.
        service._vehicle = VehicleState((4, 15, 4))
        service._environment.position = service._vehicle.position
        release.set()
        waiting = pump(service, lambda s: s["semantic_mission"]["status"] == "needs_clarification")
        self.assertIn("当前位置", waiting["semantic_mission"]["message"])
        self.assertTrue(waiting["flight"]["safety_armed"])
        self.assertIsNone(service.semantic.plan.constraints["max_altitude_m"])

    def test_newest_instruction_wins_over_late_older_response(self):
        entered, release = Event(), Event()
        def delayed():
            entered.set()
            release.wait(5)
            return program([skill("obsolete", "visit", "laboratory"), home()])
        service = self.service(delayed, program([skill("latest", "visit", "conference"), home()]))
        self.addCleanup(release.set)
        service.submit_task("去实验楼。", background=False)
        self.assertTrue(entered.wait(2))
        service.submit_task("改去会议中心。", background=False)
        release.set()
        ready = pump(service, lambda s: s["semantic_mission"]["status"] == "running")
        self.assertEqual(ready["semantic_mission"]["current_step_id"], "latest")
        self.assertEqual(len(service.agent.inputs), 2)
        self.assertEqual(ready["semantic_mission"]["revision"], 1)

    def test_retrying_rejected_update_does_not_resume_old_plan(self):
        service = self.service(program([skill("old", "inspect", "library", 60), home()]),
                               PlannerProviderError("DeepSeek HTTP 401", retryable=False),
                               program([skill("new", "visit", "conference"), home()]))
        service.submit_task("检查图书馆。", background=False)
        pump(service, lambda s: s["uav"]["z"] > 2)
        service.submit_task("改去会议中心。", background=False)
        pump(service, lambda s: s["semantic_mission"]["status"] == "error")
        service.start(background=False)
        ready = pump(service, lambda s: s["semantic_mission"]["revision"] == 2)
        self.assertEqual(ready["semantic_mission"]["current_step_id"], "new")

    def test_elapsed_condition_starts_from_new_revision_not_world_uptime(self):
        condition = dict(source="telemetry", field="elapsed_seconds", op=">=", value=1)
        service = self.service(program([skill("old", "inspect", "library", 60), home()]),
                               program([repeat([skill("wait", "visit", "library", .2)], count=None, until=condition), home()]))
        service.submit_task("检查图书馆。", background=False)
        pump(service, lambda s: s["flight"]["waypoints"] and service.waypoints.current is None
             and s["flight"]["speed_mps"] <= .02)
        prior_time = service._time
        self.assertGreater(prior_time, 1)
        service.submit_task("在这里持续检查一秒，然后返航。", background=False)
        final = pump(service, lambda s: s["status"] == "completed")
        loop = final["semantic_mission"]["loops"][-1]
        self.assertGreaterEqual(loop["iteration"], 2)
        self.assertGreater(service._time, prior_time)

    def test_navigation_prechecks_all_branch_and_loop_back_edges_without_unrolling(self):
        scenario = compact_campus()
        branch = dict(id="branch", action="branch",
                      condition=dict(source="telemetry", field="battery_percent", op=">", value=50),
                      then=[skill("left", "visit", "library")],
                      **{"else": [skill("right", "visit", "conference")]})
        parsed = semantic_plan_from_json_text(json.dumps(program([
            repeat([branch, skill("scan", "thermal_scan", "laboratory")], count=1000), home()])), task_catalog(scenario))
        context = scenario.to_dict()
        context["position"] = context["base"]
        calls = []
        def checked(planner, current, feedback):
            calls.append((tuple(current["position"].values()), tuple(current["goal"].values())))
        with patch("uav_llm_demo.semantic_navigation.checked_plan", side_effect=checked):
            validate_task_routes(parsed, scenario, context)
        base, library, lab, conference = (2, 15, 0), (4, 15, 2.2), (7, 15, 2.2), (10, 15, 2.2)
        self.assertTrue({(base, library), (base, conference), (library, lab), (conference, lab),
                         (lab, library), (lab, conference), (lab, base)}.issubset(set(calls)))
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLess(len(calls), 12)

    def test_truncated_model_program_has_specific_safe_retry_feedback(self):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content": "{partial"}}]}
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response) as post:
            with self.assertRaisesRegex(PlannerProviderError, "response length limit") as caught:
                DeepSeekAgent(api_key="offline-program-test-key").interpret_task(
                    task_catalog(compact_campus()), "重复执行巡检，最后返航。")
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 12000)
        self.assertNotIn("{partial", str(caught.exception))
        response.close.assert_called_once_with()

    def test_malformed_model_choice_keeps_safe_retry_contract(self):
        for choice in (None, [], "unexpected"):
            with self.subTest(choice=choice):
                response = Mock(status_code=200)
                response.json.return_value = {"choices": [choice]}
                with patch("uav_llm_demo.agents._post_deepseek", return_value=response):
                    with self.assertRaisesRegex(PlannerProviderError, "invalid semantic task plan") as caught:
                        DeepSeekAgent(api_key="offline-program-test-key").interpret_task(
                            task_catalog(compact_campus()), "巡检图书馆。")
                self.assertTrue(caught.exception.retryable)
                response.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
