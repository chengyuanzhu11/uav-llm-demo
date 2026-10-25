"""End-to-end control using offline LLM responses and production geometry/runtime.

These tests never call the remote API or touch the running Flask service.
"""
from dataclasses import replace
from math import pi
from unittest import TestCase
from unittest.mock import Mock, patch
import json

from tests.test_semantic_updates import OfflineSemanticAgent, compact_campus, home, program, pump, skill
from uav_llm_demo.agents import DeepSeekAgent, PlannerProviderError, SemanticPlanError
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import distance, norm, VehicleState
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.task_contract import TaskContractMonitor, TaskContractError
from uav_llm_demo.semantic_tasks import SemanticMission, semantic_plan_from_json_text, task_catalog
from uav_llm_demo.semantic_navigation import apply_task_constraints


def control(target, state="landed", name="settle", when=None, **parameters):
    return dict(id=name, action="control", target_id=target, when=when,
                control=dict(state=state, **parameters))


def review(name="assess"):
    return dict(id=name, action="review", instruction="根据检查结果决定后续工作，保持最终位置和状态。", when=None)


class GenericControlServiceTests(TestCase):
    def setUp(self):
        guard = patch("uav_llm_demo.agents._post_deepseek", side_effect=AssertionError("Remote API forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def service(self, *replies, actual=False):
        scene = replace(load_scenario("semantic_campus"), dynamic_objects=()) if actual else compact_campus()
        service = MissionService(scenario=scene, agent=OfflineSemanticAgent(replies), planner_latency_seconds=0,
                                 traffic_seed=17)
        self.addCleanup(service.close)
        return service

    def complete(self, service):
        return pump(service, lambda s: s["status"] == "completed", seconds=450)

    def test_football_center_landing_without_return_and_next_task_relaunches(self):
        instruction = "去北区足球场中央降落停留，不用返航"
        service = self.service(program([control("north_football")]),
                               program([control("north_basketball", name="next")]), actual=True)
        service.submit_task(instruction, background=False)
        state = self.complete(service)
        evidence = state["semantic_mission"]["evidence"]
        self.assertEqual([e["action"] for e in evidence], ["control"])
        self.assertEqual(service.agent.inputs[0][1], instruction)
        self.assertEqual(state["semantic_mission"]["goal_state"], {
            "target_id": "north_football", "position": dict(x=42.0, y=22.5, z=0.0),
            "state": "landed", "yaw_deg": None, "duration_seconds": 2.0, "satisfied": True})
        self.assertTrue(evidence[-1]["result"]["landed"])
        self.assertGreaterEqual(evidence[-1]["duration_seconds"], 2 - 1e-9)
        self.assertLessEqual(distance(service._vehicle.position, (42, 22.5, 0)), .18)
        self.assertLessEqual(service._vehicle.position[2] * 5, .05)
        self.assertLessEqual(norm(service._vehicle.velocity), .02)
        self.assertTrue(state["milestones"]["safe_success"])
        self.assertFalse(state["milestones"]["base_returned"])
        parked = service._vehicle.position
        for _ in range(50):
            service.advance(.1)
        self.assertEqual(service._vehicle.position, parked)
        self.assertFalse(service._safety_armed)
        self.assertEqual(len(service.agent.inputs), 1)
        # The same interface takes off from the remote field and lands elsewhere.
        service.submit_task("再去北区篮球场中央降落。", background=False)
        final = self.complete(service)
        self.assertLessEqual(distance(service._vehicle.position, (44, 13.5, 0)), .18)
        self.assertFalse(final["milestones"]["base_returned"])
        self.assertEqual(len(final["semantic_mission"]["evidence"]), 2)
        self.assertEqual(final["semantic_mission"]["goal_state"]["target_id"], "north_basketball")

    def test_land_at_plaza_then_takeoff_hold_offset_turn_and_return(self):
        nodes = [control("south_plaza", duration_seconds=.3),
                 control("south_plaza", "airborne", "lift", offset_m=[3, 0, 0],
                         altitude_m=2, yaw_deg=90, duration_seconds=.3), home()]
        service = self.service(program(nodes), actual=True)
        service.submit_task("南广场降落后起飞，向东3米在2米高处朝东停留，最后返航。", background=False)
        state = self.complete(service)
        records = state["semantic_mission"]["evidence"]
        self.assertEqual([e["action"] for e in records], ["control", "control", "return_home"])
        self.assertEqual(records[0]["result"]["state"], "landed")
        self.assertEqual(records[1]["result"]["state"], "airborne")
        self.assertLessEqual(distance(tuple(records[1]["position"].values()), (49.5,82,.4)), .18)
        self.assertTrue(state["milestones"]["base_returned"])

    def test_airborne_completion_holds_position_and_yaw_with_safety_active(self):
        service = self.service(program([control("library", "airborne", anchor="arrival", yaw_deg=90)]))
        service.submit_task("图书馆访问点朝东悬停，不返航。", background=False)
        state = self.complete(service)
        self.assertEqual(state["semantic_mission"]["goal_state"]["state"], "airborne")
        self.assertLess(abs(service._vehicle.yaw-pi/2), .04)
        self.assertTrue(service._safety_armed)
        held = service._hold_target
        # Simulate a modest position disturbance without changing the validated goal.
        service.runtime.vehicle = VehicleState((held[0]+.1, held[1], held[2]), yaw=pi/2)
        service._commit_vehicle(service.runtime.read_state(), 0)
        for _ in range(60):
            service.advance(.1)
        self.assertLess(distance(service._vehicle.position, held), .03)
        self.assertEqual(service.state()["status"], "completed")
        self.assertFalse(service._plan_requested)
        self.assertEqual(len(service.agent.inputs), 1)

    def test_review_uses_measured_evidence_and_preserves_terminal_goal(self):
        end = control("home", name="final")
        service = self.service(program([skill("scan", "thermal_scan", "library", .1), review(), end]),
                               program([skill("follow", "inspect", "conference", .1), end]))
        service.submit_task("检查图书馆，根据结果决定复查地点，最后回基地降落。", background=False)
        state = self.complete(service)
        self.assertEqual(len(service.agent.inputs), 2)
        context = service.agent.inputs[1][0]["mission_context"]
        self.assertEqual(context["evidence"][0]["result"]["temperature_c"], 29)
        self.assertIn("observations", context)
        self.assertIn("Execution review", service.agent.inputs[1][2])
        self.assertEqual([r["action"] for r in state["semantic_mission"]["evidence"]],
                         ["thermal_scan", "review", "inspect", "control"])
        self.assertEqual(state["semantic_mission"]["goal_state"]["state"], "landed")
        self.assertEqual(service._autonomous_reviews, 1)
        self.assertTrue(any(f["kind"]=="review_accepted" for f in service.semantic.execution_feedback))

    def test_review_cannot_replace_landing_with_hovering(self):
        service = self.service(program([review(), control("home")]),
                               *[program([control("home", "airborne")]) for _ in range(3)])
        service.submit_task("评估当前情况后基地降落。", background=False)
        state = pump(service, lambda s: s["semantic_mission"]["status"] == "generation_failed")
        self.assertFalse(state["milestones"]["task_completed"])
        self.assertEqual(service._vehicle.position, service._environment.base)
        self.assertIn("自动修正仍未通过", service.semantic.message)
        self.assertEqual(len(service.agent.inputs), 4)
        self.assertEqual(service.semantic.evidence, [])

    def test_review_budget_prevents_recursive_paid_requests(self):
        repeated = program([review(), control("home")])
        service = self.service(*[repeated for _ in range(10)])
        service.submit_task("按观测评估，最后降落。", background=False)
        state = pump(service, lambda s: s["semantic_mission"]["status"] == "safety_wait")
        self.assertEqual(service._autonomous_reviews, 8)
        self.assertEqual(len(service.agent.inputs), 9)  # Initial interpretation plus eight reviews.
        self.assertFalse(state["milestones"]["task_completed"])

    def test_model_schema_correction_has_reason_and_transport_still_retries(self):
        service = self.service(SemanticPlanError("Control state must be landed or airborne."),
                               program([home()]))
        service.submit_task("回基地", background=False)
        self.complete(service)
        self.assertIn("Control state", service.agent.inputs[1][2])
        failed = self.service(*[SemanticPlanError("Unexpected task fields.") for _ in range(4)])
        failed.submit_task("测试不合法程序", background=False)
        pump(failed, lambda s:s["semantic_mission"]["status"]=="generation_failed")
        self.assertEqual(len(failed.agent.inputs), 3)
        temporary = self.service(*[PlannerProviderError("HTTP 503", retryable=True) for _ in range(4)],
                                 program([home()]))
        temporary.submit_task("回基地", background=False)
        self.complete(temporary)
        self.assertEqual(len(temporary.agent.inputs), 5)

    def test_unsafe_landing_rejected_before_takeoff(self):
        for target in ("library", "central_garden", "south_court"):
            with self.subTest(target=target):
                service = self.service(program([control(target)]), actual=True)
                service.submit_task("到目标中央降落。", background=False)
                state = pump(service, lambda s:s["semantic_mission"]["status"]=="needs_clarification")
                self.assertEqual(service._vehicle.position, service._environment.base)
                self.assertFalse(state["milestones"]["task_completed"])
                self.assertEqual(service.semantic.evidence, [])

    def test_review_in_branch_with_outer_continuation_is_valid(self):
        branch = dict(id="choice", action="branch",
                      condition=dict(source="telemetry",field="battery_percent",op=">",value=0),
                      then=[review()], **{"else": []})
        service = self.service(program([branch, home()]), program([home()]))
        service.submit_task("根据电量情况评估，最终回基地。", background=False)
        self.complete(service)
        self.assertEqual(len(service.agent.inputs), 2)

    def test_conditionally_skipped_only_action_is_not_stuck_running(self):
        condition = dict(source="telemetry", field="battery_percent", op="<", value=0)
        service = self.service(program([control("library", "airborne", when=condition)]))
        service.submit_task("电量小于零时去图书馆。", background=False)
        state = self.complete(service)
        self.assertEqual(service.semantic.evidence, [])
        self.assertEqual(state["semantic_mission"]["goal_state"]["target_id"], None)
        self.assertEqual(service._vehicle.position, service._environment.base)

    def test_previous_landing_permission_does_not_override_new_avoid_region(self):
        scene = load_scenario("semantic_campus")
        context = scene.to_dict()
        context["ground_sites"] = [dict(x=48.5,y=82,z=0), dict(x=42,y=22.5,z=0)]
        apply_task_constraints(context, scene, dict(max_altitude_m=None, avoid_regions=["south_plaza"]))
        self.assertEqual(context["ground_sites"], [dict(x=42,y=22.5,z=0)])

    def test_provider_prompt_receives_raw_instruction_and_control_geometry(self):
        request = "去北区足球场中央降落停留，不用返航"
        response = Mock(status_code=200)
        response.json.return_value = {"choices":[{"message":{"content":json.dumps(program([control("north_football")]))}}]}
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response) as post:
            plan = DeepSeekAgent(api_key="offline-test-key").interpret_task(task_catalog(load_scenario("semantic_campus")), request)
        payload = post.call_args.kwargs["json"]
        self.assertIn(request, payload["messages"][-1]["content"])
        self.assertIn("north_football", payload["messages"][-1]["content"])
        self.assertEqual(plan.steps[0]["action"], "control")
        self.assertEqual(plan.steps[0]["control"]["state"], "landed")

    def test_review_contract_keeps_heading_dwell_and_motion_path(self):
        catalog = task_catalog(compact_campus())
        def parsed(nodes):
            return semantic_plan_from_json_text(json.dumps(program(nodes)), catalog)
        initial = TaskContractMonitor(parsed([control("home", yaw_deg=90, duration_seconds=5)]), "test")
        for modified in (control("home", yaw_deg=0, duration_seconds=5),
                         control("home", yaw_deg=90, duration_seconds=1)):
            with self.assertRaises(TaskContractError):
                initial.validate_revision(parsed([modified]), after_review=False)
        node = dict(id="path", action="motion", target_id="library", when=None,
                    motion=dict(path=dict(kind="polyline", points_m=[[0,0,0],[3,0,0]])))
        with self.assertRaises(TaskContractError):
            TaskContractMonitor(parsed([node]), "test").validate_revision(parsed([home()]), after_review=False)

    def test_hover_above_ground_does_not_produce_landing_evidence(self):
        scene = load_scenario("semantic_campus")
        mission = SemanticMission(scene)
        mission.accept(semantic_plan_from_json_text(json.dumps(program([control("north_football")])),task_catalog(scene)))
        for i in range(50):
            self.assertFalse(mission.observe((42,22.5,.03),0,.1,i*.1))  # 15 cm high.
        self.assertEqual(mission.evidence, [])
        self.assertFalse(mission.goal_state["satisfied"])
        for i in range(20):
            mission.observe((42,22.5,0),0,.1,5+i*.1)
        self.assertEqual(mission.status,"completed")
        self.assertTrue(mission.goal_state["satisfied"])

    def test_open_ended_observation_review_can_choose_a_continuation(self):
        service = self.service(program([skill("scan","thermal_scan","library",.1),review()]),
                               program([control("conference","airborne")]))
        service.submit_task("检查图书馆，再根据结果自行选择下一处巡检地点。",background=False)
        self.complete(service)
        self.assertEqual(service.semantic.goal_state["target_id"],"conference")
        self.assertEqual(len(service.agent.inputs),2)
        self.assertEqual(service.agent.inputs[1][0]["mission_context"]["evidence"][0]["result"]["temperature_c"],29)
