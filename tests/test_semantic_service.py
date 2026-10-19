import json
import unittest
import requests
from dataclasses import replace
from threading import Event
from unittest.mock import Mock, patch

from uav_llm_demo import create_app
from uav_llm_demo.agents import DeepSeekAgent, GeometricPlanner
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import VehicleState, distance, norm
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.safety import ALTITUDE_EPSILON, SafetyChecker


def step(name, action, target, when=None):
    return dict(id=name, action=action, target_id=target, duration_seconds=None, when=when)


def response_plan(steps=None, constraints=None):
    return dict(status="ready", summary="按顺序执行检查并返航。", steps=steps or [
        step("library", "inspect", "library"), step("conference", "capture", "conference"),
        step("home", "return_home", "home")],
        constraints=constraints or dict(max_altitude_m=None, avoid_regions=[]))


def response(data):
    return Mock(status_code=200, json=Mock(return_value={
        "choices": [{"message": {"content": json.dumps(data, ensure_ascii=False)}}],
        "usage": dict(prompt_tokens=100, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=100,
                      completion_tokens=50, total_tokens=150)}))


def advance_until(service, predicate, seconds=240):
    for _ in range(round(seconds / .1)):
        if service._future is not None:
            try:
                service._future.result(timeout=10)
            except Exception:
                pass
        service.advance(.1)
        state = service.state()
        if predicate(state):
            return state
    raise AssertionError(f"Task did not reach expected state: {service.state()['semantic_mission']}")


class SemanticServiceTests(unittest.TestCase):
    def service(self, agent=None, **kwargs):
        scenario = replace(load_scenario("semantic_campus"), dynamic_objects=())
        service = MissionService(scenario=scenario, agent=agent or DeepSeekAgent(api_key="test-secret-key"), **kwargs)
        self.addCleanup(service.close)
        return service

    def test_full_flight_visits_two_sites_dwell_then_lands_with_one_provider_call(self):
        service = self.service()
        self.assertEqual(service._vehicle.position, service._environment.base)
        self.assertEqual(service._vehicle.position[2], 0)
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(response_plan())) as post:
            service.submit_task("先巡检图书馆，再给会议中心模拟拍照，最后返航。", background=False)
            first = advance_until(service, lambda s: len(s["semantic_mission"]["evidence"]) == 1)
            self.assertEqual(first["semantic_mission"]["current_step_id"], "conference")
            self.assertEqual(first["phase"], "fly_to_target")
            self.assertFalse(first["milestones"]["task_completed"])
            self.assertGreaterEqual(first["semantic_mission"]["evidence"][0]["duration_seconds"], 2 - 1e-9)
            final = advance_until(service, lambda s: s["status"] == "completed")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(final["semantic_mission"]["provider_calls"], 1)
        self.assertEqual([e["action"] for e in final["semantic_mission"]["evidence"]], ["inspect", "capture", "return_home"])
        self.assertEqual(final["semantic_mission"]["evidence"][0], first["semantic_mission"]["evidence"][0])
        self.assertTrue(final["milestones"]["safe_success"])
        self.assertLessEqual(distance(service._vehicle.position, service._environment.base), .18)
        self.assertLessEqual(norm(service._vehicle.velocity), .02)
        self.assertTrue(all(e["decision"]["agent_backend"] == "local" for e in final["history"]))
        self.assertGreaterEqual(final["flight"]["planner_calls"], 3)
        self.assertNotIn("test-secret-key", json.dumps(final))
        self.assertEqual(final["diagnostics"]["usage"]["requests"], 1)

    def test_conditional_followup_runs_or_skips_from_observed_site_temperature(self):
        for target, expected in (("laboratory", True), ("library", False)):
            with self.subTest(target=target):
                service = self.service()
                plan = response_plan([step("scan", "thermal_scan", target),
                    step("followup", "capture", target, dict(step_id="scan", field="anomaly", op="==", value=True)),
                    step("home", "return_home", "home")])
                with patch("uav_llm_demo.agents._post_deepseek", return_value=response(plan)):
                    service.submit_task("热检查，如果异常再模拟拍照，然后返航。", background=False)
                    state = advance_until(service, lambda s: s["semantic_mission"]["current_step_id"] == "home")
                mission = state["semantic_mission"]
                self.assertEqual(mission["evidence"][0]["result"]["anomaly"], expected)
                self.assertEqual(mission["evidence"][0]["result"]["temperature_c"], 68 if expected else 29)
                self.assertEqual(mission["steps"][1]["status"], "completed" if expected else "skipped")
                self.assertEqual(len(mission["evidence"]), 2 if expected else 1)
                self.assertFalse(state["milestones"]["task_completed"])

    def test_minimal_instruction_executes_with_default_parameters_and_returns_home(self):
        service = self.service()
        plan = response_plan([step("inspect", "inspect", "library"), step("home", "return_home", "home")])
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(plan)) as post:
            service.submit_task("巡检图书馆", background=False)
            state = advance_until(service, lambda s: s["status"] == "completed")
        self.assertEqual(post.call_count, 1)
        mission = state["semantic_mission"]
        self.assertEqual(mission["instruction"], "巡检图书馆")
        self.assertEqual(mission["constraints"], {"max_altitude_m": None, "avoid_regions": []})
        self.assertEqual([s["duration_seconds"] for s in mission["steps"]], [2, 0])
        self.assertEqual([e["action"] for e in mission["evidence"]], ["inspect", "return_home"])
        self.assertGreaterEqual(mission["evidence"][0]["duration_seconds"], 2 - 1e-9)
        expected = next(point["position"] for point in service._scenario_template.points_of_interest
                        if point["id"] == "library")
        self.assertAlmostEqual(mission["evidence"][0]["position"]["z"], expected["z"], delta=.18 / 5)
        self.assertTrue(state["milestones"]["safe_success"])

    def test_constraints_enter_every_route_and_control_safety_context(self):
        service = self.service()
        constraints = dict(max_altitude_m=25, avoid_regions=["south_plaza"])
        plan = response_plan([step("gate", "visit", "east_gate"), step("home", "return_home", "home")], constraints)
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(plan)):
            service.submit_task("避开南广场，高度不超过25米，查看东门后返航。", background=False)
            final = advance_until(service, lambda s: s["status"] == "completed")
        checker = SafetyChecker(service._context())
        self.assertEqual(checker.upper[2], 5)
        path = [tuple(p[axis] for axis in "xyz") for p in final["continuous_path"]]
        self.assertTrue(all(checker.segment_safe(a, b) for a, b in zip(path, path[1:])))
        self.assertTrue(all(p[2] <= 5 for p in path))
        left, right, north, south = next(region["bounds"] for region in service._scenario_template.semantic_regions
                                        if region["id"] == "south_plaza")
        self.assertFalse(checker.segment_safe((left - 1, (north + south) / 2, 3),
                                             (right + 1, (north + south) / 2, 3)))

    def test_closed_no_fly_corner_cannot_oscillate_between_safe_and_unsafe(self):
        scenario = load_scenario("semantic_campus")
        context = scenario.to_dict()
        left, right, north, south = next(region["bounds"] for region in scenario.semantic_regions
                                        if region["id"] == "south_plaza")
        context["no_fly_zones"] = [dict(min=dict(x=left, y=north, z=0),
                                        max=dict(x=right, y=south, z=scenario.max_altitude))]
        checker = SafetyChecker(context)
        # The segment touches the expanded (left - .2, north - .2) corner.
        endpoint = (left + 25, north - 3, 2)
        self.assertFalse(checker.segment_safe((left - 2, north, 2.2), endpoint))
        self.assertFalse(checker.segment_safe((left - 2, north + 3e-10, 2.2), endpoint))
        self.assertTrue(checker.segment_safe((left - 2, north - .01, 2.2), endpoint))

    def test_return_instruction_at_home_does_not_take_off(self):
        service = self.service()
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(
                response_plan([step("home", "return_home", "home")]))):
            service.submit_task("返航", background=False)
            state = advance_until(service, lambda s: s["status"] == "completed", 5)
        self.assertEqual(state["uav"], state["base"])
        self.assertEqual(state["metrics"]["path_length"], 0)
        self.assertEqual(len(state["semantic_mission"]["evidence"]), 1)

    def test_return_within_landing_tolerance_does_not_climb_again(self):
        service = self.service()
        service.runtime.vehicle = VehicleState((*service._environment.base[:2], .02))
        service._commit_vehicle(service.runtime.vehicle, 0)
        prior_path = list(service._environment.path)
        prior_length = sum(distance(a, b) for a, b in zip(prior_path, prior_path[1:]))
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(
                response_plan([step("home", "return_home", "home")]))):
            service.submit_task("返航", background=False)
            state = advance_until(service, lambda s: s["status"] == "completed", 5)
        self.assertLessEqual(max(p["z"] for p in state["continuous_path"]), .02)
        self.assertEqual(service._environment.path[:len(prior_path)], prior_path)
        self.assertLessEqual(state["metrics"]["path_length"] - prior_length, .1)

    def test_preflight_rejects_unreachable_branch_before_takeoff(self):
        service = self.service()
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(response_plan())), \
             patch.object(GeometricPlanner, "plan", side_effect=ValueError("private raw failure")):
            service.submit_task("巡检后返航。", background=False)
            state = advance_until(service, lambda s: s["semantic_mission"]["status"] == "needs_clarification")
        self.assertEqual(state["uav"], state["base"])
        self.assertEqual(state["flight"]["plan_id"], 0)
        self.assertEqual(state["semantic_mission"]["evidence"], [])
        self.assertNotIn("private raw failure", str(state))

    def test_ten_metre_task_adjusts_observation_points_and_completes_safely(self):
        service = self.service()
        plan = response_plan(constraints=dict(max_altitude_m=10, avoid_regions=[]))
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(plan)) as post:
            service.submit_task("先巡检图书馆，高度10m，再给会议中心模拟拍照，最后返航。", background=False)
            base = service._environment.base
            cruising = advance_until(service, lambda s: distance(
                (s["uav"]["x"], s["uav"]["y"], base[2]), base) > 3)
            self.assertEqual(cruising["flight"]["stage"], "cruise")
            state = advance_until(service, lambda s: s["status"] == "completed")
        self.assertEqual(post.call_count, 1)
        self.assertTrue(state["milestones"]["safe_success"])
        self.assertEqual([e["action"] for e in state["semantic_mission"]["evidence"]],
                         ["inspect", "capture", "return_home"])
        path = [tuple(p[axis] for axis in "xyz") for p in state["continuous_path"]]
        checker = SafetyChecker(service._context())
        self.assertTrue(all(p[2] <= 2 + ALTITUDE_EPSILON for p in path))
        self.assertTrue(all(checker.segment_safe(a, b) for a, b in zip(path, path[1:])))
        self.assertLessEqual(distance(service._vehicle.position, service._environment.base), .18)

    def test_real_constraint_conflict_requires_specific_clarification_without_retries(self):
        service = self.service()
        plan = response_plan(constraints=dict(max_altitude_m=9, avoid_regions=[]))
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(plan)) as post:
            service.submit_task("高度不超过9米，巡检后返航。", background=False)
            state = advance_until(service, lambda s: s["semantic_mission"]["status"] == "needs_clarification")
            for _ in range(100):
                service.advance(.1)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(state["uav"], state["base"])
        self.assertEqual(state["semantic_mission"]["evidence"], [])
        self.assertIn("高度", state["semantic_mission"]["message"])
        self.assertNotIn("地图或约束", state["semantic_mission"]["message"])

    def test_unexpected_task_failure_is_not_mislabeled_as_user_constraint_conflict(self):
        for error_type in (RuntimeError, ValueError):
            with self.subTest(error_type=error_type.__name__):
                service = self.service()
                with patch.object(service.agent, "interpret_task", side_effect=error_type("private-debug-secret")):
                    service.submit_task("巡检后返航。", background=False)
                    state = advance_until(service, lambda s: bool(s["flight"]["error"]))
                self.assertEqual(state["semantic_mission"]["status"], "error")
                self.assertIn("内部错误", state["flight"]["error"])
                self.assertNotIn("private-debug-secret", str(state))
                self.assertEqual(state["uav"], state["base"])

    def test_request_construction_errors_are_reported_without_constraint_blame_or_secret_leak(self):
        failures = (ValueError("private-debug-secret"),
                    UnicodeEncodeError("latin-1", "private-debug-secret中", 20, 21, "private-debug-secret"),
                    *(kind("private-debug-secret") for kind in (
                        requests.exceptions.InvalidHeader, requests.exceptions.InvalidURL,
                        requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema)))
        for failure in failures:
            with self.subTest(error_type=type(failure).__name__):
                service = self.service()
                with patch("uav_llm_demo.agents._post_deepseek", side_effect=failure) as post:
                    service.submit_task("巡检后返航。", background=False)
                    state = advance_until(service, lambda s: bool(s["flight"]["error"]))
                    for _ in range(20):
                        service.advance(.1)
                self.assertEqual(post.call_count, 1)
                self.assertEqual(state["semantic_mission"]["status"], "error")
                self.assertIn("Key 设置", state["feedback"])
                self.assertNotIn("private-debug-secret", str(state))
                self.assertNotIn("test-secret-key", str(state))
                self.assertEqual(state["uav"], state["base"])
                self.assertEqual(state["diagnostics"]["usage"]["requests"], 1)
                self.assertEqual(state["diagnostics"]["usage"]["unreported_requests"], 0)

    def test_transient_failures_retry_more_than_three_times_without_mock(self):
        service = self.service()
        replies = [Mock(status_code=503)] * 4 + [response(response_plan())]
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=replies) as post:
            service.submit_task("巡检图书馆后返航。", background=False)
            state = advance_until(service, lambda s: s["flight"]["plan_id"] > 0, 60)
        self.assertEqual(post.call_count, 5)
        self.assertIsNone(state["flight"]["error"])
        self.assertEqual(state["semantic_mission"]["status"], "running")
        self.assertTrue(state["flight"]["running"])

    def test_auth_error_sanitized_and_no_retry_then_key_change_can_resume(self):
        service = self.service()
        with patch("uav_llm_demo.agents._post_deepseek", return_value=Mock(status_code=401)) as post:
            service.submit_task("巡检图书馆后返航。", background=False)
            state = advance_until(service, lambda s: bool(s["flight"]["error"]))
            for _ in range(30):
                service.advance(.1)
        self.assertEqual(post.call_count, 1)
        self.assertIn("HTTP 401", state["semantic_mission"]["message"])
        self.assertNotIn("test-secret-key", str(state))
        service.configure_agent(dict(backend="deepseek", api_key="replacement-test-key"))
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(response_plan())):
            service.start(background=False)
            state = advance_until(service, lambda s: s["flight"]["plan_id"] > 0)
        self.assertEqual(state["semantic_mission"]["status"], "running")

    def test_paused_interpretation_result_is_discarded_and_resume_reinterprets(self):
        entered, release = Event(), Event()
        service = self.service()
        def slow(*args, **kwargs):
            entered.set()
            release.wait(5)
            return response(response_plan())
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=slow) as post:
            service.submit_task("巡检后返航。", background=False)
            self.assertTrue(entered.wait(2))
            service.stop_runtime()
            release.set()
            paused = advance_until(service, lambda s: service._future is None)
            self.assertEqual(paused["semantic_mission"]["steps"], [])
            self.assertFalse(paused["flight"]["running"])
            self.assertFalse(paused["flight"]["planning"])
            service.start(background=False)
            resumed = advance_until(service, lambda s: s["flight"]["plan_id"] > 0)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(resumed["semantic_mission"]["status"], "running")

    def test_changing_key_during_paused_pending_request_does_not_resume_calls(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        service = self.service()
        def slow(*args, **kwargs):
            entered.set()
            release.wait(5)
            return response(response_plan())
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=slow) as post:
            service.submit_task("巡检后返航。", background=False)
            self.assertTrue(entered.wait(2))
            service.stop_runtime()
            state = service.configure_agent(dict(backend="deepseek", api_key="replacement-test-key"))
            self.assertFalse(state["flight"]["planning"])
            release.set()
            advance_until(service, lambda s: service._future is None)
            for _ in range(100):
                service.advance(.1)
            self.assertEqual(post.call_count, 1)
            self.assertFalse(service.state()["flight"]["running"])
            self.assertEqual(service.state()["semantic_mission"]["steps"], [])

    def test_reset_and_map_switch_discard_late_semantic_response(self):
        for action in ("reset", "switch"):
            with self.subTest(action=action):
                service = self.service()
                release, entered = Event(), Event()
                def slow(*args, **kwargs):
                    entered.set()
                    release.wait(5)
                    return response(response_plan())
                with patch("uav_llm_demo.agents._post_deepseek", side_effect=slow):
                    service.submit_task("巡检后返航。", background=False)
                    self.assertTrue(entered.wait(2))
                    state = service.reset() if action == "reset" else service.switch_scenario("urban_crossing")
                    release.set()
                    state = advance_until(service, lambda s: service._future is None)
                self.assertFalse(state["flight"]["running"])
                self.assertEqual(state["flight"]["plan_id"], 0)
                if action == "reset":
                    self.assertEqual(state["semantic_mission"]["status"], "idle")
                    self.assertEqual(state["semantic_mission"]["evidence"], [])
                else:
                    self.assertIsNone(state["semantic_mission"])
                    self.assertEqual(service.configure_agent({"backend": "mock"})["agent_backend"], "mock")

    def test_api_rejects_wrong_mode_missing_key_bad_shape_and_accepts_active_update(self):
        with patch.dict("os.environ", {}, clear=True):
            service = MissionService(scenario_id="semantic_campus")
        self.addCleanup(service.close)
        client = create_app(service=service).test_client()
        for data in (None, [], {"instruction": ""}, {"instruction": "查看广场"}):
            self.assertEqual(client.post("/api/mission/task", json=data).status_code, 400)
        self.assertEqual(client.post("/api/config", json={"backend": "mock"}).status_code, 400)
        self.assertEqual(client.post("/api/mission/plan").status_code, 400)
        self.assertEqual(client.post("/api/mission/start").status_code, 400)
        self.assertEqual(client.post("/api/runtime", json={"backend": "airsim"}).status_code, 400)
        self.assertEqual(client.post("/api/mission/goal", json={"target": dict(x=3, y=3, z=3)}).status_code, 400)
        service.configure_agent(dict(backend="deepseek", api_key="test-secret-key"))
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(response_plan())):
            service.submit_task("巡检后返航", background=False)
            original_run = service.trace.run_id
            original_mission = service.semantic.mission_id
            service.submit_task("修改任务，先检查图书馆", background=False)
            self.assertEqual(service.trace.run_id, original_run)
            self.assertEqual(service.semantic.mission_id, original_mission)
            advance_until(service, lambda state: state["semantic_mission"]["status"] == "running")
            with self.assertRaisesRegex(ValueError, "API key"):
                service.submit_task("test-secret-key", background=False)
        service.reset()
        service.switch_scenario("urban_crossing")
        self.assertEqual(client.post("/api/mission/task", json={"instruction": "查看广场"}).status_code, 400)

    def test_clarification_does_not_fly_or_count_as_completion(self):
        service = self.service()
        plan = dict(status="needs_clarification", summary="不具备投递能力，请改为巡检任务。", steps=[],
                    constraints=dict(max_altitude_m=None, avoid_regions=[]))
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(plan)):
            service.submit_task("投递包裹", background=False)
            state = advance_until(service, lambda s: s["semantic_mission"]["status"] == "needs_clarification")
        self.assertEqual(state["uav"], state["base"])
        self.assertFalse(state["milestones"]["task_completed"])
        self.assertFalse(state["flight"]["running"])
        self.assertEqual(state["history"], [])


if __name__ == "__main__":
    unittest.main()
