from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from uav_llm_demo.agents import DeepSeekAgent, GeometricPlanner, MockLLMAgent, PlannerProviderError, checked_plan
from uav_llm_demo.safety import SafetyChecker
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_navigation import apply_task_constraints, validate_task_routes
from uav_llm_demo.semantic_tasks import SemanticConstraintError, SemanticMission, SemanticPlan, semantic_plan_from_json_text, task_catalog


def scene():
    """Small schema/evidence fixture; navigation tests use the real campus."""
    points = [
        dict(id="home", label="基地", kind="base", position=dict(x=2, y=27, z=0), allowed_actions=["return_home"]),
        dict(id="library", label="图书馆", kind="library", position=dict(x=8, y=13, z=4),
             allowed_actions=["visit", "inspect", "thermal_scan", "capture"]),
        dict(id="laboratory", label="实验一号楼", kind="laboratory", position=dict(x=23, y=12, z=3),
             allowed_actions=["visit", "inspect", "thermal_scan", "capture"]),
        dict(id="conference", label="会议中心", kind="conference", position=dict(x=23, y=27, z=6),
             allowed_actions=["visit", "inspect", "capture"]),
        dict(id="south_plaza", label="南广场", kind="plaza", position=dict(x=8, y=23, z=3),
             allowed_actions=["visit", "capture"]),
    ]
    return SimpleNamespace(scenario_id="semantic_campus", min_altitude=2, max_altitude=10,
                           points_of_interest=tuple(points), semantic_regions=(
                               dict(id="south_plaza", label="南广场", bounds=[4, 12, 19, 27.5]),),
                           recommended_tasks=("巡检图书馆后返航。",))


def step(step_id, action, target_id, duration=None, when=None):
    return dict(id=step_id, action=action, target_id=target_id, duration_seconds=duration, when=when)


def proposal(steps=None):
    return dict(status="ready", summary="巡检图书馆并返航。",
                steps=steps or [step("inspect_1", "inspect", "library"), step("return_1", "return_home", "home")],
                constraints=dict(max_altitude_m=None, avoid_regions=[]))


def parse(data, scenario=None):
    return semantic_plan_from_json_text(json.dumps(data, ensure_ascii=False), task_catalog(scenario or scene()))


def position_of(scenario, target):
    position = next(point["position"] for point in scenario.points_of_interest if point["id"] == target)
    return tuple(position[axis] for axis in "xyz")


class SemanticSchemaTests(unittest.TestCase):
    def test_catalog_has_capabilities_but_no_hidden_sensor_truth(self):
        scenario = scene()
        scenario.points_of_interest[1]["temperature_c"] = 123
        catalog = task_catalog(scenario)
        self.assertNotIn("123", json.dumps(catalog))
        self.assertEqual(catalog["actions"]["thermal_scan"]["result_fields"]["temperature_c"], "number")
        self.assertNotIn("anomaly", catalog["points_of_interest"][1])
        self.assertEqual(catalog["vehicle"]["id"], "uav-1")
        self.assertEqual(catalog["altitude_range_m"], {"min": 10, "max": 50})
        self.assertIn("not a required minimum", catalog["observation_altitude_policy"])
        catalog["points_of_interest"][1]["position"]["x"] = -1
        self.assertEqual(scenario.points_of_interest[1]["position"]["x"], 8)

    def test_valid_plan_defaults_and_snapshot_isolation(self):
        plan = parse(proposal())
        self.assertEqual(plan.steps[0]["duration_seconds"], 2)
        self.assertEqual(plan.steps[1]["duration_seconds"], 0)
        exported = plan.to_dict()
        exported["steps"][0]["target_id"] = "home"
        exported["constraints"]["avoid_regions"].append("invalid")
        self.assertEqual(plan.steps[0]["target_id"], "library")
        self.assertEqual(plan.constraints["avoid_regions"], [])

    def test_catalog_defaults_match_executor_for_all_actions_and_remain_isolated(self):
        scenario = scene()
        catalog = task_catalog(scenario)
        data = proposal([step(action, action, "home" if action == "return_home" else "library")
                         for action in ("visit", "inspect", "thermal_scan", "capture", "return_home")])
        data["constraints"] = catalog["default_constraints"]
        plan = parse(data, scenario)
        self.assertEqual(plan.constraints, {"max_altitude_m": None, "avoid_regions": []})
        self.assertEqual([item["duration_seconds"] for item in plan.steps], [0, 2, 3, 1, 0])
        for item in plan.steps:
            self.assertEqual(item["duration_seconds"], catalog["actions"][item["action"]]["default_duration_seconds"])
        catalog["default_constraints"]["avoid_regions"].append("south_plaza")
        self.assertEqual(task_catalog(scenario)["default_constraints"]["avoid_regions"], [])
        self.assertEqual(SemanticMission(scenario).to_dict()["constraints"]["avoid_regions"], [])

    def test_explicit_supported_values_are_not_replaced_by_defaults(self):
        data = proposal([step("inspect", "inspect", "library", duration=7),
                         step("home", "return_home", "home")])
        data["constraints"] = {"max_altitude_m": 15, "avoid_regions": ["south_plaza"]}
        plan = parse(data)
        self.assertEqual(plan.steps[0]["duration_seconds"], 7)
        self.assertEqual(plan.constraints, data["constraints"])
        mission = SemanticMission(scene())
        mission.accept(plan)
        self.assertEqual(mission.current_goal, (8, 13, 3))

    def test_unknown_fields_targets_and_capabilities_fail_closed(self):
        variants = []
        for field, value in (("target_id", "unknown"), ("action", "drop_package"), ("id", "第一步"),
                             ("waypoints", []), ("duration_seconds", "3")):
            data = proposal()
            data["steps"][0][field] = value
            variants.append(data)
        data = proposal([step("scan", "thermal_scan", "conference"), step("home", "return_home", "home")])
        variants.append(data)
        data = proposal()
        data["ignored_request"] = "deliver goods"
        variants.append(data)
        data = proposal()
        del data["steps"][0]["when"]
        variants.append(data)
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse(data)

    def test_json_duplicate_nonfinite_and_shape_rejected(self):
        catalog = task_catalog(scene())
        texts = ["null", "[]", "{}", "{", "```json\n{}\n```", json.dumps(proposal()) + " suffix",
                 json.dumps(proposal()).replace('"status": "ready"', '"status":"ready","status":"ready"')]
        for value in (True, float("inf"), float("nan"), -1, 601, 0, 10**400):
            data = proposal()
            data["steps"][0]["duration_seconds"] = value
            texts.append(json.dumps(data))
        for text in texts:
            with self.subTest(text=text[:120]), self.assertRaises(ValueError):
                semantic_plan_from_json_text(text, catalog)

    def test_return_home_is_optional_and_still_targets_home(self):
        self.assertEqual(len(parse(proposal([step("first", "visit", "library")])).steps), 1)
        self.assertEqual(len(parse(proposal([step("first", "return_home", "home"), step("next", "visit", "library")])).steps), 2)
        variants = [proposal([step("first", "visit", "home"), step("end", "return_home", "home")]),
                    proposal([step("end", "return_home", "library")]),
                    proposal([step("end", "return_home", "home", 1)])]
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse(data)
        self.assertEqual(len(parse(proposal([step("end", "return_home", "home")])).steps), 1)

    def test_duplicate_step_ids_and_step_limit_rejected(self):
        with self.assertRaises(ValueError):
            parse(proposal([step("same", "visit", "library"), step("same", "return_home", "home")]))
        with self.assertRaises(ValueError):
            parse(proposal([step(f"step_{i}", "visit", "library") for i in range(128)] + [step("end", "return_home", "home")]))
        data = proposal()
        data["steps"] = []
        with self.assertRaises(ValueError):
            parse(data)

    def test_condition_requires_typed_guaranteed_preceding_result(self):
        condition = dict(step_id="scan", field="temperature_c", op=">", value=40)
        valid = proposal([step("scan", "thermal_scan", "laboratory"),
                          step("capture", "capture", "library", when=condition),
                          step("home", "return_home", "home")])
        self.assertEqual(parse(valid).steps[1]["when"], condition)
        for field, value in (("step_id", "capture"), ("field", "secret"), ("op", "eval"), ("value", True)):
            data = deepcopy(valid)
            data["steps"][1]["when"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                parse(data)
        data = deepcopy(valid)
        data["steps"][0]["action"] = "inspect"
        with self.assertRaises(ValueError):
            parse(data)

    def test_constraints_have_real_units_and_reject_goal_conflicts(self):
        for ceiling in (True, "20", 9, 51, float("nan")):
            data = proposal()
            data["constraints"]["max_altitude_m"] = ceiling
            with self.subTest(ceiling=ceiling), self.assertRaises(ValueError):
                parse(data)
        for ceiling in (10, 19.9, 20, 30):
            data = proposal()
            data["constraints"]["max_altitude_m"] = ceiling
            self.assertEqual(parse(data).constraints["max_altitude_m"], ceiling)
        data = proposal([step("visit", "visit", "south_plaza"), step("home", "return_home", "home")])
        data["constraints"]["avoid_regions"] = ["south_plaza"]
        with self.assertRaises(SemanticConstraintError) as caught:
            parse(data)
        self.assertEqual(caught.exception.code, "target_in_avoided_region")
        self.assertEqual(caught.exception.user_message,
                         "任务地点位于指定的避让区域内，请调整任务地点或避让区域。")
        for regions in (["unknown"], ["south_plaza", "south_plaza"], "south_plaza", [None]):
            data = proposal()
            data["constraints"]["avoid_regions"] = regions
            with self.assertRaises(ValueError):
                parse(data)

    def test_constraint_errors_have_fixed_public_codes_and_messages(self):
        for ceiling in (9, 51):
            data = proposal()
            data["constraints"]["max_altitude_m"] = ceiling
            with self.subTest(ceiling=ceiling), self.assertRaises(SemanticConstraintError) as caught:
                parse(data)
            self.assertEqual(caught.exception.code, "altitude_out_of_range")
            self.assertEqual(caught.exception.user_message, "高度上限超出地图允许的飞行范围，请调整高度要求。")
            self.assertEqual(str(caught.exception), caught.exception.user_message)

    def test_clarification_never_contains_executable_actions(self):
        data = dict(status="needs_clarification", summary="该无人机没有投递能力，请修改任务。", steps=[],
                    constraints=dict(max_altitude_m=None, avoid_regions=[]))
        self.assertEqual(parse(data).steps, ())
        mission = SemanticMission(scene())
        mission.accept(parse(data))
        self.assertEqual(mission.status, "needs_clarification")
        self.assertIsNone(mission.current_goal)
        data["steps"] = [step("home", "return_home", "home")]
        with self.assertRaises(ValueError):
            parse(data)


class SemanticEvidenceTests(unittest.TestCase):
    def mission(self, steps=None):
        mission = SemanticMission(scene(), "巡检后返航", mission_id="mission-test")
        mission.accept(parse(proposal(steps)))
        self.assertEqual(mission.status, "running")
        return mission

    def test_dwell_needs_stationary_arrival_and_resets_on_departure_or_pause(self):
        mission = self.mission()
        goal = mission.current_goal
        self.assertFalse(mission.observe(goal, .03, 2, 2))
        self.assertFalse(mission.observe((goal[0] + .061, *goal[1:]), 0, 2, 4))
        self.assertFalse(mission.observe(goal, 0, 1, 5))
        self.assertEqual(mission.status, "running")
        self.assertFalse(mission.observe(goal, 0, 1, 6, enabled=False))
        self.assertFalse(mission.observe(goal, 0, 1, 7))
        self.assertFalse(mission.observe((goal[0], goal[1], goal[2] + .04), 0, 1, 8))
        self.assertFalse(mission.observe(goal, 0, 1, 9))
        self.assertTrue(mission.observe(goal, 0, 1, 10))
        self.assertEqual(len(mission.evidence), 1)
        self.assertEqual(mission.current_step["action"], "return_home")
        self.assertNotEqual(mission.status, "completed")

    def test_evidence_bound_to_target_executor_and_mission_before_ground_completion(self):
        mission = self.mission()
        self.assertTrue(mission.observe(mission.current_goal, 0, 2, 2))
        event = deepcopy(mission.evidence[0])
        self.assertEqual((event["mission_id"], event["vehicle_id"], event["target_id"], event["action"]),
                         ("mission-test", "uav-1", "library", "inspect"))
        self.assertEqual(event["source"], "simulated_inspection")
        self.assertFalse(mission.observe((2, 27, 2), 0, 2, 4))
        self.assertFalse(mission.observe((2, 27, 0), .03, 2, 6))
        self.assertEqual(mission.evidence[0], event)
        self.assertTrue(mission.observe((2, 27, 0), 0, .05, 6.05))
        self.assertEqual(mission.status, "completed")
        self.assertIsNone(mission.current_step)
        self.assertFalse(mission.observe((2, 27, 0), 0, 1, 7))
        self.assertEqual(len(mission.evidence), 2)

    def test_anomaly_condition_executes_capture_only_after_bound_scan(self):
        for target, expected_anomaly in (("laboratory", True), ("library", False)):
            with self.subTest(target=target):
                mission = self.mission([step("scan", "thermal_scan", target),
                    step("capture", "capture", target, when=dict(step_id="scan", field="anomaly", op="==", value=True)),
                    step("home", "return_home", "home")])
                self.assertEqual(mission.evidence, [])
                self.assertTrue(mission.observe(mission.current_goal, 0, 3, 3))
                self.assertEqual(mission.evidence[0]["result"]["anomaly"], expected_anomaly)
                self.assertTrue(mission.evidence[0]["result"]["simulated"])
                if expected_anomaly:
                    self.assertEqual(mission.current_step["id"], "capture")
                    self.assertTrue(mission.observe(mission.current_goal, 0, 1, 4))
                    self.assertFalse(mission.evidence[-1]["result"]["image_generated"])
                else:
                    self.assertEqual(mission.to_dict()["steps"][1]["status"], "skipped")
                    self.assertEqual(len(mission.evidence), 1)
                self.assertEqual(mission.current_step["id"], "home")

    def test_public_snapshots_do_not_mutate_execution_and_accept_revalidates(self):
        mission = self.mission()
        snapshot = mission.to_dict()
        snapshot["steps"][0]["target_id"] = "home"
        current = mission.current_step
        current["duration_seconds"] = 0
        self.assertEqual(mission.current_goal, (8, 13, 4))
        self.assertFalse(mission.observe(mission.current_goal, 0, 1, 1))
        forged = SemanticPlan("ready", "伪造", (step("x", "launch_missile", "library"),),
                              dict(max_altitude_m=None, avoid_regions=[]))
        with self.assertRaises(ValueError):
            mission.accept(forged)
        self.assertEqual(mission.current_step["action"], "inspect")

    def test_nonfinite_telemetry_cannot_create_evidence(self):
        mission = self.mission()
        self.assertFalse(mission.observe((float("nan"), 13, 4), 0, 3, 3))
        self.assertFalse(mission.observe((10**400, 13, 4), 0, 3, 3))
        for speed, dt, now in ((float("nan"), 3, 3), (0, float("inf"), 3), (0, 3, -1)):
            with self.assertRaises(ValueError):
                mission.observe(mission.current_goal, speed, dt, now)
        self.assertEqual(mission.evidence, [])

    def test_invalid_telemetry_breaks_dwell_continuity(self):
        mission = self.mission()
        self.assertFalse(mission.observe(mission.current_goal, 0, 1, 1))
        with self.assertRaises(ValueError):
            mission.observe(mission.current_goal, float("nan"), 1, 2)
        self.assertFalse(mission.observe(mission.current_goal, 0, 1, 3))
        self.assertTrue(mission.observe(mission.current_goal, 0, 1, 4))


class SemanticProviderTests(unittest.TestCase):
    def response(self, data):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps(data)}}]}
        return response

    def test_task_interpretation_uses_shared_transport_and_no_geometry_prompt(self):
        with patch("uav_llm_demo.agents._post_deepseek", return_value=self.response(proposal())) as post:
            plan = DeepSeekAgent(api_key="test-credential").interpret_task(task_catalog(scene()), "巡检图书馆后返航。")
        self.assertEqual(plan.steps[0]["action"], "inspect")
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["max_tokens"], 12000)
        self.assertIn("Catalog:", request["json"]["messages"][1]["content"])
        self.assertIn("巡检图书馆后返航。", request["json"]["messages"][1]["content"])
        self.assertNotIn("Plan a complete UAV route", request["json"]["messages"][0]["content"])
        sent_catalog = json.loads(request["json"]["messages"][1]["content"].split("\nInstruction:\n")[0].removeprefix("Catalog:\n"))
        self.assertEqual(sent_catalog["default_constraints"], {"max_altitude_m": None, "avoid_regions": []})

    def test_semantic_usage_has_separate_prompt_version_on_success_and_failure(self):
        for response in (self.response(proposal()), self.response({}), Mock(status_code=429)):
            with self.subTest(status=response.status_code), \
                    patch("uav_llm_demo.agents._post_deepseek", return_value=response), \
                    patch("uav_llm_demo.agents.record_request") as usage:
                try:
                    DeepSeekAgent(api_key="test-credential").interpret_task(task_catalog(scene()), "巡检图书馆后返航")
                except PlannerProviderError:
                    pass
            self.assertEqual(usage.call_count, 1)
            self.assertEqual(usage.call_args.kwargs["prompt_version"], "semantic-tasks-v6")

    def test_api_errors_are_sanitized_and_invalid_semantic_output_never_becomes_route(self):
        agent = DeepSeekAgent(api_key="test-credential")
        with patch("uav_llm_demo.agents._post_deepseek", return_value=self.response({})):
            with self.assertRaisesRegex(PlannerProviderError, "invalid semantic task plan") as caught:
                agent.interpret_task(task_catalog(scene()), "检查图书馆")
        self.assertFalse(caught.exception.retryable)
        self.assertTrue(caught.exception.detail)
        with patch("uav_llm_demo.agents._post_deepseek", return_value=Mock(status_code=401, text="test-credential")):
            with self.assertRaises(PlannerProviderError) as caught:
                agent.interpret_task(task_catalog(scene()), "检查图书馆")
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn("test-credential", str(caught.exception))

    def test_impossible_constraints_need_clarification_without_provider_retry(self):
        cases = []
        for ceiling in (9, 9.99, 51):
            data = proposal()
            data["constraints"]["max_altitude_m"] = ceiling
            cases.append(data)
        data = proposal([step("visit", "visit", "south_plaza"), step("home", "return_home", "home")])
        data["constraints"]["avoid_regions"] = ["south_plaza"]
        cases.append(data)
        for data in cases:
            with self.subTest(constraints=data["constraints"]), \
                    patch("uav_llm_demo.agents._post_deepseek", return_value=self.response(data)) as post:
                with self.assertRaises(SemanticConstraintError):
                    DeepSeekAgent(api_key="test-credential").interpret_task(task_catalog(scene()), "按这些约束执行")
            self.assertEqual(post.call_count, 1)

    def test_blank_instruction_and_missing_key_do_not_call_provider(self):
        with patch.dict("os.environ", {}, clear=True), patch("uav_llm_demo.agents._post_deepseek") as post:
            for text in ("", " ", None, "x" * 2001):
                with self.assertRaises(ValueError):
                    DeepSeekAgent(api_key="test-credential").interpret_task(task_catalog(scene()), text)
            with self.assertRaises(PlannerProviderError):
                DeepSeekAgent().interpret_task(task_catalog(scene()), "检查图书馆")
        post.assert_not_called()

    def test_secret_echo_and_unicode_escaped_secret_are_rejected(self):
        marker = "test-credential"
        data = proposal()
        data["summary"] = marker
        for escaped in (False, True):
            response = self.response(data)
            if escaped:
                content = response.json.return_value["choices"][0]["message"]["content"]
                response.json.return_value["choices"][0]["message"]["content"] = content.replace(marker, "".join(f"\\u{ord(c):04x}" for c in marker))
            with patch("uav_llm_demo.agents._post_deepseek", return_value=response):
                with self.assertRaises(PlannerProviderError) as caught:
                    DeepSeekAgent(api_key=marker).interpret_task(task_catalog(scene()), "检查图书馆")
            self.assertNotIn(marker, str(caught.exception))

    def test_local_geometry_and_urban_mock_share_one_implementation(self):
        self.assertIs(MockLLMAgent.plan, GeometricPlanner.plan)
        self.assertIs(MockLLMAgent._search, GeometricPlanner._search)
        context = dict(position=dict(x=0, y=3, z=0), goal=dict(x=6, y=3, z=2),
                       base=dict(x=0, y=3, z=0), grid_size=7, altitude_range=dict(min=2, max=5),
                       buildings=[], no_fly_zones=[], dynamic_objects=[])
        direct = GeometricPlanner().plan(context)
        mock = MockLLMAgent().plan(context)
        self.assertEqual(direct, mock)
        self.assertEqual(direct.waypoints[0][:2], (0, 3))
        self.assertEqual(GeometricPlanner.backend_name, "local")
        self.assertEqual(MockLLMAgent.backend_name, "mock")

    def test_return_home_while_already_landed_does_not_take_off(self):
        context = dict(position=dict(x=0, y=3, z=0), goal=dict(x=0, y=3, z=0),
                       base=dict(x=0, y=3, z=0), grid_size=7, altitude_range=dict(min=2, max=5),
                       buildings=[], no_fly_zones=[], dynamic_objects=[])
        for planner in (GeometricPlanner(), MockLLMAgent()):
            with self.subTest(backend=planner.backend_name):
                self.assertEqual(planner.plan(context).waypoints, ((0, 3, 0),))
        context["goal"] = dict(x=1, y=3, z=0)
        with self.assertRaises(ValueError):
            GeometricPlanner().plan(context)


class SemanticNavigationTests(unittest.TestCase):
    def task(self, scenario, ceiling=10):
        data = proposal([step("inspect", "inspect", "library"),
                         step("capture", "capture", "conference"),
                         step("home", "return_home", "home")])
        data["constraints"]["max_altitude_m"] = ceiling
        return parse(data, scenario)

    def test_route_validation_preserves_all_caller_owned_nested_context(self):
        scenario = load_scenario("semantic_campus")
        plan = replace(self.task(scenario, 25), constraints=dict(max_altitude_m=25, avoid_regions=["south_plaza"]))
        context = scenario.to_dict()
        context["position"] = context["base"]
        context["no_fly_zones"] = [dict(min=dict(x=0, y=0, z=0), max=dict(x=1, y=1, z=10))]
        before = deepcopy(context)

        def mutate_working_context(planner, working, feedback):
            self.assertEqual(working["altitude_range"]["max"], 5)
            self.assertEqual(len(working["no_fly_zones"]), 2)
            working["buildings"][0]["height"] = 999
            working["no_fly_zones"][0]["max"]["z"] = 1
            working["base"]["x"] = -999

        with patch("uav_llm_demo.semantic_navigation.checked_plan", side_effect=mutate_working_context):
            self.assertIs(validate_task_routes(plan, scenario, context), plan)
        self.assertEqual(context, before)

    def _service_with_constraints(self):
        from uav_llm_demo.mission_service import MissionService

        scenario = replace(load_scenario("semantic_campus"), dynamic_objects=())
        service = MissionService(scenario=scenario, agent=DeepSeekAgent(api_key="test-context-key"))
        self.addCleanup(service.close)
        plan = replace(self.task(scenario, 25), constraints=dict(max_altitude_m=25, avoid_regions=["south_plaza"]))
        service.semantic.accept(plan)
        return service

    def test_control_contexts_do_not_share_mutable_constraint_or_building_data(self):
        service = self._service_with_constraints()
        first, second = service._context(), service._context()
        expected = deepcopy(second)
        first["altitude_range"]["max"] = -999
        first["no_fly_zones"][0]["min"]["x"] = -999
        first["buildings"][0]["height"] = 999
        first["position"]["x"] = -999
        self.assertEqual(second, expected)
        self.assertEqual(service._context(), expected)

    def test_repeated_control_contexts_do_not_accumulate_exclusion_regions(self):
        service = self._service_with_constraints()
        scenario = service._environment.scenario
        original_zones = deepcopy(scenario.no_fly_zones)
        for _ in range(5):
            context = service._context()
            self.assertEqual(len(context["no_fly_zones"]), len(original_zones) + 1)
            self.assertEqual(context["altitude_range"]["max"], 5)
        self.assertEqual(scenario.no_fly_zones, original_zones)

    def test_ten_metre_task_uses_same_goals_for_routes_and_arrival_evidence(self):
        scenario = load_scenario("semantic_campus")
        plan = self.task(scenario)
        context = scenario.to_dict()
        context["position"] = context["base"]
        self.assertIs(validate_task_routes(plan, scenario, context), plan)
        self.assertEqual(context["altitude_range"]["max"], 10)
        mission = SemanticMission(scenario)
        mission.accept(plan)
        apply_task_constraints(context, scenario, plan.constraints)
        self.assertEqual(context["altitude_range"], {"min": 2, "max": 2})
        library, conference = (position_of(scenario, target) for target in ("library", "conference"))
        expected_goals = ((*library[:2], 2), (*conference[:2], 2), scenario.base)
        default_goals = (library, conference, None)
        for index, (goal, default) in enumerate(zip(expected_goals, default_goals)):
            self.assertEqual(mission.current_goal, goal)
            context["goal"] = dict(zip("xyz", mission.current_goal))
            route = checked_plan(GeometricPlanner(), context, "")
            self.assertEqual(SafetyChecker(context).validate_path(context["position"], route.waypoints), [])
            self.assertLessEqual(max(point[2] for point in route.waypoints), 2)
            if default is not None:
                self.assertFalse(mission.observe(default, 0, 3, index * 10 + 3))
                self.assertEqual(len(mission.evidence), index)
            self.assertTrue(mission.observe(route.waypoints[-1], 0, 3, index * 10 + 6))
            self.assertEqual(mission.evidence[-1]["position"], dict(zip("xyz", goal)))
            context["position"] = context["goal"]
        self.assertEqual(mission.status, "completed")
        self.assertTrue(mission.evidence[-1]["result"]["landed"])

    def test_without_ceiling_observation_heights_stay_unchanged(self):
        scenario = load_scenario("semantic_campus")
        mission = SemanticMission(scenario)
        mission.accept(self.task(scenario, None))
        self.assertEqual(mission.current_goal, position_of(scenario, "library"))
        self.assertTrue(mission.observe(mission.current_goal, 0, 2, 2))
        self.assertEqual(mission.current_goal, position_of(scenario, "conference"))

    def test_lowered_goal_inside_building_still_rejected_by_geometry(self):
        scenario = load_scenario("semantic_campus")
        points = deepcopy(scenario.points_of_interest)
        building = next(item for item in scenario.campus_layout["buildings"] if item["id"] == "L5")
        left, right, north, south = building["footprint"]
        observation = dict(x=(left + right) / 2, y=(north + south) / 2, z=4)
        self.assertGreater(building["height"], 2)
        self.assertLess(building["height"] + .6 / 5, observation["z"])
        next(point for point in points if point["id"] == "library")["position"] = observation
        scenario = replace(scenario, points_of_interest=points)
        context = scenario.to_dict()
        context["position"] = context["base"]
        validate_task_routes(self.task(scenario, None), scenario, context)
        with self.assertRaises(SemanticConstraintError) as caught:
            validate_task_routes(self.task(scenario), scenario, context)
        self.assertEqual(caught.exception.code, "route_unreachable")
        self.assertEqual(caught.exception.user_message, "当前高度上限或避让区域使任务地点不可达，请调整约束。")

    def test_unreachable_route_preserves_no_fly_zones(self):
        scenario = load_scenario("semantic_campus")
        barrier_y = (scenario.base[1] + position_of(scenario, "library")[1]) / 2
        barrier = dict(min=dict(x=0, y=barrier_y, z=0),
                       max=dict(x=scenario.grid_size - 1, y=barrier_y + 1, z=scenario.max_altitude))
        scenario = replace(scenario, no_fly_zones=scenario.no_fly_zones + (barrier,))
        context = scenario.to_dict()
        context["position"] = context["base"]
        with self.assertRaises(SemanticConstraintError) as caught:
            validate_task_routes(self.task(scenario), scenario, context)
        self.assertEqual(caught.exception.code, "route_unreachable")

    def test_route_errors_do_not_expose_internal_exception_text(self):
        scenario = load_scenario("semantic_campus")
        context = scenario.to_dict()
        context["position"] = context["base"]
        with patch("uav_llm_demo.semantic_navigation.checked_plan", side_effect=ValueError("private-internal-detail")):
            with self.assertRaises(SemanticConstraintError) as caught:
                validate_task_routes(self.task(scenario), scenario, context)
        self.assertNotIn("private-internal-detail", str(caught.exception))
        self.assertNotIn("private-internal-detail", caught.exception.user_message)
        self.assertTrue(caught.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
