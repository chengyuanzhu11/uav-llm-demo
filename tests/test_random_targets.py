import random
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from uav_llm_demo import create_app
from uav_llm_demo.agents import DeepSeekAgent, MockLLMAgent, checked_plan
from uav_llm_demo.environment import point_to_dict
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import SCALE, norm
from uav_llm_demo.scenarios import list_scenarios, load_scenario
from uav_llm_demo.safety import SafetyChecker
from tests.test_mission_service import RecordingRuntime, pump


class RandomTargetTest(unittest.TestCase):
    def service(self, **kwargs):
        service = MissionService(**kwargs)
        self.addCleanup(service.close)
        return service

    def test_random_candidates_are_safe_reachable_and_keep_altitude(self):
        service = self.service(target_seed=0)
        for item in list_scenarios():
            template = load_scenario(item["id"])
            targets = set()
            for seed in range(12):
                service._target_rng = random.Random(seed)
                scenario = service._random_scenario(template)
                target = scenario.target
                targets.add(target)
                horizontal = norm(tuple((target[i] - scenario.base[i]) * SCALE[i] for i in range(2)))
                self.assertGreaterEqual(horizontal, 75 if template.urban_parcels else 30)
                if template.urban_parcels:
                    self.assertTrue(target[0] >= 20 or target[1] >= 20)
                    middle = (template.grid_size - 1) / 2
                    self.assertNotEqual(tuple(value < middle for value in target[:2]),
                                        tuple(value < middle for value in template.base[:2]))
                self.assertEqual(target[2], template.target[2])
                self.assertTrue(all(1 <= coordinate <= scenario.grid_size - 2 for coordinate in target[:2]))
                context = scenario.to_dict()
                context.update(position=point_to_dict(scenario.base), goal=point_to_dict(target))
                self.assertFalse(SafetyChecker(context).validate_waypoint(target))
                route = checked_plan(MockLLMAgent(), context, "")
                self.assertFalse(SafetyChecker(context).validate_path(scenario.base, route.waypoints))
            self.assertGreater(len(targets), 1)

    def test_same_seed_reproduces_target_sequence_without_repeating_on_reset(self):
        first = self.service(target_seed=7)
        second = self.service(target_seed=7)
        previous = first.state()["target"]
        self.assertEqual(previous, second.state()["target"])
        for _ in range(3):
            left, right = first.reset(), second.reset()
            self.assertEqual(left["target"], right["target"])
            self.assertNotEqual(left["target"], previous)
            self.assertEqual(left["scenario"]["target"], left["target"])
            self.assertEqual(left["observation"]["goal"], left["target"])
            pad = next(item for item in left["scene_layout"] if item["name"] == "target_pad")
            self.assertEqual(pad["position"]["x"], left["target"]["x"])
            self.assertEqual(pad["position"]["y"], left["target"]["y"])
            self.assertEqual(left["flight"]["waypoints"], [])
            self.assertEqual(left["flight"]["planner_calls"], 0)
            previous = left["target"]

    def test_deepseek_not_called_for_target_selection_and_reset_keeps_key(self):
        with patch("uav_llm_demo.agents._post_deepseek") as request:
            service = self.service(agent=DeepSeekAgent(api_key="dummy-test-credential"), target_seed=4)
            before = service.state()
            reset = service.reset()
        request.assert_not_called()
        self.assertNotEqual(before["target"], reset["target"])
        self.assertEqual(reset["agent_backend"], "deepseek")
        self.assertTrue(reset["deepseek_key_configured"])
        self.assertNotIn("dummy-test-credential", str(reset))
        self.assertEqual(reset["flight"]["planner_calls"], 0)

    def test_planning_tracks_selected_goal_and_does_not_randomize_it(self):
        planner = MockLLMAgent()
        planner.plan = Mock(wraps=planner.plan)
        service = self.service(agent=planner, target_seed=9)
        target = service.state()["target"]
        with patch.object(service.avoidance, "predictions", return_value=[]):
            service.start(background=False)
            state = pump(service, lambda item: item["flight"]["plan_id"] == 1)
            for _ in range(20):
                service.advance(.05)
        self.assertEqual(service.state()["target"], target)
        self.assertEqual(planner.plan.call_count, 1)
        self.assertEqual(planner.plan.call_args.args[0]["goal"], target)
        self.assertEqual(state["flight"]["waypoints"][-1], target)
        service.stop_runtime()
        service.configure_agent({"backend": "mock"})
        self.assertEqual(service.state()["target"], target)

    def test_runtime_switch_retains_target_and_airsim_reset_updates_scene(self):
        runtime = RecordingRuntime("airsim")
        service = self.service(target_seed=2, runtime_factory=lambda backend, config: runtime)
        target = service.state()["target"]
        switched = service.switch_runtime("airsim")
        self.assertEqual(switched["target"], target)
        self.assertEqual(point_to_dict(runtime.scene.target), target)
        reset = service.reset()
        self.assertNotEqual(reset["target"], target)
        self.assertEqual(point_to_dict(runtime.scene.target), reset["target"])
        self.assertEqual(service.switch_runtime("html")["target"], reset["target"])

    def test_no_reachable_candidate_is_rejected_instead_of_using_fixed_goal(self):
        service = self.service(target_seed=0)
        template = load_scenario("urban_crossing")
        blocked = replace(template, grid_size=12, base=(1, 1, 2), min_altitude=2, max_altitude=2,
                          target=(10, 10, 2), preferred_altitude=None, urban_parcels=(),
                          buildings=tuple((4, y, 4) for y in range(12)), dynamic_objects=(),
                          no_fly_zones=(dict(min=dict(x=0, y=3, z=0), max=dict(x=3, y=11, z=10)),))
        self.assertFalse(SafetyChecker(blocked.to_dict()).validate_waypoint(blocked.target))
        with self.assertRaisesRegex(ValueError, "No safe reachable random target"):
            service._random_scenario(blocked)

    def test_altitude_does_not_make_a_nearby_target_eligible(self):
        service = self.service(target_seed=0)
        template = replace(load_scenario("open_field"), grid_size=8, target=(6, 6, 10))
        with self.assertRaisesRegex(ValueError, "No safe reachable random target"):
            service._random_scenario(template)

    def test_start_block_is_excluded_even_when_its_far_edge_exceeds_75_metres(self):
        service = self.service(target_seed=0)
        scenario = replace(load_scenario("urban_crossing"), base=(1, 1, 2))
        context = scenario.to_dict()
        context["position"] = point_to_dict(scenario.base)
        point = (19, 19, 2)
        self.assertGreater(norm(tuple((point[i] - scenario.base[i]) * SCALE[i] for i in range(2))), 75)
        self.assertFalse(SafetyChecker(context).validate_waypoint(point))
        # If only this safe but same-block point is left, do not fall back to it.
        original = SafetyChecker.validate_waypoint
        def only_same_block(checker, target):
            return original(checker, target) if tuple(target) == point else ["blocked for test"]
        with patch.object(SafetyChecker, "validate_waypoint", only_same_block):
            with self.assertRaisesRegex(ValueError, "No safe reachable random target"):
                service._random_scenario(scenario)

    def test_reset_and_scenario_api_publish_the_new_target(self):
        service = self.service(target_seed=5)
        client = create_app(service=service).test_client()
        before = client.get("/api/state").get_json()
        reset = client.post("/api/reset")
        self.assertEqual(reset.status_code, 200)
        self.assertNotEqual(reset.get_json()["target"], before["target"])
        switched = client.post("/api/scenario", json={"scenario_id": "open_field"})
        self.assertEqual(switched.status_code, 200)
        payload = switched.get_json()
        self.assertEqual(payload["target"]["z"], 5)
        self.assertEqual(payload["target"], payload["scenario"]["target"])
        self.assertEqual(payload["target"], payload["observation"]["goal"])

    def test_unavailable_random_target_returns_error_without_resetting_mission(self):
        service = self.service(target_seed=5)
        client = create_app(service=service).test_client()
        before = client.get("/api/state").get_json()
        with patch.object(service, "_random_scenario", side_effect=ValueError("No safe reachable random target")):
            result = client.post("/api/reset")
        self.assertEqual(result.status_code, 400)
        self.assertIn("No safe reachable", result.get_json()["error"])
        after = service.state()
        self.assertEqual(after["target"], before["target"])
        self.assertEqual(after["uav"], before["uav"])
        self.assertEqual(after["flight"]["plan_id"], before["flight"]["plan_id"])


if __name__ == "__main__":
    unittest.main()
