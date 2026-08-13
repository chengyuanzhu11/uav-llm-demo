import unittest

from uav_llm_demo.agents import MockLLMAgent, decision_from_json_text
from uav_llm_demo.environment import UAVEnvironment
from uav_llm_demo.evaluation import metrics_for_events
from uav_llm_demo.benchmark import run_scenario
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.scenarios import list_scenarios, load_scenario
from uav_llm_demo import create_app


class PlanBContractTest(unittest.TestCase):
    def test_scenarios_are_named_and_data_driven(self):
        scenarios = list_scenarios()

        self.assertGreaterEqual(len(scenarios), 3)
        self.assertIn("urban_crossing", {item["id"] for item in scenarios})
        scenario = load_scenario("urban_crossing")
        self.assertEqual(scenario.base, (1, 18, 2))
        self.assertGreater(len(scenario.buildings), 0)
        self.assertGreater(len(scenario.dynamic_objects), 0)

    def test_environment_can_start_from_named_scenario(self):
        scenario = load_scenario("open_field")
        environment = UAVEnvironment(scenario)

        self.assertEqual(environment.scenario_id, "open_field")
        self.assertEqual(environment.position, scenario.base)
        self.assertEqual(environment.target, scenario.target)

    def test_mock_agent_emits_a_skill_and_waypoint(self):
        observation = UAVEnvironment().get_observation()

        decision = MockLLMAgent().decide(observation)

        self.assertIn(decision.skill, {"navigate_to", "return_home", "hover"})
        self.assertIsNotNone(decision.waypoint)
        self.assertIn("skill", decision.to_dict())
        self.assertIn("waypoint", decision.to_dict())

    def test_structured_response_accepts_skill_and_waypoint(self):
        decision = decision_from_json_text(
            '{"skill":"navigate_to","action":"move_east",'
            '"waypoint":{"x":2,"y":18,"z":2},'
            '"reason":"Advance toward the target.","risk":"low",'
            '"expected_result":"Reach the next waypoint."}'
        )

        self.assertEqual(decision.skill, "navigate_to")
        self.assertEqual(decision.waypoint, {"x": 2.0, "y": 18.0, "z": 2.0})

    def test_service_state_exposes_metrics_and_safety_trace(self):
        service = MissionService(agent=MockLLMAgent())

        state = service.step()

        self.assertIn("metrics", state)
        self.assertIn("evaluation", state)
        self.assertIn("safety_intervened", state["history"][0]["decision"])

    def test_service_can_switch_named_scenario(self):
        service = MissionService(agent=MockLLMAgent())

        state = service.switch_scenario("open_field")

        self.assertEqual(state["scenario_id"], "open_field")
        self.assertEqual(state["uav"], state["base"])
        self.assertEqual(state["step_count"], 0)

    def test_metrics_count_safety_interventions(self):
        metrics = metrics_for_events(
            [
                {"decision": {"safety_intervened": True}, "feedback": "blocked"},
                {"decision": {"safety_intervened": False}, "feedback": "ok"},
            ],
            final_status="completed",
            path_length=4.5,
            battery_used=3,
        )

        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["steps"], 2)
        self.assertEqual(metrics["safety_interventions"], 1)
        self.assertEqual(metrics["path_length"], 4.5)

    def test_scenario_api_lists_and_switches_scenarios(self):
        client = create_app().test_client()

        listed = client.get("/api/scenarios")
        switched = client.post("/api/scenario", json={"scenario_id": "open_field"})

        self.assertEqual(listed.status_code, 200)
        self.assertIn("open_field", {item["id"] for item in listed.get_json()["scenarios"]})
        self.assertEqual(switched.status_code, 200)
        self.assertEqual(switched.get_json()["scenario_id"], "open_field")

    def test_invalid_scenario_api_returns_bad_request(self):
        response = create_app().test_client().post(
            "/api/scenario",
            json={"scenario_id": "does-not-exist"},
        )

        self.assertEqual(response.status_code, 400)

    def test_benchmark_runs_a_named_scenario(self):
        result = run_scenario("open_field", max_steps=100)

        self.assertEqual(result["scenario_id"], "open_field")
        self.assertTrue(result["metrics"]["success"])
        self.assertGreater(result["metrics"]["steps"], 0)


if __name__ == "__main__":
    unittest.main()
