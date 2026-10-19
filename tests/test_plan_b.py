from tests.scenario_fixtures import empty_airspace
import unittest

from uav_llm_demo import create_app
from uav_llm_demo.agents import MockLLMAgent
from uav_llm_demo.benchmark import run_scenario
from uav_llm_demo.environment import UAVEnvironment
from uav_llm_demo.evaluation import metrics_for_events
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.scenarios import list_scenarios, load_scenario


class PlanContractTest(unittest.TestCase):
    def test_scenarios_are_named_and_data_driven(self):
        scenarios = list_scenarios()

        self.assertEqual({item["id"] for item in scenarios}, {"urban_crossing", "semantic_campus"})
        self.assertIn("urban_crossing", {item["id"] for item in scenarios})
        scenario = load_scenario("urban_crossing")
        self.assertEqual(scenario.base, (1, 18, 0))
        self.assertGreater(len(scenario.buildings), 0)
        self.assertGreater(len(scenario.dynamic_objects), 0)

    def test_urban_buildings_leave_orthogonal_streets_clear(self):
        scenario = load_scenario("urban_crossing")
        road_cells = {int(center) + offset for center in scenario.urban_street_centers
                      for offset in (-1, 0, 1, 2)}

        self.assertTrue(all(x not in road_cells and y not in road_cells
                            for x, y, _ in scenario.buildings))

    def test_environment_can_start_from_named_scenario(self):
        scenario = load_scenario("semantic_campus")
        environment = UAVEnvironment(scenario)

        self.assertEqual(environment.scenario_id, "semantic_campus")
        self.assertEqual(environment.position, scenario.base)
        self.assertEqual(environment.target, scenario.target)

    def test_mock_planner_emits_waypoint_sequence_only(self):
        scenario = load_scenario("urban_crossing")
        context = scenario.to_dict()
        context.update(
            position=context["base"],
            goal=context["target"],
            task="Navigate to target",
        )

        plan = MockLLMAgent().plan(context)
        payload = plan.to_dict()

        self.assertGreaterEqual(len(plan.waypoints), 1)
        self.assertEqual(set(payload), {"waypoints", "reason", "risk", "expected_result"})
        self.assertNotIn("action", payload)
        self.assertEqual(plan.waypoints[-1], scenario.target)

    def test_service_state_exposes_continuous_metrics_and_safety_trace(self):
        service = MissionService(agent=MockLLMAgent())
        self.addCleanup(service.close)

        state = service.state()

        self.assertIn("metrics", state)
        self.assertIn("evaluation", state)
        self.assertIn("max_acceleration_mps2", state["flight"])
        self.assertEqual(state["metrics"]["planner_calls"], 0)

    def test_service_can_switch_named_scenario(self):
        service = MissionService(agent=MockLLMAgent())
        self.addCleanup(service.close)

        state = service.switch_scenario("semantic_campus")

        self.assertEqual(state["scenario_id"], "semantic_campus")
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
        app = create_app()
        self.addCleanup(app.extensions["mission_service"].close)
        client = app.test_client()

        listed = client.get("/api/scenarios")
        switched = client.post("/api/scenario", json={"scenario_id": "semantic_campus"})

        self.assertEqual(listed.status_code, 200)
        self.assertIn("semantic_campus", {item["id"] for item in listed.get_json()["scenarios"]})
        self.assertEqual(switched.status_code, 200)
        self.assertEqual(switched.get_json()["scenario_id"], "semantic_campus")

    def test_invalid_scenario_api_returns_bad_request(self):
        app = create_app()
        self.addCleanup(app.extensions["mission_service"].close)
        response = app.test_client().post(
            "/api/scenario",
            json={"scenario_id": "does-not-exist"},
        )

        self.assertEqual(response.status_code, 400)

    def test_benchmark_runs_a_named_scenario(self):
        result = run_scenario("test_airspace", scenario=empty_airspace(), max_seconds=120)

        self.assertEqual(result["scenario_id"], "test_airspace")
        self.assertTrue(result["metrics"]["success"])
        self.assertGreater(result["metrics"]["control_ticks"], 0)
        self.assertGreaterEqual(result["metrics"]["planner_calls"], 2)


if __name__ == "__main__":
    unittest.main()
