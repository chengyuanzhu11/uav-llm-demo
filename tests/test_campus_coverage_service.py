"""New campus places execute with the real route planner and HTML controller."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from tests.test_semantic_updates import OfflineSemanticAgent, program, skill, home
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import distance, norm
from uav_llm_demo.scenarios import load_scenario


class CampusCoverageServiceTests(unittest.TestCase):
    def test_new_building_and_surface_targets_execute_then_land_under_ten_metre_ceiling(self):
        targets = ["c4", "r3", "d14", "a5", "north_basketball", "l6", "central_garden"]
        reply = program([skill(f"visit_{target}", "inspect", target) for target in targets] + [home()],
                        ceiling=10)
        agent = OfflineSemanticAgent([reply])
        scenario = replace(load_scenario("semantic_campus"), dynamic_objects=())
        service = MissionService(scenario=scenario, agent=agent,
                                 planner_latency_seconds=0, traffic_seed=17)
        self.addCleanup(service.close)
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=AssertionError("Remote API forbidden")):
            service.submit_task("限高10米，依次巡检C4、R3、D14、A5、北区篮球场、L6和中心花园，然后返航。",
                                background=False)
            maximum_altitude = 0.0
            for _ in range(12000):
                if service._future is not None:
                    service._future.result(timeout=15)
                service.advance(.1)
                maximum_altitude = max(maximum_altitude, service._vehicle.position[2] * 5)
                self.assertNotEqual(service._environment.status, "error", service._error)
                if service._environment.status == "completed":
                    break
        final = service.state()
        self.assertEqual(final["status"], "completed", service.semantic.message)
        self.assertEqual([item["target_id"] for item in service.semantic.evidence], targets + ["home"])
        self.assertTrue(final["milestones"]["safe_success"])
        self.assertLessEqual(maximum_altitude, 10.000001)
        self.assertLessEqual(distance(service._vehicle.position, service._environment.base), .18)
        self.assertLessEqual(norm(service._vehicle.velocity), .02)
        self.assertEqual(len(agent.inputs), 1)


if __name__ == "__main__":
    unittest.main()
