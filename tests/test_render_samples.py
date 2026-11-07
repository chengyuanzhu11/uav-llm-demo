"""Display samples are measured history, isolated from the control state."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from tests.scenario_fixtures import empty_airspace
from uav_llm_demo.environment import point_to_dict
from uav_llm_demo.html_runtime import HtmlRuntime
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import VehicleState


class GazeboStub(HtmlRuntime):
    backend_name = "gazebo"

    def configure_scene(self, scenario):
        self.scenario = scenario

    def status(self):
        return dict(backend="gazebo", connected=True, supports_wind=True, source="test",
                    landed=self.vehicle.position[2] <= .036, armed=False)


class RenderSamplesTest(unittest.TestCase):
    def setUp(self):
        self.scenario = replace(empty_airspace(), grid_size=8,
                                base=(1.0, 6.0, 0.0), target=(6.0, 1.0, 3.0))
        self.service = MissionService(scenario=self.scenario, traffic_seed=17,
                                      runtime_factory=lambda backend, config: GazeboStub())
        self.addCleanup(self.service.close)

    def set_vehicle(self, position=(2.0, 2.0, 2.0), velocity=(1.0, 0.0, 0.0), yaw=0.7):
        vehicle = VehicleState(position=position, velocity=velocity, yaw=yaw)
        self.service.runtime.vehicle = vehicle
        self.service._commit_vehicle(vehicle, 0)
        return vehicle

    def test_initial_sample_matches_measured_vehicle_and_traffic(self):
        state = self.service.state()
        self.assertEqual(len(state["motion_samples"]), 1)
        sample = state["motion_samples"][0]
        self.assertEqual(sample["t"], 0)
        self.assertEqual(sample["path_index"], 0)
        self.assertEqual(sample["position"], list(self.scenario.base))
        self.assertEqual(sample["velocity"], [0.0, 0.0, 0.0])
        self.assertEqual(sample["yaw"], 0)
        self.assertTrue(sample["landed"])
        bird = self.service._environment.dynamic_objects[0]
        self.assertEqual(sample["traffic"], {bird.object_id: [*bird.position, *bird.velocity]})
        self.assertEqual(state["path_start_index"], 0)

    def test_same_time_replaces_last_sample_without_changing_prior_snapshot(self):
        self.service.advance(0.05)
        old = self.service.state()
        vehicle = self.set_vehicle()
        for _ in range(5):
            self.service._publish_state()
        state = self.service.state()
        self.assertEqual(len(state["motion_samples"]), 2)
        self.assertEqual([s["t"] for s in state["motion_samples"]], [0, 0.05])
        current = state["motion_samples"][-1]
        self.assertEqual(current["position"], list(vehicle.position))
        self.assertEqual(current["velocity"], list(vehicle.velocity))
        self.assertEqual(current["yaw"], vehicle.yaw)
        self.assertFalse(current["landed"])
        self.assertEqual(old["motion_samples"][-1]["position"], list(self.scenario.base))

    def test_paused_vehicle_deceleration_and_traffic_are_still_sampled(self):
        self.set_vehicle()
        paused = self.service.stop_runtime()
        self.assertFalse(paused["flight"]["running"])
        before = paused["motion_samples"][-1]
        self.service.advance(0.05)
        after = self.service.state()["motion_samples"][-1]
        self.assertEqual(after["t"], 0.05)
        self.assertGreater(after["position"][0], before["position"][0])
        self.assertGreater(after["velocity"][0], 0)
        self.assertLess(after["velocity"][0], before["velocity"][0])
        self.assertNotEqual(after["traffic"], before["traffic"])
        for _ in range(30):
            self.service.advance(0.05)
        resting = self.service.state()["motion_samples"][-1]
        self.assertTrue(all(abs(v) < 1e-10 for v in resting["velocity"]))
        self.service.advance(0.05)
        still = self.service.state()["motion_samples"][-1]
        self.assertEqual(still["position"], resting["position"])
        self.assertNotEqual(still["traffic"], resting["traffic"])

    def test_history_keeps_last_32_control_ticks_even_when_stationary(self):
        for _ in range(50):
            self.service.advance(0.05)
        samples = self.service.state()["motion_samples"]
        self.assertEqual(len(samples), 32)
        self.assertAlmostEqual(samples[0]["t"], 0.95)
        self.assertAlmostEqual(samples[-1]["t"], 2.5)
        self.assertTrue(all(a["t"] < b["t"] for a, b in zip(samples, samples[1:])))
        self.assertTrue(all(s["position"] == list(self.scenario.base) for s in samples))

    def test_state_mutation_cannot_change_buffer_control_or_another_snapshot(self):
        self.service.advance(0.05)
        original = self.service.state()
        changed = self.service.state()
        changed["motion_samples"][0]["position"][0] = -999
        changed["motion_samples"][-1]["velocity"][0] = -999
        changed["motion_samples"][-1]["traffic"]["bird-1"][0] = -999
        changed["motion_samples"].clear()
        changed["scenario"]["base"]["x"] = -999
        self.service._publish_state()
        self.assertEqual(self.service.state()["scenario"], original["scenario"])
        self.assertEqual(self.service.state()["motion_samples"], original["motion_samples"])
        self.assertEqual(self.service._vehicle.position, self.scenario.base)
        self.assertEqual(self.service._vehicle.velocity, (0, 0, 0))
        self.assertNotEqual(self.service._environment.dynamic_objects[0].position[0], -999)

    def test_scene_metadata_refreshes_and_distance_survives_display_window(self):
        original = self.service.state()
        x, y, z = self.scenario.base
        self.set_vehicle((x+1, y, z))
        self.set_vehicle((x+1, y, z+1))
        # Trimming the renderer's history must not erase measured flight cost.
        self.service._environment.path = [self.service._vehicle.position]
        self.service._environment.continuous_path = [point_to_dict(self.service._vehicle.position)]
        self.service._environment.scenario = replace(self.scenario, name="Updated scene")
        self.service._publish_state()
        measured = self.service.state()
        self.assertEqual(measured["metrics"]["path_length"], 8)
        self.assertEqual(measured["scenario"]["name"], "Updated scene")
        self.assertEqual(original["scenario"]["name"], self.scenario.name)
        with patch.object(self.service, "_random_scenario", return_value=self.scenario):
            reset = self.service.reset()
        self.assertEqual(reset["metrics"]["path_length"], 0)

    def test_path_window_tracks_absolute_origin_and_only_moves_when_truncated(self):
        env = self.service._environment
        env.path = [self.scenario.base] * 9999
        env.continuous_path = [point_to_dict(self.scenario.base) for _ in range(9999)]
        self.set_vehicle(position=(2.0, 2.0, 2.0))
        self.service._publish_state()
        state = self.service.state()
        self.assertEqual(len(state["continuous_path"]), 10000)
        self.assertEqual(state["path_start_index"], 0)
        self.assertEqual(state["motion_samples"][-1]["path_index"], 9999)
        self.service._time += 0.05
        self.set_vehicle(position=(2.1, 2.0, 2.0))
        self.service._publish_state()
        self.service._time += 0.05
        self.set_vehicle(position=(2.2, 2.0, 2.0))
        self.service._publish_state()
        truncated = self.service.state()
        self.assertEqual(len(truncated["continuous_path"]), 10000)
        self.assertEqual(truncated["path_start_index"], 2)
        self.assertEqual([s["path_index"] for s in truncated["motion_samples"]],
                         [9999, 10000, 10001])
        self.assertEqual(truncated["continuous_path"][-2:], [
            dict(x=2.1, y=2.0, z=2.0), dict(x=2.2, y=2.0, z=2.0)])
        self.service._commit_vehicle(self.service._vehicle, 0)
        self.service._publish_state()
        self.assertEqual(self.service.state()["path_start_index"], 2)

    def test_reset_clears_sample_history_and_path_origin(self):
        for _ in range(3):
            self.service.advance(0.05)
        self.service._path_start_index = 100
        with patch.object(self.service, "_random_scenario", return_value=self.scenario):
            state = self.service.reset()
        self.assertEqual(state["path_start_index"], 0)
        self.assertEqual(state["continuous_path"], [point_to_dict(self.scenario.base)])
        self.assertEqual(len(state["motion_samples"]), 1)
        self.assertEqual(state["motion_samples"][0]["t"], 0)
        self.assertEqual(state["motion_samples"][0]["path_index"], 0)

    def test_scenario_switch_does_not_carry_samples_across_coordinates(self):
        self.service.advance(0.05)
        replacement = replace(self.scenario, scenario_id="test_other", base=(6.0, 6.0, 0.0))
        with patch("uav_llm_demo.mission_service.load_scenario", return_value=replacement), \
                patch.object(self.service, "_random_scenario", return_value=replacement):
            state = self.service.switch_scenario("test_other")
        self.assertEqual(state["scenario_id"], "test_other")
        self.assertEqual(len(state["motion_samples"]), 1)
        self.assertEqual(state["motion_samples"][0]["position"], list(replacement.base))
        self.assertEqual(state["motion_samples"][0]["t"], 0)
        self.assertEqual(state["path_start_index"], 0)

    def test_external_telemetry_is_sampled_and_runtime_switch_starts_fresh(self):
        self.service.advance(0.05)
        gazebo = self.service.switch_runtime("gazebo")
        self.assertEqual(gazebo["motion_samples"][0]["t"], 0)
        self.assertEqual(gazebo["motion_samples"][0]["position"], list(self.service.runtime.read_state().position))
        self.assertEqual(gazebo["dynamic_objects"], [])
        self.assertEqual(gazebo["motion_samples"][0]["traffic"], {})
        self.assertEqual(gazebo["observation"]["dynamic_objects"], [])
        self.service.advance(0.05)
        self.assertGreater(self.service.state()["motion_samples"][-1]["t"], 0)
        html = self.service.switch_runtime("html")
        self.assertEqual(len(html["motion_samples"]), 1)
        self.assertEqual(html["motion_samples"][0]["t"], 0)
        self.assertEqual(html["path_start_index"], 0)
        self.assertTrue(html["dynamic_objects"])
        self.assertTrue(html["motion_samples"][0]["traffic"])


if __name__ == "__main__":
    unittest.main()
