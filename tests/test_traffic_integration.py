"""Observation boundaries and metric prediction are enforced in the live service."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.scenario_fixtures import empty_airspace
from uav_llm_demo.avoidance import LocalAvoidance
from uav_llm_demo.environment import DynamicFlyingObject
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import VehicleState
from uav_llm_demo.safety import SafetyChecker
from uav_llm_demo.traffic_observation import TrafficObservationConfig


class FrozenTraffic:
    def sync(self, environment):
        return None


class TrafficIntegrationTest(unittest.TestCase):
    def service(self, **config):
        service = MissionService(scenario=empty_airspace(), traffic_seed=19,
                                 observation_config=TrafficObservationConfig(**config))
        self.addCleanup(service.close)
        service._traffic = FrozenTraffic()
        service.runtime.vehicle = VehicleState((10, 10, 2))
        service._commit_vehicle(service.runtime.vehicle, 0)
        service._environment.dynamic_objects = [
            DynamicFlyingObject("visible", "bird", (12, 10, 2), (0, 0, 0)),
            DynamicFlyingObject("far", "bird", (18, 10, 2), (0, 0, 0)),
        ]
        return service

    def test_renderer_sees_world_but_planner_and_avoider_only_receive_delayed_sensor(self):
        service = self.service(range_m=10, occlusion=False)
        bird = service._environment.dynamic_objects[0]
        bird.motion_prediction = (999, 999, 999)
        service.advance(.05)
        self.assertEqual(service._context()["dynamic_objects"], [])
        self.assertEqual(service.avoidance.predictions(service._vehicle, service._time), [])
        self.assertEqual(len(service.state()["dynamic_objects"]), 2)
        service.advance(.1)
        sensed = service._context()["dynamic_objects"]
        self.assertEqual([item["id"] for item in sensed], ["visible"])
        self.assertAlmostEqual(sensed[0]["observation_age_seconds"], .1)
        self.assertEqual(sensed[0]["predicted_position"], dict(x=12, y=10, z=2))
        self.assertEqual(sensed[0]["distance_unit"], "m")
        self.assertAlmostEqual(sensed[0]["distance_m"], 6)
        track = service.avoidance.predictions(service._vehicle, service._time)[0]
        self.assertAlmostEqual(track.radius_m, sensed[0]["separation_distance_m"])
        self.assertEqual(service.state()["observation"]["dynamic_objects"], sensed)

    def test_occlusion_loses_track_then_expires_without_reading_truth(self):
        service = self.service(occlusion=True, latency_seconds=0)
        service.advance(.1)
        self.assertIn("visible", service.avoidance.predictor.object_ids)
        service._environment.buildings = {(11, 10): 4}
        service._environment.dynamic_objects[0].position = (13, 10, 2)
        for _ in range(14):
            service.advance(.1)
        tracked = {item["id"]: item for item in service._context()["dynamic_objects"]}
        self.assertEqual(tracked["visible"]["position"]["x"], 12)
        for _ in range(3):
            service.advance(.1)
        self.assertNotIn("visible", service.avoidance.predictor.object_ids)
        self.assertNotIn("visible", {item["id"] for item in service._context()["dynamic_objects"]})

    def test_reset_discards_all_previous_observation_history(self):
        service = self.service(occlusion=False, latency_seconds=0)
        service.advance(.1)
        self.assertTrue(service.avoidance.predictor.object_ids)
        service.reset()
        self.assertEqual(service.avoidance.predictor.object_ids, ())
        self.assertEqual(service._traffic_observer.latest(0), [])

    def test_horizontal_uncertainty_does_not_create_false_vertical_extent(self):
        lower_bound = LocalAvoidance._uncertain_swept_distance((0, 0, 8), (1, 0, 7), 15, .2)
        self.assertAlmostEqual(lower_bound, 6.8)
        self.assertEqual(LocalAvoidance._uncertain_swept_distance((-5, 0, 2), (5, 0, -2), .1, .1), 0)

    def test_opt_in_route_snapshot_check_uses_same_metric_separation(self):
        checker = SafetyChecker(dict(grid_size=20, altitude_range=dict(min=2, max=10), buildings=[],
            goal=dict(x=14, y=10, z=2), dynamic_objects=[dict(radius_m=2.1,
                position=dict(x=12, y=11, z=2))]))
        # 3 m center distance clears the 2.5 m contact boundary but violates
        # the shared 3.1 m separation boundary.
        self.assertTrue(checker.validate_path((10, 10, 2), [(14, 10, 2)], check_dynamic=True))

    def test_run_journal_records_start_and_final_minimum_of_each_incident_kind(self):
        service = self.service(occlusion=False, latency_seconds=0)
        records = []
        service.trace.writer = SimpleNamespace(emit=records.append, dropped=0, errors=0)
        service._safety_armed = True
        bird = service._environment.dynamic_objects[0]
        bird.position = (10.2, 10, 2)
        with patch.object(service.avoidance, "choose", return_value=None):
            service.advance(.1)
            bird.position = (18, 10, 2)
            service.advance(.1)
        for kind in ("dynamic_separation_violation", "dynamic_envelope_contact"):
            events = [record["incident"] for record in records if record["event"] == kind]
            self.assertEqual(len(events), 2)
            self.assertTrue(events[0]["active"])
            self.assertFalse(events[1]["active"])
            self.assertGreater(events[1]["duration_seconds"], 0)
            self.assertAlmostEqual(events[1]["minimum_center_distance_m"], .6)


if __name__ == "__main__":
    unittest.main()
