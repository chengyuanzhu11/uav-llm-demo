"""Every tracking sample, including entry, must honor explicit task heading."""
from dataclasses import replace
from math import atan2, degrees, pi, radians
import unittest
from unittest.mock import patch

from tests.test_semantic_motion_service import advance, motion
from tests.test_semantic_updates import OfflineSemanticAgent, compact_campus, home, program, pump
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.spatial_motion import spatial_catalog


class MotionEntryHeadingTests(unittest.TestCase):
    def setUp(self):
        for target in ("uav_llm_demo.agents._post_deepseek", "requests.sessions.Session.request"):
            guard = patch(target, side_effect=AssertionError("Network is forbidden in heading tests"))
            guard.start()
            self.addCleanup(guard.stop)

    def service(self, node):
        service = MissionService(scenario=compact_campus(),
                                 agent=OfflineSemanticAgent([program([node, home()])]),
                                 planner_latency_seconds=0, traffic_seed=17)
        self.addCleanup(service.close)
        return service

    @staticmethod
    def circuit():
        return motion(dict(kind="curve", x="6*cos(t)", y="6*sin(t)", z=0,
                           t_start=0, t_end=2*pi), "library",
                      heading=dict(mode="face_target", target_id="library"), speed=1.5)

    def collect_heading_errors(self, service, desired_yaw):
        errors = []
        for _ in range(2200):
            final = advance(service)
            if service._motion_phase == "tracking":
                vehicle = service._vehicle
                errors.append(abs((vehicle.yaw-desired_yaw(vehicle.position)+pi) % (2*pi)-pi))
            if final["status"] == "completed":
                break
        self.assertEqual(final["status"], "completed", service.semantic.message)
        self.assertGreater(len(errors), 20)
        self.assertLessEqual(errors[0], radians(2))
        self.assertLessEqual(max(errors), .2)
        self.assertTrue(final["milestones"]["safe_success"])
        records = final["semantic_mission"]["evidence"]
        self.assertEqual([item["action"] for item in records], ["motion", "return_home"])
        self.assertGreater(records[0]["result"]["heading_samples"], 0)
        self.assertLessEqual(records[0]["result"]["max_heading_error_deg"], degrees(.2))
        return errors

    def test_face_target_is_aligned_from_the_first_tracking_sample(self):
        service = self.service(self.circuit())
        center = next(item["center_grid"] for item in spatial_catalog(compact_campus())["objects"]
                      if item["id"] == "library")
        service.submit_task("绕图书馆一圈，机头始终朝向图书馆中心，然后返航。", background=False)
        # Transit approaches the entry from the west, while the required heading
        # at that entry faces west: the old code began tracking about 180 deg off.
        self.collect_heading_errors(service, lambda position:
                                    atan2(center[0]-position[0], -(center[1]-position[1])))

    def test_fixed_heading_is_aligned_from_the_first_tracking_sample(self):
        node = motion(dict(kind="polyline", points_m=[[0, 0, 0], [9, 0, 0]]), "library",
                      heading=dict(mode="fixed", yaw_deg=0), speed=1.5)
        service = self.service(node)
        service.submit_task("机头始终向北，沿图书馆旁折线侧飞，然后返航。", background=False)
        self.collect_heading_errors(service, lambda _: 0.0)

    def test_pause_and_resume_preserve_explicit_heading_and_verified_progress(self):
        service = self.service(self.circuit())
        service.submit_task("绕图书馆，机头保持朝向中心，随后返航。", background=False)
        pump(service, lambda _: service._motion_phase == "tracking"
             and service._motion_follower.progress_m > 3)
        follower = service._motion_follower
        before = follower.progress_m
        service.stop_runtime()
        for _ in range(12):
            advance(service)
        self.assertEqual(follower.progress_m, before)
        service.start(background=False)
        center = next(item["center_grid"] for item in spatial_catalog(compact_campus())["objects"]
                      if item["id"] == "library")
        self.collect_heading_errors(service, lambda position:
                                    atan2(center[0]-position[0], -(center[1]-position[1])))
        self.assertTrue(follower.completed)

    def test_midflight_heading_violation_cannot_be_erased_by_later_alignment(self):
        service = self.service(self.circuit())
        service.submit_task("绕图书馆一圈，机头始终朝向中心，然后返航。", background=False)
        pump(service, lambda _: service._motion_phase == "tracking"
             and service._motion_follower.progress_m > 2)
        # Inject only a measured yaw disturbance. Preserve position and velocity,
        # so neither a teleport nor a route-completion defect explains rejection.
        disturbed = replace(service._vehicle, yaw=service._vehicle.yaw+pi/2)
        service.runtime.vehicle = disturbed
        service._commit_vehicle(disturbed, 0)
        final = pump(service, lambda state: state["semantic_mission"]["status"] in
                     {"completed", "safety_wait", "error"}, seconds=180)
        self.assertEqual(final["semantic_mission"]["status"], "safety_wait")
        self.assertFalse(final["milestones"]["task_completed"])
        self.assertFalse(final["milestones"]["safe_success"])
        self.assertFalse(final["semantic_mission"]["task_contract"]["satisfied"])
        motion_records = [record for record in service.semantic.evidence if record["action"] == "motion"]
        self.assertEqual(len(motion_records), 1)
        self.assertGreater(motion_records[0]["result"]["heading_samples"], 0)
        self.assertGreater(motion_records[0]["result"]["max_heading_error_deg"], degrees(.2))
        self.assertNotIn("return_home", [record["action"] for record in service.semantic.evidence])


if __name__ == "__main__":
    unittest.main()
