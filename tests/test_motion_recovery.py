"""Service-level recovery checks using local observations and real motion control."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from tests.test_avoidance import StraightTraffic
from tests.test_semantic_motion_service import motion, advance
from tests.test_semantic_updates import OfflineSemanticAgent, compact_campus, home, program, pump
from uav_llm_demo.avoidance import LocalAvoidance
from uav_llm_demo.dynamic_risk import physical_radius_m, separation_distance_m
from uav_llm_demo.environment import DynamicFlyingObject
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import SCALE, VehicleState, distance
from uav_llm_demo.safety import SafetyChecker


class MotionRecoveryTests(unittest.TestCase):
    def setUp(self):
        guard = patch("uav_llm_demo.agents._post_deepseek",
                      side_effect=AssertionError("Remote API forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def service(self, *replies, scenario=None):
        service = MissionService(scenario=scenario or compact_campus(),
                                 agent=OfflineSemanticAgent(replies),
                                 planner_latency_seconds=0, traffic_seed=17)
        self.addCleanup(service.close)
        return service

    def test_resume_behind_building_rejoins_unfinished_path_without_losing_progress(self):
        node = motion(dict(kind="polyline", points_m=[[0, 0, 0], [0, 9, 0]]),
                      "library", speed=1)
        scenario = replace(compact_campus(), buildings=((7, 16, 4),))
        service = self.service(program([node, home()]), scenario=scenario)
        service.submit_task("沿指定轨迹飞行后返航。", background=False)
        pump(service, lambda _: service._motion_follower is not None
             and service._motion_follower.progress_m > 2)
        follower = service._motion_follower
        before = follower.progress_m
        anchor = follower.rejoin_point
        service.stop_runtime()
        for _ in range(10):
            advance(service)
        # Model the result of safety movement during a pause, now behind a building.
        # This is an injected recovery pose, not a simulated bird encounter.
        displaced = VehicleState((10, anchor[1], anchor[2]))
        service.runtime.vehicle = displaced
        service._commit_vehicle(displaced, 0)
        self.assertFalse(SafetyChecker(service._context()).segment_safe(displaced.position, anchor))
        service.start(background=False)
        saw_transfer = False
        for _ in range(2200):
            state = advance(service)
            self.assertIsNone(service._error)
            if service._motion_follower is follower:
                self.assertGreaterEqual(follower.progress_m, before)
                if service._motion_phase == "transit":
                    saw_transfer = True
                    self.assertEqual(follower.progress_m, before)
                    self.assertEqual(service._environment.target, anchor)
            if state["status"] == "completed":
                break
        self.assertTrue(saw_transfer)
        self.assertEqual(state["status"], "completed", service.semantic.message)
        self.assertTrue(follower.completed)
        self.assertTrue(state["milestones"]["safe_success"])
        self.assertEqual([item["action"] for item in service.semantic.evidence], ["motion", "return_home"])

    def test_ceiling_touching_turn_is_rejected_before_takeoff_and_safe_margin_can_fly(self):
        unsafe = motion(dict(kind="polyline", points_m=[[0, 0, 0], [0, 0, 19], [18, 0, 19]]),
                        "library", speed=4)
        safe = motion(dict(kind="polyline", points_m=[[0, 0, 0], [0, 0, 18], [18, 0, 18]]),
                      "library", speed=4)
        service = self.service(program([unsafe, home()], ceiling=30),
                               program([safe, home()], ceiling=30))
        service.submit_task("先上升到三十米，再向东飞行。", background=False)
        state = pump(service, lambda _: service.semantic.status in {"needs_clarification", "safety_wait"})
        self.assertEqual(service._vehicle.position, service._environment.base)
        self.assertEqual(service.semantic.evidence, [])
        self.assertFalse(service._running)
        self.assertIsNone(service._error)

        # A changed explicit instruction supplies the margin; execution must not
        # silently modify the original requested height to manufacture a safe path.
        service.submit_task("改为上升到二十九米，再向东飞行后返航。", background=False)
        maximum = 0.0
        for _ in range(1800):
            state = advance(service)
            maximum = max(maximum, service._vehicle.position[2] * SCALE[2])
            self.assertIsNone(service._error)
            if state["status"] == "completed":
                break
        self.assertEqual(state["status"], "completed", service.semantic.message)
        self.assertGreater(maximum, 28.9)
        self.assertLessEqual(maximum, 30)
        self.assertTrue(state["milestones"]["safe_success"])

    def test_real_observed_bird_detour_preserves_progress_then_completes_motion(self):
        node = motion(dict(kind="polyline", points_m=[[0, 0, 0], [30, 0, 0]]),
                      "library", speed=2.5)
        service = self.service(program([node, home()]))
        service.submit_task("沿轨迹飞行，避让鸟群后继续并返航。", background=False)
        pump(service, lambda _: service._motion_follower is not None
             and service._motion_follower.progress_m > 5)
        follower = service._motion_follower
        position = service._vehicle.position
        start = (position[0] + 6, position[1], position[2])
        bird = DynamicFlyingObject("incoming", "bird", start, (-1, 0, 0))
        service._environment.dynamic_objects = [bird]
        service._traffic = StraightTraffic(lambda: service._time, start=start, velocity=(-1, 0, 0))
        minimum = float("inf")
        avoidance_frames = 0
        max_offset = 0.0
        for _ in range(2500):
            before_vehicle, before_bird = service._vehicle, bird.position
            before_progress = follower.progress_m
            was_active = service.avoidance.active
            state = advance(service)
            after_vehicle = service._vehicle
            relative_start = tuple((before_bird[i] - before_vehicle.position[i]) * SCALE[i] for i in range(3))
            relative_end = tuple((bird.position[i] - after_vehicle.position[i]) * SCALE[i] for i in range(3))
            minimum = min(minimum, LocalAvoidance._swept_distance(relative_start, relative_end))
            self.assertIsNone(service._error)
            if service._motion_follower is follower:
                self.assertGreaterEqual(follower.progress_m, before_progress)
                if was_active or service.avoidance.active:
                    avoidance_frames += 1
                    self.assertEqual(follower.progress_m, before_progress)
                max_offset = max(max_offset, distance(after_vehicle.position,
                    (after_vehicle.position[0], position[1], position[2])))
            if state["status"] == "completed":
                break
        self.assertGreater(avoidance_frames, 5)
        self.assertGreater(max_offset, 1.0)
        self.assertGreaterEqual(minimum, separation_distance_m(physical_radius_m(bird)))
        self.assertTrue(follower.completed)
        self.assertEqual(state["status"], "completed", service.semantic.message)
        self.assertTrue(state["milestones"]["safe_success"])
        self.assertEqual([item["action"] for item in service.semantic.evidence], ["motion", "return_home"])
        self.assertEqual(len(service.agent.inputs), 1)


if __name__ == "__main__":
    unittest.main()
