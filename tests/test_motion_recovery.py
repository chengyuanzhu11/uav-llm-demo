"""Service-level recovery checks using local observations and real motion control."""
from dataclasses import replace
from copy import deepcopy
from math import pi
import unittest
from unittest.mock import patch

from tests.test_avoidance import StraightTraffic
from tests.test_semantic_motion_service import motion, advance
from tests.test_semantic_updates import OfflineSemanticAgent, compact_campus, home, program, pump, skill
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

    def service(self, *replies, scenario=None, requirements=None):
        service = MissionService(scenario=scenario or compact_campus(),
                                 agent=OfflineSemanticAgent(replies, requirements=requirements),
                                 planner_latency_seconds=0, traffic_seed=17)
        self.addCleanup(service.close)
        return service

    def test_event_review_preserves_started_geometry_and_continues_after_auxiliary_work(self):
        node = motion(dict(kind="curve", x="3*cos(t)", y="3*sin(t)", z=2,
                           t_start=0, t_end=2*pi), "laboratory",
                      heading=dict(mode="fixed", yaw_deg=0), speed=1)
        initial = program([node, home()])
        requirements = deepcopy(initial)
        requirements["steps"][0]["motion"]["path"]["flexible"] = dict(
            scale_groups=[["x", "y"]], offset_axes=["z"])
        changed = deepcopy(node)
        changed["motion"]["path"].update(x="4*cos(t)", y="4*sin(t)")
        continued = deepcopy(node)
        continued["id"] = "continued_flight"
        repaired = program([skill("check", "thermal_scan", "laboratory"), continued, home()])
        service = self.service(initial, program([changed, home()]), repaired,
                               program([continued, home()]), requirements=lambda _: requirements)
        service.submit_task("绕一圈后返航，半径和高度由执行方案确定。", background=False)
        pump(service, lambda _: service._motion_follower is not None
             and service._motion_follower.progress_m > 6)
        follower = service._motion_follower
        progress, duration = follower.progress_m, service._motion_duration
        heading_samples, heading_error = service._motion_heading_samples, service._motion_max_heading_error_deg
        first_execution = service.semantic.current_step["execution_id"]
        for _ in range(2):
            service._recover_semantic_event("no_progress", "Injected persistent blockage")
        pump(service, lambda _: service._motion_phase == "tracking" and not service._plan_requested)
        progress, duration = follower.progress_m, service._motion_duration
        heading_samples, heading_error = service._motion_heading_samples, service._motion_max_heading_error_deg
        service._last_dynamic_plan_at = service._time - 20
        def stalled_avoidance(vehicle, target, *args, **kwargs):
            service.avoidance.active = True
            return target
        # Escalate through the actual control-tick handoff, after this tick has
        # already selected a motion controller and before it records progress.
        with patch.object(service.avoidance, "stalled", return_value=True), \
                patch.object(service.avoidance, "choose", side_effect=stalled_avoidance):
            service.advance(.05)
        self.assertEqual(follower.progress_m, progress)
        pump(service, lambda _: service.semantic.revision == 2 and service._motion_follower is follower)
        self.assertEqual(service._motion_phase, "transit")
        self.assertEqual(follower.progress_m, progress)
        self.assertEqual(service._motion_duration, duration)
        self.assertEqual(service._motion_heading_samples, heading_samples)
        self.assertEqual(service._motion_max_heading_error_deg, heading_error)
        self.assertEqual(service._environment.target, follower.rejoin_point)
        self.assertEqual([item["action"] for item in service.semantic.evidence], ["thermal_scan"])
        self.assertEqual(service.semantic.current_step["continuation_of"], first_execution)
        self.assertIn("already-started motion", service.agent.inputs[2][2])
        self.assertAlmostEqual(service.agent.inputs[1][0]["mission_context"]["started_motions"][0]["progress_m"], progress)

        # A second review links to the same measured prefix without resetting its
        # metrics, including when the first continuation used a different node ID.
        pump(service, lambda _: follower.progress_m > progress + 3)
        second_execution = service.semantic.current_step["execution_id"]
        second_progress = follower.progress_m
        for _ in range(3):
            service._recover_semantic_event("no_progress", "Injected next blockage after observation")
        pump(service, lambda _: service.semantic.revision == 3 and service._motion_follower is follower)
        self.assertEqual(follower.progress_m, second_progress)
        final_execution = service.semantic.current_step["execution_id"]
        state = pump(service, lambda state: state["status"] == "completed")
        self.assertTrue(state["milestones"]["safe_success"])
        evidence = next(item for item in service.semantic.evidence if item["action"] == "motion")
        self.assertEqual(evidence["result"]["tracking_executions"],
                         [first_execution, second_execution, final_execution])
        self.assertEqual(evidence["continuation_of"], second_execution)
        self.assertAlmostEqual(evidence["result"]["distance_m"], follower.total_length_m)
        self.assertGreater(evidence["duration_seconds"], duration)
        self.assertGreater(evidence["result"]["heading_samples"], heading_samples)
        self.assertGreaterEqual(evidence["result"]["max_heading_error_deg"], heading_error)
        self.assertEqual(len(service.agent.inputs), 4)

    def test_user_update_cancels_suspended_motion_without_crediting_its_partial_path(self):
        node = motion(dict(kind="polyline", points_m=[[0, 0, 0], [0, 9, 0]]),
                      "library", speed=1)
        service = self.service(program([node, home()]), program([home()]))
        service.submit_task("沿指定轨迹飞行后返航。", background=False)
        pump(service, lambda _: service._motion_follower is not None
             and service._motion_follower.progress_m > 2)
        cancelled = service.semantic.current_step["execution_id"]
        for _ in range(3):
            service._recover_semantic_event("no_progress", "Injected persistent blockage")
        self.assertEqual(len(service._suspended_motions), 1)
        service.submit_task("取消剩余轨迹，直接返航。", background=False)
        state = pump(service, lambda state: state["status"] == "completed")
        self.assertTrue(state["milestones"]["safe_success"])
        self.assertEqual(service._suspended_motions, [])
        self.assertEqual([item["action"] for item in service.semantic.evidence], ["return_home"])
        self.assertEqual(next(item for item in service.semantic._steps
                              if item["execution_id"] == cancelled)["status"], "cancelled")

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
