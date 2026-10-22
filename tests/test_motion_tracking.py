import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from uav_llm_demo.airsim_runtime import AirSimRuntime
from uav_llm_demo.html_runtime import HtmlRuntime
from uav_llm_demo.motion import TrajectoryController, VehicleState, distance, norm
from uav_llm_demo.path_tracking import PathFollower


class PathTrackingTests(unittest.TestCase):
    DT = .05

    def simulate(self, points, *, speed=3, max_seconds=220):
        follower = PathFollower(points, speed)
        runtime = HtmlRuntime()
        runtime.reset(points[0])
        controller = TrajectoryController(max_speed=speed)
        history = []
        for _ in range(round(max_seconds / self.DT)):
            state = runtime.read_state()
            target = follower.target(state)
            before = follower.progress_m
            # Merely reading targets (e.g. rendering) must not consume the path.
            self.assertEqual(target, follower.target(state))
            self.assertEqual(before, follower.progress_m)
            velocity = controller.velocity_command(state, target, self.DT)
            after = runtime.drive(velocity, self.DT)
            follower.update(after, self.DT)
            history.append((after, follower.progress_m))
            self.assertGreaterEqual(follower.progress_m, before)
            self.assertLessEqual(norm(velocity), speed + 1e-8)
            self.assertLessEqual(norm(tuple(velocity[i] - state.velocity[i] for i in range(3))), .10000001)
            self.assertLessEqual(abs(velocity[2]), controller.max_climb_speed + 1e-8)
            if follower.completed:
                break
        self.assertTrue(follower.completed, follower.snapshot())
        self.assertLessEqual(norm(runtime.read_state().velocity), follower.END_SPEED_MPS)
        self.assertLessEqual(distance(runtime.read_state().position, points[-1]), follower.END_DISTANCE_M)
        return follower, history

    def test_two_closed_laps_follow_order_without_skipping_or_stopping_at_samples(self):
        # Each lap is 94 m, with the same start, intermediate lap boundary and end.
        points = [(20 + 5 * math.cos(4 * math.pi * i / 360),
                   20 + 5 * math.sin(4 * math.pi * i / 360), 4) for i in range(361)]
        follower, history = self.simulate(points)
        angles = [0.0] + [math.atan2(s.position[1] - 20, s.position[0] - 20) for s, _ in history]
        angular_distance = sum((b - a + math.pi) % (2 * math.pi) - math.pi for a, b in zip(angles, angles[1:]))
        actual = sum(distance(a.position, b.position) for (a, _), (b, _) in zip(history, history[1:]))
        self.assertGreater(angular_distance, 4 * math.pi - .01)
        self.assertGreater(actual, .95 * follower.total_length_m)
        self.assertLess(max(abs(distance(s.position, (20, 20, 4)) - 15) for s, _ in history), .8)
        self.assertGreater(min(norm(s.velocity) for s, p in history if 10 < p < follower.total_length_m - 10), 1.0)
        self.assertTrue(any(.48 < p / follower.total_length_m < .52 and distance(s.position, points[0]) < 1.0
                            for s, p in history))

    def test_figure_eight_crossing_never_jumps_to_another_branch(self):
        points = [(20 + 5 * math.sin(2 * math.pi * i / 360),
                   20 + 5 * math.sin(2 * math.pi * i / 360) * math.cos(2 * math.pi * i / 360), 4)
                  for i in range(361)]
        follower, history = self.simulate(points)
        progress = [0.0] + [p for _, p in history]
        self.assertLess(max(b - a for a, b in zip(progress, progress[1:])), .6)
        self.assertTrue(any(.45 < p / follower.total_length_m < .55 and distance(s.position, points[0]) < .5
                            for s, p in history))
        actual = sum(distance(a.position, b.position) for (a, _), (b, _) in zip(history, history[1:]))
        self.assertGreater(actual, .94 * follower.total_length_m)

    def test_sharp_polyline_visits_each_corner_and_stops_only_at_end(self):
        points = [(10, 10, 4), (20, 10, 4), (20, 20, 4), (10, 20, 4), (10, 10, 4)]
        follower, history = self.simulate(points)
        for corner in points[1:-1]:
            self.assertLess(min(distance(s.position, corner) for s, _ in history), .21)
        self.assertFalse(follower.recovering)
        actual = sum(distance(a.position, b.position) for (a, _), (b, _) in zip(history, history[1:]))
        self.assertGreater(actual, 118)

    def test_spiral_tracks_height_and_order_with_bounded_vertical_speed(self):
        points = [(20 + 4 * math.cos(4 * math.pi * i / 360),
                   20 + 4 * math.sin(4 * math.pi * i / 360), 3 + i / 360 * 3) for i in range(361)]
        follower, history = self.simulate(points)
        self.assertGreater(history[-1][0].position[2], 5.99)
        self.assertGreater(follower.total_length_m, 150)

    def test_avoidance_freezes_progress_and_rejoins_the_unfinished_interval(self):
        points = [(10, 10, 4), (30, 10, 4)]
        follower = PathFollower(points, 3)
        runtime = HtmlRuntime()
        runtime.reset(points[0])
        controller = TrajectoryController(max_speed=3)
        for _ in range(100):
            state = runtime.read_state()
            after = runtime.drive(controller.velocity_command(state, follower.target(state), self.DT), self.DT)
            follower.update(after, self.DT)
        before = follower.progress_m
        anchor = follower.rejoin_point
        detour = (runtime.read_state().position[0], 13, 4)
        for _ in range(70):
            state = runtime.read_state()
            after = runtime.drive(controller.velocity_command(state, detour, self.DT), self.DT)
            follower.update(after, self.DT, avoiding=True)
        self.assertEqual(follower.progress_m, before)
        self.assertEqual(follower.target(runtime.read_state()), anchor)
        saw_rejoin = False
        for _ in range(2500):
            state = runtime.read_state()
            was_recovering = follower.recovering
            after = runtime.drive(controller.velocity_command(state, follower.target(state), self.DT), self.DT)
            follower.update(after, self.DT)
            if was_recovering:
                self.assertEqual(follower.progress_m, before)
            if was_recovering and not follower.recovering:
                saw_rejoin = True
                self.assertLessEqual(distance(after.position, anchor), follower.REJOIN_M)
            if follower.completed:
                break
        self.assertTrue(saw_rejoin)
        self.assertTrue(follower.completed)

    def test_a_teleport_or_waiting_at_closed_endpoint_cannot_complete_laps(self):
        points = [(10, 10, 4), (20, 10, 4), (20, 20, 4), (10, 20, 4), (10, 10, 4)]
        follower = PathFollower(points, 3)
        for _ in range(100):
            follower.update(VehicleState(points[0]), self.DT)
        self.assertEqual(follower.progress_m, 0)
        follower.update(VehicleState(points[2]), self.DT)
        self.assertTrue(follower.recovering)
        self.assertEqual(follower.progress_m, 0)
        self.assertFalse(follower.completed)
        self.assertEqual(follower.rejoin_point, points[0])

    def test_far_outside_corridor_does_not_credit_progress(self):
        follower = PathFollower([(10, 10, 4), (20, 10, 4)], 3)
        follower.update(VehicleState((10, 10, 4)), self.DT)
        for index in range(1, 21):
            follower.update(VehicleState((10, 10 + index * .02, 4), (0, 1.2, 0)), self.DT)
        self.assertTrue(follower.recovering)
        self.assertEqual(follower.progress_m, 0)

    def test_invalid_paths_and_speeds_fail_explicitly(self):
        for speed in [0, -1, 4.01, float("nan"), True]:
            with self.assertRaises(ValueError):
                PathFollower([(0, 0, 3), (1, 1, 3)], speed)
        for points in [None, [None, (0, 0, 3)], [], [(0, 0, 3)], [(0, 0, 3), (0, 0, 3)], [(0, 0, 3), (1, float("inf"), 3)]]:
            with self.assertRaises(ValueError):
                PathFollower(points, 2)


class IndependentHeadingTests(unittest.TestCase):
    def test_sideways_flight_can_keep_north_heading(self):
        runtime = HtmlRuntime()
        for _ in range(20):
            state = runtime.drive((3, 0, 0), .05, yaw=0)
        self.assertAlmostEqual(state.yaw, 0)
        self.assertGreater(state.position[0], 1)
        self.assertEqual(state.velocity, (3, 0, 0))

    def test_explicit_heading_uses_shortest_arc_and_bounded_rate(self):
        runtime = HtmlRuntime()
        runtime.vehicle = VehicleState((1, 1, 3), yaw=math.radians(179))
        state = runtime.drive((0, 0, 0), .05, yaw=math.radians(-179))
        self.assertAlmostEqual(state.yaw, math.radians(-179))
        state = runtime.drive((0, 0, 0), .05, yaw=0)
        self.assertAlmostEqual(state.yaw, math.radians(-174.5))

    def test_default_heading_retains_velocity_based_html_behavior(self):
        runtime = HtmlRuntime()
        state = runtime.drive((3, 0, 0), .05)
        self.assertAlmostEqual(state.yaw, math.pi / 2)

    def test_airsim_explicit_heading_converts_radians_to_ned_degrees(self):
        runtime = AirSimRuntime()
        self.addCleanup(runtime.close)
        sent = []
        runtime.client = SimpleNamespace(moveByVelocityAsync=lambda *args, **kwargs: sent.append((args, kwargs)))
        runtime.api = SimpleNamespace(DrivetrainType=SimpleNamespace(MaxDegreeOfFreedom="free"),
                                      YawMode=lambda is_rate, angle: (is_rate, angle))
        runtime.connected = True
        with patch.object(runtime, "_vehicle_state", return_value=VehicleState((1, 1, 3))):
            runtime.drive((2, -1, .5), .05, yaw=math.pi / 2)
            runtime.drive((2, -1, .5), .05)
        self.assertEqual(sent[0][0], (1, 2, -.5, .3))
        self.assertEqual(sent[0][1]["yaw_mode"], (False, 90.0))
        self.assertEqual(sent[1][1]["yaw_mode"], (False, 0.0))

    def test_nonfinite_heading_rejected_before_flight(self):
        for runtime in (HtmlRuntime(), AirSimRuntime()):
            self.addCleanup(runtime.close)
            with self.assertRaises(ValueError):
                runtime.drive((0, 0, 0), .05, yaw=float("nan"))


if __name__ == "__main__":
    unittest.main()
