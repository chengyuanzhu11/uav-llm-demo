import time
import unittest
from threading import Event

from uav_llm_demo.agents import MockLLMAgent
from uav_llm_demo.avoidance import LocalAvoidance
from uav_llm_demo.environment import DynamicFlyingObject
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import SCALE, VehicleState, distance, norm
from uav_llm_demo.safety import SafetyChecker
from uav_llm_demo.traffic_motion import sample_motion


class StraightTraffic:
    def __init__(self, clock, start=(16, 10, 2), velocity=(-1, 0, 0)):
        self.clock, self.started = clock, clock()
        self.models = {"incoming": dict(phase=0, segments=[dict(start=0, duration=30,
            controls=[tuple(start[i] + velocity[i] * t for i in range(3)) for t in (0, 10, 20, 30)])])}

    def sync(self, env):
        elapsed = self.clock() - self.started
        bird = env.dynamic_objects[0]
        bird.position, bird.velocity = sample_motion(self.models[bird.object_id], elapsed)
        bird.motion_prediction = sample_motion(self.models[bird.object_id], elapsed + 1)[0]
        return dict(elapsed=elapsed, models=self.models)


def encounter_service(velocity=(3, 0, 0), agent=None):
    service = MissionService(scenario_id="open_field", agent=agent)
    service.runtime.vehicle = VehicleState((10, 10, 2), velocity)
    service._commit_vehicle(service.runtime.vehicle, 0)
    service._environment.dynamic_objects = [DynamicFlyingObject("incoming", "bird", (16, 10, 2), (-1, 0, 0))]
    service._traffic = StraightTraffic(lambda: service._time)
    service.update_goal(dict(x=18, y=10, z=2))
    service._plan_requested = False
    service.waypoints.load([(18, 10, 2)], service._vehicle.position)
    service.replanning.accepted(service._context(), service._time)
    service._running = service._safety_armed = True
    service._plan_id = 1
    return service


class AvoidanceTest(unittest.TestCase):
    def simulate(self, service, seconds):
        minimum, active, sideways = float("inf"), False, 0
        checker = SafetyChecker(service._context())
        for _ in range(round(seconds / .05)):
            before = service._vehicle
            bird_before = service._environment.dynamic_objects[0].position
            service.advance(.05)
            after = service._vehicle
            self.assertTrue(checker.segment_safe(before.position, after.position))
            self.assertLessEqual(norm(after.velocity), service.controller.max_speed + 1e-8)
            self.assertLessEqual(norm(tuple(after.velocity[i] - before.velocity[i] for i in range(3))), .10000001)
            bird_after = service._environment.dynamic_objects[0].position
            minimum = min(minimum, LocalAvoidance._swept_distance(
                tuple((bird_before[i] - before.position[i]) * SCALE[i] for i in range(3)),
                tuple((bird_after[i] - after.position[i]) * SCALE[i] for i in range(3))))
            active |= service.avoidance.active
            sideways = max(sideways, distance(after.position, (10, 10, 2)))
        return minimum, active, sideways

    def test_head_on_detours_then_rejoins_without_api_calls(self):
        service = encounter_service()
        self.addCleanup(service.close)
        minimum, active, sideways = self.simulate(service, 7)
        self.assertTrue(active)
        self.assertGreater(sideways, 1)
        self.assertGreater(minimum, 3.1)
        self.assertEqual(service._planner_calls, 0)
        self.assertFalse(service.avoidance.active)
        self.assertTrue(service._running)
        self.assertIsNone(service._error)

    def test_incoming_bird_while_paused_still_triggers_escape(self):
        service = encounter_service(velocity=(0, 0, 0))
        self.addCleanup(service.close)
        service.stop_runtime()
        minimum, active, sideways = self.simulate(service, 7)
        self.assertTrue(active)
        self.assertGreater(sideways, 1)
        self.assertGreater(minimum, 3.1)
        self.assertFalse(service._running)
        self.assertEqual(service._planner_calls, 0)

    def test_pending_api_does_not_block_local_avoidance(self):
        release = Event()
        class SlowPlanner(MockLLMAgent):
            def plan(self, context, feedback=""):
                release.wait(30)
                return super().plan(context, feedback)
        service = encounter_service(velocity=(0, 0, 0), agent=SlowPlanner())
        self.addCleanup(service.close)
        self.addCleanup(release.set)
        service._request_replan("manual_plan")
        minimum, active, sideways = self.simulate(service, 7)
        self.assertTrue(active)
        self.assertGreater(sideways, 1)
        self.assertGreater(minimum, 3.1)
        self.assertEqual(service._planner_calls, 1)
        self.assertFalse(service._future.done())

    def test_repeated_planning_failures_keep_retrying_with_local_protection(self):
        class FailedPlanner(MockLLMAgent):
            def plan(self, context, feedback=""):
                raise RuntimeError("secret must not leak")
        service = encounter_service(velocity=(0, 0, 0), agent=FailedPlanner())
        self.addCleanup(service.close)
        service._traffic = StraightTraffic(lambda: service._time, start=(18, 18, 2), velocity=(0, 0, 0))
        service._request_replan("manual_plan")
        for _ in range(500):
            time.sleep(.001)
            service.advance(.05)
            if len(service._environment.history) >= 5:
                break
        self.assertIsNone(service._error)
        self.assertNotIn("secret", str(service._environment.history))
        self.assertGreaterEqual(service._planner_calls, 5)
        service._traffic = StraightTraffic(lambda: service._time)
        minimum, active, sideways = self.simulate(service, 7)
        self.assertTrue(active)
        self.assertGreater(sideways, 1)
        self.assertGreater(minimum, 3.1)
        self.assertGreaterEqual(service._planner_calls, 5)
        self.assertTrue(service._running)
        self.assertTrue(service._plan_requested)
        self.assertTrue(service._safety_armed)

    def test_building_envelopes_remain_safe_during_escape(self):
        service = encounter_service(velocity=(0, 0, 0))
        self.addCleanup(service.close)
        # Narrow corridor: sideways choices intersect buildings; forward/back
        # remain possible, and every actually commanded segment must be safe.
        service._environment.buildings = {(x, y): 4 for x in range(6, 16) for y in (9, 11)}
        minimum, active, _ = self.simulate(service, 7)
        self.assertTrue(active)
        self.assertGreater(minimum, 3.1)
        self.assertIsNone(service._error)

    def test_persistent_blockage_spaces_high_level_calls(self):
        service = encounter_service(velocity=(0, 0, 0))
        self.addCleanup(service.close)
        service._traffic = StraightTraffic(lambda: service._time, start=(18, 10, 2), velocity=(0, 0, 0))
        called_at = []
        for _ in range(650):
            previous = service._planner_calls
            time.sleep(.001)
            service.advance(.05)
            if service._planner_calls != previous:
                called_at.append(service._time)
        self.assertTrue(called_at)
        self.assertLessEqual(len(called_at), 3)
        self.assertTrue(all(b - a >= 9.99 for a, b in zip(called_at, called_at[1:])))
        self.assertIsNone(service._error)

    def test_two_incoming_birds_are_checked_together(self):
        service = encounter_service(velocity=(0, 0, 0))
        self.addCleanup(service.close)
        checker = SafetyChecker(service._context())
        # Orthogonal incoming traffic, in metres converted to grid units.
        tracks = [(3.1, [(14 - i * .15, 10, 2) for i in range(21)]),
                  (3.1, [(10, 14 - i * .15, 2) for i in range(21)])]
        target = service.avoidance.choose(service._vehicle, None, service.controller, checker, tracks, 0)
        self.assertIsNotNone(target)
        result = service.avoidance.rollout(service._vehicle, target, service.controller, checker, tracks)
        self.assertGreaterEqual(result.clearance, service.avoidance.BUFFER)

    def test_far_traffic_does_not_change_mission_target(self):
        service = encounter_service(velocity=(0, 0, 0))
        self.addCleanup(service.close)
        service._traffic = StraightTraffic(lambda: service._time, start=(2, 2, 2), velocity=(0, 0, 0))
        service.advance(.05)
        self.assertFalse(service.avoidance.active)
        self.assertEqual(service._planner_calls, 0)
        self.assertEqual(service.waypoints.current, (18, 10, 2))

    def test_no_escape_is_reported_and_static_bounds_are_not_violated(self):
        controller = MissionService(scenario_id="open_field")
        self.addCleanup(controller.close)
        checker = SafetyChecker(dict(grid_size=2, altitude_range=dict(min=2, max=2), buildings=[]))
        local = LocalAvoidance()
        state = VehicleState((.5, .5, 2))
        tracks = [(3.1, [state.position] * 21)]
        target = local.choose(state, None, controller.controller, checker, tracks, 0)
        self.assertIsNone(target)
        self.assertTrue(local.limited)
        self.assertIn("no collision-free", local.message)

    def test_reset_and_runtime_fault_disable_protection(self):
        service = encounter_service()
        self.addCleanup(service.close)
        service.advance(.05)
        service._fail("Runtime unavailable")
        position = service._vehicle.position
        self.simulate(service, .5)
        self.assertEqual(service._vehicle.position, position)
        self.assertFalse(service._safety_armed)
        service.reset()
        self.assertFalse(service.avoidance.active)
        self.assertFalse(service._safety_armed)


if __name__ == "__main__":
    unittest.main()
