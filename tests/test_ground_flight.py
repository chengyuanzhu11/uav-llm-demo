from tests.scenario_fixtures import empty_airspace
import unittest
from unittest.mock import patch
from dataclasses import replace

from uav_llm_demo.airsim_runtime import AirSimRuntime
from uav_llm_demo.agents import MockLLMAgent, checked_plan
from uav_llm_demo.benchmark import run_scenario
from uav_llm_demo.motion import TrajectoryController, VehicleState
from uav_llm_demo.safety import SafetyChecker


def scene():
    return replace(empty_airspace(), grid_size=7, base=(0, 3, 0),
                   target=(6, 3, 3), dynamic_objects=(), buildings=())


def context(scenario):
    data = scenario.to_dict()
    data.update(position=data['base'], goal=data['target'])
    return data


class GroundFlightTests(unittest.TestCase):
    def test_ground_is_only_allowed_over_base_and_transition_is_vertical(self):
        checker = SafetyChecker(context(scene()))
        self.assertFalse(checker.validate_waypoint((0, 3, 0)))
        self.assertTrue(checker.validate_waypoint((2, 3, 0)))
        self.assertTrue(checker.validate_waypoint((0, 3, -.01)))
        self.assertTrue(checker.segment_safe((0, 3, 0), (0, 3, 2.2)))
        self.assertFalse(checker.segment_safe((0, 3, 0), (6, 3, 3)))
        self.assertFalse(checker.segment_safe((6, 3, 3), (0, 3, 0)))

    def test_outbound_and_return_include_safe_ground_transitions(self):
        ctx = context(scene())
        outbound = checked_plan(MockLLMAgent(), ctx, '')
        self.assertEqual(outbound.waypoints[0], (0, 3, 2.2))
        self.assertEqual(outbound.waypoints[-1], (6, 3, 3))
        ctx.update(position=ctx['target'], goal=ctx['base'])
        inbound = checked_plan(MockLLMAgent(), ctx, '')
        self.assertEqual(inbound.waypoints[-2:], ((0, 3, 2.2), (0, 3, 0)))

    def test_cruise_boundary_tolerates_roundoff_without_allowing_real_low_flight(self):
        ctx = context(replace(scene(), min_altitude=2, max_altitude=2))
        checker = SafetyChecker(ctx)
        for delta in (-1e-10, 0, 1e-10):
            point = (6, 3, 2 + delta)
            self.assertEqual(checker.validate_waypoint(point), [])
            self.assertTrue(checker.segment_safe((0, 3, 2 + delta), point))
        for delta in (-1e-5, 1e-5):
            point = (6, 3, 2 + delta)
            self.assertTrue(checker.validate_waypoint(point))
            self.assertFalse(checker.segment_safe((0, 3, 2 + delta), (6, 3, 2)))
        self.assertFalse(checker.segment_safe((0, 3, 0), (6, 3, 2)))

    def test_three_dimensional_route_climbs_over_full_width_wall(self):
        s = replace(scene(), buildings=tuple((3,y,4) for y in range(7)))
        plan = checked_plan(MockLLMAgent(), context(s), '')
        self.assertTrue(any(p[2] > 4.12 for p in plan.waypoints))
        self.assertFalse(SafetyChecker(context(s)).validate_path(s.base, plan.waypoints))

    def test_airsim_ground_reset_does_not_automatically_take_off(self):
        runtime = AirSimRuntime()
        try:
            with patch.object(runtime, '_reset'), patch.object(runtime, '_ensure_airborne') as takeoff, \
                 patch.object(runtime, '_read_telemetry', return_value={}), \
                 patch.object(runtime, 'status', return_value={}):
                runtime._reset_to_base((1, 18, 0))
                takeoff.assert_not_called()
                runtime._reset_to_base((1, 18, 2))
                takeoff.assert_called_once_with((1, 18, 2))
        finally:
            runtime.close()

    def test_vertical_speed_limits_are_independent_of_horizontal_speed(self):
        controller = TrajectoryController()
        for target, limit in [((0,0,10),2), ((0,0,0),1)]:
            state = VehicleState((0,0,5))
            for _ in range(100):
                velocity = controller.velocity_command(state, target, .05)
                self.assertLessEqual(abs(velocity[2]), limit + 1e-9)
                state = VehicleState(state.position, velocity)

    def test_round_trip_only_completes_after_ground_return(self):
        result = run_scenario('test_airspace', scenario=scene(), max_seconds=120)
        self.assertEqual(result['status'], 'completed', result['termination_reason'])
        self.assertTrue(result['milestones']['target_reached'])
        self.assertTrue(result['milestones']['base_returned'])
        self.assertTrue(result['milestones']['stationary'])
        self.assertTrue(result['milestones']['safe_success'])


if __name__ == '__main__':
    unittest.main()
