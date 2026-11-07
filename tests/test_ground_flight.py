from tests.scenario_fixtures import empty_airspace
import unittest
from unittest.mock import patch
from dataclasses import replace

from uav_llm_demo.agents import MockLLMAgent, checked_plan
from uav_llm_demo.benchmark import run_scenario
from uav_llm_demo.motion import TrajectoryController, VehicleState, norm, SCALE
from uav_llm_demo.safety import SafetyChecker


def scene():
    return replace(empty_airspace(), grid_size=7, base=(0, 3, 0),
                   target=(6, 3, 3), dynamic_objects=(), buildings=())


def context(scenario):
    data = scenario.to_dict()
    data.update(position=data['base'], goal=data['target'])
    return data


class GroundFlightTests(unittest.TestCase):
    def test_landing_recentres_before_descending_and_resists_lateral_drift(self):
        from math import hypot
        checker = SafetyChecker(context(scene()))
        target = scene().base
        controller = TrajectoryController(response_seconds=.5)
        radius = checker.landing_radius(target)
        state = VehicleState((target[0]+.6/3, target[1], 2.2))
        command = controller.velocity_command(state, target, .05, landing_radius=radius)
        self.assertLess(command[0], 0)
        self.assertEqual(command[2], 0)  # Correct the edge offset before descent.
        for _ in range(1500):
            command = controller.velocity_command(state, target, .05, landing_radius=radius)
            self.assertLessEqual(norm(command), controller.max_speed+1e-9)
            self.assertLessEqual(abs(command[2]), controller.max_descent_speed+1e-9)
            self.assertLessEqual(norm(tuple((command[i]-state.velocity[i])/.05 for i in range(3))),
                                 controller.max_acceleration+1e-9)
            # Persistent actuator/velocity disturbance, measured on every tick.
            actual = (command[0]+.01, command[1]-.005, command[2])
            point = tuple(state.position[i]+(state.velocity[i]+actual[i])*.05/(2*SCALE[i]) for i in range(3))
            point = (*point[:2], max(0., point[2]))
            self.assertTrue(checker.segment_safe(state.position, point))
            state = VehicleState(point, actual)
            if point[2]*5 < .02:
                break
        self.assertLess(state.position[2]*5, .02)
        self.assertLess(hypot((state.position[0]-target[0])*3,(state.position[1]-target[1])*3), .15)

    def test_ground_is_only_allowed_over_base_and_transition_is_vertical(self):
        checker = SafetyChecker(context(scene()))
        self.assertFalse(checker.validate_waypoint((0, 3, 0)))
        rejection = checker.validate_waypoint((2, 3, 0))
        self.assertTrue(rejection)
        self.assertIn("z=0.000 m not in [10.000, 50.000] m", rejection[0])
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
