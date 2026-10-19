"""Landing approaches and dynamically trackable shortcuts share flight guards."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from tests.scenario_fixtures import empty_airspace
from uav_llm_demo.agents import MockLLMAgent
from uav_llm_demo.avoidance import LocalAvoidance
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import (MAX_TRACKING_DEVIATION_M, ReplanningManager, TrajectoryController,
                                 VehicleState, WaypointManager, segment_deviation)
from uav_llm_demo.safety import SafetyChecker


def landing_context():
    context = replace(empty_airspace(), grid_size=40, dynamic_objects=()).to_dict()
    context.update(position=dict(x=4.37, y=16.59, z=9.44), goal=context["base"])
    return context


class ReturnReconnectionTests(unittest.TestCase):
    def test_landing_entry_cannot_be_skipped_even_with_a_clear_descent(self):
        manager = WaypointManager()
        manager.load([(1, 18, 10), (1, 18, 2.2), (1, 18, 0)], (25, 8, 6))
        aircraft = VehicleState((4.371672884, 16.594918066, 9.437903718), (-3.5, 1.45, .94))
        self.assertFalse(manager.reconnect(aircraft, SafetyChecker(landing_context()), lambda *args: True))
        self.assertEqual(manager.index, 0)
        self.assertFalse(manager.update(VehicleState((1, 18, 10), (-3, 0, 0))))
        self.assertTrue(manager.update(VehicleState((1, 18, 10))))
        self.assertEqual(manager.index, 1)
        # After the stable approach, an ordinary collinear descent can advance.
        self.assertTrue(manager.reconnect(VehicleState((1, 18, 2.3), (0, 0, -1)),
                                          SafetyChecker(landing_context()), lambda *args: True))
        self.assertEqual(manager.index, 2)

    def test_landing_entry_still_allows_skipping_earlier_transit_waypoints(self):
        manager = WaypointManager()
        manager.load([(20, 10, 10), (8, 13, 10), (1, 18, 10), (1, 18, 2.2), (1, 18, 0)],
                     (25, 8, 10))
        self.assertTrue(manager.reconnect(VehicleState((24, 9, 10)),
                                         SafetyChecker(landing_context()), lambda *args: True))
        self.assertEqual(manager.index, 2)
        self.assertEqual(manager.skipped, 2)

    def test_single_ground_goal_is_not_misclassified_or_skipped(self):
        manager = WaypointManager()
        manager.load([(1, 18, 0)], (1, 18, .02))
        self.assertFalse(manager.reconnect(VehicleState((1, 18, .02)),
                                          SafetyChecker(landing_context()), lambda *args: True))
        self.assertTrue(manager.update(VehicleState((1, 18, 0))))

    def test_metric_deviation_includes_height_and_closed_segment_endpoints(self):
        self.assertEqual(segment_deviation((0, 0, .6), (0, 0, 0), (1, 0, 0)), 3)
        self.assertEqual(segment_deviation((2, 0, 0), (0, 0, 0), (1, 0, 0)), 3)
        self.assertEqual(segment_deviation((0, 0, .6), (0, 0, 0), (0, 0, 0)), 3)
        context = landing_context()
        manager, replanner = WaypointManager(), ReplanningManager()
        manager.load([(1, 18, 6)], (8, 18, 6))
        replanner.accepted(context, 0)
        checker = SafetyChecker(context)
        self.assertIsNone(replanner.should_replan(context, VehicleState((4, 18, 6.6)), manager, checker, 1))
        self.assertEqual(replanner.should_replan(context, VehicleState((4, 18, 6.61)), manager, checker, 1),
                         "tracking_deviation_exceeded_3m")

    def test_high_momentum_shortcut_is_rejected_before_it_triggers_deviation(self):
        scenario = replace(empty_airspace(), grid_size=40, target=(1, 18, 2.2), dynamic_objects=())
        service = MissionService(scenario=scenario, agent=MockLLMAgent())
        self.addCleanup(service.close)
        aircraft = VehicleState((4.371672884, 16.594918066, 9.437903718), (-3.495869, 1.456838, .94499))
        service.runtime.vehicle = aircraft
        service._commit_vehicle(aircraft, 0)
        service.waypoints.load([(1, 18, 10), (1, 18, 2.2)], aircraft.position)
        service.replanning.accepted(service._context(), 0)
        service._running = service._safety_armed = True
        service._plan_id = 1
        checker = SafetyChecker(service._context())
        unsafe_turn = LocalAvoidance().rollout(aircraft, scenario.target, TrajectoryController(), checker, [], capture=True)
        self.assertIsNotNone(unsafe_turn)
        self.assertGreater(max(segment_deviation(p, aircraft.position, scenario.target) for p in unsafe_turn.path),
                           MAX_TRACKING_DEVIATION_M)
        with patch.object(service.agent, "plan", side_effect=AssertionError("No planning is needed")):
            service.advance(.05)
        self.assertEqual(service.waypoints.index, 0)
        self.assertFalse(service._plan_requested)
        self.assertIsNone(service._future)


if __name__ == "__main__":
    unittest.main()
