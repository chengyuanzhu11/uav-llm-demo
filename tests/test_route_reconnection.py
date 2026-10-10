import unittest
import time

from uav_llm_demo.avoidance import LocalAvoidance
from uav_llm_demo.evaluation import path_length
from uav_llm_demo.motion import VehicleState, WaypointManager, TrajectoryController
from uav_llm_demo.safety import SafetyChecker
from test_avoidance import encounter_service, StraightTraffic


def checker(buildings=()):
    return SafetyChecker(dict(grid_size=20, altitude_range=dict(min=2, max=10), buildings=list(buildings)))


class RouteReconnectionTest(unittest.TestCase):
    def test_detour_rejoins_goal_without_returning_to_old_waypoint(self):
        service = encounter_service(velocity=(0, 0, 0))
        self.addCleanup(service.close)
        service._traffic = StraightTraffic(lambda: service._time, start=(2, 2, 2), velocity=(0, 0, 0))
        service.waypoints.load([(8, 10, 2), (12, 10, 2), (18, 10, 2)], (6, 10, 2))
        service.avoidance.active = True
        service.avoidance.target = (10, 10, 2)
        service.avoidance.since = 0
        service.advance(.05)
        self.assertEqual(service.waypoints.index, 2)
        self.assertEqual(service.waypoints.skipped, 2)
        for _ in range(20):
            service.advance(.05)
        self.assertGreater(service._vehicle.position[0], 10)
        self.assertEqual(service._planner_calls, 0)

    def test_shortcut_cannot_cross_building_or_dynamic_conflict(self):
        manager = WaypointManager()
        manager.load([(8, 8, 2), (14, 10, 2)], (8, 10, 2))
        state = VehicleState((10, 10, 2))
        self.assertFalse(manager.reconnect(state, checker([dict(x=12, y=10, height=4)]), lambda *args: True))
        self.assertFalse(manager.reconnect(state, checker(), lambda *args: False))
        self.assertEqual(manager.index, 0)

    def test_transit_can_advance_without_stopping_but_goal_cannot(self):
        manager = WaypointManager()
        manager.load([(10, 10, 2), (14, 10, 2)], (8, 10, 2))
        self.assertTrue(manager.reconnect(VehicleState((9.9, 10, 2), (2, 0, 0)), checker(), lambda *args: True))
        self.assertFalse(manager.update(VehicleState((14, 10, 2), (1, 0, 0))))
        self.assertTrue(manager.update(VehicleState((14, 10, 2))))

    def test_climbing_competes_with_a_safe_horizontal_retreat(self):
        local = LocalAvoidance()
        control = TrajectoryController()
        state = VehicleState((10, 10, 2))
        bounds = checker(dict(x=x, y=y, height=3) for x in range(6, 17) for y in (9, 11))
        tracks = [(3.1, [(12, 10, 2)] * 41)]
        self.assertGreater(local.rollout(state, (8, 10, 2), control, bounds, tracks).clearance, local.BUFFER)
        target = local.choose(state, (18, 10, 2), control, bounds, tracks, 0)
        self.assertGreater(target[0], state.position[0])
        self.assertGreater(target[2], state.position[2])
        self.assertGreaterEqual(local.rollout(state, target, control, bounds, tracks).clearance, local.BUFFER)

    def test_terminal_checks_traffic_after_arrival(self):
        local = LocalAvoidance()
        state = VehicleState((10, 10, 2))
        target = (10.2, 10, 2)
        control = TrajectoryController()
        # The threat arrives after the ordinary three-second horizon.
        tracks = [(1.0, [(10.2, 15 - index * .15, 2) for index in range(41)])]
        self.assertGreater(local.rollout(state, target, control, checker(), tracks).clearance, local.BUFFER)
        self.assertLess(local.rollout(state, target, control, checker(), tracks, terminal=True).clearance, local.BUFFER)
        occupied = [(3.1, [target] * 41)]
        selected = local.choose(state, target, control, checker(), occupied, 0, terminal=True)
        self.assertNotEqual(selected, target)
        self.assertTrue(local.active)

    def test_progress_prevents_timer_only_replanning(self):
        local = LocalAvoidance()
        local.active = True
        manager = WaypointManager()
        manager.load([(18, 10, 2)], (1, 10, 2))
        for second in range(20):
            self.assertFalse(local.stalled(manager, VehicleState((1 + second * .2, 10, 2)), second))
        self.assertTrue(local.stalled(manager, VehicleState((4.8, 10, 2)), 28))

    def test_position_units_are_converted_for_path_metrics(self):
        self.assertEqual(path_length([dict(x=0, y=0, z=0), dict(x=1, y=0, z=0), dict(x=1, y=0, z=1)]), 8)

    def test_occupied_goal_is_approached_after_traffic_leaves(self):
        service = encounter_service(velocity=(0, 0, 0))
        self.addCleanup(service.close)
        service._environment.target = (14, 10, 2)
        service.waypoints.load([(11, 10, 2), (14, 10, 2)], service._vehicle.position)
        service.replanning.accepted(service._context(), 0)

        class DepartingTraffic:
            def sync(self, environment):
                bird = environment.dynamic_objects[0]
                bird.position = (14, 10 + max(0, service._time - 8), 2)
                bird.velocity = (0, 1 if service._time >= 8 else 0, 0)
                bird.motion_prediction = tuple(p + v for p, v in zip(bird.position, bird.velocity))
                return None

        service._traffic = DepartingTraffic()
        minimum = float("inf")
        for _ in range(700):
            time.sleep(.001)
            service.advance(.05)
            bird = service._environment.dynamic_objects[0]
            from uav_llm_demo.motion import distance
            minimum = min(minimum, distance(service._vehicle.position, bird.position) - 3.1)
            if service._time < 8:
                self.assertEqual(service._environment.phase, "fly_to_target")
            if service._environment.phase == "returning_home":
                break
        self.assertEqual(service._environment.phase, "returning_home")
        self.assertGreater(minimum, 0)
        self.assertLessEqual(service._planner_calls, 2)
        self.assertIsNone(service._error)


if __name__ == "__main__":
    unittest.main()
