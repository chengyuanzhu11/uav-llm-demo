"""Validated ground sites share one geometry/controller safety contract."""
import unittest
from copy import deepcopy

from uav_llm_demo.agents import GeometricPlanner, checked_plan
from uav_llm_demo.avoidance import LocalAvoidance
from uav_llm_demo.html_runtime import HtmlRuntime
from uav_llm_demo.motion import TrajectoryController, VehicleState, WaypointManager, distance, norm
from uav_llm_demo.safety import SafetyChecker


def context():
    return dict(grid_size=12, altitude_range=dict(min=2, max=6), buildings=[], no_fly_zones=[],
                base=dict(x=1, y=2, z=0), position=dict(x=1, y=2, z=0),
                goal=dict(x=9, y=8, z=0), ground_sites=[dict(x=9, y=8, z=0)], dynamic_objects=[])


class GroundSiteTests(unittest.TestCase):
    def test_ground_to_ground_routes_have_distinct_vertical_transitions(self):
        ctx = context()
        checker = SafetyChecker(ctx)
        self.assertEqual(checker.ground_sites, ((1, 2, 0), (9, 8, 0)))
        plan = checked_plan(GeometricPlanner(), ctx, "")
        self.assertEqual(plan.waypoints, ((1, 2, 2.2), (9, 8, 2.2), (9, 8, 0)))
        self.assertEqual(checker.validate_path((1, 2, 0), plan.waypoints), [])
        ctx.update(position=ctx["goal"], goal=ctx["base"])
        returned = checked_plan(GeometricPlanner(), ctx, "")
        self.assertEqual(returned.waypoints, ((9, 8, 2.2), (1, 2, 2.2), (1, 2, 0)))

    def test_low_flight_cannot_connect_distinct_valid_columns(self):
        checker = SafetyChecker(context())
        for a, b in (((1, 2, 0), (9, 8, 0)), ((1, 2, 1), (9, 8, 1)),
                     ((1, 2, 0), (9, 8, 2.2)), ((1, 2, 2.2), (9, 8, 0))):
            with self.subTest(a=a, b=b):
                self.assertFalse(checker.segment_safe(a, b))
                self.assertFalse(checker.segment_safe(b, a))
        self.assertTrue(checker.segment_safe((9, 8, 0), (9, 8, 2.2)))
        self.assertFalse(checker.segment_safe((9, 8, 0), (9.251, 8, 2.2)))
        self.assertTrue(checker.validate_waypoint((5, 5, 0)))

    def test_invalid_or_blocked_ground_sites_do_not_grant_low_altitude_access(self):
        for sites in (None, {}, [dict(x=9, y=8, z=1)], [dict(x=12, y=8, z=0)],
                      [dict(x=True, y=8, z=0)], [dict(x=float("nan"), y=8, z=0)]):
            ctx = context()
            ctx["ground_sites"] = sites
            with self.subTest(sites=sites), self.assertRaises(ValueError):
                SafetyChecker(ctx)
        blocked = context()
        blocked["buildings"] = [dict(x=9, y=8, height=1)]
        with self.assertRaisesRegex(ValueError, "column"):
            SafetyChecker(blocked)
        blocked = context()
        blocked["no_fly_zones"] = [dict(min=dict(x=8, y=7, z=.9), max=dict(x=10, y=9, z=1.1))]
        with self.assertRaisesRegex(ValueError, "column"):
            SafetyChecker(blocked)

    def test_landing_approach_is_preserved_for_any_site(self):
        checker = SafetyChecker(context())
        manager = WaypointManager()
        manager.load([(4, 4, 3), (9, 8, 3), (9, 8, 2.2), (9, 8, 0)], (1, 2, 3))
        self.assertTrue(manager.reconnect(VehicleState((4, 4, 3)), checker, lambda *args: True))
        self.assertEqual(manager.index, 1)
        self.assertFalse(manager.reconnect(VehicleState((9, 8, 3), (1, 0, 0)), checker, lambda *args: True))
        self.assertFalse(manager.update(VehicleState((9, 8, 3), (1, 0, 0))))
        self.assertTrue(manager.update(VehicleState((9, 8, 3))))
        self.assertTrue(manager.reconnect(VehicleState((9, 8, 2.3)), checker, lambda *args: True))
        self.assertEqual(manager.index, 3)

    def test_no_explicit_sites_retains_base_only_behavior(self):
        ctx = context()
        ctx.pop("ground_sites")
        checker = SafetyChecker(ctx)
        self.assertEqual(checker.ground_sites, ((1, 2, 0),))
        self.assertTrue(checker.validate_waypoint((9, 8, 0)))
        self.assertTrue(checker.segment_safe((1, 2, 0), (1, 2, 2.2)))
        with self.assertRaises(ValueError):
            checked_plan(GeometricPlanner(), ctx, "")

    def test_local_avoidance_rejects_low_lateral_escape_at_remote_site(self):
        checker = SafetyChecker(context())
        local = LocalAvoidance()
        state = VehicleState((9, 8, .5))
        controller = TrajectoryController()
        self.assertIsNone(local.rollout(state, (8, 8, .5), controller, checker, []))
        self.assertIsNotNone(local.rollout(state, (9, 8, 2.2), controller, checker, []))
        self.assertIsNotNone(local.rollout(state, None, controller, checker, []))

    def test_controller_lands_at_remote_site_then_relaunches_and_returns(self):
        ctx = context()
        runtime, manager, controller = HtmlRuntime(), WaypointManager(), TrajectoryController()
        runtime.reset((1, 2, 0))
        checker = SafetyChecker(ctx)
        for destination in ((9, 8, 0), (1, 2, 0)):
            leg = deepcopy(ctx)
            leg.update(position=dict(zip("xyz", runtime.read_state().position)), goal=dict(zip("xyz", destination)))
            plan = checked_plan(GeometricPlanner(), leg, "")
            manager.load(plan.waypoints, runtime.read_state().position)
            for _ in range(4000):
                before = runtime.read_state()
                # Exercise the same shortcut and braking checks as flight execution.
                manager.reconnect(before, checker, lambda point, final: LocalAvoidance().rollout(
                    before, point, controller, checker, [], terminal=final and distance(before.position, point) <= 6) is not None)
                after = runtime.drive(controller.velocity_command(before, manager.current, .05), .05)
                self.assertTrue(checker.segment_safe(before.position, after.position), (before, after))
                manager.update(after)
                if manager.current is None:
                    break
            self.assertIsNone(manager.current)
            self.assertLessEqual(distance(runtime.read_state().position, destination), .18)
            self.assertLessEqual(norm(runtime.read_state().velocity), .02)


if __name__ == "__main__":
    unittest.main()
