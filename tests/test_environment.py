import unittest

from uav_llm_demo.agents import MockLLMAgent
from uav_llm_demo.environment import BASE, UAVEnvironment
from uav_llm_demo.motion import TrajectoryController, VehicleState, WaypointManager
from uav_llm_demo.safety import SafetyChecker


def planner_context(environment):
    context = environment.get_observation().to_dict()
    context.update(
        grid_size=environment.grid_size,
        buildings=[
            {"x": x, "y": y, "height": height}
            for (x, y), height in environment.buildings.items()
        ],
        base={"x": environment.base[0], "y": environment.base[1], "z": environment.base[2]},
        target={"x": environment.target[0], "y": environment.target[1], "z": environment.target[2]},
        no_fly_zones=[],
        orientation=environment.kinematics["orientation"],
        task="Navigate to target",
    )
    return context


class EnvironmentContractTest(unittest.TestCase):
    def test_observation_describes_waypoint_planning_not_discrete_actions(self):
        observation = UAVEnvironment().get_observation()

        self.assertIn("waypoint", observation.to_text().lower())
        self.assertNotIn("valid_actions", observation.to_dict())
        self.assertEqual(observation.position, BASE)

    def test_snapshot_preserves_fractional_continuous_position(self):
        environment = UAVEnvironment()
        environment.position = (1.25, 17.5, 2.1)

        snapshot = environment.snapshot()

        self.assertEqual(snapshot["uav"], {"x": 1.25, "y": 17.5, "z": 2.1})
        self.assertEqual(snapshot["observation"]["position"], snapshot["uav"])

    def test_dynamic_objects_are_included_in_observation(self):
        observation = UAVEnvironment().get_observation().to_dict()

        self.assertGreater(len(observation["dynamic_objects"]), 0)
        self.assertIn("predicted_position", observation["dynamic_objects"][0])
        self.assertIn("nearest_dynamic_obstacle", observation)

    def test_mock_route_is_safe_for_every_continuous_segment(self):
        environment = UAVEnvironment()
        plan = MockLLMAgent().plan(planner_context(environment))

        errors = SafetyChecker(planner_context(environment)).validate_path(
            environment.position, plan.waypoints
        )

        self.assertEqual(errors, [])
        self.assertEqual(plan.waypoints[-1], tuple(float(v) for v in environment.target))


class MotionContractTest(unittest.TestCase):
    def test_descent_settles_without_crossing_base_altitude(self):
        from uav_llm_demo.html_runtime import HtmlRuntime
        for control_dt in (0.05, 0.1, 0.2):
            runtime = HtmlRuntime()
            runtime.reset((18.0, 2.0, 5.0))
            controller = TrajectoryController()
            for _ in range(1000):
                command = controller.velocity_command(runtime.read_state(), (1.0, 18.0, 2.0), control_dt)
                vehicle = runtime.drive(command, control_dt)
                self.assertGreaterEqual(vehicle.position[2], 2.0 - 1e-10)
            self.assertAlmostEqual(vehicle.position[2], 2.0, places=6)
            self.assertLess(sum(value * value for value in vehicle.velocity) ** 0.5, 0.02)

    def test_controller_limits_acceleration_and_speed(self):
        controller = TrajectoryController(max_speed=4.0, max_acceleration=2.0)
        state = VehicleState((1.0, 1.0, 2.0))

        command = controller.velocity_command(state, (10.0, 1.0, 2.0), 0.05)

        self.assertLessEqual(sum(value * value for value in command) ** 0.5, 0.1 + 1e-9)
        next_state = VehicleState(state.position, command)
        next_command = controller.velocity_command(next_state, (10.0, 1.0, 2.0), 0.05)
        self.assertLessEqual(sum(value * value for value in next_command) ** 0.5, 4.0 + 1e-9)

    def test_waypoint_manager_advances_only_after_settling(self):
        manager = WaypointManager()
        manager.load(((2.0, 1.0, 2.0), (3.0, 1.0, 2.0)), (1.0, 1.0, 2.0))

        self.assertFalse(manager.update(VehicleState((1.95, 1.0, 2.0), (1.0, 0, 0))))
        self.assertTrue(manager.update(VehicleState((2.0, 1.0, 2.0), (0.0, 0, 0))))
        self.assertEqual(manager.current, (3.0, 1.0, 2.0))

    def test_controller_rejects_unbounded_control_period(self):
        with self.assertRaises(ValueError):
            TrajectoryController().velocity_command(
                VehicleState((1.0, 1.0, 2.0)), (2.0, 1.0, 2.0), 0.21
            )


if __name__ == "__main__":
    unittest.main()
