import unittest
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace

from uav_llm_demo import _validated_decision, airsim_runtime
from uav_llm_demo.agents import AgentDecision, MockLLMAgent, decision_from_json_text
from uav_llm_demo.airsim_runtime import (
    AirSimRuntime,
    AirSimRuntimeConfig,
    AirSimRpcWorker,
    calculate_path_timeout,
    grid_to_ned,
    ned_to_grid,
    within_position_tolerance,
)
from uav_llm_demo.environment import (
    ACTIONS,
    BASE,
    TARGET,
    TRAJECTORY_SAMPLES,
    UAVEnvironment,
    minimum_jerk_trajectory,
)


class UAVEnvironmentTest(unittest.TestCase):
    def test_uav_cannot_cross_obstacle(self):
        env = UAVEnvironment()
        env.position = (5, 5, 2)

        feedback = env.apply_action("move_east")

        self.assertEqual(env.position, (5, 5, 2))
        self.assertIn("intersects", feedback)

    def test_action_consumes_battery(self):
        env = UAVEnvironment()

        env.apply_action("hover")

        self.assertEqual(env.battery, 99)

    def test_reaching_target_switches_to_returning_home(self):
        env = UAVEnvironment()
        env.position = (17, 2, 5)

        env.apply_action("move_east")

        self.assertEqual(env.position, TARGET)
        self.assertEqual(env.phase, "returning_home")
        self.assertEqual(env.status, "active")

    def test_return_home_primitive_completes_at_base(self):
        env = UAVEnvironment()
        env.position = (2, 18, 2)
        env.phase = "returning_home"

        env.apply_action("return_home")

        self.assertEqual(env.position, BASE)
        self.assertEqual(env.status, "completed")

    def test_observation_explains_screen_coordinate_direction(self):
        env = UAVEnvironment()
        env.position = (16, 18, 5)

        observation = env.get_observation()

        self.assertEqual(observation.to_dict()["goal_delta"], {"dx": 2, "dy": -16, "dz": 0})
        self.assertEqual(
            observation.to_dict()["goal_direction"],
            "east/right and north/up-map",
        )
        self.assertIn(
            "Goal delta: dx=2, dy=-16, dz=0; goal is east/right and north/up-map",
            observation.to_text(),
        )

    def test_returning_home_limits_actions_to_return_home(self):
        env = UAVEnvironment()
        env.position = (1, 17, 2)
        env.phase = "returning_home"

        observation = env.get_observation()
        prompt = MockLLMAgent().build_prompt(observation)

        self.assertEqual(observation.valid_actions, ("return_home", "hover"))
        self.assertIn("只能从这些动作中选择一个：return_home, hover", prompt)

    def test_ascend_changes_altitude(self):
        env = UAVEnvironment()

        env.apply_action("ascend")

        self.assertEqual(env.position, (1, 18, 3))
        self.assertEqual(env.battery, 98)

    def test_dynamic_objects_move_every_step(self):
        env = UAVEnvironment()
        before = [item.position for item in env.dynamic_objects]

        env.apply_action("hover")

        after = [item.position for item in env.dynamic_objects]
        self.assertNotEqual(before, after)

    def test_observation_contains_predicted_dynamic_traffic(self):
        env = UAVEnvironment()

        observation = env.get_observation()

        self.assertEqual(len(observation.dynamic_objects), 3)
        self.assertIn("predicted_position", observation.dynamic_objects[0])
        self.assertIsNotNone(observation.nearest_dynamic_obstacle)

    def test_dynamic_object_blocks_the_next_flight_cell(self):
        env = UAVEnvironment()
        env.dynamic_objects[0].position = (2.0, 17.0, 2.0)
        env.dynamic_objects[0].velocity = (0.0, 1.0, 0.0)

        feedback = env.apply_action("move_east")

        self.assertNotEqual(env.position, (2, 18, 2))
        self.assertIn("Dynamic object", feedback)

    def test_action_generates_minimum_jerk_trajectory(self):
        env = UAVEnvironment()
        start = env.position

        env.apply_action("move_east")

        self.assertEqual(len(env.last_trajectory), TRAJECTORY_SAMPLES)
        self.assertEqual(
            env.last_trajectory[0],
            {"x": start[0], "y": start[1], "z": start[2]},
        )
        self.assertEqual(
            env.last_trajectory[-1],
            {"x": env.position[0], "y": env.position[1], "z": env.position[2]},
        )
        first_increment = (
            env.last_trajectory[1]["x"] - env.last_trajectory[0]["x"]
        )
        middle_increment = (
            env.last_trajectory[16]["x"] - env.last_trajectory[15]["x"]
        )
        last_increment = (
            env.last_trajectory[-1]["x"] - env.last_trajectory[-2]["x"]
        )
        self.assertLess(first_increment, middle_increment)
        self.assertLess(last_increment, middle_increment)
        self.assertEqual(len(env.continuous_path), TRAJECTORY_SAMPLES)


class TrajectoryAndAirSimMappingTest(unittest.TestCase):
    def test_settled_path_does_not_switch_to_hover_controller(self):
        class Vector3r:
            def __init__(self, x, y, z):
                self.x_val = x
                self.y_val = y
                self.z_val = z

        class FakeApi:
            DrivetrainType = SimpleNamespace(MaxDegreeOfFreedom=1)

            @staticmethod
            def YawMode(is_rate, yaw_or_rate):
                return SimpleNamespace(
                    is_rate=is_rate,
                    yaw_or_rate=yaw_or_rate,
                )

        FakeApi.Vector3r = Vector3r

        class CompletedFuture:
            def join(self):
                return None

        class FakeClient:
            def __init__(self):
                self.state_reads = 0
                self.hover_calls = 0

            def getMultirotorState(self, vehicle_name):
                self.state_reads += 1
                settled = self.state_reads > 1
                position = Vector3r(3.0 if settled else 0.0, 0.0, -10.0)
                velocity = Vector3r(0.1 if settled else 0.0, 0.0, 0.0)
                zero = Vector3r(0.0, 0.0, 0.0)
                orientation = SimpleNamespace(
                    x_val=0.0,
                    y_val=0.0,
                    z_val=0.0,
                    w_val=1.0,
                )
                return SimpleNamespace(
                    kinematics_estimated=SimpleNamespace(
                        position=position,
                        orientation=orientation,
                        linear_velocity=velocity,
                        angular_velocity=zero,
                        linear_acceleration=zero,
                        angular_acceleration=zero,
                    ),
                    timestamp=self.state_reads,
                    landed_state=1,
                )

            def moveOnPathAsync(self, *args, **kwargs):
                return CompletedFuture()

            def hoverAsync(self, vehicle_name):
                self.hover_calls += 1
                return CompletedFuture()

            def cancelLastTask(self, vehicle_name):
                return None

        runtime = AirSimRuntime(AirSimRuntimeConfig(command_timeout=8.0))
        client = FakeClient()
        runtime.api = FakeApi
        runtime.client = client
        runtime.connected = True
        try:
            runtime._fly_trajectory([(1.0, 18.0, 2.0), (1.0, 17.0, 2.0)])
            runtime._hover()
        finally:
            runtime._worker.close()

        self.assertEqual(client.hover_calls, 0)

    def test_execution_path_rebases_from_actual_airsim_position(self):
        self.assertTrue(hasattr(airsim_runtime, "rebase_grid_path"))
        path = airsim_runtime.rebase_grid_path(
            [
                (1.0, 17.0, 2.0),
                (1.0, 16.0, 2.0),
            ],
            actual_start=(1.0, 16.6, 2.0),
            samples=9,
        )

        self.assertEqual(path[0], (1.0, 16.6, 2.0))
        self.assertEqual(path[-1], (1.0, 16.0, 2.0))
        self.assertTrue(
            all(path[index][1] >= path[index + 1][1] for index in range(8))
        )

    def test_short_airsim_segments_use_stable_velocity_cap(self):
        self.assertTrue(hasattr(airsim_runtime, "choose_segment_velocity"))
        self.assertEqual(
            airsim_runtime.choose_segment_velocity(
                distance_m=3.0,
                configured_velocity_mps=4.0,
            ),
            1.5,
        )
        self.assertEqual(
            airsim_runtime.choose_segment_velocity(
                distance_m=3.0,
                configured_velocity_mps=1.0,
            ),
            1.0,
        )
        self.assertEqual(
            airsim_runtime.choose_segment_velocity(
                distance_m=10.0,
                configured_velocity_mps=4.0,
            ),
            4.0,
        )

    def test_airsim_arrival_requires_position_and_velocity_to_settle(self):
        self.assertTrue(hasattr(airsim_runtime, "telemetry_has_settled"))
        target = (3.0, 0.0, -10.0)
        self.assertFalse(
            airsim_runtime.telemetry_has_settled(
                {
                    "position_ned": {"x": 3.05, "y": 0.0, "z": -10.0},
                    "linear_velocity": {"x": 1.2, "y": 0.0, "z": 0.0},
                },
                target,
            )
        )
        self.assertTrue(
            airsim_runtime.telemetry_has_settled(
                {
                    "position_ned": {"x": 3.08, "y": 0.0, "z": -10.0},
                    "linear_velocity": {"x": 0.12, "y": 0.0, "z": 0.0},
                },
                target,
            )
        )
        self.assertTrue(
            airsim_runtime.telemetry_has_settled(
                {
                    "position_ned": {"x": 3.08, "y": 0.0, "z": -10.45},
                    "linear_velocity": {"x": 0.08, "y": 0.0, "z": 0.01},
                },
                target,
            )
        )
        self.assertFalse(
            airsim_runtime.telemetry_has_settled(
                {
                    "position_ned": {"x": 3.08, "y": 0.0, "z": -10.55},
                    "linear_velocity": {"x": 0.08, "y": 0.0, "z": 0.01},
                },
                target,
            )
        )
        self.assertFalse(
            airsim_runtime.telemetry_has_settled(
                {
                    "position_ned": {"x": 3.3, "y": 0.0, "z": -10.0},
                    "linear_velocity": {"x": 0.08, "y": 0.0, "z": 0.01},
                },
                target,
            )
        )

    def test_airsim_disconnect_does_not_call_unreachable_server(self):
        class Transport:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class UnreachableClient:
            def __init__(self):
                self.client = Transport()
                self.remote_call_attempted = False

            def hoverAsync(self, **kwargs):
                self.remote_call_attempted = True
                raise TimeoutError("AirSim server is gone")

        runtime = AirSimRuntime()
        client = UnreachableClient()
        runtime.client = client
        runtime.connected = True
        runtime.armed = True

        runtime.disconnect()

        self.assertFalse(client.remote_call_attempted)
        self.assertTrue(client.client.closed)
        self.assertFalse(runtime.connected)
        self.assertFalse(runtime.armed)

    def test_airsim_rpc_worker_cancels_pending_tasks_when_closed(self):
        worker = AirSimRpcWorker()
        pending_task = worker.call(
            lambda: asyncio.get_event_loop().create_task(asyncio.sleep(60))
        )

        worker.close()

        self.assertTrue(pending_task.cancelled())

    def test_short_airsim_path_uses_bounded_completion_timeout(self):
        self.assertAlmostEqual(
            calculate_path_timeout(
                distance_m=3.0,
                velocity_mps=1.5,
                maximum_timeout=8.0,
            ),
            7.0,
        )
        self.assertEqual(
            calculate_path_timeout(
                distance_m=40.0,
                velocity_mps=4.0,
                maximum_timeout=8.0,
            ),
            8.0,
        )

    def test_airsim_target_tolerance_accepts_wind_tracking_error(self):
        self.assertTrue(
            within_position_tolerance(
                {"x": 3.55, "y": 0.25, "z": -9.7},
                (3.0, 0.0, -10.0),
                tolerance=0.75,
            )
        )
        self.assertFalse(
            within_position_tolerance(
                {"x": 3.8, "y": 0.0, "z": -10.0},
                (3.0, 0.0, -10.0),
                tolerance=0.75,
            )
        )

    def test_airsim_rpc_worker_serializes_calls_from_flask_threads(self):
        worker = AirSimRpcWorker()
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                execution_threads = list(
                    callers.map(
                        lambda _: worker.call(get_ident),
                        range(2),
                    )
                )
        finally:
            worker.close()

        self.assertEqual(execution_threads[0], execution_threads[1])
        self.assertNotEqual(execution_threads[0], get_ident())

    def test_minimum_jerk_trajectory_preserves_endpoints(self):
        trajectory = minimum_jerk_trajectory((1, 2, 3), (6, 4, 8), 41)

        self.assertEqual(trajectory[0], (1.0, 2.0, 3.0))
        self.assertEqual(trajectory[-1], (6.0, 4.0, 8.0))
        self.assertTrue(
            all(
                trajectory[index][0] <= trajectory[index + 1][0]
                for index in range(len(trajectory) - 1)
            )
        )

    def test_grid_and_ned_coordinates_round_trip(self):
        point = (13.25, 7.5, 4.2)

        ned = grid_to_ned(point, meters_per_cell=2.5, meters_per_altitude_level=4.0)
        restored = ned_to_grid(
            ned,
            meters_per_cell=2.5,
            meters_per_altitude_level=4.0,
        )

        for expected, actual in zip(point, restored):
            self.assertAlmostEqual(expected, actual)


class MockLLMAgentTest(unittest.TestCase):
    def test_agent_outputs_legal_action(self):
        env = UAVEnvironment()
        agent = MockLLMAgent()

        decision = agent.decide(env.get_observation())

        self.assertIn(decision.action, ACTIONS)
        self.assertTrue(decision.reason)
        self.assertTrue(decision.expected_result)

    def test_agent_can_finish_full_mission(self):
        env = UAVEnvironment()
        agent = MockLLMAgent()

        for _ in range(120):
            decision = agent.decide(env.get_observation())
            self.assertIn(decision.action, ACTIONS)
            env.apply_action(decision.action)
            if env.status != "active":
                break

        self.assertEqual(env.status, "completed")
        self.assertEqual(env.position, BASE)


class DeepSeekParsingTest(unittest.TestCase):
    def test_deepseek_json_response_is_validated(self):
        decision = decision_from_json_text(
            """
            ```json
            {
              "action": "move_north",
              "reason": "目标在北侧，且下一格安全。",
              "risk": "LOW",
              "expected_result": "UAV 向北移动一格。"
            }
            ```
            """
        )

        self.assertEqual(decision.action, "move_north")
        self.assertEqual(decision.risk, "low")

    def test_invalid_deepseek_action_falls_back_to_hover(self):
        decision = decision_from_json_text(
            '{"action":"fly_fast","reason":"bad action","risk":"unknown","expected_result":"none"}'
        )

        self.assertEqual(decision.action, "hover")
        self.assertEqual(decision.risk, "medium")

    def test_executor_arbitration_prevents_backtracking(self):
        env = UAVEnvironment()
        env.position = (2, 18, 2)
        observation = env.get_observation()
        deepseek_decision = AgentDecision(
            action="move_west",
            reason="Move west.",
            risk="low",
            expected_result="UAV moves west.",
        )

        validated = _validated_decision(env, observation, deepseek_decision)

        self.assertIn(validated.action, {"move_east", "move_north", "ascend"})
        self.assertNotEqual(validated.action, "move_west")
        self.assertIn("Executor arbitration", validated.reason)


if __name__ == "__main__":
    unittest.main()
