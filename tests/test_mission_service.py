import unittest

from uav_llm_demo.agents import MockLLMAgent
from uav_llm_demo.environment import BASE
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.runtime import RuntimeExecution, RuntimeUnavailableError


class RecordingRuntime:
    def __init__(self, backend_name="html"):
        self.backend_name = backend_name
        self.executions = 0
        self.closed = False
        self.reset_calls = []
        self.telemetry = None

    def reset(self, base=BASE):
        self.reset_calls.append(base)
        return self.status()

    def execute(self, trajectory):
        self.executions += 1
        points = tuple(dict(point) for point in trajectory)
        final = points[-1]
        kinematics = {
            "position": final,
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            "source": f"{self.backend_name}-test",
        }
        if self.backend_name == "airsim":
            self.telemetry = {**kinematics, "position_grid": final}
        return RuntimeExecution(points, 0.9, kinematics)

    def stop(self):
        return None

    def status(self):
        return {
            "backend": self.backend_name,
            "connected": True,
            "supports_wind": self.backend_name == "airsim",
            "telemetry": self.telemetry,
        }

    def close(self):
        self.closed = True


class FailingRuntime(RecordingRuntime):
    def execute(self, trajectory):
        raise RuntimeUnavailableError("test execution failed")


class MissionServiceTest(unittest.TestCase):
    def test_defaults_to_html_runtime(self):
        service = MissionService(agent=MockLLMAgent())

        state = service.state()

        self.assertEqual(state["runtime_backend"], "html")
        self.assertTrue(state["runtime"]["connected"])
        self.assertNotIn("simulator_backend", state)
        self.assertNotIn("logical_uav", state)

    def test_step_commits_runtime_execution_result(self):
        runtime = RecordingRuntime()
        service = MissionService(agent=MockLLMAgent(), runtime=runtime)
        before = service.state()["step_count"]

        after = service.step()

        self.assertEqual(after["step_count"], before + 1)
        self.assertEqual(runtime.executions, 1)
        self.assertEqual(after["kinematics"]["source"], "html-test")

    def test_failed_runtime_execution_does_not_commit_candidate_state(self):
        service = MissionService(agent=MockLLMAgent(), runtime=FailingRuntime())
        before = service.state()

        with self.assertRaises(RuntimeUnavailableError):
            service.step()

        after = service.state()
        self.assertEqual(after["uav"], before["uav"])
        self.assertEqual(after["battery"], before["battery"])
        self.assertEqual(after["step_count"], before["step_count"])

    def test_switch_to_airsim_resets_after_candidate_is_ready(self):
        created = RecordingRuntime("airsim")
        service = MissionService(
            agent=MockLLMAgent(),
            runtime_factory=lambda backend, config: created,
        )
        service.step()

        state = service.switch_runtime(
            "airsim",
            {"host": "127.0.0.1", "port": 41451},
        )

        self.assertEqual(created.reset_calls, [BASE])
        self.assertEqual(state["runtime_backend"], "airsim")
        self.assertEqual(state["uav"], state["base"])
        self.assertEqual(state["step_count"], 0)

    def test_failed_airsim_switch_preserves_html_runtime_and_mission(self):
        def failing_factory(backend, config):
            raise RuntimeUnavailableError("AirSim is offline")

        service = MissionService(
            agent=MockLLMAgent(),
            runtime_factory=failing_factory,
        )
        service.step()
        before = service.state()

        with self.assertRaises(RuntimeUnavailableError):
            service.switch_runtime("airsim", {})

        after = service.state()
        self.assertEqual(after["runtime_backend"], "html")
        self.assertEqual(after["uav"], before["uav"])
        self.assertEqual(after["step_count"], before["step_count"])

    def test_switch_back_to_html_closes_airsim_and_resets(self):
        airsim = RecordingRuntime("airsim")
        service = MissionService(agent=MockLLMAgent(), runtime=airsim)

        state = service.switch_runtime("html", {})

        self.assertTrue(airsim.closed)
        self.assertEqual(state["runtime_backend"], "html")
        self.assertEqual(state["uav"], state["base"])
        self.assertEqual(state["step_count"], 0)


if __name__ == "__main__":
    unittest.main()
