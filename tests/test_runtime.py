import unittest

from uav_llm_demo.airsim_runtime import AirSimRuntime, AirSimRuntimeError
from uav_llm_demo.html_runtime import HtmlRuntime


class HtmlRuntimeTest(unittest.TestCase):
    def test_is_connected_without_an_external_process(self):
        runtime = HtmlRuntime()

        status = runtime.status()

        self.assertEqual(runtime.backend_name, "html")
        self.assertTrue(status["connected"])
        self.assertFalse(status["supports_wind"])

    def test_executes_the_planned_trajectory(self):
        runtime = HtmlRuntime()
        planned = [
            {"x": 1.0, "y": 18.0, "z": 2.0},
            {"x": 2.0, "y": 18.0, "z": 2.0},
        ]

        result = runtime.execute(planned)

        self.assertEqual(list(result.trajectory), planned)
        self.assertGreater(result.trajectory_duration, 0)
        self.assertEqual(result.kinematics["source"], "html")

    def test_stop_and_close_are_idempotent(self):
        runtime = HtmlRuntime()

        runtime.stop()
        runtime.close()
        runtime.close()

        self.assertTrue(runtime.status()["connected"])


class AirSimRuntimeContractTest(unittest.TestCase):
    def test_reports_common_runtime_capabilities(self):
        runtime = AirSimRuntime()
        self.addCleanup(runtime.disconnect)

        status = runtime.status()

        self.assertEqual(runtime.backend_name, "airsim")
        self.assertTrue(status["supports_wind"])
        self.assertEqual(status["source"], "airsim")

    def test_execute_wraps_existing_flight_result(self):
        runtime = AirSimRuntime()
        self.addCleanup(runtime.disconnect)
        trajectory = [
            {"x": 1.0, "y": 18.0, "z": 2.0},
            {"x": 2.0, "y": 18.0, "z": 2.0},
        ]
        runtime.fly_trajectory = lambda points: {
            "trajectory": list(points),
            "trajectory_duration": 0.75,
            "kinematics": {"source": "airsim"},
        }

        result = runtime.execute(trajectory)

        self.assertEqual(list(result.trajectory), trajectory)
        self.assertEqual(result.trajectory_duration, 0.75)
        self.assertEqual(result.kinematics["source"], "airsim")

    def test_stop_during_execution_signals_the_active_flight(self):
        runtime = AirSimRuntime()
        self.addCleanup(runtime.disconnect)
        runtime._executing.set()

        runtime.stop()

        self.assertTrue(runtime._stop_requested.is_set())

    def test_active_flight_cancels_when_stop_is_requested(self):
        class CompletedFuture:
            def join(self):
                return None

        class Client:
            def __init__(self):
                self.cancelled = False
                self.hovered = False

            def cancelLastTask(self, vehicle_name):
                self.cancelled = True

            def hoverAsync(self, vehicle_name):
                self.hovered = True
                return CompletedFuture()

        runtime = AirSimRuntime()
        self.addCleanup(runtime.disconnect)
        runtime.client = Client()
        runtime.connected = True
        runtime.api = object()
        runtime._stop_requested.set()
        runtime._read_telemetry = lambda: {"position_grid": {"x": 1, "y": 18, "z": 2}}

        with self.assertRaisesRegex(AirSimRuntimeError, "stopped"):
            runtime._cancel_if_stopped()

        self.assertTrue(runtime.client.cancelled)
        self.assertTrue(runtime.client.hovered)

    def test_close_releases_airsim_api_control(self):
        class Transport:
            def close(self):
                return None

        class Client:
            def __init__(self):
                self.client = Transport()
                self.released_vehicle = None

            def enableApiControl(self, enabled, vehicle_name):
                if not enabled:
                    self.released_vehicle = vehicle_name

        runtime = AirSimRuntime()
        runtime.client = Client()
        runtime.connected = True
        runtime.armed = True

        runtime.close()

        self.assertEqual(runtime.client.released_vehicle, "Drone1")
        self.assertFalse(runtime.connected)


if __name__ == "__main__":
    unittest.main()
