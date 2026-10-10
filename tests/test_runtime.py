import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace

from uav_llm_demo.airsim_runtime import (
    AirSimRuntime,
    AirSimRpcWorker,
    grid_to_ned,
    ned_to_grid,
)
from uav_llm_demo.html_runtime import HtmlRuntime


class HtmlRuntimeTest(unittest.TestCase):
    def test_is_connected_without_an_external_process(self):
        runtime = HtmlRuntime()

        status = runtime.status()

        self.assertEqual(runtime.backend_name, "html")
        self.assertTrue(status["connected"])
        self.assertFalse(status["supports_wind"])

    def test_drive_updates_fractional_position_and_velocity(self):
        runtime = HtmlRuntime()

        first = runtime.drive((3.0, 0.0, 0.0), 0.5)
        second = runtime.drive((3.0, 0.0, 0.0), 0.5)

        self.assertAlmostEqual(first.position[0], 1.25)
        self.assertAlmostEqual(second.position[0], 1.75)
        self.assertEqual(second.velocity, (3.0, 0.0, 0.0))
        self.assertEqual(second.acceleration, (0.0, 0.0, 0.0))

    def test_stop_and_close_are_idempotent(self):
        runtime = HtmlRuntime()

        runtime.stop()
        runtime.close()
        runtime.close()

        self.assertTrue(runtime.status()["connected"])


class AirSimRuntimeContractTest(unittest.TestCase):
    def test_reports_common_runtime_capabilities_without_importing_client(self):
        runtime = AirSimRuntime()
        self.addCleanup(runtime.close)

        status = runtime.status()

        self.assertEqual(runtime.backend_name, "airsim")
        self.assertTrue(status["supports_wind"])
        self.assertEqual(status["source"], "airsim")
        self.assertFalse(status["connected"])

    def test_stop_cancels_current_velocity_lease_and_hovers(self):
        class Client:
            def __init__(self):
                self.cancelled = False
                self.hovered = False

            def cancelLastTask(self, vehicle_name):
                self.cancelled = True

            def hoverAsync(self, vehicle_name):
                self.hovered = True

        runtime = AirSimRuntime()
        self.addCleanup(runtime.close)
        runtime.client = Client()
        runtime.api = SimpleNamespace()
        runtime.connected = True

        runtime.stop()

        self.assertTrue(runtime.client.cancelled)
        self.assertTrue(runtime.client.hovered)

    def test_disconnect_releases_transport_without_remote_hover(self):
        class Transport:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class Client:
            def __init__(self):
                self.client = Transport()
                self.remote_call_attempted = False

        runtime = AirSimRuntime()
        client = Client()
        runtime.client = client
        runtime.connected = True
        runtime.armed = True

        runtime.disconnect()

        self.assertFalse(client.remote_call_attempted)
        self.assertTrue(client.client.closed)
        self.assertFalse(runtime.connected)
        self.assertFalse(runtime.armed)

    def test_rpc_worker_serializes_calls_on_one_thread(self):
        worker = AirSimRpcWorker()
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                execution_threads = list(callers.map(lambda _: worker.call(get_ident), range(2)))
        finally:
            worker.close()

        self.assertEqual(execution_threads[0], execution_threads[1])
        self.assertNotEqual(execution_threads[0], get_ident())

    def test_rpc_worker_cancels_pending_async_task_when_closed(self):
        worker = AirSimRpcWorker()
        pending_task = worker.call(
            lambda: asyncio.get_event_loop().create_task(asyncio.sleep(60))
        )

        worker.close()

        self.assertTrue(pending_task.cancelled())

    def test_grid_and_ned_coordinates_round_trip(self):
        point = (13.25, 7.5, 4.2)

        ned = grid_to_ned(point, meters_per_cell=2.5, meters_per_altitude_level=4.0)
        restored = ned_to_grid(ned, meters_per_cell=2.5, meters_per_altitude_level=4.0)

        for expected, actual in zip(point, restored):
            self.assertAlmostEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
