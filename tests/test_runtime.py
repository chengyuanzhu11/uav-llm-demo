import unittest
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




if __name__ == "__main__":
    unittest.main()
