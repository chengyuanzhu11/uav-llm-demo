import unittest

from uav_llm_demo import create_app
from uav_llm_demo.agents import MockLLMAgent
from uav_llm_demo.mission_service import MissionService
from tests.test_mission_service import FailingRuntime, RecordingRuntime


class RuntimeApiTest(unittest.TestCase):
    def test_app_starts_in_html_without_constructing_airsim(self):
        def forbidden_factory(backend, config):
            raise AssertionError("AirSim must be lazy")

        response = create_app(runtime_factory=forbidden_factory).test_client().get(
            "/api/state"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["runtime_backend"], "html")

    def test_step_runs_in_default_html_runtime(self):
        client = create_app().test_client()

        response = client.post("/api/step")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_backend"], "html")
        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(len(payload["history"]), 1)

    def test_old_simulator_routes_are_removed(self):
        client = create_app().test_client()

        for path in (
            "/api/simulator/config",
            "/api/simulator/stop",
            "/api/simulator/wind",
        ):
            with self.subTest(path=path):
                self.assertEqual(client.post(path, json={}).status_code, 404)

    def test_html_wind_is_an_unsupported_capability(self):
        response = create_app().test_client().post(
            "/api/runtime/wind",
            json={"north": 1, "east": 0, "down": 0},
        )

        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.get_json()["error"].startswith("Runtime error:"))

    def test_runtime_execution_failure_returns_503_without_commit(self):
        service = MissionService(
            agent=MockLLMAgent(),
            runtime=FailingRuntime(),
        )
        client = create_app(service=service).test_client()
        before = client.get("/api/state").get_json()

        response = client.post("/api/step")

        after = client.get("/api/state").get_json()
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.get_json()["error"].startswith("Runtime error:"))
        self.assertEqual(after["uav"], before["uav"])
        self.assertEqual(after["step_count"], before["step_count"])

    def test_switch_runtime_uses_nested_config_and_resets(self):
        airsim = RecordingRuntime("airsim")
        app = create_app(runtime_factory=lambda backend, config: airsim)
        client = app.test_client()
        client.post("/api/step")

        response = client.post(
            "/api/runtime",
            json={
                "backend": "airsim",
                "config": {"host": "127.0.0.1", "port": 41451},
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_backend"], "airsim")
        self.assertEqual(payload["step_count"], 0)

    def test_agent_config_does_not_echo_api_key(self):
        response = create_app().test_client().post(
            "/api/config",
            json={
                "backend": "deepseek",
                "api_key": "secret-for-test",
                "model": "deepseek-v4-flash",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["agent_backend"], "deepseek")
        self.assertNotIn("secret-for-test", str(payload))


if __name__ == "__main__":
    unittest.main()
