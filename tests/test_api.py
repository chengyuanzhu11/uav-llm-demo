import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from uav_llm_demo import create_app
from uav_llm_demo.agents import MockLLMAgent
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.gazebo_runtime import GazeboRuntime
from uav_llm_demo.motion import VehicleState
from uav_llm_demo.scenarios import load_scenario
from tests.test_mission_service import FailingRuntime, RecordingRuntime


def wait_for(service, predicate, limit=300):
    for _ in range(limit):
        time.sleep(0.001)
        service.advance(service.CONTROL_DT)
        state = service.state()
        if predicate(state):
            return state
    return service.state()


class RuntimeApiTest(unittest.TestCase):
    def make_app(self, service=None, **kwargs):
        app = create_app(service=service, **kwargs)
        self.addCleanup(app.extensions["mission_service"].close)
        return app

    def test_app_starts_in_html_without_constructing_gazebo(self):
        def forbidden_factory(backend, config):
            raise AssertionError("Gazebo must be lazy")

        app = self.make_app(runtime_factory=forbidden_factory)
        response = app.test_client().get("/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["runtime_backend"], "html")

    def test_discrete_step_endpoint_is_removed(self):
        response = self.make_app().test_client().post("/api/step")
        self.assertEqual(response.status_code, 404)

    def test_retired_crossing_demo_endpoint_cannot_replace_current_mission(self):
        app = self.make_app()
        service = app.extensions["mission_service"]
        client = app.test_client()
        before = service.state()
        run_id = service.trace.run_id
        environment = service._environment

        response = client.post("/api/demo/crossing", json={})

        self.assertEqual(response.status_code, 404)
        after = service.state()
        self.assertIs(service._environment, environment)
        self.assertEqual(service.trace.run_id, run_id)
        # Idle bird traffic advances with wall time; only mission state must stay fixed.
        for field in ("scenario_id", "agent_backend", "uav", "target", "base", "phase"):
            self.assertEqual(after[field], before[field])
        for field in ("running", "planner_calls", "plan_id", "waypoints"):
            self.assertEqual(after["flight"][field], before["flight"][field])
        self.assertNotIn("demo", before)

    def test_mission_start_returns_planning_state(self):
        app = self.make_app()
        response = app.test_client().post("/api/mission/start")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_backend"], "html")
        self.assertTrue(payload["flight"]["running"])
        self.assertGreaterEqual(payload["flight"]["planner_calls"], 1)

    def test_retired_physics_controls_cannot_reset_or_stop_the_runtime(self):
        app = self.make_app()
        service = app.extensions["mission_service"]
        runtime, run_id = service.runtime, service.trace.run_id
        with patch.object(runtime, "stop") as stop, patch.object(runtime, "reset") as reset:
            response = app.test_client().post("/api/runtime", json={"backend": "physics"})
            self.assertEqual(response.status_code, 400)
            self.assertIn("'html' or 'gazebo'", response.get_json()["error"])
            response = app.test_client().post("/api/runtime/wind", json={"north": 1})
            self.assertEqual(response.status_code, 410)
            self.assertEqual(response.get_json()["error_code"], "runtime_removed")
            stop.assert_not_called()
            reset.assert_not_called()
        self.assertIs(service.runtime, runtime)
        self.assertEqual(service.trace.run_id, run_id)

    def test_runtime_failure_is_reported_without_commit(self):
        service = MissionService(agent=MockLLMAgent(), runtime=FailingRuntime())
        app = self.make_app(service=service)
        client = app.test_client()
        before = client.get("/api/state").get_json()
        client.post("/api/mission/start")

        after = wait_for(service, lambda state: state["flight"]["mode"] == "error")
        self.assertEqual(after["uav"], before["uav"])
        self.assertEqual(after["step_count"], before["step_count"])

    def test_native_state_conflicts_preserve_flight_and_reset_succeeds_after_landing(self):
        # Exercise the real adapter guards and API, without a transport or model call.
        runtime = GazeboRuntime()
        scenario = load_scenario("visual_garden")
        runtime._scene = scenario
        runtime._mav = Mock()
        runtime.camera = SimpleNamespace(time=20., close=Mock())
        runtime.read_state = Mock(return_value=VehicleState(scenario.base))
        runtime.read_telemetry = Mock(return_value={})
        runtime.connected = True
        runtime._last_message = time.monotonic()
        runtime.stop = Mock(wraps=runtime.stop)
        factory = Mock(side_effect=AssertionError("Must reuse the existing Gazebo connection"))
        service = MissionService(runtime=runtime, runtime_factory=factory, scenario_id="visual_garden")
        client = self.make_app(service=service).test_client()
        service._running = True
        service._publish_state()
        before = service.state()
        seed = service._traffic_seed
        runtime._command = (1., .2, 0.)
        for landed, armed, position, code in (
            (False, True, (2, 12, 1), "runtime_requires_landed"),
            (True, True, scenario.base, "runtime_requires_landed"),
            (None, False, scenario.base, "runtime_requires_landed"),
            (True, False, (5, 12, 0), "runtime_requires_home"),
        ):
            with self.subTest(landed=landed, armed=armed, position=position):
                runtime._landed, runtime.armed = landed, armed
                runtime.read_state.return_value = VehicleState(position)
                response = client.post("/api/reset")
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.get_json()["error_code"], code)
                if code == "runtime_requires_landed":
                    response = client.post("/api/runtime", json={"backend": "html"})
                    self.assertEqual(response.status_code, 409)
                    self.assertEqual(response.get_json()["error_code"], code)
                self.assertEqual(service.state(), before)
                self.assertEqual(service._traffic_seed, seed)
                runtime.stop.assert_not_called()
                self.assertEqual(runtime._command, (1., .2, 0.))
        response = client.post("/api/scenario", json={"scenario_id": "semantic_campus"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error_code"], "runtime_world_mismatch")
        self.assertEqual(service.state(), before)
        runtime.stop.assert_not_called()

        runtime._landed, runtime.armed = True, False
        runtime.read_state.return_value = VehicleState(scenario.base)
        response = client.post("/api/runtime", json={"backend": "gazebo", "config": {
            "world": "visual_garden", "port": 14540, "vehicle_name": "x500_rgbd_0"}})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["flight"]["running"])
        self.assertEqual(runtime._command, (0., 0., 0.))
        runtime.stop.assert_called_once()
        factory.assert_not_called()
        self.assertEqual(client.post("/api/runtime", json={"backend": "html"}).status_code, 200)

    def test_switch_runtime_uses_nested_config_and_resets(self):
        gazebo = RecordingRuntime("gazebo")
        app = self.make_app(runtime_factory=lambda backend, config: gazebo)
        client = app.test_client()

        response = client.post(
            "/api/runtime",
            json={
                "backend": "gazebo",
                "config": {"world": "visual_garden", "port": 14540},
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_backend"], "gazebo")
        self.assertEqual(payload["step_count"], 0)

    def test_agent_config_does_not_echo_api_key(self):
        response = self.make_app().test_client().post(
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
        self.assertTrue(payload["deepseek_key_configured"])
        self.assertNotIn("secret-for-test", str(payload))

    def test_agent_config_reuses_process_key_without_resetting_mission(self):
        service = MissionService(agent=MockLLMAgent())
        app = self.make_app(service=service)
        client = app.test_client()
        # Drive ticks explicitly: background flight may legitimately move between HTTP requests.
        with patch.object(service, "_ensure_loop"):
            client.post("/api/mission/start")
        wait_for(service, lambda state: state["step_count"] > 0)
        before = client.get("/api/state").get_json()

        selected = client.post(
            "/api/config",
            json={
                "backend": "deepseek",
                "api_key": "secret-for-test",
                "model": "deepseek-v4-flash",
            },
        )
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.get_json()["step_count"], before["step_count"])
        self.assertEqual(selected.get_json()["uav"], before["uav"])

        mock_state = client.post("/api/config", json={"backend": "mock"})
        self.assertEqual(mock_state.status_code, 200)
        self.assertEqual(mock_state.get_json()["agent_backend"], "mock")

        reused = client.post("/api/config", json={"backend": "deepseek"})
        self.assertEqual(reused.status_code, 200)
        self.assertEqual(reused.get_json()["agent_backend"], "deepseek")
        self.assertEqual(service.agent.api_key, "secret-for-test")
        self.assertNotIn("secret-for-test", str(reused.get_json()))

    def test_agent_config_can_initialize_from_environment_key(self):
        with patch.dict(
            "os.environ",
            {
                "DEEPSEEK_API_KEY": "environment-secret",
                "DEEPSEEK_MODEL": "environment-model",
            },
        ):
            service = MissionService(agent=MockLLMAgent())
            app = self.make_app(service=service)
            response = app.test_client().post("/api/config", json={"backend": "deepseek"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.agent.model, "environment-model")
        self.assertNotIn("environment-secret", str(response.get_json()))

    def test_agent_config_missing_key_is_atomic(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            service = MissionService(agent=MockLLMAgent())
            app = self.make_app(service=service)
            client = app.test_client()
            before = client.get("/api/state").get_json()

            response = client.post("/api/config", json={"backend": "deepseek"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(service.agent.backend_name, "mock")
        self.assertEqual(client.get("/api/state").get_json()["agent_backend"], before["agent_backend"])

    def test_agent_config_rejects_unimplemented_backend(self):
        app = self.make_app(service=MissionService(agent=MockLLMAgent()))
        response = app.test_client().post("/api/config", json={"backend": "openai"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(app.extensions["mission_service"].agent.backend_name, "mock")

    def test_planning_trace_records_the_agent_that_produced_it(self):
        service = MissionService(agent=MockLLMAgent())
        app = self.make_app(service=service)
        client = app.test_client()
        client.post("/api/mission/start")
        state = wait_for(service, lambda item: item["flight"]["plan_id"] == 1)

        self.assertEqual(state["history"][0]["decision"]["agent_backend"], "mock")
        self.assertEqual(state["evaluation"]["events"][0]["agent_backend"], "mock")


if __name__ == "__main__":
    unittest.main()
