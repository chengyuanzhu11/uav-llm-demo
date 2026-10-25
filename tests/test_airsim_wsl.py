import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from uav_llm_demo import create_app, _build_runtime
from uav_llm_demo.airsim_connection import default_airsim_host, validate_airsim_host, windows_path
from uav_llm_demo.airsim_runtime import AirSimRuntime, AirSimRuntimeConfig
from uav_llm_demo.scenarios import SCENARIOS


class AirSimWslTest(unittest.TestCase):
    def test_gateway_and_local_hosts_only(self):
        with patch('uav_llm_demo.airsim_connection.windows_host', return_value='172.25.16.1'), \
             patch('uav_llm_demo.airsim_connection.is_wsl', return_value=True), \
             patch.dict('os.environ', {}, clear=True):
            self.assertEqual(default_airsim_host(), '172.25.16.1')
            self.assertEqual(validate_airsim_host('localhost'), 'localhost')
            for host in ['0.0.0.0', '8.8.8.8', '192.168.1.100', 'https://127.0.0.1']:
                with self.assertRaises(ValueError):
                    validate_airsim_host(host)

    def test_frontend_default_does_not_override_environment(self):
        with patch.dict('os.environ', {'AIRSIM_HOST':'127.0.0.1', 'AIRSIM_PORT':'41452', 'AIRSIM_VEHICLE_NAME':'TestDrone'}):
            app = create_app()
            self.addCleanup(app.extensions['mission_service'].close)
            page = app.test_client().get('/').get_data(as_text=True)
            runtime = _build_runtime('airsim', {})
            self.addCleanup(runtime.close)
            self.assertIn('id="airsimHost" value="127.0.0.1"', page)
            self.assertIn('value="41452"', page)
            self.assertEqual((runtime.config.host, runtime.config.port, runtime.config.vehicle_name),
                             ('127.0.0.1', 41452, 'TestDrone'))

    def test_connect_and_close_never_acquire_control(self):
        client = Mock()
        client.ping.return_value = True
        api = SimpleNamespace(MultirotorClient=Mock(return_value=client))
        runtime = AirSimRuntime(AirSimRuntimeConfig(host='127.0.0.1'))
        with patch('uav_llm_demo.airsim_runtime.import_airsim_client', return_value=(api, 'cosysairsim')), \
             patch.object(runtime, '_read_telemetry', return_value={}):
            runtime.connect()
            runtime.close()
        client.enableApiControl.assert_not_called()
        client.armDisarm.assert_not_called()
        client.takeoffAsync.assert_not_called()
        client.reset.assert_not_called()
        client.client.close.assert_called_once()

    def test_failed_texture_preflight_preserves_existing_scene(self):
        runtime = AirSimRuntime(AirSimRuntimeConfig(host='127.0.0.1'))
        self.addCleanup(runtime.close)
        runtime.client = Mock()
        runtime.api = Mock()
        runtime.connected = True
        runtime.client.simListAssets.return_value = ['Cube']
        with patch('uav_llm_demo.airsim_runtime.texture_directory', side_effect=PermissionError('unwritable')):
            with self.assertRaises(PermissionError):
                runtime.configure_scene(SCENARIOS['urban_crossing'])
        runtime.client.simDestroyObject.assert_not_called()
        runtime.client.simSpawnObject.assert_not_called()

    def test_linux_private_texture_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('uav_llm_demo.airsim_connection.is_wsl', return_value=True), \
             patch('uav_llm_demo.airsim_connection.subprocess.run', return_value=SimpleNamespace(stdout='\\\\wsl.localhost\\Ubuntu\\tmp\\textures')):
            with self.assertRaises(ValueError):
                windows_path(Path(folder))

    def test_only_small_pad_contact_errors_are_normalized(self):
        try:
            import cosysairsim as api
        except ImportError:
            self.skipTest("Optional AirSim client is not installed")
        runtime = AirSimRuntime(AirSimRuntimeConfig(host='127.0.0.1'))
        self.addCleanup(runtime.close)
        runtime.api = api
        runtime.connected = True
        runtime._scene = SCENARIOS['urban_crossing']
        runtime._origin = (500.0, 500.0, -100.0)
        state = api.MultirotorState()
        runtime.client = Mock()
        runtime.client.getMultirotorState.return_value = state
        for below, landed, expected in [(0.035, api.LandedState.Landed, 0.0),
                                        (0.2, api.LandedState.Landed, -0.04),
                                        (0.035, api.LandedState.Flying, 0.0)]:
            with self.subTest(below=below, landed=landed):
                state.landed_state = landed
                state.kinematics_estimated.position = api.Vector3r(500, 500, -100 + below)
                telemetry = runtime.read_telemetry()
                self.assertAlmostEqual(telemetry['position_grid']['z'], expected)
                self.assertAlmostEqual(telemetry['position_ned']['z'], -100 + below)

        state.kinematics_estimated.position = api.Vector3r(500, 503, -99.965)
        self.assertAlmostEqual(runtime.read_telemetry()['position_grid']['z'], -0.007)
        state.kinematics_estimated.position = api.Vector3r(500, 500, -99.965)
        state.kinematics_estimated.linear_velocity = api.Vector3r(0, 0, 0.05)
        self.assertEqual(runtime.read_state().velocity[2], 0.0)
        self.assertEqual(runtime.read_telemetry()['linear_velocity']['z'], 0.05)
