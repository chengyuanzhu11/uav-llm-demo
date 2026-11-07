"""Focused checks for new adapter boundaries; real flight is verify_gazebo.py."""
from collections import deque
from types import SimpleNamespace
import math
import threading
import time
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET

from uav_llm_demo.gazebo_runtime import GazeboRuntime, GazeboRuntimeConfig
from uav_llm_demo.runtime import RuntimeStateError
from uav_llm_demo.scenarios import load_scenario


class GazeboRuntimeTests(unittest.TestCase):
    def test_campus_world_uses_shared_visuals_and_solid_geometry(self):
        from scripts.build_gazebo_scene import build
        from uav_llm_demo.simulation_scene import campus_collision_geometry
        campus = load_scenario("semantic_campus")
        self.assertEqual(campus.gazebo_world, campus.scenario_id)
        self.assertIsNone(load_scenario("urban_crossing").gazebo_world)
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.build_gazebo_scene.export_visual", return_value=Path(directory)/'scenery.glb'):
                root = ET.parse(build(campus.scenario_id, directory))
            models = {model.attrib['name']: model for model in root.findall('world/model')}
            self.assertEqual(len(root.findall('.//visual')), 1)
            self.assertTrue(models['campus_visual'].find('link/visual/geometry/mesh/uri').text.endswith('/scenery.glb'))
            for solid in campus_collision_geometry(campus.campus_layout):
                model = models[solid['name']]
                pose = [float(v) for v in model.findtext('pose').split()[:3]]
                expected = [solid['position'][axis]*scale for axis,scale in zip('xyz', (3,-3,5))]
                self.assertEqual(pose, expected)
                self.assertEqual([float(v) for v in model.findtext('link/collision/geometry/box/size').split()],
                                 list(solid['size'].values()))
            floor = models['ground']
            self.assertAlmostEqual(float(floor.findtext('pose').split()[2])+
                float(floor.findtext('link/collision/geometry/box/size').split()[2])/2, 0)

    def test_world_pose_is_independent_of_px4_estimator_drift_and_stale_link_fails(self):
        from uav_llm_demo.runtime import RuntimeUnavailableError
        runtime = GazeboRuntime()
        runtime._scene = load_scenario("visual_garden")
        runtime._ground_down = -.25
        runtime._position = SimpleNamespace(x=6, y=9, z=-5.25, vx=2, vy=3, vz=-1)
        runtime._last_position = time.monotonic()
        runtime._landed = False
        runtime._attitude = SimpleNamespace(yaw=.4)
        runtime._ground_pose_z=-.01
        runtime.camera=SimpleNamespace(vehicle_state=lambda: ((15,-30,4.99),(3,2,1),None,3.25))
        state = runtime.read_state()
        self.assertEqual(state.position, (5, 10, 1))
        self.assertEqual(state.velocity, (3, -2, 1))
        self.assertEqual(state.yaw, .4)
        runtime._position.z=100  # Estimator reset/drift is not physical movement.
        self.assertEqual(runtime.read_state(),state)
        runtime._last_position -= 3
        with self.assertRaises(RuntimeUnavailableError):
            runtime.read_state()

    def test_world_mismatch_and_invalid_config_fail_before_connect(self):
        runtime = GazeboRuntime()
        with self.assertRaises(RuntimeStateError):
            runtime.configure_scene(load_scenario("semantic_campus"))
        self.assertIsNone(runtime._mav)
        for config in ({"port": True}, {"world": None}, {"vehicle_name": "../bad"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                GazeboRuntimeConfig(**config)

    def test_camera_pose_composes_parent_transforms_and_uses_child_timestamp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Install requirements-gazebo.txt for camera calibration checks")
        from uav_llm_demo.gazebo_camera import GazeboCamera
        camera = object.__new__(GazeboCamera)
        camera.world = "visual_garden"
        camera.vehicle = "vehicle"
        camera.time = 3.25
        camera._clock_published = time.monotonic()
        camera.vehicle_sample = None
        camera._vehicle_path = deque(maxlen=4)
        camera._poses = deque(maxlen=4)
        camera._condition = threading.Condition()

        def pose(name, parent, position, yaw):
            return SimpleNamespace(name=name, header=SimpleNamespace(
                stamp=SimpleNamespace(sec=3, nsec=250_000_000),
                data=[SimpleNamespace(key="frame_id", value=[parent]),
                      SimpleNamespace(key="child_frame_id", value=[name])]),
                position=SimpleNamespace(**dict(zip("xyz", position))),
                orientation=SimpleNamespace(x=0, y=0, z=math.sin(yaw/2), w=math.cos(yaw/2)))

        camera._pose(SimpleNamespace(pose=[
            pose("vehicle", "visual_garden", (10, 20, 5), math.pi/2),
            pose("vehicle::base_link", "vehicle", (0, 0, .25), 0),
            pose("vehicle::camera_link", "vehicle::base_link", (1, 0, .1), 0)]))
        stamp, point, rotation, _ = camera._poses[-1]
        self.assertEqual(stamp, 3.25)
        np.testing.assert_allclose(point, (10, 21, 5.35), atol=1e-9)
        np.testing.assert_allclose(rotation[:, 0], (0, 1, 0), atol=1e-9)
        from uav_llm_demo.runtime import RuntimeUnavailableError
        self.assertEqual(camera.vehicle_state()[3], 3.25)
        camera.time += .3  # A freshly delivered callback can contain old geometry.
        with self.assertRaises(RuntimeUnavailableError):
            camera.vehicle_state()
        camera.time = 3.25
        camera._clock_published -= 1  # Pose and clock backed up together.
        with self.assertRaises(RuntimeUnavailableError):
            camera.vehicle_state()
        # Late samples document the flown path but must not renew a command.
        body = pose("vehicle", "visual_garden", (11, 20, 5), 0)
        body.header.stamp.nsec = 500_000_000
        camera.time = 4.
        camera._pose(SimpleNamespace(pose=[body]))
        self.assertEqual([t for t,_ in camera.vehicle_path(3.25, 4.)], [3.5])
        from uav_llm_demo.runtime import StaleFeedbackError
        with self.assertRaises(StaleFeedbackError):
            camera.vehicle_state()
        camera.vehicle_sample_received -= 1.1
        with self.assertRaises(RuntimeUnavailableError) as loss:
            camera.vehicle_state()
        self.assertNotIsInstance(loss.exception, StaleFeedbackError)
