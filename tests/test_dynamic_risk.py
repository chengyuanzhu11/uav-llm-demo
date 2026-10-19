import unittest
from types import SimpleNamespace

from uav_llm_demo.dynamic_risk import (AVOIDANCE_BUFFER_M, contact_distance_m,
                                     physical_radius_m, risk_levels, separation_distance_m)
from uav_llm_demo.environment import DynamicFlyingObject, UAVEnvironment


class DynamicRiskTest(unittest.TestCase):
    def test_legacy_radius_keeps_existing_safety_threshold(self):
        bird = DynamicFlyingObject("bird", "bird", (1, 1, 2), (0, 0, 0), radius=.7)
        self.assertAlmostEqual(physical_radius_m(bird), 2.1)
        self.assertAlmostEqual(contact_distance_m(physical_radius_m(bird)), 2.5)
        self.assertAlmostEqual(separation_distance_m(physical_radius_m(bird)), 3.1)
        self.assertEqual(AVOIDANCE_BUFFER_M, .3)
        data = bird.to_dict()
        self.assertEqual(data["radius"], .7)
        self.assertEqual(data["radius_unit"], "horizontal_grid_cell")
        self.assertAlmostEqual(data["radius_m"], 2.1)

    def test_explicit_metric_radius_and_invalid_values(self):
        self.assertEqual(physical_radius_m(SimpleNamespace(radius=.7, radius_m=1.2)), 1.2)
        for invalid in (-1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                physical_radius_m(SimpleNamespace(radius_m=invalid))

    def test_metric_distance_respects_vertical_scale(self):
        horizontal = risk_levels((1, 0, 0), (0, 0, 0), (0, 0, 0), radius_m=0)
        vertical = risk_levels((0, 0, 1), (0, 0, 0), (0, 0, 0), radius_m=0)
        self.assertEqual(horizontal["distance"], 3)
        self.assertEqual(vertical["distance"], 5)
        self.assertEqual(vertical["distance_unit"], "m")

    def test_relative_motion_distinguishes_approaching_from_receding(self):
        toward = risk_levels((3, 0, 0), (0, 0, 0), (-1, 0, 0), radius_m=.5)
        away = risk_levels((3, 0, 0), (0, 0, 0), (1, 0, 0), radius_m=.5)
        self.assertEqual(toward["closest_approach_seconds"], 3)
        self.assertEqual(toward["predicted_distance"], 0)
        self.assertEqual(toward["threat_level"], "high")
        self.assertEqual(away["predicted_distance"], 9)
        self.assertEqual(away["threat_level"], "low")
        self.assertEqual(away["closing_speed_mps"], -3)

    def test_uav_velocity_is_metric_and_participates_in_relative_risk(self):
        co_moving = risk_levels((3, 0, 0), (0, 0, 0), (1, 0, 0), (3, 0, 0), radius_m=.5)
        approaching = risk_levels((3, 0, 0), (0, 0, 0), (1, 0, 0), (6, 0, 0), radius_m=.5)
        self.assertEqual(co_moving["relative_speed_mps"], 0)
        self.assertEqual(co_moving["predicted_distance"], 9)
        self.assertEqual(approaching["predicted_distance"], 0)

    def test_contact_and_separation_risk_are_distinct(self):
        contact = risk_levels((.5, 0, 0), (0, 0, 0), (0, 0, 0), radius_m=2.1)
        separation_only = risk_levels((.95, 0, 0), (0, 0, 0), (0, 0, 0), radius_m=2.1)
        self.assertEqual(contact["threat_level"], "critical")
        self.assertEqual(separation_only["threat_level"], "high")
        self.assertGreater(separation_only["contact_clearance_m"], 0)
        self.assertLess(separation_only["separation_clearance_m"], 0)

    def test_environment_uses_vehicle_metric_velocity_and_declares_units(self):
        env = UAVEnvironment()
        env.position = (10, 10, 2)
        env.dynamic_objects = [DynamicFlyingObject("bird", "bird", (13, 10, 2), (1, 0, 0))]
        env.kinematics["linear_velocity"] = {"x": 6, "y": 0, "z": 0}
        observed = env.get_observation().dynamic_objects[0]
        self.assertEqual(observed["distance"], 9)
        self.assertEqual(observed["predicted_distance"], 0)
        self.assertEqual(observed["distance_unit"], "m")
        self.assertEqual(observed["risk_prediction"], "relative_constant_velocity")

    def test_nearest_current_risk_uses_clearance_not_future_crossing(self):
        env = UAVEnvironment()
        env.position = (10, 10, 2)
        env.dynamic_objects = [DynamicFlyingObject("near", "bird", (11, 10, 2), (1, 0, 0)),
                               DynamicFlyingObject("future", "bird", (13, 10, 2), (-1, 0, 0))]
        observation = env.get_observation()
        self.assertEqual(observation.nearest_dynamic_obstacle["id"], "near")
        self.assertEqual(observation.dynamic_objects[1]["predicted_distance_m"], 0)


class DynamicRiskFrontendTest(unittest.TestCase):
    def test_current_risk_uses_backend_thresholds_and_labels_units(self):
        from tests.test_frontend_contract import FrontendContractTest
        FrontendContractTest.run_frontend_functions(self,
            ["currentDynamicRiskLevel", "dynamicDistanceM", "formatDynamicRisk", "formatNumber", "isRecord"], r"""
const envelope = {contact_distance_m: 2.5, separation_distance_m: 3.1, near_miss_distance_m: 4.1};
assert.equal(currentDynamicRiskLevel(2.4, envelope), 'critical');
assert.equal(currentDynamicRiskLevel(2.8, envelope), 'high');
assert.equal(currentDynamicRiskLevel(3.6, envelope), 'medium');
assert.equal(currentDynamicRiskLevel(5, envelope), 'low');
assert.equal(currentDynamicRiskLevel(null, envelope), 'unknown');
assert.equal(currentDynamicRiskLevel(2, {}), 'unknown');
assert.equal(dynamicDistanceM({distance: 2}), null);
assert.equal(dynamicDistanceM({distance: 6, distance_unit: 'm'}), 6);
const risk = formatDynamicRisk({...envelope, id:'bird', distance_m: 5, predicted_distance_m: 0,
  threat_level:'critical', separation_clearance_m:1.9}, true);
assert.match(risk, /当前 low/);
assert.match(risk, /中心距 5.00 m/);
assert.match(risk, /安全净空 1.90 m/);
assert.doesNotMatch(risk, /critical|predicted/);
""")

    def test_threejs_envelopes_use_metric_api_radii_and_update_in_place(self):
        import json
        from pathlib import Path
        from tests.test_frontend_contract import FrontendContractTest
        vendor = (Path(__file__).resolve().parents[1] / "uav_llm_demo/static/vendor/three.module.js").as_uri()
        FrontendContractTest.run_frontend_functions(self,
            ["createBird", "updateBirdEnvelope"],
            "const THREE = await import(" + json.dumps(vendor) + ");\n" + r"""
const WORLD_SCALE = 3;
const bird = createBird({id:'bird', radius:.7, radius_m:2.1,
  contact_distance_m:2.5, separation_distance_m:3.1, near_miss_distance_m:4.1});
assert.equal(bird.userData.contactBubble.scale.x, 2.5);
assert.equal(bird.userData.safetyBubble.scale.x, 3.1);
assert.equal(bird.userData.contactBubble.geometry.parameters.radius, 1);
assert.equal(bird.userData.safetyBubble.geometry.parameters.radius, 1);
assert.equal(bird.userData.anatomy.scale.x, 2.1/1.65);
const originalGeometry = bird.userData.safetyBubble.geometry;
updateBirdEnvelope(bird, {radius_m:1, contact_distance_m:1.4, separation_distance_m:2,
  near_miss_distance_m:3});
assert.equal(bird.userData.safetyBubble.scale.x, 2);
assert.equal(bird.userData.safetyBubble.geometry, originalGeometry);
updateBirdEnvelope(bird, {radius:.7});
assert.equal(bird.userData.contact_distance_m, null);
assert.equal(bird.userData.safetyBubble.scale.x, 0);
assert.equal(bird.userData.anatomy.scale.x, (.7*3)/1.65);
""")


if __name__ == "__main__":
    unittest.main()
