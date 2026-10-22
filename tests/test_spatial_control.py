"""Reusable control goals must come from geometry and verified support surfaces."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json
import shutil
import subprocess
import unittest

from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.safety import GROUND_COLUMN_RADIUS_M, SAFETY_MARGIN
from uav_llm_demo.spatial_control import (
    CONTROL_DEFAULTS, LANDING_CLEARANCE_M, control_catalog, normalize_control,
    resolve_control_goal,
)


class SpatialControlTests(unittest.TestCase):
    def setUp(self):
        self.scene = load_scenario("semantic_campus")
        self.catalog = control_catalog(self.scene)

    def goal(self, target="north_football", constraints=None, scenario=None, **control):
        return resolve_control_goal({"id": "control", "action": "control", "target_id": target,
                                     "control": control, "when": None}, scenario or self.scene,
                                    constraints or {})

    def test_every_place_including_home_has_real_center_and_arrival(self):
        objects = {item["id"]: item for item in self.catalog["objects"]}
        self.assertEqual(set(objects), {point["id"] for point in self.scene.points_of_interest})
        self.assertEqual(objects["library"]["center_grid"], [54, 67, 0])
        self.assertEqual(objects["library"]["arrival_grid"], [53, 74, 3])
        self.assertEqual(objects["north_football"]["surface_ids"], ["north-football"])
        self.assertEqual(len(self.catalog["surfaces"]), len(self.scene.campus_layout["surfaces"]))
        objects["library"]["center_grid"][0] = -1
        self.assertEqual(self.scene.campus_layout["spatial_objects"]["library"]["center_grid"], [54, 67])

    def test_catalog_does_not_require_execution_geometry_in_minimal_fixture(self):
        point = {"id": "test", "label": "test", "position": {"x": 1, "y": 2, "z": 3}}
        scene = SimpleNamespace(min_altitude=2, max_altitude=10, points_of_interest=[point])
        catalog = control_catalog(scene)
        self.assertEqual(catalog["objects"][0]["arrival_grid"], [1, 2, 3])
        self.assertEqual(catalog["surfaces"], [])

    def test_defaults_and_null_optional_values_are_normalized_without_mutation(self):
        raw = {"duration_seconds": None, "altitude_m": None, "yaw_deg": None}
        self.assertEqual(normalize_control(raw, "north_football", {"control": self.catalog}, {}), CONTROL_DEFAULTS)
        self.assertEqual(raw, {"duration_seconds": None, "altitude_m": None, "yaw_deg": None})
        self.assertEqual(normalize_control(None, "home", self.catalog, {}), CONTROL_DEFAULTS)

    def test_landing_reuses_flat_support_across_sports_and_plaza(self):
        expected = {"north_football": (42, 22.5, 0), "north_basketball": (44, 13.5, 0),
                    "south_plaza": (48.5, 82, 0), "home": (48, 88, 0)}
        for target, position in expected.items():
            with self.subTest(target=target):
                goal = self.goal(target, state="landed", duration_seconds=12)
                self.assertEqual(goal, {"position": position, "ground_site": position,
                                        "state": "landed", "duration_seconds": 12.0})

    def test_unseen_surface_and_place_identifiers_work_without_named_rules(self):
        layout = deepcopy(self.scene.campus_layout)
        for collection in (layout["places"], layout["surfaces"]):
            for item in collection:
                if item["id"] in ("north_football", "north-football"):
                    item["id"] = "independent-test-ground"
                if item.get("surface_ids") == ["north-football"]:
                    item["surface_ids"] = ["independent-test-ground"]
        layout["spatial_objects"]["independent-test-ground"] = layout["spatial_objects"].pop("north_football")
        points = deepcopy(self.scene.points_of_interest)
        for point in points:
            if point["id"] == "north_football":
                point["id"] = "independent-test-ground"
        scene = replace(self.scene, campus_layout=layout, points_of_interest=points)
        self.assertEqual(self.goal("independent-test-ground", scenario=scene, state="landed")["position"], (42, 22.5, 0))

    def test_offset_uses_metres_and_exact_position_without_relocation(self):
        result = self.goal(state="landed", offset_m=[1.5, -3, 0])
        self.assertEqual(result["position"], (42.5, 21.5, 0))
        result = self.goal(state="airborne", altitude_m=15, offset_m=[3, 0, 2])
        self.assertEqual(result["position"], (43, 22.5, 3.4))

    def test_arrival_anchor_preserves_existing_observation_coordinates(self):
        result = self.goal("library", anchor="arrival", altitude_m=15)
        self.assertEqual(result["position"], (53, 74, 3))
        self.assertIsNone(result["ground_site"])

    def test_omitted_altitude_clears_shared_roof_but_explicit_height_is_not_rewritten(self):
        default = self.goal("library")
        self.assertGreater(default["position"][2] * 5, 18 + LANDING_CLEARANCE_M)
        self.assertEqual(default["position"][:2], (54, 67))
        with self.assertRaisesRegex(ValueError, "building"):
            self.goal("library", altitude_m=15)
        with self.assertRaisesRegex(ValueError, "ceiling"):
            self.goal("library", constraints={"max_altitude_m": 18})

    def test_rooftop_and_canopy_landings_are_rejected(self):
        for target in ("library", "central_garden", "south_court"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "building or roof"):
                self.goal(target, state="landed")

    def test_water_inside_ground_surface_is_not_landing_support(self):
        with self.assertRaisesRegex(ValueError, "water"):
            self.goal("west_garden", state="landed", offset_m=[-6, -9, 0])

    def test_tree_inside_plaza_blocks_entire_descent(self):
        with self.assertRaisesRegex(ValueError, "tree"):
            self.goal("south_plaza", state="landed", offset_m=[-10.5, -6, 0])

    def test_roads_near_ground_surface_edge_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "road"):
            self.goal("north_court", state="landed", offset_m=[-25.8, -26.7, 0])

    def test_surface_boundary_requires_full_vehicle_support(self):
        # Exact painted edge: center x 42 to left edge 39.61, in metre offsets.
        with self.assertRaisesRegex(ValueError, "support margin"):
            self.goal(state="landed", offset_m=[(39.61 - 42) * 3, 0, 0])

    def test_missing_surfaces_do_not_implicitly_authorize_arbitrary_landings(self):
        scene = replace(self.scene, campus_layout=None)
        with self.assertRaisesRegex(ValueError, "known ground"):
            self.goal(state="landed", scenario=scene)
        self.assertEqual(self.goal("home", state="landed", scenario=scene)["position"], (48, 88, 0))
        with self.assertRaisesRegex(ValueError, "known ground"):
            self.goal("home", state="landed", scenario=scene, offset_m=[3, 0, 0])

    def test_airborne_and_landed_states_cannot_be_confused_by_altitude(self):
        for parameters in ({"altitude_m": 10}, {"offset_m": [0, 0, 1]}):
            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, "ground altitude"):
                self.goal(state="landed", **parameters)
        for parameters in ({"altitude_m": 0}, {"offset_m": [0, 0, -10]}):
            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, "floor"):
                self.goal("home", **parameters)

    def test_low_hover_uses_verified_ground_column_without_opening_low_transit(self):
        goal = self.goal(state="airborne", altitude_m=2)
        self.assertEqual(goal["position"], (42, 22.5, .4))
        self.assertEqual(goal["ground_site"], (42, 22.5, 0))
        self.assertEqual(goal["state"], "airborne")
        with self.assertRaisesRegex(ValueError, "building or roof"):
            self.goal("library", altitude_m=2)

    def test_max_height_floor_and_avoid_regions_apply_to_control(self):
        with self.assertRaisesRegex(ValueError, "ceiling"):
            self.goal(altitude_m=16, constraints={"max_altitude_m": 15})
        with self.assertRaisesRegex(ValueError, "minimum transit"):
            self.goal(state="landed", constraints={"max_altitude_m": 5})
        with self.assertRaisesRegex(ValueError, "avoided region"):
            self.goal("south_plaza", state="landed", constraints={"avoid_regions": ["south_plaza"]})
        with self.assertRaisesRegex(ValueError, "no-fly"):
            self.goal("south_plaza", constraints={"avoid_regions": ["south_plaza"]})

    def test_low_no_fly_volume_blocks_descent_even_if_cruise_goal_is_clear(self):
        scene = replace(self.scene, no_fly_zones=({"min": {"x": 41, "y": 21, "z": .3},
                                                 "max": {"x": 43, "y": 24, "z": 1}},))
        self.goal(scenario=scene)
        with self.assertRaisesRegex(ValueError, "vertical approach"):
            self.goal(state="landed", scenario=scene)

    def test_finite_heading_is_normalized(self):
        for raw, expected in ((270, -90), (-270, 90), (180, -180), (720, 0)):
            self.assertEqual(normalize_control({"yaw_deg": raw}, "home", self.catalog, {})["yaw_deg"], expected)

    def test_bad_model_parameters_are_rejected_explicitly(self):
        for value in ({"state": "magic"}, {"anchor": "nearest"}, {"offset_m": [0, 1]},
                      {"offset_m": [True, 0, 0]}, {"altitude_m": float("inf")},
                      {"duration_seconds": 601}, {"duration_seconds": -1}, {"yaw_deg": float("nan")},
                      {"unbounded_code": "anything"}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_control(value, "home", self.catalog, {})

    def test_landing_margin_covers_whole_access_column_plus_vehicle_safety(self):
        self.assertEqual(LANDING_CLEARANCE_M, GROUND_COLUMN_RADIUS_M + SAFETY_MARGIN)
        # Center is inside the football support, but the permitted column is not.
        edge = 40 - .39
        with self.assertRaisesRegex(ValueError, "support margin"):
            self.goal(state="landed", offset_m=[(edge + .4 - 42) * 3, 0, 0])
        # The centerline misses this tree, while an allowed off-center descent does not.
        tree_x, tree_y = 45, 80
        tree_radius = .38 * 3
        x = tree_x + (tree_radius + LANDING_CLEARANCE_M - .05) / 3
        with self.assertRaisesRegex(ValueError, "tree"):
            self.goal("south_plaza", state="landed", offset_m=[(x - 48.5) * 3, (tree_y - 82) * 3, 0])

    def test_stadium_terrace_and_canopy_block_landing_columns(self):
        obstacles = self.scene.campus_layout["ground_obstacles"]
        self.assertEqual(len(obstacles), 10)
        self.assertEqual({item["kind"] for item in obstacles},
                         {"spectator_stand", "spectator_canopy", "gate_pier", "bridge_rail", "monument_sign"})
        for x in (20.15, 20.7):
            # x=20.7 is east of the actual canopy; its descent column still clips it.
            with self.subTest(x=x), self.assertRaisesRegex(ValueError, "ground structure"):
                self.goal("stadium", state="landed", offset_m=[(x - 24.5) * 3, 0, 0])
        self.assertEqual(self.goal("stadium", state="landed")["position"], (24.5, 83.5, 0))

    def test_ground_structure_bounds_are_consumed_without_kind_specific_permissions(self):
        layout = deepcopy(self.scene.campus_layout)
        # Move the existing monument box above a known clear support. The same
        # geometric validator must see it, independently of its name or kind.
        obstacle = next(item for item in layout["ground_obstacles"] if item["kind"] == "monument_sign")
        center = [(lo + hi) / 2 for lo, hi in zip(obstacle["lower_grid"], obstacle["upper_grid"])]
        for bound in (obstacle["lower_grid"], obstacle["upper_grid"]):
            bound[0] += 42 - center[0]
            bound[1] += 22.5 - center[1]
        scene = replace(self.scene, campus_layout=layout)
        with self.assertRaisesRegex(ValueError, "ground structure"):
            self.goal(scenario=scene, state="landed")
        self.assertEqual(self.goal(state="landed")["position"], (42, 22.5, 0))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for shared geometry rendering")
    def test_renderer_consumes_exact_shared_ground_obstacle_bounds(self):
        root = Path(__file__).resolve().parents[1]
        scene = {"grid_size": self.scene.grid_size, "campus_layout": deepcopy(self.scene.campus_layout),
                 "points_of_interest": list(self.scene.points_of_interest)}
        # A data-only change must reach the actual Three.js meshes without an
        # accompanying renderer constant or duplicate obstacle implementation.
        for bound in ("lower_grid", "upper_grid"):
            scene["campus_layout"]["ground_obstacles"][0][bound][0] += .7
        script = r'''
import assert from 'node:assert/strict';
const THREE = await import(''' + repr((root / "uav_llm_demo/static/vendor/three.module.js").as_uri()) + r''');
const {buildSemanticCampus} = await import(''' + repr((root / "uav_llm_demo/static/semantic_campus.js").as_uri()) + r''');
const context = new Proxy({}, {get: (object,key) => key in object ? object[key] : () => context});
globalThis.document = {createElement: () => ({width:0,height:0,getContext: () => context})};
const scene = ''' + json.dumps(scene) + r''';
const group = buildSemanticCampus(THREE,new THREE.Group(),scene);
const meshes = group.children.filter(object => object.userData.groundObstacle);
assert.equal(meshes.length,scene.campus_layout.ground_obstacles.length);
const half=(scene.grid_size-1)/2;
const close=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-5,`${actual} != ${expected}`);
for(const obstacle of scene.campus_layout.ground_obstacles) {
  const mesh=meshes.find(item=>item.userData.groundObstacle===obstacle.id);
  assert.ok(mesh);
  const actual=new THREE.Box3().setFromObject(mesh);
  const [left,north,bottom]=obstacle.lower_grid;
  const [right,south,top]=obstacle.upper_grid;
  [actual.min.x,actual.min.y,actual.min.z].forEach((value,i)=>close(value,[(left-half)*3,bottom*5,(north-half)*3][i]));
  [actual.max.x,actual.max.y,actual.max.z].forEach((value,i)=>close(value,[(right-half)*3,top*5,(south-half)*3][i]));
}
const sign=group.getObjectByName('campus_south_gate_sign');
const decal=group.getObjectByName('campus_south_gate_name');
close(decal.geometry.parameters.width,sign.geometry.parameters.width);
close(decal.geometry.parameters.height,sign.geometry.parameters.height);
'''
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_world_bounds_are_checked_after_offsets(self):
        with self.assertRaisesRegex(ValueError, "bounds"):
            self.goal(offset_m=[300, 0, 0])


if __name__ == "__main__":
    unittest.main()
