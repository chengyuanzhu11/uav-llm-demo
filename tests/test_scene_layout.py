import json
import unittest
from uav_llm_demo.scene_layout import AIRSIM_ORIGIN, scene_layout
from uav_llm_demo.scenarios import SCENARIOS
from uav_llm_demo.airsim_runtime import AirSimRuntime, grid_to_ned, ned_to_grid
from uav_llm_demo.agents import MockLLMAgent
from uav_llm_demo.safety import SafetyChecker


class SharedSceneTests(unittest.TestCase):
    def test_sidewalks_stop_at_cross_streets(self):
        layout = scene_layout(SCENARIOS["urban_crossing"])
        for prefix, axis, road_prefixes in (
            ("sidewalk_ns_", "y", ("road_east_west", "road_service_ew_")),
            ("sidewalk_ew_", "x", ("road_north_south", "road_service_ns_")),
        ):
            for sidewalk in (item for item in layout if item["name"].startswith(prefix)):
                start = sidewalk["position"][axis] - sidewalk["size"][axis] / 6
                end = sidewalk["position"][axis] + sidewalk["size"][axis] / 6
                self.assertGreater(end, start)
                for road in (item for item in layout if item["name"].startswith(road_prefixes)):
                    road_start = road["position"][axis] - road["size"][axis] / 6
                    road_end = road["position"][axis] + road["size"][axis] / 6
                    self.assertTrue(end <= road_start + 1e-9 or start >= road_end - 1e-9,
                                    (sidewalk["name"], road["name"]))

    def test_urban_prompt_uses_compact_parcel_volumes_without_weakening_safety(self):
        scenario = SCENARIOS["urban_crossing"]
        context = scenario.to_dict()
        context.update(position=context["base"], goal=context["target"])
        prompt = MockLLMAgent().build_prompt(context)
        payload = json.loads(prompt.split("Scene:\n", 1)[1].split("\nMission:\n", 1)[0])
        self.assertNotIn("buildings", payload)
        self.assertNotIn("urban_parcels", payload)
        self.assertEqual(len(payload["building_volumes"]), 44)
        self.assertLess(len(prompt), 13000)
        checker = SafetyChecker(context)
        for volume in payload["building_volumes"]:
            center = tuple((volume["min"][axis] + volume["max"][axis]) / 2
                           for axis in ("x", "y"))
            self.assertTrue(checker.validate_waypoint((*center, 2.0)))

    def test_urban_parcels_drive_collision_cells_and_street_geometry(self):
        scenario = SCENARIOS["urban_crossing"]
        layout = scene_layout(scenario)
        names = {item["name"] for item in layout}
        self.assertEqual(scenario.grid_size, 40)
        self.assertEqual(len(scenario.urban_parcels), 32)
        self.assertIn("road_north_south", names)
        self.assertIn("road_east_west", names)
        self.assertEqual(len([name for name in names if name.startswith("cross_")]), 96)
        for center in scenario.urban_street_centers:
            self.assertTrue(any(item["name"].startswith("road_north_south")
                                and item["position"]["x"] == center for item in layout))
            self.assertTrue(any(item["name"].startswith("road_east_west")
                                and item["position"]["y"] == center for item in layout))
        cells = scenario.building_map
        for parcel in scenario.urban_parcels:
            left, right, north, south = parcel.footprint
            self.assertGreaterEqual(right, left)
            self.assertGreaterEqual(south, north)
            for x in range(left, right + 1):
                for y in range(north, south + 1):
                    self.assertGreaterEqual(cells[x, y], parcel.podium_height)
            if parcel.tower:
                tx0, tx1, ty0, ty1, height = parcel.tower
                self.assertLess(left, tx0)
                self.assertLess(tx1, right)
                self.assertLess(north, ty0)
                self.assertLess(ty1, south)
                self.assertGreater(height, parcel.podium_height)
                for x in range(tx0, tx1 + 1):
                    for y in range(ty0, ty1 + 1):
                        self.assertEqual(cells[x, y], height)
        context = scenario.to_dict()
        context["goal"] = context["target"]
        checker = SafetyChecker(context)
        self.assertFalse(checker.validate_waypoint(scenario.base))
        self.assertFalse(checker.validate_waypoint(scenario.target))

    def test_every_scenario_has_exact_building_geometry(self):
        for scenario in SCENARIOS.values():
            objects = scene_layout(scenario)
            self.assertEqual(len({o["name"] for o in objects}), len(objects))
            buildings = [o for o in objects if o["name"].startswith("building_")]
            self.assertEqual(len(buildings), len(scenario.buildings))
            for obj, (x, y, height) in zip(buildings, scenario.buildings):
                self.assertEqual(obj["position"], dict(x=x, y=y, z=height / 2))
                self.assertAlmostEqual(obj["size"]["z"], height * 5)
                self.assertAlmostEqual(obj["size"]["x"], 2.34)
            ground = objects[0]
            self.assertEqual(ground["position"]["z"] * 5 + ground["size"]["z"] / 2, 0)

    def test_relocated_map_round_trip_preserves_grid_positions(self):
        runtime = AirSimRuntime()
        try:
            runtime._origin = AIRSIM_ORIGIN
            for point in ((1, 18, 2), (18, 2, 5), (6, 5, 4)):
                actual = runtime._grid_to_ned(point)
                local = tuple(actual[i] - AIRSIM_ORIGIN[i] for i in range(3))
                self.assertEqual(local, grid_to_ned(point))
                self.assertEqual(ned_to_grid(local), point)
        finally:
            runtime.close()
