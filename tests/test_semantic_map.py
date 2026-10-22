import unittest
from collections import Counter
from copy import deepcopy

from uav_llm_demo.agents import GeometricPlanner
from uav_llm_demo.campus_layout import CAMPUS_LAYOUT, campus_building_cells
from uav_llm_demo.safety import SafetyChecker, _intersects
from uav_llm_demo.scenarios import SCENARIOS, list_scenarios, load_scenario
from uav_llm_demo.scene_layout import scene_layout
from uav_llm_demo.semantic_tasks import task_catalog, semantic_plan_from_json_text
from uav_llm_demo.spatial_motion import compile_motion
from uav_llm_demo.semantic_navigation import validate_motion_path
import json


class SemanticCampusMapTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario("semantic_campus")
        self.context = self.scenario.to_dict()
        self.points = {item["id"]: item for item in self.context["points_of_interest"]}

    def test_public_catalog_contains_only_two_supported_maps(self):
        self.assertEqual(set(SCENARIOS), {"urban_crossing", "semantic_campus"})
        self.assertEqual({item["id"] for item in list_scenarios()}, set(SCENARIOS))
        urban = load_scenario("urban_crossing")
        self.assertEqual(urban.mission_mode, "navigation")
        self.assertEqual((urban.grid_size, urban.base, urban.target), (40, (1, 18, 0), (38, 2, 5)))
        self.assertIsNone(urban.to_dict()["campus_layout"])
        self.assertTrue(urban.urban_parcels)
        self.assertEqual(self.context["mission_mode"], "semantic")
        self.assertEqual(self.scenario.urban_parcels, ())
        self.assertEqual(self.scenario.urban_street_centers, ())
        for removed in ("open_field", "narrow_corridor"):
            with self.assertRaises(ValueError):
                load_scenario(removed)

    def test_public_catalog_has_campus_capabilities_without_hidden_sensor_values(self):
        self.assertTrue({
            "home", "conference", "library", "teaching", "exhibition", "research_one",
            "research_two", "laboratory", "stadium", "east_gate", "north_teaching",
        } <= set(self.points))
        self.assertEqual(len(self.points), 56)
        self.assertEqual(self.points["home"]["position"], self.context["base"])
        self.assertEqual(self.points["home"]["allowed_actions"], ["return_home"])
        for site in ("library", "laboratory"):
            self.assertIn("thermal_scan", self.points[site]["allowed_actions"])
        for item in self.points.values():
            self.assertEqual(set(item), {"id", "label", "code", "kind", "position", "allowed_actions"})
        for recommendation in self.context["recommended_tasks"]:
            self.assertIn("返航", recommendation)
        self.assertGreater(len({point["position"]["z"] for point in self.points.values()}), 4)

    def test_catalog_mutation_cannot_change_shared_geometry_or_future_requests(self):
        original = deepcopy(self.context)
        self.context["points_of_interest"][0]["position"]["x"] = 900
        self.context["semantic_regions"][0]["bounds"][0] = -900
        self.context["campus_layout"]["buildings"][0]["footprint"][0] = -999
        self.context["campus_layout"]["roofs"][0]["skylights"][0][0] = -999
        self.context["campus_layout"]["roads"][0]["points"][0][0] = -999
        self.context["campus_layout"]["trees"][0]["x"] = -999
        self.assertEqual(self.scenario.to_dict(), original)
        self.assertEqual(self.scenario.buildings, campus_building_cells())

    def test_place_catalog_covers_every_model_once_and_uses_safe_arrivals(self):
        layout = self.context["campus_layout"]
        places = layout["places"]
        self.assertEqual({place["id"] for place in places}, set(self.points))
        self.assertEqual(len({place["code"] for place in places}), len(places))
        self.assertEqual(Counter(identifier for place in places for identifier in place["building_ids"]),
                         Counter(item["id"] for item in layout["buildings"]))
        self.assertEqual(Counter(identifier for place in places for identifier in place["surface_ids"]),
                         Counter(item["id"] for item in layout["surfaces"]))
        self.assertEqual(sum(bool(place["building_ids"]) for place in places), 41)
        checker = SafetyChecker(self.context, clearance=1.1)
        for place in places:
            with self.subTest(place=place["id"]):
                point = self.points[place["id"]]
                self.assertEqual(point["label"], place["label"])
                self.assertEqual(point["position"], place["position"])
                self.assertEqual(checker.validate_waypoint(point["position"]), [])
                if place["building_ids"]:
                    self.assertNotEqual(place["label_position"], place["position"])
        self.assertEqual({point["id"] for point in self.points.values()
                          if "thermal_scan" in point["allowed_actions"]}, {"library", "laboratory"})
        self.assertEqual({point["id"]: point["code"] for point in task_catalog(self.scenario)["points_of_interest"]},
                         {place["id"]: place["code"] for place in places})

    def test_all_sites_connect_to_base_and_cross_district_routes_are_reachable(self):
        planner = GeometricPlanner()
        checker = SafetyChecker(self.context)
        for point in self.points.values():
            self.assertEqual(checker.validate_waypoint(point["position"]), [], point["id"])
        # Both directions to the same hub prove connectivity for every place.
        # Avoid thousands of redundant full searches as the shared map grows.
        pairs = [(self.points["home"], site) for site in self.points.values() if site["id"] != "home"]
        pairs += [(site, self.points["home"]) for site in self.points.values() if site["id"] != "home"]
        pairs += [(self.points[start], self.points[end]) for start, end in
                  (("d14", "l6"), ("c4", "c8"), ("r3", "south_canteen"),
                   ("a5", "stadium"), ("north_basketball", "laboratory"))]
        for start, goal in pairs:
            context = {**self.context, "position": start["position"], "goal": goal["position"]}
            route = planner.plan(context)
            self.assertEqual(SafetyChecker(context).validate_path(start["position"], route.waypoints), [],
                             (start["id"], goal["id"]))

    def test_every_published_default_boundary_is_safe_and_program_compatible(self):
        catalog = task_catalog(self.scenario)
        self.assertEqual({item["id"] for item in catalog["spatial"]["objects"]}, set(self.points) - {"home"})
        for point in self.points.values():
            if point["id"] == "home":
                continue
            node = dict(id="move", action="motion", target_id=point["id"],
                        motion=dict(path=dict(kind="boundary")), when=None)
            raw = dict(status="ready", summary="绕行后返航", steps=[node,
                dict(id="home",action="return_home",target_id="home",duration_seconds=None,when=None)],
                constraints=dict(max_altitude_m=None,avoid_regions=[]))
            with self.subTest(place=point["id"]):
                plan = semantic_plan_from_json_text(json.dumps(raw), catalog)
                motion = compile_motion(plan.steps[0], self.scenario, self.scenario.base, plan.constraints)
                validate_motion_path(motion, self.context)

    def test_all_outdoor_sites_remain_reachable_under_ten_metre_ceiling(self):
        for site in self.points.values():
            if site["id"] == "home":
                continue
            goal = {**site["position"], "z": 2}
            context = {**self.context, "altitude_range": {"min": 2, "max": 2},
                       "position": self.context["base"], "goal": goal}
            route = GeometricPlanner().plan(context)
            self.assertEqual(SafetyChecker(context).validate_path(context["position"], route.waypoints), [], site["id"])
            self.assertTrue(all(point[2] <= 2 for point in route.waypoints), site["id"])

    def test_optional_campus_region_avoidance_keeps_library_reachable(self):
        self.assertEqual(self.context["no_fly_zones"], [])
        regions = {region["id"]: region for region in self.context["semantic_regions"]}
        self.assertEqual(set(regions), {"south_plaza", "south_residential", "north_residential"})
        left, right, north, south = regions["south_plaza"]["bounds"]
        context = {**self.context, "position": self.context["base"], "goal": self.points["library"]["position"],
                   "no_fly_zones": [dict(min=dict(x=left, y=north, z=0),
                                         max=dict(x=right, y=south, z=10))]}
        route = GeometricPlanner().plan(context)
        self.assertEqual(SafetyChecker(context).validate_path(context["position"], route.waypoints), [])

    def test_visual_solids_and_glazed_roofs_match_conservative_collision_cells(self):
        cells = self.scenario.building_map
        self.assertEqual(self.scenario.buildings, campus_building_cells(CAMPUS_LAYOUT))
        for item in [*CAMPUS_LAYOUT["buildings"], *CAMPUS_LAYOUT["roofs"]]:
            left, right, north, south = item["footprint"]
            self.assertTrue(0 <= left <= right < self.scenario.grid_size)
            self.assertTrue(0 <= north <= south < self.scenario.grid_size)
            for x in range(left, right + 1):
                for y in range(north, south + 1):
                    if any(l <= x <= r and n <= y <= s for l, r, n, s in item.get("holes", [])):
                        continue
                    self.assertGreaterEqual(cells[x, y], item["height"])
        checker = SafetyChecker(self.context)
        self.assertTrue(checker.validate_waypoint((44, 62, 2)), "Conservative roof columns must block under-roof shortcuts")
        self.assertTrue(checker.validate_waypoint((49, 62, 2)), "Glazed south skylights cannot be flown through")
        self.assertTrue(checker.validate_waypoint((49, 37, 2)), "Glazed north skylights cannot be flown through")
        self.assertTrue(checker.validate_waypoint((49, 54, 2)), "The linking roof blocks unsafe underpasses")
        self.assertEqual(checker.validate_waypoint((44, 62, 4)), [], "Overflight must clear the maximum roof height")
        for item in CAMPUS_LAYOUT["trees"]:
            self.assertLess(item["height"], self.scenario.min_altitude)
            home = self.context["base"]
            self.assertGreater((item["x"] - home["x"]) ** 2 + (item["y"] - home["y"]) ** 2,
                               (item["radius"] + 0.3) ** 2)

    def test_campus_roads_do_not_cross_physical_building_footprints(self):
        for road in CAMPUS_LAYOUT["roads"]:
            edge = road["width"] / 2 + 0.39
            for building in CAMPUS_LAYOUT["buildings"]:
                left, right, north, south = building["footprint"]
                lower, upper = (left - edge, north - edge), (right + edge, south + edge)
                for start, end in zip(road["points"], road["points"][1:]):
                    self.assertFalse(_intersects(start, end, lower, upper),
                                     (road["id"], building["id"]))

    def test_export_uses_shared_building_cells_and_ground_launch_apron(self):
        cells = self.scenario.building_map
        objects = scene_layout(self.scenario)
        self.assertEqual(len({item["name"] for item in objects}), len(objects))
        buildings = [item for item in objects if item["name"].startswith("building_")]
        self.assertEqual(len(buildings), len(cells))
        for item in buildings:
            point = item["position"]
            self.assertEqual(item["size"]["z"], cells[point["x"], point["y"]] * 5)
        launch = next(item for item in objects if item["name"] == "campus_launch_apron")
        self.assertLessEqual(launch["position"]["z"] * 5 + launch["size"]["z"] / 2, 0)
        self.assertFalse(any(item["name"].startswith("road_north_south") for item in objects))


if __name__ == "__main__":
    unittest.main()
