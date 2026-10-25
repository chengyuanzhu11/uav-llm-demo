"""General motion geometry and evidence, independent of named maneuver handlers."""
from copy import deepcopy
import json
import math
import unittest

from uav_llm_demo.campus_layout import CAMPUS_LAYOUT
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.safety import SAFETY_MARGIN, SafetyChecker, metric_distance
from uav_llm_demo.path_tracking import PathFollower
from uav_llm_demo.spatial_motion import MOTION_LIMITS, compile_motion, spatial_catalog
from uav_llm_demo.semantic_tasks import SemanticConstraintError, SemanticMission, semantic_plan_from_json_text, task_catalog


def motion(path=None, heading=None, identifier="movement", target="stadium"):
    result = {"id": identifier, "action": "motion", "target_id": target,
              "motion": {"path": path or {"kind": "boundary", "laps": 2}}, "when": None}
    if heading is not None:
        result["motion"]["heading"] = heading
    return result


def program(node):
    return {"status": "ready", "summary": "完成轨迹后返航。",
            "steps": [node, {"id": "home", "action": "return_home", "target_id": "home",
                            "duration_seconds": None, "when": None}],
            "constraints": {"max_altitude_m": None, "avoid_regions": []}}


def parse(node, scenario=None, constraints=None):
    scenario = scenario or load_scenario("semantic_campus")
    raw = program(node)
    if constraints is not None:
        raw["constraints"] = constraints
    return semantic_plan_from_json_text(json.dumps(raw), task_catalog(scenario))


def compile_node(node, constraints=None):
    scenario = load_scenario("semantic_campus")
    plan = parse(node, scenario, constraints)
    return compile_motion(plan.steps[0], scenario, scenario.base, plan.constraints)


class MotionGeometryTests(unittest.TestCase):
    def test_boundary_uses_actual_render_geometry_and_preserves_two_laps(self):
        scene = load_scenario("semantic_campus")
        track = next(item for item in CAMPUS_LAYOUT["surfaces"] if item["kind"] == "running_track")
        objects = {item["id"]: item for item in spatial_catalog(scene)["objects"]}
        self.assertEqual(objects["stadium"]["boundary_grid"], track["geometry"]["centerline_grid"])
        once = compile_node(motion({"kind": "boundary"}))
        twice = compile_node(motion())
        self.assertAlmostEqual(twice.length_m, 2 * once.length_m)
        self.assertEqual(twice.points[0], twice.points[len(once.points) - 1])
        self.assertEqual(twice.points[0], twice.points[-1])
        self.assertGreater(twice.points[1][1], twice.points[0][1])
        self.assertEqual(twice.speed_mps, 2.5)
        checker = SafetyChecker(scene.to_dict())
        self.assertTrue(all(checker.segment_safe(a, b) for a, b in zip(twice.points, twice.points[1:])))
        objects["stadium"]["boundary_grid"][0][0] = -100
        self.assertNotEqual(track["geometry"]["centerline_grid"][0][0], -100)

    def test_library_omitted_height_clears_actual_roof_and_tracking_corridor(self):
        scene = load_scenario("semantic_campus")
        catalog = task_catalog(scene)
        library = next(item for item in catalog["spatial"]["objects"] if item["id"] == "library")
        self.assertAlmostEqual(library["boundary_default_altitude_m"], 19.5)
        compiled = compile_node(motion({"kind": "boundary"}, target="library"))
        checker = SafetyChecker(scene.to_dict(), clearance=SAFETY_MARGIN + PathFollower.CORRIDOR_M)
        self.assertTrue(all(checker.segment_safe(a, b) for a, b in zip(compiled.points, compiled.points[1:])))
        self.assertAlmostEqual(compiled.points[0][2] * 5, 19.5)
        with self.assertRaises(SemanticConstraintError) as conflict:
            compile_node(motion({"kind": "boundary"}, target="library"),
                         {"max_altitude_m": 18, "avoid_regions": []})
        self.assertEqual(conflict.exception.code, "motion_path_unsafe")
        # Explicitly requested lower height is preserved, then rejected by route safety.
        lower = compile_node(motion({"kind": "boundary", "altitude_m": 15}, target="library"))
        self.assertEqual(lower.points[0][2] * 5, 15)
        self.assertFalse(all(checker.segment_safe(a, b) for a, b in zip(lower.points, lower.points[1:])))
        ordinary = {"id": "inspect", "action": "inspect", "target_id": "library",
                    "duration_seconds": None, "when": None}
        mission = SemanticMission(scene)
        mission.accept(parse(ordinary, scene))
        self.assertEqual(mission.current_goal, (53, 74, 3))

    def test_direction_is_clockwise_in_east_south_frame(self):
        cw = compile_node(motion({"kind": "boundary", "direction": "clockwise"}))
        ccw = compile_node(motion({"kind": "boundary", "direction": "counterclockwise"}))
        area = lambda points: sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:]))
        self.assertGreater(area(cw.points), 0)
        self.assertLess(area(ccw.points), 0)
        self.assertAlmostEqual(cw.length_m, ccw.length_m)

    def test_generic_curve_composes_figure_eight_and_spiral(self):
        eight = compile_node(motion({"kind": "curve", "x": "8*sin(t)", "y": "4*sin(2*t)",
                                     "z": 4, "t_start": 0, "t_end": 2 * math.pi}))
        spiral = compile_node(motion({"kind": "curve", "x": "8*cos(t)", "y": "8*sin(t)",
                                      "z": "t/2", "t_start": 0, "t_end": 4 * math.pi}))
        self.assertLess(metric_distance(eight.points[0], eight.points[-1]), 1e-8)
        self.assertAlmostEqual((spiral.points[-1][2] - spiral.points[0][2]) * 5, 2 * math.pi)
        self.assertGreater(spiral.length_m, eight.length_m)
        self.assertTrue(all(metric_distance(a, b) <= .75000001
                            for a, b in zip(spiral.points, spiral.points[1:])))

    def test_polyline_and_heading_use_meter_offsets_from_published_center(self):
        item = motion({"kind": "polyline", "points_m": [[0, 0, 4], [6, 3, 4], [6, 3, 9]]},
                      {"mode": "face_target", "target_id": "stadium"})
        compiled = compile_node(item)
        center = next(obj["center_grid"] for obj in spatial_catalog(load_scenario("semantic_campus"))["objects"]
                      if obj["id"] == "stadium")
        self.assertEqual(compiled.points[0], (center[0], center[1], center[2] + .8))
        self.assertEqual(compiled.points[-1], (center[0] + 2, center[1] + 1, center[2] + 1.8))
        self.assertEqual(compiled.heading, {"mode": "face_point", "point": tuple(center)})
        self.assertAlmostEqual(compiled.length_m, math.sqrt(45) + 5)

    def test_linear_boundary_height_preserves_first_and_final_absolute_altitude(self):
        compiled = compile_node(motion({"kind": "boundary", "laps": 1, "altitude_m": 12, "end_altitude_m": 20},
                                       {"mode": "fixed", "yaw_deg": -90}))
        self.assertAlmostEqual(compiled.points[0][2] * 5, 12)
        self.assertAlmostEqual(compiled.points[-1][2] * 5, 20)
        self.assertTrue(all(a[2] <= b[2] for a, b in zip(compiled.points, compiled.points[1:])))
        self.assertEqual(compiled.heading, {"mode": "fixed", "yaw_deg": -90})
        self.assertTrue(all(metric_distance(a, b) <= .75000001
                            for a, b in zip(compiled.points, compiled.points[1:])))
        self.assertIn("zero north", spatial_catalog(load_scenario("semantic_campus"))["heading_modes"]["fixed"])

    def test_altitude_and_map_constraints_never_silently_rewrite_explicit_geometry(self):
        variants = [
            motion({"kind": "boundary", "altitude_m": 55}),
            motion({"kind": "boundary", "altitude_m": 5}),
            motion({"kind": "polyline", "points_m": [[0, 0, 0], [1000, 0, 0]]}),
            motion({"kind": "curve", "x": "8*cos(t)", "y": "8*sin(t)", "z": 50, "t_start": 0, "t_end": 6.28}),
        ]
        for node in variants:
            with self.subTest(node=node), self.assertRaises(ValueError):
                compile_node(node)
        with self.assertRaises(ValueError):
            compile_node(motion({"kind": "boundary", "altitude_m": 15}),
                         {"max_altitude_m": 12, "avoid_regions": []})
        with self.assertRaises(ValueError):
            compile_node(motion({"kind": "polyline", "points_m": [[0, 0, 0], [2, 0, 0]]}),
                         {"max_altitude_m": 10, "avoid_regions": []})

    def test_expression_cannot_execute_code_or_exhaust_work_budget(self):
        expressions = ["__import__('os').system('whoami')", "t.__class__", "[x for x in range(2)]",
                       "True", "sin(t, t)", "t**9999", "t**t", "1/(t-1)", "8*sin(1000*t)",
                       "1e999", "1/1e-200", "t/(1e-200*t+1e-200)",
                       "(" * 100 + "t" + ")" * 100 + "+x"]
        for expression in expressions:
            node = motion({"kind": "curve", "x": expression, "y": "t", "z": 0, "t_start": 0, "t_end": 2})
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                compile_node(node)

    def test_counts_speed_and_degenerate_paths_fail_closed(self):
        variants = [motion({"kind": "boundary", "laps": count}) for count in (0, -1, True, 1.5, 21)]
        for speed in (0, -1, True, 4.1, math.nan):
            node = motion()
            node["motion"]["speed_mps"] = speed
            variants.append(node)
        variants.extend(motion(heading=invalid) for invalid in ({}, [], {"mode": "face_target", "target_id": []}))
        variants += [motion({"kind": "polyline", "points_m": [[0, 0, 0], [0, 0, 0]]}),
                     motion({"kind": "curve", "x": 0, "y": 0, "z": 0, "t_start": 0, "t_end": 1}),
                     motion({"kind": "curve", "x": "t", "y": 0, "z": 0, "t_start": 1, "t_end": 0}),
                     motion({"kind": "polyline", "points_m": [[0, 0, 0]] * (MOTION_LIMITS["max_points"] + 1)})]
        for node in variants:
            with self.subTest(node=node), self.assertRaises(ValueError):
                compile_node(node)


class MotionEvidenceTests(unittest.TestCase):
    def test_stationary_arrival_cannot_claim_motion_completion(self):
        scene = load_scenario("semantic_campus")
        mission = SemanticMission(scene)
        mission.accept(parse(motion(), scene))
        self.assertIsNone(mission.current_goal)
        for _ in range(3):
            self.assertFalse(mission.observe((26, 84, 2.2), 0, 30, 100))
        self.assertEqual(mission.current_step["action"], "motion")
        self.assertFalse(mission.evidence)

    def test_completion_is_execution_bound_and_requires_full_measured_traversal(self):
        scene = load_scenario("semantic_campus")
        mission = SemanticMission(scene)
        plan = parse(motion(), scene)
        compiled = compile_motion(plan.steps[0], scene, scene.base, plan.constraints)
        mission.accept(plan)
        metrics = {"execution_id": mission.current_step["execution_id"], "completed": True,
                   "covered_distance_m": compiled.length_m, "total_length_m": compiled.length_m,
                   "duration_seconds": 64}
        for replacement in ({"execution_id": "stale"}, {"completed": False}, {"covered_distance_m": 0}):
            self.assertFalse(mission.complete_motion(compiled.points[-1], 100, {**metrics, **replacement}))
        self.assertTrue(mission.complete_motion(compiled.points[-1], 100, metrics))
        self.assertEqual(mission.current_step["action"], "return_home")
        evidence = mission.evidence[0]
        self.assertEqual(evidence["source"], "runtime_trajectory_tracking")
        self.assertEqual(evidence["result"]["distance_m"], compiled.length_m)
        self.assertEqual(evidence["result"]["progress"], 1)
        self.assertEqual(evidence["duration_seconds"], 64)
        self.assertFalse(mission.complete_motion(compiled.points[-1], 101, metrics))

    def test_motion_composes_with_existing_loops_and_conditions(self):
        scene = load_scenario("semantic_campus")
        node = {"id": "repeat_motion", "action": "repeat", "count": 2, "until": None,
                "when": None, "steps": [motion({"kind": "boundary", "laps": 1})]}
        data = program(node)
        data["steps"].insert(1, {"id": "inspect", "action": "inspect", "target_id": "stadium",
                               "duration_seconds": None,
                               "when": {"step_id": "movement", "field": "progress", "op": "==", "value": 1}})
        plan = semantic_plan_from_json_text(json.dumps(data), task_catalog(scene))
        compiled = compile_motion(plan.steps[0]["steps"][0], scene, scene.base, plan.constraints)
        mission = SemanticMission(scene)
        mission.accept(plan)
        identities = []
        for i in range(2):
            step = mission.current_step
            identities.append(step["execution_id"])
            self.assertEqual(step["loop_iterations"], {"repeat_motion": i + 1})
            self.assertTrue(mission.complete_motion(compiled.points[-1], 100 + i,
                            {"execution_id": step["execution_id"], "completed": True,
                             "covered_distance_m": compiled.length_m,
                             "total_length_m": compiled.length_m, "duration_seconds": 32}))
        self.assertNotEqual(*identities)
        self.assertEqual(mission.current_step["id"], "inspect")
        self.assertEqual(len(mission.evidence), 2)
        self.assertEqual(mission.to_dict()["loops"][0]["status"], "completed")

    def test_revised_task_does_not_accept_old_motion_evidence(self):
        scene = load_scenario("semantic_campus")
        mission = SemanticMission(scene)
        plan = parse(motion(), scene)
        mission.accept(plan)
        old_execution = mission.current_step["execution_id"]
        mission.accept(plan, preserve_evidence=True)
        self.assertFalse(mission.complete_motion((27.8, 82.5, 2.2), 100,
                         {"execution_id": old_execution, "completed": True,
                          "covered_distance_m": 80, "total_length_m": 80, "duration_seconds": 32}))
        self.assertEqual(mission.evidence, [])
