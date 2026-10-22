"""Shared transport schema and repair diagnostics, independent of any task wording."""
from copy import deepcopy
import json
import unittest

from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_tasks import SemanticConstraintError, semantic_plan_from_json_text, task_catalog
from uav_llm_demo.spatial_motion import normalize_motion
from uav_llm_demo.spatial_control import normalize_control
from uav_llm_demo.task_protocol import TaskProtocolError, validate_structure


def motion(target="east_garden", path=None, identifier="move"):
    return {"id": identifier, "action": "motion", "target_id": target,
            "motion": {"path": path or {"kind": "boundary"}}, "when": None}


def program(*nodes):
    return {"status": "ready", "summary": "执行任务。", "steps": list(nodes),
            "constraints": {"max_altitude_m": None, "avoid_regions": []}}


class TaskProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = task_catalog(load_scenario("semantic_campus"))

    def parse(self, value):
        return semantic_plan_from_json_text(json.dumps(value, ensure_ascii=False), self.catalog)

    def error(self, value, path, code=None):
        with self.assertRaises(TaskProtocolError) as caught:
            self.parse(value)
        error = caught.exception
        self.assertEqual(error.diagnostic["path"], path)
        self.assertIn(path, str(error))
        if code is not None:
            self.assertEqual(error.diagnostic["code"], code)
        return error.diagnostic

    def test_catalog_schema_is_the_contract_parser_uses(self):
        schema = self.catalog["task_program_schema"]
        self.assertEqual(schema["additionalProperties"], False)
        nodes = schema["$defs"]["node"]["oneOf"]
        for variant in nodes:
            self.assertFalse(variant["additionalProperties"])
            self.assertIn("action", variant["required"])
        value = program(motion())
        validate_structure(value, schema)
        plan = self.parse(value)
        validate_structure(plan.to_dict(), schema)

    def test_missing_and_unknown_motion_fields_describe_required_shape(self):
        value = program(motion())
        value["steps"][0]["motion"] = {"kind": "boundary", "laps": 1}
        error = self.error(value, "$.steps[0].motion.kind", "object_fields")
        self.assertEqual(error["missing"], ["path"])
        self.assertEqual(error["unexpected"], ["kind", "laps"])
        self.assertEqual(error["allowed"], ["path", "speed_mps", "heading"])

    def test_root_and_nested_constraints_reject_fields(self):
        value = program(motion())
        value["command"] = "arbitrary"
        self.error(value, "$.command", "object_fields")
        del value["command"]
        del value["constraints"]["avoid_regions"]
        self.error(value, "$.constraints.avoid_regions", "object_fields")

    def test_avoidance_region_items_validate_before_duplicate_detection(self):
        value = program(motion())
        value["constraints"]["avoid_regions"] = [{}]
        self.error(value, "$.constraints.avoid_regions[0]", "wrong_type")
        region = self.catalog["regions"][0]["id"]
        value["constraints"]["avoid_regions"] = [region, region]
        self.error(value, "$.constraints.avoid_regions", "duplicate_item")

    def test_unknown_motion_fields_locate_nested_branch_in_repeat(self):
        movement = motion()
        movement["motion"]["path"]["radius_m"] = 10
        branch = {"id": "choice", "action": "branch",
                  "condition": {"source": "telemetry", "field": "battery_percent", "op": ">", "value": 20},
                  "then": [movement], "else": []}
        repeat = {"id": "loop", "action": "repeat", "count": 2, "until": None, "when": None, "steps": [branch]}
        error = self.error(program(repeat), "$.steps[0].steps[0].then[0].motion.path.radius_m")
        self.assertEqual(error["unexpected"], ["radius_m"])
        self.assertNotIn("radius_m", error["allowed"])

    def test_wrong_types_are_not_coerced(self):
        for field, value in (("laps", "1"), ("laps", True), ("altitude_m", "12"), ("direction", [])):
            with self.subTest(field=field, value=value):
                candidate = program(motion(path={"kind": "boundary", field: value}))
                self.error(candidate, f"$.steps[0].motion.path.{field}")

    def test_heading_fields_and_missing_discriminators_are_precise(self):
        node = motion()
        node["motion"]["heading"] = {"mode": "fixed"}
        self.error(program(node), "$.steps[0].motion.heading.yaw_deg")
        node["motion"]["heading"] = {"mode": "tangent", "target_id": "east_garden"}
        self.error(program(node), "$.steps[0].motion.heading.target_id")
        node["motion"] = {"path": {"laps": 1}}
        self.error(program(node), "$.steps[0].motion.path.kind")

    def test_curve_and_polyline_inner_errors_are_field_addressed(self):
        curve = {"kind": "curve", "x": "cos(t)", "y": "sin(t)", "z": 5, "t_start": 0, "t_end": 6.28}
        malformed = {**curve, "x": "__import__('os')"}
        self.error(program(motion(path=malformed)), "$.steps[0].motion.path.x", "invalid_expression")
        self.error(program(motion(path={**curve, "t_end": -1})), "$.steps[0].motion.path.t_end")
        line = {"kind": "polyline", "points_m": [[0, 0, 5], [10, False, 5]]}
        self.error(program(motion(path=line)), "$.steps[0].motion.path.points_m[1][1]", "wrong_type")

    def test_control_errors_retain_nested_coordinate_paths(self):
        node = {"id": "land", "action": "control", "target_id": "stadium",
                "control": {"offset_m": [0, "east", 0]}, "when": None}
        self.error(program(node), "$.steps[0].control.offset_m[1]", "wrong_type")
        node["control"] = {"state": "landed", "altitude_m": 5}
        self.error(program(node), "$.steps[0].control.altitude_m", "goal_state_conflict")

    def test_condition_reference_and_result_type_errors_locate_leaf(self):
        node = motion()
        node["when"] = {"all": [{"source": "telemetry", "field": "battery_percent", "op": ">", "value": "30"}]}
        self.error(program(node), "$.steps[0].when.all[0].value", "wrong_type")
        node["when"] = {"not": {"step_id": "future", "field": "completed", "op": "==", "value": True}}
        self.error(program(node), "$.steps[0].when.not.step_id", "invalid_reference")

    def test_same_boundary_defaults_apply_to_multiple_places(self):
        for target in ("east_garden", "west_garden", "stadium", "library"):
            with self.subTest(target=target):
                normalized = self.parse(program(motion(target))).steps[0]["motion"]
                self.assertEqual(normalized["path"]["direction"], "clockwise")
                self.assertEqual(normalized["path"]["laps"], 1)
                self.assertEqual(normalized["heading"], {"mode": "tangent"})
                self.assertEqual(normalized["speed_mps"], 2.5)

    def test_null_control_and_motion_options_keep_existing_defaults(self):
        node = motion()
        node["motion"].update(speed_mps=None, heading=None)
        node["motion"]["path"].update(altitude_m=None, end_altitude_m=None)
        normalized = self.parse(program(node)).steps[0]["motion"]
        self.assertEqual(normalized["speed_mps"], 2.5)
        self.assertEqual(normalized["path"]["altitude_m"], normalized["path"]["end_altitude_m"])
        control = {"id": "hold", "action": "control", "target_id": "stadium", "control": None, "when": None}
        normalized = self.parse(program(control)).steps[0]["control"]
        self.assertEqual(normalized["state"], "airborne")
        self.assertEqual(normalized["duration_seconds"], 2.0)

    def test_direct_normalizers_share_the_published_contract(self):
        with self.assertRaises(TaskProtocolError) as caught:
            normalize_motion({"path": {"kind": "boundary", "radius_m": 4}}, "east_garden", self.catalog, {})
        self.assertEqual(caught.exception.path, "$.motion.path.radius_m")
        with self.assertRaises(TaskProtocolError) as caught:
            normalize_control({"speed_mps": 4}, "stadium", self.catalog, {})
        self.assertEqual(caught.exception.path, "$.control.speed_mps")

    def test_geometry_ceiling_conflicts_remain_semantic_constraints(self):
        value = program(motion("library"))
        value["constraints"]["max_altitude_m"] = 18
        with self.assertRaises(SemanticConstraintError) as caught:
            self.parse(value)
        self.assertEqual(caught.exception.code, "motion_path_unsafe")

    def test_catalog_schema_mutation_does_not_relax_local_validation(self):
        catalog = deepcopy(self.catalog)
        catalog["task_program_schema"] = {}
        value = program(motion())
        value["steps"][0]["motion"]["path"]["radius"] = 10
        with self.assertRaises(TaskProtocolError):
            semantic_plan_from_json_text(json.dumps(value), catalog)

    def test_error_diagnostics_are_bounded_without_candidate_echo(self):
        unknown = "field" * 10000
        value = program(motion())
        value["steps"][0]["motion"][unknown] = "secret payload must not be echoed"
        with self.assertRaises(TaskProtocolError) as caught:
            self.parse(value)
        diagnostic = json.dumps(caught.exception.diagnostic)
        self.assertLess(len(diagnostic), 2000)
        self.assertNotIn("secret payload", diagnostic)


if __name__ == "__main__":
    unittest.main()
