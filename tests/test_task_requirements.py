"""Requirements retain source intent without acquiring execution defaults."""
from copy import deepcopy
import json
import unittest

from tests.semantic_fixtures import fixture_audit
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_tasks import task_catalog
from uav_llm_demo.task_protocol import TaskProtocolError, validate_structure
from uav_llm_demo.task_requirements import (
    TaskRequirementsError, audit_requirements_from_json_text,
    requirement_schema, requirements_from_json_text, compare_requirement_assertions,
)


class TaskRequirementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = task_catalog(load_scenario("semantic_campus"))

    def document(self, instruction, *steps, constraints=None):
        constraints = {} if constraints is None else constraints
        return {"status": "ready", "summary": "保留原任务要求。", "steps": list(steps),
                "constraints": constraints, "coverage": [{"quote": instruction,
                    "requirement_ids": [step["id"] for step in steps], "constraint_keys": list(constraints)}]}

    def parse(self, document, instruction):
        return requirements_from_json_text(json.dumps(document, ensure_ascii=False), self.catalog, instruction)

    def test_sparse_obligations_preserve_relative_motion_without_execution_defaults(self):
        instruction = "顺时针绕东侧花园两圈，升高5米，然后悬停；高度不超过40米。"
        document = self.document(instruction,
            {"id": "laps", "action": "motion", "target_id": "east_garden",
             "motion": {"path": {"kind": "boundary", "direction": "clockwise", "laps": 2, "altitude_delta_m": 5}}},
            {"id": "hold", "action": "control", "control": {"state": "airborne"}},
            constraints={"max_altitude_m": 40})
        requirements = self.parse(document, instruction)
        self.assertEqual(requirements.to_dict(), document)
        self.assertEqual(requirements.instruction, instruction)
        self.assertNotIn('"default":', json.dumps(requirement_schema()))
        exported = requirements.to_dict()
        exported["steps"][0]["motion"]["path"]["laps"] = 8
        self.assertEqual(requirements.steps[0]["motion"]["path"]["laps"], 2)

    def test_condition_composition_reviews_and_freeform_geometry_remain_available(self):
        instruction = "观测后按条件重复路径，随后复核；其余保留。"
        condition = {"step_id": "scan", "field": "temperature_c", "op": ">", "value": 40}
        document = self.document(instruction,
            {"id": "scan", "action": "thermal_scan", "target_id": "laboratory"},
            {"id": "choice", "action": "branch", "condition": condition, "then": [
                {"id": "cycle", "action": "repeat", "count": 2, "until": None, "steps": [
                    {"id": "curve", "action": "motion", "target_id": "east_garden", "motion": {
                        "path": {"kind": "curve", "x": "4*cos(t)", "y": "4*sin(t)", "z": 10,
                                 "t_start": 0, "t_end": 6.28,
                                 "flexible": {"scale_groups": [["x", "y"]], "offset_axes": ["z"]}}}}]}], "else": [
                {"id": "polyline", "action": "motion", "target_id": "east_garden", "motion": {
                    "path": {"kind": "polyline", "points_m": [[0, 0, 5], [2, 3, 6]],
                             "flexible": {"scale_groups": [["x", "y"], ["z"]], "offset_axes": []}}}}]},
            {"id": "review", "action": "review", "instruction": "根据证据安排后续工作。"})
        # A modification's literal source need not separately restate inherited leaves.
        document["coverage"][0]["requirement_ids"] = ["review"]
        self.assertEqual(self.parse(document, instruction).to_dict(), document)
        document["steps"][0]["when"] = condition
        with self.assertRaises(TaskRequirementsError) as caught:
            self.parse(document, instruction)
        self.assertEqual(caught.exception.diagnostic["path"], "$.steps[0].when.step_id")

    def test_reference_geometry_cannot_contradict_relative_height_requirement(self):
        instruction = "沿曲线逐渐升高4米。"
        document = self.document(instruction, dict(id="move", action="motion", target_id="stadium",
            motion=dict(path=dict(kind="curve", x="sin(t)", y="sin(2*t)", z="0", t_start=0, t_end=1,
                                  altitude_delta_m=4, flexible=dict(scale_groups=[["x"], ["y"]], offset_axes=["z"])))))
        with self.assertRaises(TaskRequirementsError) as caught:
            self.parse(document, instruction)
        self.assertEqual(caught.exception.code, "inconsistent_height_requirement")
        document["steps"][0]["motion"]["path"]["z"] = "4*t"
        self.assertEqual(self.parse(document, instruction).steps[0]["motion"]["path"]["altitude_delta_m"], 4)
        path = document["steps"][0]["motion"]["path"]
        path.pop("z")
        path["altitude_trend"] = "nondecreasing"
        self.assertNotIn("z", requirements_from_json_text(json.dumps(document), self.catalog, instruction,
                         for_extraction=True).steps[0]["motion"]["path"])
        path.pop("flexible")
        with self.assertRaises(TaskRequirementsError) as caught:
            requirements_from_json_text(json.dumps(document), self.catalog, instruction, for_extraction=True)
        self.assertEqual(caught.exception.diagnostic["path"], "$.steps[0].motion.path.flexible")
        path["flexible"] = dict(scale_groups=[["x"], ["y"]], offset_axes=["z"])
        path.update(t_start=-1, t_end=1000)
        with self.assertRaises(TaskRequirementsError) as caught:
            requirements_from_json_text(json.dumps(document), self.catalog, instruction, for_extraction=True)
        self.assertEqual(caught.exception.diagnostic["path"], "$.steps[0].motion.path.t_end")
        self.assertEqual(caught.exception.code, "parameter_range")

    def test_source_guards_reject_missing_text_fabricated_quotes_and_bad_links(self):
        instruction = "Inspect the library, then return home."
        document = self.document(instruction, {"id": "inspect", "action": "inspect", "target_id": "library"})
        mutations = [
            ({"quote": "Inspect the library"}, "source_coverage_missing"),
            ({"quote": "Inspect another building"}, "invalid_source"),
            ({"requirement_ids": ["missing"]}, "invalid_reference"),
            ({"constraint_keys": ["max_altitude_m"]}, "invalid_reference"),
        ]
        for patch, code in mutations:
            with self.subTest(code=code, patch=patch):
                value = deepcopy(document)
                value["coverage"][0].update(patch)
                with self.assertRaises(TaskRequirementsError) as caught:
                    self.parse(value, instruction)
                self.assertEqual(caught.exception.code, code)
        # Punctuation and whitespace are not additional obligations.
        punctuation = self.document("Inspect the library", document["steps"][0])
        self.parse(punctuation, "  Inspect the library! \n")

    def test_semantic_audit_locally_rejects_empty_approval_and_missing_obligation(self):
        instruction = "Inspect the library, then return home."
        incomplete = self.document(instruction, {"id": "inspect", "action": "inspect", "target_id": "library"})
        requirements = self.parse(incomplete, instruction)
        complete = deepcopy(incomplete)
        complete["steps"].append({"id": "home", "action": "return_home", "target_id": "home"})
        audit = fixture_audit(complete, instruction)
        # A provider's approval cannot override missing structured requirements.
        result = audit_requirements_from_json_text(json.dumps(audit), requirements, instruction)
        self.assertEqual(result.status, "revision_required")
        self.assertTrue(any("home.action" in item["message"] for item in result.issues))
        self.assertTrue(any(item["requirement_id"] == "inspect" for item in result.confirmed_assertions))
        for patch, code in [({"assertions": []}, "missing_assertions"),
                            ({"assertions": [{**audit["assertions"][0], "quote": "invented instruction"}]}, "invalid_source"),
                            ({"assertions": [{**audit["assertions"][0], "expected_json": "{bad}"}]}, "invalid_assertion")]:
            with self.subTest(code=code), self.assertRaises(TaskRequirementsError) as caught:
                audit_requirements_from_json_text(json.dumps({**audit, **patch}), requirements, instruction)
            self.assertEqual(caught.exception.code, code)
        with self.assertRaises(TaskRequirementsError) as caught:
            audit_requirements_from_json_text(json.dumps(audit), requirements, instruction + " changed")
        self.assertEqual(caught.exception.code, "source_mismatch")

    def test_source_assertions_check_quantities_branches_order_and_ending(self):
        instruction = "Observe below 17 metres, fly a half clockwise circuit rising 3 metres, then sample if hotter than 31; otherwise inspect, then return."
        original = self.document(instruction,
            dict(id="see", action="visual", target_id="east_garden", visual=dict(description="object", max_distance_m=17)),
            dict(id="move", action="motion", target_id="east_garden", motion=dict(path=dict(kind="boundary", laps=.5,
                 direction="clockwise", altitude_delta_m=3, altitude_trend="nondecreasing"))),
            dict(id="scan", action="thermal_scan", target_id="library"),
            dict(id="choose", action="branch", condition=dict(step_id="scan", field="temperature_c", op=">", value=31),
                 then=[dict(id="sample", action="capture", target_id="library")],
                 **{"else": [dict(id="inspect", action="inspect", target_id="conference")]}),
            dict(id="home", action="return_home", target_id="home"))
        audit = fixture_audit(original, instruction)
        expected = self.parse(original, instruction)
        accepted = audit_requirements_from_json_text(json.dumps(audit), expected, instruction)
        self.assertEqual(accepted.status, "approved")
        mutations = [
            lambda d: d["steps"][0]["visual"].update(max_distance_m=50),
            lambda d: d["steps"][1]["motion"]["path"].update(laps=2),
            lambda d: d["steps"][1]["motion"]["path"].update(altitude_delta_m=4),
            lambda d: d["steps"][3]["condition"].update(value=25),
            lambda d: d["steps"][3].update(then=d["steps"][3]["else"], **{"else": d["steps"][3]["then"]}),
            lambda d: d["steps"].reverse(),
            lambda d: d["steps"][-1].update(action="control", control=dict(state="airborne")),
        ]
        for mutate in mutations:
            candidate = deepcopy(original)
            mutate(candidate)
            with self.subTest(candidate=candidate):
                # Mechanical comparison also guards a repaired draft before its
                # ordinary structure/reference validation runs.
                differences, _, _ = compare_requirement_assertions(candidate, accepted.confirmed_assertions, instruction)
                self.assertTrue(differences)
        unaudited = deepcopy(audit)
        unaudited["assertions"] = [a for a in unaudited["assertions"] if a["requirement_id"] != "home"]
        rejected = audit_requirements_from_json_text(json.dumps(unaudited), expected, instruction)
        self.assertTrue(any(item["code"] == "unaudited_requirement" for item in rejected.issues))

    def test_unsupported_parameters_and_invalid_json_fail_with_shared_diagnostics(self):
        instruction = "沿花园曲线飞行。"
        document = self.document(instruction, {"id": "curve", "action": "motion", "target_id": "east_garden",
                                              "motion": {"path": {"kind": "curve", "x": "cos(t)"}}})
        for field, value in (("samples", 256), ("x", "__import__('os')")):
            with self.subTest(field=field):
                invalid = deepcopy(document)
                invalid["steps"][0]["motion"]["path"][field] = value
                with self.assertRaises(TaskRequirementsError) as caught:
                    self.parse(invalid, instruction)
                self.assertEqual(caught.exception.diagnostic["path"], f"$.steps[0].motion.path.{field}")
        reference = {"kind": "curve", "x": "cos(t)", "y": "sin(t)", "z": 10, "t_start": 0, "t_end": 6.28}
        for flexible in (
            {"scale_groups": [[]], "offset_axes": []},
            {"scale_groups": [["x", "x"]], "offset_axes": []},
            {"scale_groups": [["x", "y"], ["x"]], "offset_axes": []},
            {"scale_groups": [["q"]], "offset_axes": []},
            {"scale_groups": [], "offset_axes": ["z", "z"]},
            {"scale_groups": [["x"]]},
        ):
            with self.subTest(flexible=flexible):
                invalid = deepcopy(document)
                invalid["steps"][0]["motion"]["path"] = {**reference, "flexible": flexible}
                with self.assertRaises(TaskRequirementsError):
                    self.parse(invalid, instruction)
        for geometry in ({"kind": "curve", "x": "cos(t)"}, {"kind": "polyline"}, {"kind": "boundary"}):
            with self.subTest(geometry=geometry):
                invalid = deepcopy(document)
                invalid["steps"][0]["motion"]["path"] = {
                    **geometry, "flexible": {"scale_groups": [["x", "y"]], "offset_axes": ["z"]}}
                with self.assertRaises(TaskRequirementsError) as caught:
                    self.parse(invalid, instruction)
                self.assertEqual(caught.exception.code,
                    "object_fields" if geometry["kind"] == "boundary" else "incomplete_reference_geometry")
        for geometry, missing in (({"kind": "curve", "x": "cos(t)"}, ["x", "y", "t_start", "t_end"]),
                                  ({"kind": "polyline"}, ["points_m"])):
            with self.subTest(reference=geometry):
                invalid = deepcopy(document)
                invalid["steps"][0]["motion"]["path"] = geometry
                # Older persisted requirements may be sparse; extraction must
                # publish the complete reference already required by validation.
                validate_structure(invalid, requirement_schema(for_extraction=False))
                geometry["flexible"] = {"scale_groups": [], "offset_axes": []}
                with self.assertRaises(TaskProtocolError) as caught:
                    validate_structure(invalid, requirement_schema(for_extraction=True))
                self.assertEqual(caught.exception.code, "object_fields")
                self.assertEqual(caught.exception.diagnostic["missing"], [key for key in missing if key not in geometry])
        with self.assertRaises(TaskRequirementsError) as caught:
            requirements_from_json_text('{"status":"ready","status":"ready"}', self.catalog, instruction)
        self.assertEqual(caught.exception.code, "duplicate_field")
        clarification = {"status": "needs_clarification", "summary": "请明确对象。", "steps": [], "constraints": {}, "coverage": []}
        self.assertEqual(self.parse(clarification, instruction).status, "needs_clarification")


if __name__ == "__main__":
    unittest.main()
