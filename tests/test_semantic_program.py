"""Task program semantics: decisions use completed evidence, never planned sensor truth."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from uav_llm_demo.semantic_tasks import (
    PROGRAM_LIMITS, SemanticExecutionError, SemanticMission, iter_plan_actions,
    semantic_plan_from_json_text, task_catalog,
)
from tests.test_semantic_tasks import parse, proposal, scene, step


def compare(step_id, field="temperature_c", op=">", value=40):
    return dict(step_id=step_id, field=field, op=op, value=value)


def repeat(body, count=3, until=None, when=None, step_id="loop"):
    return dict(id=step_id, action="repeat", count=count, until=until, steps=body, when=when)


def branch(condition, yes, no=None, step_id="choice"):
    return {"id": step_id, "action": "branch", "condition": condition, "then": yes, "else": no or []}


def telemetry(field, op, value):
    return dict(source="telemetry", field=field, op=op, value=value)


def program(nodes):
    return proposal([*nodes, step("home", "return_home", "home")])


def mission(nodes, telemetry_values=None):
    value = SemanticMission(scene(), "自由任务描述", mission_id="program-mission")
    value.accept(parse(program(nodes)), telemetry=telemetry_values)
    return value


def finish(value, now=10, telemetry_values=None):
    current = value.current_step
    return value.observe(value.current_goal, 0, max(.05, current["duration_seconds"]), now,
                         telemetry=telemetry_values)


class ProgramSchemaTests(unittest.TestCase):
    def test_arbitrary_numeric_thresholds_cross_target_boolean_composition(self):
        condition = {"all": [compare("scan", value=40),
                             {"any": [compare("scan", "anomaly", "==", False),
                                      {"not": compare("scan", "temperature_c", "<=", 20)}]}]}
        value = parse(program([step("scan", "thermal_scan", "laboratory"),
                               step("record", "capture", "library", when=condition)]))
        self.assertEqual(value.steps[1]["when"], condition)
        self.assertEqual([item["id"] for item in iter_plan_actions(value)], ["scan", "record", "home"])

    def test_ordered_comparisons_reject_booleans_strings_and_wrong_types(self):
        for condition in (compare("scan", "anomaly", ">", True), compare("scan", value="40"),
                          compare("scan", value=True), compare("scan", "source", "==", "secret"),
                          {"all": []}, {"any": []}, {"not": None},
                          {"step_id": "scan", "field": "anomaly", "equals": True}):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                parse(program([step("scan", "thermal_scan", "laboratory"),
                               step("capture", "capture", "library", when=condition)]))

    def test_telemetry_fields_are_explicit_and_numeric(self):
        for condition in (telemetry("battery_percent", "<=", 20), telemetry("elapsed_seconds", ">=", 300)):
            self.assertEqual(parse(program([step("visit", "visit", "library", when=condition)])).steps[0]["when"], condition)
        for condition in (telemetry("future_position", ">", 3), telemetry("battery_percent", "==", False)):
            with self.assertRaises(ValueError):
                parse(program([step("visit", "visit", "library", when=condition)]))

    def test_forward_and_conditionally_missing_observations_rejected(self):
        variants = [
            [step("record", "capture", "library", when=compare("scan")), step("scan", "thermal_scan", "laboratory")],
            [step("scan", "thermal_scan", "laboratory", when=telemetry("battery_percent", ">", 50)),
             step("record", "capture", "library", when=compare("scan"))],
            [branch(telemetry("battery_percent", ">", 50), [step("scan", "thermal_scan", "laboratory")]),
             step("record", "capture", "library", when=compare("scan"))],
            [repeat([step("scan", "thermal_scan", "laboratory")], when=telemetry("battery_percent", ">", 50)),
             step("record", "capture", "library", when=compare("scan"))],
        ]
        for nodes in variants:
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                parse(program(nodes))

    def test_repeat_until_can_read_body_result_but_body_cannot_read_future(self):
        parsed = parse(program([repeat([step("scan", "thermal_scan", "library")], count=4,
                                       until=compare("scan", op="<", value=40))]))
        self.assertEqual(parsed.steps[0]["until"]["value"], 40)
        with self.assertRaises(ValueError):
            parse(program([repeat([step("record", "capture", "library", when=compare("scan")),
                                   step("scan", "thermal_scan", "library")])]))

    def test_global_ids_and_structure_budgets(self):
        with self.assertRaises(ValueError):
            parse(program([repeat([step("same", "visit", "library")], step_id="same")]))
        with self.assertRaises(ValueError):
            parse(program([branch(telemetry("battery_percent", ">", 20), [step("same", "visit", "library")],
                                  [step("same", "visit", "laboratory")])]))
        for count in (0, -1, True, 1.5, 1001):
            with self.subTest(count=count), self.assertRaises(ValueError):
                parse(program([repeat([step("visit", "visit", "library")], count=count)]))
        nested = step("visit", "visit", "library")
        for index in range(8):
            nested = repeat([nested], step_id=f"nested_{index}")
        with self.assertRaises(ValueError):
            parse(program([nested]))

    def test_early_return_inside_branch_or_loop_rejected(self):
        for nodes in ([repeat([step("land", "return_home", "home")])],
                      [branch(telemetry("battery_percent", "<", 20), [step("land", "return_home", "home")])]):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                parse(program(nodes))

    def test_user_durations_are_supported_below_defaults_and_up_to_budget(self):
        value = parse(program([step("scan", "thermal_scan", "laboratory", .25),
                               step("wait", "visit", "library", 600)]))
        self.assertEqual(value.steps[0]["duration_seconds"], .25)
        self.assertEqual(value.steps[1]["duration_seconds"], 600)

    def test_historical_references_need_explicit_known_evidence_and_typed_field(self):
        value = mission([step("scan", "thermal_scan", "laboratory")])
        finish(value)
        record = value.evidence[0]
        condition = {"evidence_id": record["evidence_id"], "field": "temperature_c", "op": ">", "value": 40}
        data = program([step("record", "capture", "library", when=condition)])
        with self.assertRaises(ValueError):
            parse(data)
        catalog = task_catalog(scene())
        catalog["mission_context"] = value.catalog_context()
        self.assertEqual(semantic_plan_from_json_text(json.dumps(data), catalog).steps[0]["when"], condition)
        condition["evidence_id"] = "unknown"
        with self.assertRaises(ValueError):
            semantic_plan_from_json_text(json.dumps(data), catalog)

    def test_iter_actions_traverses_both_branch_arms_and_loop_bodies(self):
        value = parse(program([branch(telemetry("battery_percent", ">", 20),
                                      [repeat([step("inspect", "inspect", "library")])],
                                      [step("visit", "visit", "laboratory")])]))
        self.assertEqual([node["id"] for node in iter_plan_actions(value)], ["inspect", "visit", "home"])


class ProgramExecutionTests(unittest.TestCase):
    def test_threshold_is_distinct_from_fixed_anomaly_flag(self):
        value = mission([step("scan", "thermal_scan", "library"),
                         step("capture", "capture", "laboratory", when=compare("scan", op=">", value=25))])
        finish(value)
        self.assertFalse(value.evidence[0]["result"]["anomaly"])
        self.assertEqual(value.current_step["id"], "capture")
        finish(value)
        finish(value)
        self.assertEqual(value.status, "completed")

    def test_repeat_executes_each_skill_with_distinct_bound_evidence(self):
        value = mission([repeat([step("scan", "thermal_scan", "laboratory"),
                                 step("record", "capture", "library", when=compare("scan"))], count=3)])
        for index in range(6):
            self.assertEqual(value.current_step["loop_iterations"], {"loop": index // 2 + 1})
            finish(value, now=index + 10)
        self.assertEqual(value.current_step["action"], "return_home")
        self.assertEqual(len({item["execution_id"] for item in value.evidence}), 6)
        self.assertEqual(len({item["evidence_id"] for item in value.evidence}), 6)
        self.assertEqual(value.to_dict()["loops"][0]["iteration"], 3)
        self.assertEqual(value.to_dict()["loops"][0]["status"], "completed")
        finish(value)
        self.assertEqual(value.status, "completed")
        self.assertEqual([item["id"] for item in value.to_dict()["steps"]],
                         ["scan", "record", "scan", "record", "scan", "record", "home"])

    def test_until_is_evaluated_after_body_and_uses_latest_iteration(self):
        value = mission([repeat([step("scan", "thermal_scan", "library")], count=5,
                                 until=compare("scan", value=40))])
        with patch.object(SemanticMission, "_result", side_effect=[({"temperature_c": 20.0}, "test_sensor"),
                                                                 ({"temperature_c": 45.0}, "test_sensor")]):
            finish(value)
            self.assertEqual(value.current_step["loop_iterations"], {"loop": 2})
            finish(value)
        self.assertEqual(value.current_step["action"], "return_home")
        self.assertEqual(value.to_dict()["loops"][0]["iteration"], 2)

    def test_loop_stops_on_real_elapsed_telemetry_after_body(self):
        value = mission([repeat([step("visit", "visit", "library")], count=None,
                                 until=telemetry("elapsed_seconds", ">=", 20))],
                        telemetry_values={"elapsed_seconds": 0, "battery_percent": 100})
        finish(value, 10, {"elapsed_seconds": 10, "battery_percent": 90})
        self.assertEqual(value.current_step["loop_iterations"], {"loop": 2})
        finish(value, 20, {"elapsed_seconds": 20, "battery_percent": 80})
        self.assertEqual(value.current_step["id"], "home")

    def test_branch_decision_is_frozen_on_entry(self):
        value = mission([branch(telemetry("battery_percent", ">", 50),
                                 [step("first", "visit", "library"), step("second", "visit", "laboratory")],
                                 [step("other", "visit", "conference")])],
                        telemetry_values={"battery_percent": 90})
        finish(value, telemetry_values={"battery_percent": 10})
        self.assertEqual(value.current_step["id"], "second")
        finish(value, telemetry_values={"battery_percent": 10})
        self.assertEqual(value.current_step["id"], "home")
        self.assertNotIn("other", [item["id"] for item in value.to_dict()["steps"]])

    def test_nested_repeats_have_independent_iteration_identity(self):
        value = mission([repeat([repeat([step("visit", "visit", "library")], count=2, step_id="inner")], count=2)])
        for expected in ({"loop": 1, "inner": 1}, {"loop": 1, "inner": 2},
                         {"loop": 2, "inner": 1}, {"loop": 2, "inner": 2}):
            self.assertEqual(value.current_step["loop_iterations"], expected)
            finish(value)
        self.assertEqual(value.current_step["id"], "home")

    def test_false_repeat_condition_skips_whole_body_without_evidence(self):
        value = mission([repeat([step("visit", "visit", "library")], when=telemetry("battery_percent", ">", 50))],
                        telemetry_values={"battery_percent": 20})
        self.assertEqual(value.current_step["id"], "home")
        self.assertEqual(value.evidence, [])
        self.assertEqual(value.to_dict()["loops"][0]["status"], "skipped")

    def test_loop_budget_halts_without_false_completion_and_return_remains_possible(self):
        with patch.dict(PROGRAM_LIMITS, max_loop_iterations=2):
            value = mission([repeat([step("visit", "visit", "library")], count=None)])
            finish(value)
            with self.assertRaises(SemanticExecutionError) as caught:
                finish(value)
            self.assertEqual(caught.exception.code, "execution_budget")
        self.assertEqual(value.status, "safety_wait")
        self.assertEqual(len(value.evidence), 2)
        self.assertIsNone(value.current_step)
        self.assertFalse(any(record["action"] == "return_home" for record in value.evidence))
        value.accept(parse(program([])), preserve_evidence=True)
        self.assertEqual(value.current_step["action"], "return_home")
        finish(value)
        self.assertEqual(value.status, "completed")

    def test_action_budget_does_not_block_emergency_task_return(self):
        with patch.dict(PROGRAM_LIMITS, max_actions=1):
            value = mission([step("visit", "visit", "library"), step("second", "visit", "laboratory")])
            with self.assertRaises(SemanticExecutionError):
                finish(value)
            value.accept(parse(program([])), preserve_evidence=True)
            self.assertEqual(value.current_step["action"], "return_home")
            finish(value)
        self.assertEqual(value.status, "completed")

    def test_missing_telemetry_safely_waits_without_creating_result(self):
        value = SemanticMission(scene())
        with self.assertRaises(SemanticExecutionError) as caught:
            value.accept(parse(program([step("visit", "visit", "library", when=telemetry("battery_percent", ">", 50))])))
        self.assertEqual(caught.exception.code, "missing_telemetry")
        self.assertEqual(value.status, "safety_wait")
        self.assertEqual(value.evidence, [])


class TaskRevisionTests(unittest.TestCase):
    def test_revision_cancels_partial_skill_preserves_history_and_unique_occurrences(self):
        value = mission([step("scan", "thermal_scan", "laboratory"), step("record", "capture", "library", 2)])
        finish(value)
        old = deepcopy(value.evidence)
        value.observe(value.current_goal, 0, 1, 11)
        value.accept(parse(program([step("record", "capture", "conference", 2)])), preserve_evidence=True)
        self.assertEqual(value.revision, 2)
        self.assertEqual(value.mission_id, "program-mission")
        self.assertEqual(value.evidence, old)
        records = value.to_dict()["steps"]
        self.assertEqual([item["status"] for item in records[:3]], ["completed", "cancelled", "cancelled"])
        self.assertNotEqual(records[1]["execution_id"], value.current_step["execution_id"])
        self.assertFalse(value.observe(value.current_goal, 0, 1, 12))
        self.assertTrue(value.observe(value.current_goal, 0, 1, 13))
        self.assertEqual(value.evidence[-1]["revision"], 2)
        self.assertEqual(value.evidence[-1]["target_id"], "conference")

    def test_old_step_id_cannot_substitute_for_new_revision_scan(self):
        value = mission([step("scan", "thermal_scan", "laboratory")])
        finish(value)
        replacement = parse(program([step("scan", "thermal_scan", "library"),
                                     step("record", "capture", "laboratory", when=compare("scan", value=40))]))
        value.accept(replacement, preserve_evidence=True)
        self.assertEqual(value.current_step["id"], "scan")
        finish(value)
        self.assertEqual(value.current_step["id"], "home")
        self.assertFalse(value.evidence[-1]["result"]["anomaly"])

    def test_old_explicit_evidence_can_support_new_task_condition(self):
        value = mission([step("scan", "thermal_scan", "laboratory")])
        finish(value)
        condition = {"evidence_id": value.evidence[0]["evidence_id"], "field": "temperature_c", "op": ">=", "value": 65}
        catalog = task_catalog(scene())
        catalog["mission_context"] = value.catalog_context()
        new = semantic_plan_from_json_text(json.dumps(program([step("record", "capture", "library", when=condition)])), catalog)
        value.accept(new, preserve_evidence=True)
        self.assertEqual(value.current_step["id"], "record")
        finish(value)
        self.assertEqual([item["revision"] for item in value.evidence], [1, 2])

    def test_clarification_preserves_old_plan_steps_evidence_but_cannot_execute(self):
        value = mission([step("scan", "thermal_scan", "laboratory"), step("record", "capture", "library")])
        finish(value)
        original_plan, original_steps = value.plan.to_dict(), deepcopy(value.to_dict()["steps"])
        clarification = proposal()
        clarification.update(status="needs_clarification", summary="没有投放能力，请修改任务。", steps=[])
        value.accept(parse(clarification), preserve_evidence=True)
        self.assertEqual(value.status, "needs_clarification")
        self.assertEqual(value.plan.to_dict(), original_plan)
        self.assertEqual(value.to_dict()["plan_revision"], 1)
        self.assertEqual(value.to_dict()["revision"], 2)
        self.assertEqual(value.to_dict()["steps"], original_steps)
        self.assertEqual(len(value.evidence), 1)
        self.assertIsNone(value.current_step)
        self.assertFalse(value.observe((8, 13, 4), 0, 3, 20))
        self.assertEqual([item["id"] for item in value.catalog_context()["uncompleted_steps"]], ["record", "home"])

    def test_interpreting_context_keeps_accepted_instruction_and_active_skill(self):
        value = mission([step("scan", "thermal_scan", "laboratory")])
        accepted = value.instruction
        execution = value.current_step["execution_id"]
        value.instruction = "取消当前项，改去图书馆"
        value.status = "interpreting"
        self.assertIsNone(value.current_step)
        context = value.catalog_context()
        self.assertEqual(context["accepted_instruction"], accepted)
        self.assertEqual(context["current_step"]["execution_id"], execution)
        clarification = proposal()
        clarification.update(status="needs_clarification", summary="请确认地点", steps=[])
        value.accept(parse(clarification), preserve_evidence=True)
        self.assertEqual(value.accepted_instruction, accepted)
        self.assertEqual(value.catalog_context()["current_step"]["execution_id"], execution)

    def test_foreign_mission_or_vehicle_evidence_cannot_enter_context_or_conditions(self):
        value = mission([step("scan", "thermal_scan", "laboratory")])
        finish(value)
        for field, foreign in (("mission_id", "another-mission"), ("vehicle_id", "uav-2")):
            record = deepcopy(value.evidence[0])
            record["evidence_id"] = "foreign-evidence"
            record[field] = foreign
            value.evidence.append(record)
            self.assertNotIn(record, value.catalog_context()["evidence"])
            condition = {"evidence_id": "foreign-evidence", "field": "temperature_c", "op": ">", "value": 40}
            catalog = task_catalog(scene())
            catalog["mission_context"] = {"mission_id": value.mission_id, "evidence": value.evidence}
            with self.assertRaises(ValueError):
                semantic_plan_from_json_text(json.dumps(program([step("record", "capture", "library", when=condition)])), catalog)
            with self.assertRaises(SemanticExecutionError):
                value._evaluate(condition)
            value.evidence.pop()

    def test_catalog_context_and_snapshots_cannot_change_live_records(self):
        value = mission([repeat([step("scan", "thermal_scan", "laboratory")], count=2)])
        finish(value)
        context = value.catalog_context()
        context["evidence"][0]["result"]["temperature_c"] = -1
        context["plan"]["steps"][0]["count"] = 1000
        state = value.to_dict()
        state["loops"][0]["iteration"] = 500
        state["program"][0]["steps"][0]["target_id"] = "home"
        self.assertEqual(value.evidence[0]["result"]["temperature_c"], 68)
        self.assertEqual(value.plan.steps[0]["count"], 2)
        self.assertEqual(value.to_dict()["loops"][0]["iteration"], 2)
        self.assertEqual(value.current_goal, (23, 12, 3))


if __name__ == "__main__":
    unittest.main()
