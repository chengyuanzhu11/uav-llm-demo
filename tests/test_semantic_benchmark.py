"""Independent oracle adversarial checks; never exercise a remote provider."""
from copy import deepcopy
import json
import math
import time
import unittest
from unittest.mock import patch

from uav_llm_demo.agents import PlannerProviderError
from uav_llm_demo.semantic_benchmark import (
    _BudgetedAgent, evaluate_oracle, load_dataset, redact, run_case, summarize,
)


WORLD = {"home": dict(center=[0, 0, 0], arrival=[0, 0, 0]),
         "library": dict(center=[5, 5, 0], arrival=[5, 5, 2]),
         "garden": dict(center=[5, 5, 2], arrival=[5, 5, 2],
                        boundary=[[4, 4], [6, 4], [6, 6], [4, 6]])}


def evidence(action="inspect", target="library", when=1, eid="one", result=None, source=None, **extra):
    sources = dict(inspect="simulated_inspection", thermal_scan="simulated_thermal_sensor", capture="simulated_capture",
                   motion="runtime_trajectory_tracking", return_home="runtime_telemetry", control="runtime_telemetry")
    results = dict(inspect={"inspected": True}, thermal_scan={"simulated": True, "temperature_c": 29},
                   capture={"simulated": True, "captured": True},
                   motion={"completed": True}, return_home={"returned": True}, control={"completed": True})
    return dict(mission_id="mission", vehicle_id="uav-1", execution_id=eid, action=action,
                target_id=target, simulation_time=when, duration_seconds=1,
                result=results[action] if result is None else result, source=source or sources[action], **extra)


def predicate(action="inspect", target="library", count=1, **extra):
    return dict(type="evidence", action=action, target=target, count=count, **extra)


def sample(t=0, position=(0, 0, 0), speed=0, yaw=0, eid=None, phase=None):
    return dict(t=t, position=list(position), speed_mps=speed, yaw=yaw, execution_id=eid, phase=phase)


class OracleTests(unittest.TestCase):
    def evaluate(self, req, records=(), samples=()):
        return evaluate_oracle(req, evidence=list(records), telemetry=list(samples), world=WORLD, mission_id="mission")

    def test_completion_without_required_work_fails(self):
        self.assertFalse(self.evaluate(predicate(), [evidence("return_home", "home")])["satisfied"])
        self.assertTrue(self.evaluate(predicate(), [evidence()])["satisfied"])

    def test_wrong_target_source_vehicle_mission_or_unsuccessful_result_fails(self):
        changes = [dict(target_id="garden"), dict(source="model_claim"), dict(vehicle_id="uav-2"),
                   dict(mission_id="old"), dict(result={"inspected": False})]
        for change in changes:
            with self.subTest(change=change):
                row = {**evidence(), **change}
                self.assertFalse(self.evaluate(predicate(), [row])["satisfied"])

    def test_duplicate_evidence_cannot_satisfy_multiple_executions(self):
        row = evidence()
        self.assertFalse(self.evaluate(predicate(count=2), [row, deepcopy(row)])["satisfied"])
        self.assertTrue(self.evaluate(predicate(count=2), [row, evidence(eid="two")])["satisfied"])

    def test_order_is_measured_not_asserted_by_program(self):
        req = dict(type="sequence", items=[predicate(), predicate("return_home", "home")])
        self.assertTrue(self.evaluate(req, [evidence(when=1), evidence("return_home", "home", when=2, eid="return")])["satisfied"])
        self.assertFalse(self.evaluate(req, [evidence(when=3), evidence("return_home", "home", when=2, eid="return")])["satisfied"])

    def test_occurrences_distinguish_interleaved_loops(self):
        req = dict(type="sequence", items=[predicate(occurrence=1), predicate("thermal_scan", occurrence=1),
                                            predicate(occurrence=2), predicate("thermal_scan", occurrence=2)])
        good = [evidence(when=1), evidence("thermal_scan", when=2, eid="scan1"),
                evidence(when=3, eid="two"), evidence("thermal_scan", when=4, eid="scan2")]
        self.assertTrue(self.evaluate(req, good)["satisfied"])
        good[1]["simulation_time"] = 3.5
        self.assertFalse(self.evaluate(req, good)["satisfied"])

    def test_observed_condition_requires_only_the_selected_branch(self):
        req = dict(type="if", observation=predicate("thermal_scan"), field="temperature_c", op=">", value=40,
                   then=predicate("inspect", "garden"), **{"else": predicate("inspect", "library")})
        cold = evidence("thermal_scan", when=1)
        self.assertTrue(self.evaluate(req, [cold, evidence(when=2, eid="inspection")])["satisfied"])
        hot = {**cold, "result": {"simulated": True, "temperature_c": 68}}
        self.assertFalse(self.evaluate(req, [hot, evidence(when=2, eid="inspection")])["satisfied"])
        self.assertTrue(self.evaluate(req, [hot, evidence(target="garden", when=2, eid="inspection")])["satisfied"])
        self.assertFalse(self.evaluate(req, [evidence(when=2)])["satisfied"])

    def test_condition_cannot_use_an_action_before_observation(self):
        req = dict(type="if", observation=predicate("thermal_scan"), field="temperature_c", op=">", value=20,
                   then=predicate(), **{"else": predicate("inspect", "garden")})
        self.assertFalse(self.evaluate(req, [evidence(when=1), evidence("thermal_scan", when=2, eid="scan")])["satisfied"])

    def test_dataset_conditionals_require_return_after_the_chosen_branch(self):
        cases = {case["id"]: case for case in load_dataset()["cases"]}
        for case_id, target in (("dev_conditional_false", "library"),
                                ("holdout_conditional_true", "laboratory")):
            for temperature in (29, 68):
                with self.subTest(case=case_id, temperature=temperature):
                    scan = evidence("thermal_scan", target, when=4, eid="scan",
                                    result={"simulated": True, "temperature_c": temperature})
                    scan["duration_seconds"] = 3
                    branch = (evidence("capture", target, when=6, eid="branch") if temperature > 40
                              else evidence(target="conference", when=6, eid="branch"))
                    branch["duration_seconds"] = 2
                    home = evidence("return_home", "home", when=8, eid="home")
                    records = [scan, branch, home]
                    oracle = cases[case_id]["oracle"]
                    self.assertTrue(self.evaluate(oracle, records, [sample(9)])["satisfied"])
                    # All required actions exist and final pose is home, but
                    # returning before branch completion violates "finally".
                    home["simulation_time"] = 5
                    self.assertFalse(self.evaluate(oracle, records, [sample(9)])["satisfied"])

    def test_forbidden_action_is_not_hidden_by_successful_final_goal(self):
        self.assertFalse(self.evaluate(predicate("return_home", "home", 0), [evidence("return_home", "home")])["satisfied"])

    def test_landing_checks_physical_pose_speed_and_dwell(self):
        req = dict(type="goal", target="home", state="landed", dwell_seconds=1)
        good = [sample(i / 10) for i in range(12)]
        self.assertTrue(self.evaluate(req, samples=good)["satisfied"])
        for bad in ([sample(1, (0, 0, 2))], [sample(1, speed=1)], [sample(1, (.2, 0, 0))], good[:3]):
            self.assertFalse(self.evaluate(req, samples=bad)["satisfied"])

    def test_hover_altitude_heading_requires_measured_state(self):
        req = dict(type="goal", target="library", state="airborne", altitude_m=10, yaw_deg=90)
        self.assertTrue(self.evaluate(req, samples=[sample(1, (5, 5, 2), yaw=math.pi / 2)])["satisfied"])
        self.assertFalse(self.evaluate(req, samples=[sample(1, (5, 5, 2), yaw=0)])["satisfied"])

    def test_intermediate_pose_is_bound_to_its_completed_execution(self):
        req = dict(type="pose", target="library", state="airborne", altitude_m=10, yaw_deg=90)
        records = [evidence("control", eid="correct")]
        self.assertTrue(self.evaluate(req, records, [sample(1, (5, 5, 2), yaw=math.pi / 2, eid="correct")])["satisfied"])
        self.assertFalse(self.evaluate(req, records, [sample(1, (5, 5, 2), yaw=math.pi / 2, eid="different")])["satisfied"])

    def square_samples(self, laps=1, sign=1, size=1):
        points = []
        for i in range(80 * laps + 1):
            angle = sign * 2 * math.pi * i / 80
            x, y = math.cos(angle), math.sin(angle)
            scale = size / max(abs(x), abs(y))
            points.append(sample(i / 10, (5 + x * scale, 5 + y * scale, 2), eid="ring", phase="tracking"))
        return points

    def test_direction_laps_and_boundary_are_measured_from_flight(self):
        req = dict(type="circuit", target="garden", direction="clockwise", laps=1)
        rows = [evidence("motion", "garden", 10, "ring")]
        self.assertTrue(self.evaluate(req, rows, self.square_samples())["satisfied"])
        for samples in (self.square_samples(sign=-1), self.square_samples(laps=2), self.square_samples(size=.2)):
            self.assertFalse(self.evaluate(req, rows, samples)["satisfied"])
        self.assertFalse(self.evaluate(req, rows, [sample()])["satisfied"])

    def test_two_laps_can_be_two_independent_completed_motion_nodes(self):
        req = dict(type="circuit", target="garden", direction="clockwise", laps=2)
        samples = self.square_samples()
        samples += [{**s, "execution_id": "second", "t": s["t"] + 10} for s in self.square_samples()]
        rows = [evidence("motion", "garden", 9, "ring"), evidence("motion", "garden", 19, "second")]
        self.assertTrue(self.evaluate(req, rows, samples)["satisfied"])
        rows[1]["execution_id"] = "wrong"
        self.assertFalse(self.evaluate(req, rows, samples)["satisfied"])

    def test_heading_must_be_aligned_from_start_of_motion_tracking(self):
        req = dict(type="circuit", target="garden", direction="clockwise", laps=1, heading="face_center")
        samples = self.square_samples()
        for s in samples:
            s["yaw"] = math.atan2(5 - s["position"][0], -(5 - s["position"][1]))
        rows = [evidence("motion", "garden", 10, "ring")]
        self.assertTrue(self.evaluate(req, rows, samples)["satisfied"])
        samples[0]["yaw"] += 1
        self.assertFalse(self.evaluate(req, rows, samples)["satisfied"])

    def test_all_failures_stay_in_denominator_and_expected_negative_is_separate(self):
        statuses = ["completed", "timeout", "clarification", "failed"]
        results = [dict(status=s, oracle_success=i == 0, service_success=i == 0, false_completion=False,
                        interpretation_calls=1, wall_seconds=1, expected_oracle_success=True, regression_pass=i == 0)
                   for i, s in enumerate(statuses)]
        summary = summarize(results)
        self.assertEqual(summary["runs"], 4)
        self.assertEqual(summary["success_rate"], .25)
        self.assertEqual(summary["outcomes"], {s: 1 for s in statuses})
        self.assertIsNone(summarize([])["success_rate"])

    def test_sensitive_fields_and_provider_echoes_are_redacted(self):
        secret = "test-private-key-value"
        cleaned = redact(dict(instruction="do " + secret, nested={"api_key": secret},
                              error="Bearer opaque-token", token="other"), (secret,))
        self.assertNotIn(secret, json.dumps(cleaned))
        self.assertNotIn("opaque-token", json.dumps(cleaned))
        self.assertEqual(cleaned["token"], "[redacted]")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        guard = patch("uav_llm_demo.agents._post_deepseek", side_effect=AssertionError("Network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_campus_executor_detects_omission_without_sending_oracle_to_model(self):
        case = next(c for c in load_dataset()["cases"] if c["id"] == "holdout_initial_omission")
        seen = []
        original = _BudgetedAgent.interpret_task

        def recording(agent, catalog, instruction, feedback="", **kwargs):
            seen.append((deepcopy(catalog), instruction))
            return original(agent, catalog, instruction, feedback, **kwargs)

        with patch.object(_BudgetedAgent, "interpret_task", recording):
            report = run_case(case, wall_budget=30)
        self.assertTrue(report["service_success"])
        self.assertFalse(report["oracle_success"])
        self.assertTrue(report["false_completion"])
        self.assertTrue(report["regression_pass"])
        self.assertEqual(report["interpretation_calls"], 1)
        self.assertNotIn("offline-benchmark-placeholder", json.dumps(report))
        self.assertEqual(seen[0][1], case["instruction"])
        self.assertNotIn("oracle", seen[0][0])
        self.assertNotIn("offline_responses", seen[0][0])

    def test_remote_virtual_or_missing_key_fails_before_service(self):
        case = load_dataset()["cases"][0]
        with self.assertRaises(ValueError):
            run_case(case, mode="virtual", backend="deepseek")
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            run_case(case, mode="realtime", backend="deepseek")

    def test_budget_prevents_extra_calls_and_expired_requests(self):
        agent = _BudgetedAgent(replies=[], api_key="test-placeholder", max_calls=1)
        with self.assertRaises(PlannerProviderError):
            agent.interpret_task({}, "任务")
        with self.assertRaises(PlannerProviderError):
            agent.interpret_task({}, "任务")
        self.assertEqual(agent.calls, 1)
        self.assertTrue(agent.budget_exhausted)
        expired = _BudgetedAgent(replies=[], api_key="test-placeholder", deadline=time.monotonic() - 1)
        with self.assertRaises(PlannerProviderError):
            expired.interpret_task({}, "任务")
        self.assertEqual(expired.calls, 0)

    def test_runtime_failure_is_a_result_and_does_not_drop_the_case(self):
        case = next(c for c in load_dataset()["cases"] if c["id"] == "holdout_initial_omission")
        with patch("uav_llm_demo.semantic_benchmark.MissionService.submit_task", side_effect=RuntimeError("private-value")):
            result = run_case(case)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["termination_reason"], "runtime_error")
        self.assertFalse(result["oracle_success"])
        self.assertFalse(result["regression_pass"])
        self.assertNotIn("private-value", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
