"""Independent oracle adversarial checks; never exercise a remote provider."""
from copy import deepcopy
import json
import math
import time
import unittest
from unittest.mock import patch

from uav_llm_demo.agents import PlannerProviderError
from uav_llm_demo.semantic_benchmark import (
    _BudgetedAgent, _usage_cost, compare_policies, evaluate_oracle, load_dataset, redact,
    recovery_metrics, run_case, summarize,
)
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_tasks import task_catalog


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


def sample(t=0, position=(0, 0, 0), speed=0, yaw=0, eid="execution", phase=None):
    return dict(t=t, position=list(position), speed_mps=speed, yaw=yaw, execution_id=eid, phase=phase, action=None if phase == "tracking" else "control")


class OracleTests(unittest.TestCase):
    def evaluate(self, req, records=(), samples=(), **context):
        return evaluate_oracle(req, evidence=list(records), telemetry=list(samples), world=WORLD, mission_id="mission", **context)

    def test_completion_without_required_work_fails(self):
        self.assertFalse(self.evaluate(predicate(), [evidence("return_home", "home")])["satisfied"])
        self.assertTrue(self.evaluate(predicate(), [evidence()])["satisfied"])

    def test_targetless_review_requires_valid_execution_evidence_before_work(self):
        review = predicate("review", None, result={"reviewed": True})
        req = dict(type="sequence", items=[review, predicate()])
        inspection = evidence(when=2, eid="inspection")
        verified = evidence("review", None, when=1, eid="review",
                            result={"reviewed": True}, source="validated_execution_review")
        self.assertFalse(self.evaluate(req, [inspection])["satisfied"])
        self.assertTrue(self.evaluate(req, [verified, inspection])["satisfied"])
        for change in ({"source": "model_claim"}, {"result": {"reviewed": False}},
                       {"simulation_time": 3}, {"target_id": "home"}):
            with self.subTest(change=change):
                self.assertFalse(self.evaluate(req, [{**verified, **change}, inspection])["satisfied"])
        for invalid in (predicate("inspect", None), {k: v for k, v in review.items() if k != "target"}):
            with self.assertRaises(ValueError):
                self.evaluate(invalid)

    def test_native_landing_requires_disarmed_contact_for_the_whole_dwell(self):
        requirement = dict(type="goal", target="home", state="landed", dwell_seconds=1)
        rows = [{**sample(t), "runtime": "gazebo", "landed": True, "armed": t < 1} for t in (i / 10 for i in range(11))]
        self.assertFalse(self.evaluate(requirement, samples=rows)["satisfied"])
        rows.extend({**sample(t / 10), "runtime": "gazebo", "landed": True, "armed": False} for t in range(11, 21))
        self.assertTrue(self.evaluate(requirement, samples=rows)["satisfied"])
        rows[-1]["landed"] = False
        self.assertFalse(self.evaluate(requirement, samples=rows)["satisfied"])

    def test_zero_dwell_goal_uses_final_native_pose_after_action_clears(self):
        req = dict(type="goal", target="home", state="landed")
        active = {**sample(1), "runtime": "gazebo", "landed": True, "armed": True}
        terminal = {**sample(2, eid=None), "action": None, "runtime": "gazebo", "landed": True, "armed": False}
        result = self.evaluate(req, samples=[active, terminal])
        self.assertTrue(result["satisfied"])
        self.assertEqual((result["first"], result["last"]), (2, 2))
        for change in ({"armed": True}, {"landed": False}, {"speed_mps": .1}, {"position": [1, 0, 0]}):
            with self.subTest(change=change):
                self.assertFalse(self.evaluate(req, samples=[terminal, {**terminal, "t": 3, **change}])["satisfied"])
        idle = [{**terminal, "t": i/10} for i in range(21)]
        self.assertFalse(self.evaluate({**req, "dwell_seconds": 1}, samples=idle)["satisfied"])
        self.assertFalse(self.evaluate({**req, "type": "pose"}, samples=idle)["satisfied"])
        sequence = dict(type="sequence", items=[predicate(), req])
        self.assertFalse(self.evaluate(sequence, [evidence(when=3)], samples=[terminal])["satisfied"])

    def test_wrong_target_source_vehicle_mission_or_unsuccessful_result_fails(self):
        changes = [dict(target_id="garden"), dict(source="model_claim"), dict(vehicle_id="uav-2"),
                   dict(mission_id="old"), dict(result={"inspected": False}),
                   dict(observation={"status": "unknown"}),
                   dict(observation={"status": "valid", "observed_at": 0, "valid_until": .5, "confidence": 1})]
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
        cold.update(evidence_id="original", observation=dict(status="valid", observed_at=0, valid_until=1.5, confidence=1))
        fresh = evidence("thermal_scan", when=3, eid="refresh", evidence_id="fresh", refresh_of="original",
                         result={"simulated": True, "temperature_c": 68},
                         observation=dict(status="valid", observed_at=2, valid_until=5, confidence=1))
        rows = [cold, evidence(when=1.4, eid="cold_inspection"), fresh,
                evidence(target="garden", when=4, eid="hot_inspection")]
        decisions = [dict(simulation_time=1.2, requested_evidence_id="original", resolved_evidence_id="original",
                          field="temperature_c", observed_value=29),
                     dict(simulation_time=3.2, requested_evidence_id="original", resolved_evidence_id="fresh",
                          field="temperature_c", observed_value=68)]
        self.assertTrue(self.evaluate(predicate("thermal_scan"), rows)["satisfied"])
        # The later hot refresh must not rewrite the earlier cold decision.
        earlier = self.evaluate(req, rows, condition_evaluations=decisions)
        self.assertTrue(earlier["satisfied"])
        self.assertEqual(earlier["measured"]["selected_branch"], "else")
        self.assertEqual(self.evaluate(req, rows)["measured"]["selected_branch"], "else")
        later = {**req, "decision_occurrence": 2}
        measured = self.evaluate(later, rows, condition_evaluations=decisions)
        self.assertTrue(measured["satisfied"])
        self.assertEqual(measured["measured"]["observation_evidence_id"], "fresh")
        self.assertFalse(self.evaluate(later, rows, condition_evaluations=decisions,
                                       condition_evaluations_dropped=1)["satisfied"])
        decisions[1]["simulation_time"] = 2.5  # Referencing a future reading is invalid.
        self.assertFalse(self.evaluate(later, rows, condition_evaluations=decisions)["satisfied"])
        decisions[1]["simulation_time"] = 3.2
        fresh["refresh_of"] = "unbound"
        self.assertFalse(self.evaluate(later, rows, condition_evaluations=decisions)["satisfied"])

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
        self.assertFalse(self.evaluate(req, samples=[{**row, "mission_id": "another-mission"} for row in good])["satisfied"])

    def test_hover_altitude_heading_requires_measured_state(self):
        req = dict(type="goal", target="library", state="airborne", altitude_m=10, yaw_deg=90)
        self.assertTrue(self.evaluate(req, samples=[sample(1, (5, 5, 2), yaw=math.pi / 2)])["satisfied"])
        self.assertFalse(self.evaluate(req, samples=[sample(1, (5, 5, 2), yaw=0)])["satisfied"])

    def test_pose_dwell_and_order_are_independent_of_control_segmentation(self):
        pose = dict(type="pose", target="library", state="airborne", altitude_m=10,
                    yaw_deg=90, dwell_seconds=1)
        req = dict(type="sequence", items=[pose, dict(type="goal", target="home", state="landed", dwell_seconds=1)])
        hover = [sample(i / 10, (5, 5, 2), yaw=math.pi / 2, eid=str(i // 3)) for i in range(11)]
        ground = [sample(2 + i / 10) for i in range(11)]
        self.assertTrue(self.evaluate(req, samples=hover + ground)["satisfied"])
        for bad in ([{**r, "yaw": 0} for r in hover], hover[:3],
                    [{**r, "position": [5, 5, 3]} for r in hover]):
            self.assertFalse(self.evaluate(req, samples=bad + ground)["satisfied"])
        self.assertFalse(self.evaluate(req, samples=ground + [{**r, "t": r["t"] + 4} for r in hover])["satisfied"])
        repeated = dict(type="sequence", items=[dict(type="pose", target="home", state="landed", dwell_seconds=1), pose])
        initial = [sample(i / 10) for i in range(11)]
        self.assertTrue(self.evaluate(repeated, samples=initial + [{**r, "t": r["t"] + 2} for r in hover])["satisfied"])
        same_pose = dict(type="pose", target="home", state="landed", dwell_seconds=1)
        consecutive = dict(type="sequence", items=[same_pose, same_pose])
        continuous = [sample(i / 10, eid=str(i // 4)) for i in range(31)]
        measured = self.evaluate(consecutive, samples=continuous)
        self.assertTrue(measured["satisfied"])
        self.assertAlmostEqual(measured["children"][0]["last"], 1)
        self.assertAlmostEqual(measured["children"][1]["last"], 2)
        self.assertFalse(self.evaluate(consecutive, samples=continuous[:16])["satisfied"])
        self.assertFalse(self.evaluate({**same_pose, "occurrence": 2}, samples=continuous)["satisfied"])
        second_run = continuous[:11] + [sample(1.1, speed=1)] + continuous[20:]
        self.assertTrue(self.evaluate({**same_pose, "occurrence": 2}, samples=second_run)["satisfied"])

    def test_dwell_excludes_planning_wait_and_unobserved_time(self):
        for kind in ("pose", "goal"):
            req = dict(type=kind, target="home", state="landed", dwell_seconds=1)
            idle = [{**sample(i / 10), "execution_id": None, "action": None} for i in range(21)]
            active = [sample(2 + i / 10, eid=str(i // 3)) for i in range(11)]
            with self.subTest(kind=kind):
                self.assertFalse(self.evaluate(req, samples=idle)["satisfied"])
                self.assertFalse(self.evaluate(req, samples=idle + active[:3])["satisfied"])
                self.assertFalse(self.evaluate(req, samples=[sample(0), sample(50)])["satisfied"])
                self.assertTrue(self.evaluate(req, samples=idle + active)["satisfied"])
                interrupted = active[:5] + [{**active[5], "execution_id": None, "action": None}] + active[6:]
                self.assertFalse(self.evaluate(req, samples=interrupted)["satisfied"])
                terminal = {**sample(4), "execution_id": None, "action": None}
                if kind == "goal":
                    self.assertTrue(self.evaluate(req, samples=active + [terminal])["satisfied"])
                    self.assertFalse(self.evaluate(req, samples=active[:3] + [terminal])["satisfied"])

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
        half = {**req, "laps": .5}
        self.assertTrue(self.evaluate(half, rows, self.square_samples()[:41])["satisfied"])
        self.assertFalse(self.evaluate(half, rows, self.square_samples())["satisfied"])
        for samples in (self.square_samples(sign=-1), self.square_samples(laps=2), self.square_samples(size=.2)):
            self.assertFalse(self.evaluate(req, rows, samples)["satisfied"])
        self.assertFalse(self.evaluate(req, rows, [sample()])["satisfied"])
        square = self.square_samples()
        reversing = [{**s, "t": i / 10} for i, s in enumerate(square[:21] + list(reversed(square[:21])) + square)]
        self.assertFalse(self.evaluate(req, [evidence("motion", "garden", 20, "ring")], reversing)["satisfied"])
        original = [{**s, "execution_id": "cancelled", "action": "motion", "target_id": "garden", "user_update_index": 0}
                    for s in self.square_samples()[:21]]
        for continuation, user_update in ((False, False), (True, False), (False, True)):
            with self.subTest(continuation=continuation, user_update=user_update):
                remaining = self.square_samples()[20:] if continuation else self.square_samples()
                resumed = [{**s, "execution_id": "resumed", "t": s["t"] + 4, "action": "motion",
                            "target_id": "garden", "user_update_index": int(user_update)} for s in remaining]
                result = {"completed": True}
                if continuation:
                    result["tracking_executions"] = ["cancelled", "resumed"]
                finished = [evidence("motion", "garden", 20, "resumed", result=result)]
                predicate = {**req, **({"after_update": 1} if user_update else {})}
                measured = self.evaluate(predicate, finished, original + resumed)
                self.assertEqual(measured["satisfied"], continuation or user_update)
                if not continuation and not user_update:
                    self.assertAlmostEqual(measured["measured"]["measured_signed_laps"], 1.25)
                    self.assertGreater(measured["measured"]["uncredited_tracking_distance_m"], 0)
                    recovery = recovery_metrics(req, finished, [], "mission", telemetry=original + resumed)
                    self.assertGreater(recovery["uncredited_tracking_distance_m"], 0)
                    # Reusing an execution ID across a transit/reset must not
                    # invent reverse tracking that cancels the extra quarter.
                    same_id = [{**s, "execution_id": "resumed"} for s in original]
                    gap = {**sample(3, eid="resumed", phase="transit"), "action": "motion", "target_id": "garden"}
                    repeated = self.evaluate(req, finished, same_id + [gap] + resumed)
                    self.assertFalse(repeated["satisfied"])
                    self.assertAlmostEqual(repeated["measured"]["measured_signed_laps"], 1.25)

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

    def test_all_failures_stay_in_denominator(self):
        statuses = ["completed", "timeout", "clarification", "failed"]
        results = [dict(status=s, oracle_success=i == 0, service_success=i == 0, false_completion=False,
                        interpretation_calls=1, wall_seconds=1, regression_pass=i == 0)
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

    def test_recovery_metrics_exclude_required_repeats_and_unsuccessful_attempts(self):
        reviews = [dict(event="semantic_execution_feedback", simulation_seconds=2.5,
                        feedback=dict(kind="review_requested"))]
        rows = [evidence(when=1), evidence(when=2, eid="two"), evidence(when=3, eid="extra"),
                evidence(when=4, eid="unknown", result={"inspected": False})]
        result = recovery_metrics(predicate(count=2), rows, reviews, "mission")
        self.assertEqual(result["repeated_completed_actions"], 1)
        self.assertEqual(result["duplicate_evidence"][0]["execution_id"], "extra")
        self.assertEqual(recovery_metrics(predicate(count=3), rows, reviews, "mission")["repeated_completed_actions"], 0)
        original = evidence("thermal_scan", when=1, evidence_id="original")
        refresh = evidence("thermal_scan", when=3, eid="refresh", evidence_id="fresh", refresh_of="original",
                           observation=dict(status="valid", observed_at=3, valid_until=5, confidence=1))
        self.assertEqual(recovery_metrics(predicate("thermal_scan"), [original, refresh], reviews, "mission")["repeated_completed_actions"], 0)
        refresh["refresh_of"] = "unbound"
        self.assertEqual(recovery_metrics(predicate("thermal_scan"), [original, refresh], reviews, "mission")["repeated_completed_actions"], 1)
        usage = dict(usage_missing=1, unreported_requests=0, prompt_cache_hit_tokens=0,
                     prompt_cache_miss_tokens=10, completion_tokens=10)
        prices = dict(input_hit=1, input_miss=2, output=3, currency="test")
        self.assertIsNone(_usage_cost(dict(usage=usage), prices, offline=False)["estimated_cost"])

    def test_paired_comparison_rejects_changed_conditions_and_reports_unmatched_runs(self):
        def result(policy, case="first", success=False):
            return dict(case_id=case, status="completed", oracle_success=success, service_success=True,
                        false_completion=not success, interpretation_calls=1, wall_seconds=1,
                        protocol=dict(policy=policy, seed=17, source_unchanged=True, mode="virtual", control_dt=.05))
        before, after = result("single_pass"), result("contract", success=True)
        comparison = compare_policies([before, after, result("single_pass", "unmatched")])
        pair = comparison["comparisons"]["contract"]
        self.assertEqual(pair["paired_runs"], 1)
        self.assertEqual(pair["recovered"], [dict(case_id="first", seed=17)])
        self.assertEqual(pair["unmatched_baseline"], [dict(case_id="unmatched", seed=17)])
        after["protocol"]["control_dt"] = .1
        with self.assertRaisesRegex(ValueError, "control_dt"):
            compare_policies([before, after])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        guard = patch("uav_llm_demo.agents._post_deepseek", side_effect=AssertionError("Network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_campus_executor_repairs_initial_omissions_without_sending_oracle_to_model(self):
        fixture = next(c for c in load_dataset()["cases"] if c["id"] == "holdout_initial_omission")
        original = _BudgetedAgent.interpret_task
        for repaired in (True, False):
            with self.subTest(repaired=repaired):
                case, seen = deepcopy(fixture), []
                if not repaired:
                    case["offline_responses"] = [deepcopy(case["offline_responses"][0]) for _ in range(3)]

                def recording(agent, catalog, instruction, feedback="", **kwargs):
                    seen.append((deepcopy(catalog), instruction))
                    return original(agent, catalog, instruction, feedback, **kwargs)

                with patch.object(_BudgetedAgent, "interpret_task", recording):
                    report = run_case(case, max_calls=8, wall_budget=60)
                self.assertEqual(report["service_success"], repaired)
                self.assertEqual(report["oracle_success"], repaired)
                self.assertFalse(report["false_completion"])
                self.assertEqual(report["regression_pass"], repaired)
                self.assertEqual(report["status"], "completed" if repaired else "failed")
                self.assertEqual(report["interpretation_stage_calls"],
                                 dict(requirements=2, audit=2, plan=2 if repaired else 3))
                self.assertEqual(report["interpretation_calls"], sum(report["interpretation_stage_calls"].values()))
                self.assertNotIn("offline-benchmark-placeholder", json.dumps(report))
                self.assertEqual(seen[0][1], case["instruction"])
                self.assertIn("task_requirements", seen[0][0])
                self.assertNotIn("oracle", seen[0][0])
                self.assertNotIn("offline_responses", seen[0][0])

    def test_real_model_requires_realtime_and_configured_provider_before_service(self):
        case = load_dataset()["cases"][0]
        with self.assertRaises(ValueError):
            run_case(case, mode="virtual", backend="deepseek")
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            run_case(case, mode="realtime", backend="deepseek")
        local = dict(LLM_MODEL="Qwen/Qwen3-VL-4B-Instruct", LLM_BASE_URL="http://127.0.0.1:8001/v1")
        with patch.dict("os.environ", local, clear=True), \
                patch("uav_llm_demo.gazebo_session.GazeboSession") as factory:
            session = factory.return_value
            session.log_dir, session.manifest = None, {}
            session.start.side_effect = RuntimeError("fixture startup failure")
            result = run_case(case, mode="realtime", backend="deepseek", runtime_backend="gazebo")
            session.start.assert_called_once()
            session.close.assert_called_once()
            self.assertEqual(result["termination_reason"], "runtime_startup")
        with self.assertRaises(ValueError):
            run_case(case, control_delay_ms=120)
        with self.assertRaises(ValueError):
            run_case(case, control_dt=.2)
        with self.assertRaisesRegex(ValueError, "retired physics"):
            run_case({**case, "physics": {"wind_mps": [2, -1, 0]}})
        for options in ({}, {"mode": "realtime", "traffic": "fixed"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Gazebo requires"):
                run_case(case, runtime_backend="gazebo", **options)

    def test_native_startup_failure_is_counted_and_owned_session_is_closed(self):
        from uav_llm_demo.gazebo_runtime import GazeboRuntime, GazeboRuntimeConfig
        runtime = GazeboRuntime(GazeboRuntimeConfig(world="semantic_campus"))
        with patch("uav_llm_demo.gazebo_session.GazeboSession") as factory:
            session = factory.return_value
            session.runtime, session.log_dir, session.manifest = runtime, None, {}
            session.start.side_effect = RuntimeError("Private diagnostic")
            result = run_case(load_dataset()["cases"][0], mode="realtime", runtime_backend="gazebo")
            session.close.assert_called_once()
        self.assertEqual(result["termination_reason"], "runtime_startup")
        self.assertEqual(result["protocol"]["runtime"], "gazebo")
        self.assertEqual(result["telemetry"], [])
        self.assertFalse(result["service_success"])
        self.assertEqual(summarize([result])["runs"], 1)
        self.assertNotIn("Private diagnostic", json.dumps(result))

    def test_single_pass_baseline_exposes_false_completion_without_reading_requirements(self):
        case = next(c for c in load_dataset()["cases"] if c["id"] == "holdout_initial_omission")
        for policy in ("single_pass", "current_repair", "event_review"):
            with self.subTest(policy=policy):
                result = run_case(case, policy=policy, wall_budget=30)
                self.assertEqual(result["status"], "completed")
                self.assertFalse(result["oracle_success"])
                self.assertTrue(result["false_completion"])
                self.assertEqual(result["interpretation_stage_calls"], {"plan": 1})
                self.assertEqual(result["protocol"]["requirement_source"], "plan_derived_baseline")

    def test_budget_prevents_extra_calls_and_expired_requests(self):
        case = load_dataset()["cases"][0]
        catalog = task_catalog(load_scenario("semantic_campus"))
        for limit in (1, 2):
            with self.subTest(limit=limit):
                agent = _BudgetedAgent(replies=case["offline_responses"], requirements=case["offline_requirements"],
                                       audits=case["offline_audits"], api_key="test-placeholder", max_calls=limit)
                requirements = agent.extract_task_requirements(catalog, case["instruction"])
                if limit == 1:
                    with self.assertRaises(PlannerProviderError):
                        agent.audit_task_requirements(catalog, case["instruction"], requirements)
                else:
                    agent.audit_task_requirements(catalog, case["instruction"], requirements)
                    with self.assertRaises(PlannerProviderError):
                        agent.interpret_task(catalog, case["instruction"])
                self.assertEqual(agent.calls, limit)
                self.assertEqual(sum(agent.stage_calls.values()), limit)
                self.assertTrue(agent.budget_exhausted)
        expired = _BudgetedAgent(replies=[], api_key="test-placeholder", deadline=time.monotonic() - 1)
        with self.assertRaises(PlannerProviderError):
            expired.extract_task_requirements({}, "任务")
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
