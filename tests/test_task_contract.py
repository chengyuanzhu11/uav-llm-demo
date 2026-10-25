"""General specification preservation and evidence-driven completion checks."""
from copy import deepcopy
from itertools import count
from types import SimpleNamespace
import unittest

from uav_llm_demo.task_contract import TaskContractError, TaskContractMonitor


_ids = count(1)


def action(name, kind="inspect", target="place_a", duration=2, when=None):
    return dict(id=name, action=kind, target_id=target, duration_seconds=duration, when=when)


def review(name="review"):
    return dict(id=name, action="review", instruction="Assess remaining work.", when=None)


def home(name="home"):
    return action(name, "return_home", "home", 0)


def program(nodes, constraints=None):
    return dict(status="ready", summary="Task", steps=nodes,
                constraints=constraints or dict(max_altitude_m=None, avoid_regions=[]))


def repeat(nodes, count_value=3, until=None, name="loop"):
    return dict(id=name, action="repeat", count=count_value, until=until, steps=nodes, when=None)


def condition(step_id, threshold=40):
    return dict(step_id=step_id, field="temperature_c", op=">", value=threshold)


def branch(test, yes, no, name="choice"):
    return dict(id=name, action="branch", condition=test, then=yes, **{"else": no})


def evidence(monitor, node, *, result=None, **changes):
    node = {**deepcopy(node), "execution_id": "execution_" + str(next(_ids)), "revision": monitor.revision}
    kind = node["action"]
    source = {"inspect": "simulated_inspection", "thermal_scan": "simulated_thermal_sensor",
              "capture": "simulated_capture", "motion": "runtime_trajectory_tracking",
              "review": "validated_execution_review"}.get(kind, "runtime_telemetry")
    results = {"inspect": {"inspected": True}, "visit": {"visited": True},
               "thermal_scan": {"temperature_c": 29}, "capture": {"captured": True},
               "return_home": {"returned": True, "landed": True}, "review": {"reviewed": True},
               "motion": {"completed": True, "distance_m": 100, "total_length_m": 100, "progress": 1}}
    if kind == "control":
        results[kind] = dict(completed=True, state=node["control"]["state"], landed=node["control"]["state"] == "landed")
    item = dict(evidence_id="evidence_" + str(next(_ids)), mission_id=monitor.mission_id,
                vehicle_id=monitor.vehicle_id, revision=monitor.revision,
                step_id=node["id"], action=kind, target_id=node.get("target_id"),
                execution_id=node["execution_id"], source=source, simulation_time=10,
                duration_seconds=node.get("duration_seconds", node.get("control", {}).get("duration_seconds", 0)),
                position=dict(x=0, y=0, z=0), result=deepcopy(results[kind] if result is None else result))
    item.update(changes)
    return node, item


def finish(monitor, node, **changes):
    bound, item = evidence(monitor, node, **changes)
    matched = monitor.record(bound, item)
    return item, matched


def revise(monitor, proposal, current_review=None):
    monitor.validate_revision(proposal)
    finish(monitor, current_review or review())
    monitor.accept_revision(proposal, monitor.revision + 1)


class TaskContractPreservationTests(unittest.TestCase):
    def monitor(self, nodes, **kwargs):
        return TaskContractMonitor(program(nodes), "mission", **kwargs)

    def test_id_renaming_and_auxiliary_actions_preserve_bound_obligations(self):
        required = action("required")
        monitor = self.monitor([review(), required, home()])
        aux, renamed, landed = action("aux", "capture", "place_b"), action("renamed"), home("land")
        proposal = program([aux, renamed, landed])
        revise(monitor, proposal)
        self.assertFalse(finish(monitor, aux)[1])
        self.assertTrue(finish(monitor, renamed)[1])
        self.assertTrue(finish(monitor, landed)[1])
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_same_action_auxiliary_cannot_steal_required_measurement(self):
        scan = action("scan", "thermal_scan", duration=3)
        capture = action("capture", "capture", "place_b", when=condition("scan"))
        monitor = self.monitor([review(), scan, capture, home()])
        aux, actual = action("aux", "thermal_scan", duration=3), action("actual", "thermal_scan", duration=3)
        capture2 = action("captured", "capture", "place_b", when=condition("actual"))
        revise(monitor, program([aux, actual, capture2, home("land")]))
        self.assertFalse(finish(monitor, aux, result={"temperature_c": 68})[1])
        self.assertTrue(finish(monitor, actual, result={"temperature_c": 29})[1])
        finish(monitor, home("land"))
        self.assertTrue(monitor.verdict()["satisfied"])
        self.assertEqual(monitor.snapshot()["skipped_occurrences"], 1)

    def test_no_trailing_motion_after_terminal_landing(self):
        monitor = self.monitor([review(), home()])
        with self.assertRaises(TaskContractError):
            monitor.validate_revision(program([home("renamed"), action("leave", "visit")]))

    def test_terminal_open_review_cannot_silently_disappear(self):
        monitor = self.monitor([review(), action("inspect"), review("decide_next")])
        with self.assertRaises(TaskContractError):
            monitor.validate_revision(program([action("renamed")]))
        revise(monitor, program([action("renamed"), review("assessment")]))
        finish(monitor, action("renamed"))
        self.assertFalse(monitor.verdict()["satisfied"])
        revise(monitor, program([home("land")]), review("assessment"))
        finish(monitor, home("land"))
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_hard_constraints_remain_frozen(self):
        monitor = self.monitor([review(), home()])
        with self.assertRaisesRegex(TaskContractError, "hard constraints"):
            monitor.validate_revision(program([home()], dict(max_altitude_m=30, avoid_regions=[])))

    def test_count_loops_allow_unrolling_without_losing_occurrences(self):
        inspect = action("inspect")
        monitor = self.monitor([review(), repeat([inspect], 3), home()])
        for count_value in (1, 2):
            with self.assertRaises(TaskContractError):
                monitor.validate_revision(program([repeat([action("new")], count_value), home()]))
        proposal = program([action("one"), action("two"), action("three"), home("land")])
        revise(monitor, proposal)
        for node in proposal["steps"]:
            finish(monitor, node)
        self.assertTrue(monitor.verdict()["satisfied"])
        self.assertEqual(monitor.snapshot()["matched_occurrences"], 5)

    def test_large_count_has_compact_repair_context_and_bounded_nonrecursive_match(self):
        monitor = self.monitor([review(), repeat([action("inspect")], 1000), home()])
        residual = monitor.remaining_steps(after_review=True)
        self.assertLess(len(residual), 5)
        self.assertTrue(monitor.validate_revision(program(residual))["accepted"])

    def test_unselected_branch_never_creates_or_satisfies_an_obligation(self):
        scan = action("scan", "thermal_scan", duration=3)
        yes, no = action("yes", target="place_hot"), action("no", target="place_cool")
        monitor = self.monitor([scan, review(), branch(condition("scan"), [yes], [no]), home()])
        finish(monitor, scan, result={"temperature_c": 29})
        with self.assertRaises(TaskContractError):
            monitor.validate_revision(program([yes, home()]))
        revise(monitor, program([action("selected", target="place_cool"), home("land")]))
        finish(monitor, action("selected", target="place_cool"))
        finish(monitor, home("land"))
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_future_branch_predicate_and_arms_are_preserved_under_renaming(self):
        scan = action("scan", "thermal_scan", duration=3)
        yes, no = action("yes", target="place_hot"), action("no", target="place_cool")
        monitor = self.monitor([review(), scan, branch(condition("scan"), [yes], [no]), home()])
        scan2 = action("reading", "thermal_scan", duration=3)
        choice = branch(condition("reading"), [action("positive", target="place_hot")], [action("negative", target="place_cool")])
        wrong = deepcopy(choice)
        wrong["condition"]["value"] = 100
        with self.assertRaises(TaskContractError):
            monitor.validate_revision(program([scan2, wrong, home("land")]))
        revise(monitor, program([scan2, choice, home("land")]))
        finish(monitor, scan2, result={"temperature_c": 68})
        finish(monitor, choice["then"][0])
        finish(monitor, home("land"))
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_until_uses_latest_iteration_and_may_end_after_review(self):
        scan, assessment = action("scan", "thermal_scan", duration=3), review()
        monitor = self.monitor([repeat([scan, assessment], None, condition("scan")), home()])
        finish(monitor, scan, result={"temperature_c": 29})
        with self.assertRaises(TaskContractError):
            monitor.validate_revision(program([home()]))
        # Continue using the exported residual DSL, then observe the new reading.
        residual = monitor.remaining_steps(after_review=True)
        proposal = program(residual)
        revise(monitor, proposal)
        next_scan = next(node for node in residual if node["action"] == "thermal_scan")
        finish(monitor, next_scan, result={"temperature_c": 68})
        next_review = next(node for node in residual if node["action"] == "review")
        revise(monitor, program([home("land")]), next_review)
        finish(monitor, home("land"))
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_when_false_skips_requirement_and_all_skipped_can_complete(self):
        when = dict(source="telemetry", field="battery_percent", op="<", value=10)
        monitor = self.monitor([action("conditional", when=when)], telemetry=dict(battery_percent=90))
        self.assertTrue(monitor.verdict()["satisfied"])
        self.assertEqual(monitor.snapshot()["skipped_occurrences"], 1)

    def test_snapshot_does_not_evaluate_or_advance_conditions(self):
        when = dict(source="telemetry", field="battery_percent", op="<", value=10)
        monitor = self.monitor([action("conditional", when=when)], telemetry=dict(battery_percent=90))
        before = deepcopy(monitor._stack)
        self.assertFalse(monitor.snapshot()["satisfied"])
        self.assertEqual(monitor._stack, before)

    def test_open_review_adds_new_frozen_continuation(self):
        monitor = self.monitor([review()])
        followup = program([review("again"), action("required"), home()])
        revise(monitor, followup)
        with self.assertRaises(TaskContractError):
            monitor.validate_revision(program([home()]))
        revise(monitor, program([action("renamed"), home("land")]), review("again"))
        finish(monitor, action("renamed"))
        finish(monitor, home("land"))
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_until_can_read_current_review_result_and_still_requires_real_review_record(self):
        reviewed = dict(step_id="review", field="reviewed", op="==", value=True)
        monitor = self.monitor([repeat([review()], None, reviewed), home()])
        candidate = program([home("land")])
        self.assertTrue(monitor.validate_revision(candidate)["accepted"])
        self.assertFalse(monitor.verdict()["satisfied"])
        with self.assertRaises(TaskContractError):
            monitor.accept_revision(candidate, 2)
        real, _ = finish(monitor, review())
        self.assertEqual(real["result"], {"reviewed": True})
        self.assertEqual(real["source"], "validated_execution_review")
        monitor.accept_revision(candidate, 2)
        finish(monitor, home("land"))
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_review_constant_folding_preserves_unresolved_predicates(self):
        guard = {"all": [dict(step_id="review", field="reviewed", op="==", value=True),
                          dict(source="telemetry", field="battery_percent", op="<", value=10)]}
        monitor = self.monitor([review(), action("before"), action("conditional", when=guard), home()],
                               telemetry=dict(battery_percent=90))
        remaining = monitor.remaining_steps(after_review=True)
        conditional = next(node for node in remaining if node.get("when"))
        self.assertEqual(conditional["when"], dict(source="telemetry", field="battery_percent", op="<", value=10))
        with self.assertRaises(TaskContractError):
            monitor.validate_revision(program([action("before"), action("conditional"), home()]))

class TaskContractEvidenceTests(unittest.TestCase):
    def test_return_home_requires_actual_contact_not_nearby_hover(self):
        scene = SimpleNamespace(points_of_interest=[dict(id="home", position=dict(x=0, y=0, z=0))])
        node = home()
        monitor = TaskContractMonitor(program([node]), "mission", scenario=scene)
        bound, item = evidence(monitor, node, position=dict(x=0, y=0, z=.02))
        with self.assertRaises(TaskContractError):
            monitor.record(bound, item)
        self.assertFalse(monitor.verdict()["satisfied"])

    def test_motion_report_must_match_independently_compiled_geometry(self):
        from uav_llm_demo.scenarios import load_scenario

        scene = load_scenario("semantic_campus")
        node = dict(id="route", action="motion", target_id="east_garden", when=None,
                    motion=dict(path=dict(kind="boundary", direction="clockwise", laps=1,
                                          altitude_m=10.9, end_altitude_m=10.9),
                                speed_mps=2.5, heading=dict(mode="tangent")))
        monitor = TaskContractMonitor(program([node]), "mission", scenario=scene)
        length, end = next(iter(monitor._motion_geometry.values()))
        self.assertGreater(length, 20)
        bound, item = evidence(monitor, node, position=dict(zip("xyz", end)),
                               result=dict(completed=True, distance_m=1, total_length_m=1, progress=1))
        with self.assertRaises(TaskContractError):
            monitor.record(bound, item)
        correct = TaskContractMonitor(program([node]), "mission", scenario=scene)
        finish(correct, node, position=dict(zip("xyz", end)),
               result=dict(completed=True, distance_m=length, total_length_m=length, progress=1))
        self.assertTrue(correct.verdict()["satisfied"])

    def test_prior_revision_foreign_vehicle_and_duplicate_evidence_fail(self):
        node = action("inspect")
        for changes in ({"revision": 0}, {"mission_id": "foreign"}, {"vehicle_id": "uav-2"},
                        {"source": "model_claim"}, {"target_id": "place_b"}, {"duration_seconds": 1}):
            monitor = TaskContractMonitor(program([node, home()]), "mission")
            bound, item = evidence(monitor, node, **changes)
            with self.subTest(changes=changes), self.assertRaises(TaskContractError):
                monitor.record(bound, item)
            self.assertFalse(monitor.snapshot()["satisfied"])
        monitor = TaskContractMonitor(program([repeat([node], 2), home()]), "mission")
        bound, item = evidence(monitor, node)
        monitor.record(bound, item)
        with self.assertRaises(TaskContractError):
            monitor.record(bound, item)

    def test_old_user_revision_evidence_cannot_satisfy_new_obligation(self):
        node = action("inspect")
        old = TaskContractMonitor(program([node]), "mission")
        item, _ = finish(old, node)
        new = TaskContractMonitor(program([node]), "mission", revision=2, evidence=[item])
        self.assertFalse(new.verdict()["satisfied"])
        bound, current = evidence(new, node)
        new.record(bound, current)
        self.assertTrue(new.verdict()["satisfied"])

    def test_motion_preserves_direction_laps_geometry_heading_and_full_progress(self):
        node = dict(id="route", action="motion", target_id="place_a", when=None,
                    motion=dict(path=dict(kind="boundary", direction="clockwise", laps=2,
                                          altitude_m=15, end_altitude_m=20),
                                speed_mps=2.5, heading=dict(mode="fixed", yaw_deg=90)))
        for key, value in (("direction", "counterclockwise"), ("laps", 1), ("end_altitude_m", 15)):
            monitor = TaskContractMonitor(program([review(), node, home()]), "mission")
            changed = deepcopy(node)
            changed["motion"]["path"][key] = value
            with self.subTest(key=key), self.assertRaises(TaskContractError):
                monitor.validate_revision(program([changed, home()]))
        monitor = TaskContractMonitor(program([node]), "mission")
        bound, item = evidence(monitor, node, result=dict(completed=True, distance_m=1, total_length_m=100, progress=1))
        with self.assertRaises(TaskContractError):
            monitor.record(bound, item)
        self.assertFalse(monitor.verdict()["satisfied"])

if __name__ == "__main__":
    unittest.main()
