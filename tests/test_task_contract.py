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
        return TaskContractMonitor(program(nodes), "mission", plan=program(nodes), **kwargs)

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

    def test_no_trailing_motion_after_final_required_destination(self):
        for final in (home(), action("last_visit", "visit"), action("last_inspection")):
            with self.subTest(action=final["action"]):
                monitor = self.monitor([review(), final])
                with self.assertRaises(TaskContractError):
                    monitor.validate_revision(program([final, action("leave", "visit", "place_b")]))

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
        required = program([review(), home()], dict(max_altitude_m=25, avoid_regions=["restricted"]))
        monitor = TaskContractMonitor(required, "mission", plan=required)
        with self.assertRaisesRegex(TaskContractError, "hard constraints"):
            monitor.validate_revision(program([home()], dict(max_altitude_m=30, avoid_regions=[])))
        tightened = program([home()], dict(max_altitude_m=20, avoid_regions=["restricted", "courtyard"]))
        self.assertTrue(monitor.validate_revision(tightened)["accepted"])

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

    def test_open_review_continuation_keeps_execution_binding_without_new_user_obligations(self):
        monitor = self.monitor([review()])
        followup = program([review("again"), action("required"), home()])
        revise(monitor, followup)
        self.assertEqual(monitor.remaining_steps(), [])
        self.assertTrue(monitor.validate_revision(program([home()]))["accepted"])
        revise(monitor, program([home("land")]), review("again"))
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

    def test_first_plan_must_cover_independent_requirements_and_explicit_counts(self):
        required = program([dict(id="required", action="inspect", target_id="place_a"), home()])
        for candidate in (program([home()]), program([action("extra", "capture"), action("done"), home()])):
            with self.subTest(candidate=candidate), self.assertRaises(TaskContractError):
                TaskContractMonitor(required, "mission", plan=candidate)
        with self.assertRaises(TypeError):
            TaskContractMonitor(required, "mission")
        counted = self.monitor([review(), repeat([action("scan", "thermal_scan")], 2), home()])
        with self.assertRaises(TaskContractError):
            counted.validate_revision(program([repeat([action("new", "thermal_scan")], 3), home()]))

    def test_sparse_motion_parameters_can_change_but_requirements_and_execution_cannot(self):
        from uav_llm_demo.scenarios import load_scenario

        scene = load_scenario("semantic_campus")
        requirement = dict(id="circle", action="motion", target_id="east_garden",
                           motion=dict(path=dict(kind="boundary", direction="counterclockwise", laps=2,
                                                 altitude_delta_m=4)))
        route = dict(id="route", action="motion", target_id="east_garden", when=None,
                     motion=dict(path=dict(kind="boundary", direction="counterclockwise", laps=2,
                                           altitude_m=10.9, end_altitude_m=14.9),
                                 speed_mps=2.5, heading=dict(mode="tangent")))
        required, initial = program([review(), requirement, home()]), program([review(), route, home()])
        monitor = TaskContractMonitor(required, "mission", plan=initial, scenario=scene)
        changed = deepcopy(route)
        changed["motion"]["speed_mps"] = 1.5
        changed["motion"]["path"].update(altitude_m=12.9, end_altitude_m=16.9)
        for field, value in (("direction", "clockwise"), ("laps", 1), ("end_altitude_m", 18.9)):
            wrong = deepcopy(changed)
            wrong["motion"]["path"][field] = value
            with self.subTest(field=field), self.assertRaises(TaskContractError):
                monitor.validate_revision(program([wrong, home()]))
        revise(monitor, program([changed, home()]))
        altered_runtime = deepcopy(changed)
        altered_runtime["motion"]["speed_mps"] = 3
        with self.assertRaises(TaskContractError):
            finish(deepcopy(monitor), altered_runtime)
        length, end = list(monitor._motion_geometry.values())[-1]
        finish(monitor, changed, position=dict(zip("xyz", end)),
               result=dict(completed=True, distance_m=length, total_length_m=length, progress=1))
        finish(monitor, home(), position=scene.base)
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_explicit_control_height_and_yaw_cannot_be_changed_by_other_execution_fields(self):
        requirement = dict(id="hold", action="control", target_id="place_a",
                           control=dict(altitude_m=10, yaw_deg=360, state="airborne"))
        node = dict(id="hold", action="control", target_id="place_a", when=None,
                    control=dict(anchor="center", altitude_m=10, offset_m=[0, 0, 0], yaw_deg=0,
                                 state="airborne", duration_seconds=2))
        required = program([requirement])
        for changes in ({"offset_m": [0, 0, 2]}, {"yaw_deg": 90}):
            wrong = deepcopy(node)
            wrong["control"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(TaskContractError):
                TaskContractMonitor(required, "mission", plan=program([wrong]))
        equivalent = deepcopy(node)
        equivalent["control"].update(altitude_m=8, offset_m=[0, 0, 2], duration_seconds=4)
        monitor = TaskContractMonitor(required, "mission", plan=program([equivalent]))
        with self.assertRaises(TaskContractError):
            monitor.validate_revision(program([equivalent, action("leave", "visit", "place_b")]), after_review=False)
        finish(monitor, equivalent, measured_yaw_deg=0)
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_explicit_geometry_accepts_conservative_equivalence_without_freezing_sampling(self):
        uniform_xy = dict(scale_groups=[["x", "y"]], offset_axes=["z"])
        geometries = [
            # Shape, dimensions and vertical profile are independent obligations.
            (dict(kind="curve", x="sin(t)", y="sin(2*t)", t_start=0, t_end=2,
                  altitude_delta_m=4, altitude_trend="nondecreasing",
                  flexible=dict(scale_groups=[["x"], ["y"]], offset_axes=["x", "y"])),
             dict(kind="curve", x="3*sin(t)", y="4*sin(2*t)", z="20+t**2", t_start=0, t_end=2),
             [dict(z="20+t"), dict(z="20+2*t+sin(2*pi*t)")]),
            (dict(kind="curve", x="t*2", y=3, z="10+t", t_start=0, t_end=2, altitude_delta_m=2),
             dict(kind="curve", x="2.0*t", y="3+0*t", z="t+10.0", t_start=0, t_end=2, samples=512),
             [dict(x="3*t")]),
            (dict(kind="polyline", points_m=[[0, 0, 10], [4, 0, 12]], altitude_delta_m=2),
             dict(kind="polyline", points_m=[[0, 0, 10], [2, 0, 11], [4, 0, 12]]),
             [dict(points_m=[[0, 0, 10], [2, 1, 11], [4, 0, 12]])]),
            (dict(kind="polyline", points_m=[dict(x=0, y=0, z=10), dict(x=4, y=0, z=12)], altitude_delta_m=2),
             dict(kind="polyline", points_m=[[0, 0, 10], [2, 0, 11], [4, 0, 12]]),
             [dict(points_m=[[0, 0, 10], [2, 1, 11], [4, 0, 12]])]),
            (dict(kind="curve", x="sin(t)", y="sin(2*t)", z=10, t_start=0, t_end=6.28, flexible=uniform_xy),
             dict(kind="curve", x="3*sin(t)", y="3*sin(2*t)", z=18, t_start=0, t_end=6.28, samples=300),
             [dict(y="3*cos(t)"), dict(x="3*sin(2*t)"), dict(x="-3*sin(t)"),
              dict(x="3*sin(t)+1"), dict(y="4*sin(2*t)"), dict(t_end=12.56)]),
            # A zero constant axis leaves the group's scale to its other axis.
            (dict(kind="curve", x=0, y="t", z=10, t_start=0, t_end=2, flexible=uniform_xy),
             dict(kind="curve", x=0, y="2*t", z=18, t_start=0, t_end=2, samples=128),
             [dict(x=1), dict(y="-2*t")]),
            # Without permitted x offset, a nonzero constant also constrains scale.
            (dict(kind="curve", x=2, y="t", z=10, t_start=0, t_end=2, flexible=uniform_xy),
             dict(kind="curve", x=6, y="3*t", z=18, t_start=0, t_end=2, samples=128),
             [dict(x=8)]),
            (dict(kind="curve", x="sin(t)", y="cos(t)", z="10+t", t_start=0, t_end=2,
                  altitude_delta_m=4, flexible=dict(scale_groups=[["x", "y"], ["z"]], offset_axes=["z"])),
             dict(kind="curve", x="3*sin(t)", y="3*cos(t)", z="2*(t+3)+14", t_start=0, t_end=2, samples=128),
             [dict(z="20+3*t"), dict(x="4*sin(t)")]),
            # Explicit radius stays fixed even when altitude is free.
            (dict(kind="curve", x="5*sin(t)", y="5*cos(t)", z=10, t_start=0, t_end=6.28,
                  flexible=dict(scale_groups=[], offset_axes=["z"])),
             dict(kind="curve", x="5*sin(t)", y="5*cos(t)", z=20, t_start=0, t_end=6.28, samples=128),
             [dict(x="8*sin(t)", y="8*cos(t)")]),
            (dict(kind="polyline", points_m=[[0, 0, 10], [1, 0, 10], [1, 1, 10]], flexible=uniform_xy),
             dict(kind="polyline", points_m=[[0, 0, 20], [1, 0, 20], [2, 0, 20], [2, 2, 20]]),
             [dict(points_m=points) for points in (
                 [[0, 0, 20], [2, 0, 20], [2, 3, 20]],
                 [[0, 0, 20], [-2, 0, 20], [-2, -2, 20]],
                 [[1, 0, 20], [3, 0, 20], [3, 2, 20]])]),
        ]
        # Small coefficients can produce metre-scale differences over t's range.
        for options in ({}, dict(flexible=dict(scale_groups=[], offset_axes=["z"]))):
            geometries.append((
                dict(kind="curve", x="1e-8*t**4", y=0, z=10, t_start=0, t_end=100, **options),
                dict(kind="curve", x="t**4*1e-8", y=0, z=10, t_start=0, t_end=100, samples=256),
                [dict(x="2e-8*t**4")]))
        for expected, actual, invalid_changes in geometries:
            required = program([dict(id="shape", action="motion", target_id="place_a", motion=dict(path=expected))])
            node = dict(id="route", action="motion", target_id="place_a", when=None,
                        motion=dict(path=actual, speed_mps=1, heading=dict(mode="tangent")))
            with self.subTest(reference=expected):
                TaskContractMonitor(required, "mission", plan=program([node]))
                for changes in invalid_changes:
                    changed = deepcopy(node)
                    changed["motion"]["path"].update(changes)
                    with self.subTest(changes=changes), self.assertRaises(TaskContractError):
                        TaskContractMonitor(required, "mission", plan=program([changed]))

class TaskContractEvidenceTests(unittest.TestCase):
    def test_evidence_uses_selected_plan_constraints_within_user_bounds(self):
        scene = SimpleNamespace(points_of_interest=[dict(id="place_a", position=dict(x=0, y=0, z=3))])
        node = action("inspect")
        required = program([dict(id="required", action="inspect", target_id="place_a")],
                           dict(max_altitude_m=20, avoid_regions=[]))
        plan = program([node], dict(max_altitude_m=10, avoid_regions=[]))
        monitor = TaskContractMonitor(required, "mission", plan=plan, scenario=scene)
        finish(monitor, node, position=dict(x=0, y=0, z=2))
        self.assertTrue(monitor.verdict()["satisfied"])

    def test_return_home_requires_actual_contact_not_nearby_hover(self):
        scene = SimpleNamespace(points_of_interest=[dict(id="home", position=dict(x=0, y=0, z=0))])
        node = home()
        monitor = TaskContractMonitor(program([node]), "mission", plan=program([node]), scenario=scene)
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
        monitor = TaskContractMonitor(program([node]), "mission", plan=program([node]), scenario=scene)
        length, end = next(iter(monitor._motion_geometry.values()))
        self.assertGreater(length, 20)
        bound, item = evidence(monitor, node, position=dict(zip("xyz", end)),
                               result=dict(completed=True, distance_m=1, total_length_m=1, progress=1))
        with self.assertRaises(TaskContractError):
            monitor.record(bound, item)
        correct = TaskContractMonitor(program([node]), "mission", plan=program([node]), scenario=scene)
        finish(correct, node, position=dict(zip("xyz", end)),
               result=dict(completed=True, distance_m=length, total_length_m=length, progress=1))
        self.assertTrue(correct.verdict()["satisfied"])

    def test_prior_revision_foreign_vehicle_and_duplicate_evidence_fail(self):
        node = action("inspect")
        for changes in ({"revision": 0}, {"mission_id": "foreign"}, {"vehicle_id": "uav-2"},
                        {"source": "model_claim"}, {"target_id": "place_b"}, {"duration_seconds": 1}):
            monitor = TaskContractMonitor(program([node, home()]), "mission", plan=program([node, home()]))
            bound, item = evidence(monitor, node, **changes)
            with self.subTest(changes=changes), self.assertRaises(TaskContractError):
                monitor.record(bound, item)
            self.assertFalse(monitor.snapshot()["satisfied"])
        monitor = TaskContractMonitor(program([repeat([node], 2), home()]), "mission", plan=program([repeat([node], 2), home()]))
        bound, item = evidence(monitor, node)
        monitor.record(bound, item)
        with self.assertRaises(TaskContractError):
            monitor.record(bound, item)

    def test_old_user_revision_evidence_cannot_satisfy_new_obligation(self):
        node = action("inspect")
        old = TaskContractMonitor(program([node]), "mission", plan=program([node]))
        item, _ = finish(old, node)
        new = TaskContractMonitor(program([node]), "mission", plan=program([node]), revision=2, evidence=[item])
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
            monitor = TaskContractMonitor(program([review(), node, home()]), "mission", plan=program([review(), node, home()]))
            changed = deepcopy(node)
            changed["motion"]["path"][key] = value
            with self.subTest(key=key), self.assertRaises(TaskContractError):
                monitor.validate_revision(program([changed, home()]))
        monitor = TaskContractMonitor(program([node]), "mission", plan=program([node]))
        bound, item = evidence(monitor, node, result=dict(completed=True, distance_m=1, total_length_m=100, progress=1))
        with self.assertRaises(TaskContractError):
            monitor.record(bound, item)
        self.assertFalse(monitor.verdict()["satisfied"])

if __name__ == "__main__":
    unittest.main()
