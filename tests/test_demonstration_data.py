"""Tests for evidence admission, loss boundaries, leakage and visual binding."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uav_llm_demo.demonstration_data import (
    audit_episode, build_review, canonical_sha256, decision_candidates, export_dataset, _visual_action,
)
from uav_llm_demo.interaction_trajectory import InteractionRecorder
from tests.test_interaction_trajectory import frame


def make_episode(path, *, name="task1", family="inspect", scenario="campus",
                 prompt_extra=None, finish="stop", status="applied", visual=False, background=False,
                 response_schema=None, action_override=None):
    standard = dict(id=name, family=family, scenario=scenario, instruction="Inspect the library.",
        intent=["Inspect the library"], annotation_source="agent_assisted",
        oracle=dict(type="evidence", action="inspect", target="library", count=1))
    action = action_override if action_override is not None else dict(status="ready", steps=[dict(action="inspect", target_id="library")])
    messages = [dict(role="user", content=standard["instruction"])]
    if prompt_extra:
        messages.extend(prompt_extra)
    image = frame()
    if visual:
        standard.update(evaluation_kind="visual_rgbd",
            visual_limits=dict(max_surface_error_m=.2, max_frame_age_seconds=1, max_distance_m=20))
        messages[0]["content"] = [dict(type="text", text=standard["instruction"]),
                                  dict(type="image_url", image_url=dict(url=image["rgb"]))]
        raw_args = dict(frame_id=image["frame_id"], point=[500,500], confidence=.9, reason="Visible target")
        message = dict(role="assistant", tool_calls=[dict(id="call-1", type="function",
            function=dict(name="ground", arguments=json.dumps(raw_args)))])
        action = dict(tool="ground", arguments=dict(frame_id=image["frame_id"], pixel=[500/999,500/999],confidence=.9),
                      reason="Visible target")
    else:
        message = dict(role="assistant", content=json.dumps(action))
    recorder = InteractionRecorder(path, metadata=dict(source="scripted_expert", source_sha256="fixture-source", task_id=standard["id"], scenario=standard["scenario"], task_standard_sha256=canonical_sha256(standard)))
    recorder.record("decision_input", decision_id="d1", run_id="run1", modality="vision" if visual else "text",
                    stage="visual" if visual else "plan", context=dict(instruction=standard["instruction"], mission_id=1), frames=[image] if visual else [])
    request = dict(model="offline-scripted-expert", messages=messages)
    if visual:
        from uav_llm_demo.direct_vision import TOOLS
        request["tools"] = json.loads(json.dumps(TOOLS))
        # Persisted v5 decisions remain auditable under their recorded protocol.
        ground = next(t["function"]["parameters"] for t in request["tools"] if t["function"]["name"] == "ground")
        ground["properties"].pop("box")
        ground["required"].remove("box")
    if response_schema is not None:
        request["response_format"] = dict(type="json_schema", json_schema=dict(name="requirements", schema=response_schema, strict=True))
    recorder.record("model_request", decision_id="d1", run_id="run1", model="offline-scripted-expert", request=request)
    recorder.record("model_response", decision_id="d1", run_id="run1", source="provider_override",
                    http_status=None, outcome="ok", response=dict(choices=[dict(message=message, finish_reason=finish)]))
    recorder.record("decision_result", decision_id="d1", run_id="run1", status=status, action=action)
    result = recorder.close(outcome=dict(status="completed"))
    assert result["complete"], result
    manifest = json.loads((path/"manifest.json").read_text())
    execution = dict(instruction=standard["instruction"], run_id="run1", mission_id=1,
        trajectory_events_sha256=manifest["events"]["sha256"], task_standard_sha256=canonical_sha256(standard),
        status="completed", source_sha256="fixture-source", source_unchanged=True,
        oracle_success=True, service_success=True, safe_execution=True, safety_violations=0, static_path_blocked=False,
        world={}, telemetry=[dict(t=1, mission_id=1, position=[0,0,0])],
        evidence=[dict(mission_id=1, vehicle_id="uav-1", execution_id="inspect1", evidence_id="ev1",
                       action="inspect", target_id="library", source="simulated_inspection",
                       result=dict(inspected=True), simulation_time=1, duration_seconds=1)])
    if visual:
        execution.update(landed=True, disarmed=True, safe_execution=True, localizations=[
            dict(decision_id="d1", frame_id=image["frame_id"], rgb_sha256=image["rgb_sha256"],
                 pixel=[500/999,500/999], source="gazebo_rgbd", surface_error_m=.05, frame_age_seconds=.1, distance_m=10)])
    (path/"execution.json").write_text(json.dumps(execution))
    if visual:
        annotations=[dict(frame_id=image["frame_id"], rgb_sha256=image["rgb_sha256"],
            polygon=([[0,0],[.2,0],[.2,.2],[0,.2]] if background else [[.4,.4],[.8,.4],[.8,.8],[.4,.8]]),
            reviewer="test-fixture")]
        (path/"visible_annotations.json").write_text(json.dumps(annotations))
    return standard, execution


def approve(path, standard):
    return build_review(path, standard=standard, reviewer="test-agent-reviewer",
                        response_sequences=[3], target_reviews={3:"Source clause inspect library; accepted and observed exactly once."},
                        visual_annotations=json.loads((path/"visible_annotations.json").read_text()) if (path/"visible_annotations.json").exists() else None)


class DemonstrationDataTests(unittest.TestCase):
    def test_recovery_context_is_never_a_positive_target(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"episode"
            standard, _ = make_episode(path, prompt_extra=[
                dict(role="assistant", content='{"status":"bad"}'),
                dict(role="user", content="That plan omitted the inspection. Correct it.")])
            approve(path, standard)
            report = export_dataset([path], Path(temp)/"dataset")
            self.assertEqual(report["approved_episodes"],1)
            self.assertEqual(report["decisions"],1)
            row=json.loads((Path(temp)/"dataset/pilot.jsonl").read_text())
            self.assertEqual(row["loss_scope"],"completion_only")
            self.assertEqual(row["prompt"][1]["content"],'{"status":"bad"}')
            self.assertEqual(len(row["completion"]),1)
            self.assertNotIn("bad",row["completion"][0]["content"])
            self.assertEqual(row["metadata"]["source"],"provider_override")
            self.assertFalse(report["training_started"])
            self.assertEqual(report["files"][0]["path"], "pilot.jsonl")
            self.assertEqual(report["files"][0]["sha256"], hashlib.sha256((Path(temp)/"dataset/pilot.jsonl").read_bytes()).hexdigest())

    def test_recorded_decoder_schema_and_lossless_property_order(self):
        # Actual production requirement schema contains a motion anyOf inside
        # node oneOf. The optimized runtime validator is not its general decoder.
        from uav_llm_demo.agents import _decoder_schema
        from uav_llm_demo.task_requirements import requirement_schema
        schema = _decoder_schema(requirement_schema(for_extraction=True))
        action = dict(status="ready", summary="Inspect the library.",
            steps=[dict(id="inspect1", action="inspect", target_id="library")],
            constraints={}, coverage=[dict(quote="Inspect the library.",
                requirement_ids=["inspect1"], constraint_keys=[])])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"episode"
            standard,_=make_episode(path,response_schema=schema,action_override=action)
            approve(path,standard)
            report=export_dataset([path],Path(temp)/"dataset")
            self.assertEqual(report["decisions"],1)
            row=json.loads((Path(temp)/"dataset/pilot.jsonl").read_text())
            target=row["completion"][0]["content"]
            self.assertEqual(json.loads(target),action)
            self.assertEqual(list(json.loads(target)["steps"][0]),["action","id","target_id"])
            self.assertEqual(row["metadata"]["target_normalization"],"decoder_property_order_only")
            self.assertEqual(row["metadata"]["original_response_content_sha256"],
                             hashlib.sha256(json.dumps(action).encode()).hexdigest())
            self.assertTrue((path/"standard.json").exists())
            self.assertFalse((path/"task_standard.json").exists())

    def test_no_selection_is_pending_and_never_exported(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"episode"
            standard, _=make_episode(path)
            review=build_review(path, standard=standard, reviewer="test-reviewer")
            self.assertEqual(review["status"],"pending")
            self.assertFalse((path/"review.json").exists())
            self.assertFalse((path/"standard.json").exists())
            self.assertFalse(audit_episode(path)["approved"])
            self.assertEqual(export_dataset([path],Path(temp)/"dataset")["decisions"],0)

    def test_bad_responses_and_credential_or_oracle_leakage_are_rejected(self):
        cases=[
            dict(finish="length"), dict(status="rejected"),
            dict(prompt_extra=[dict(role="user",content='{"api_key":"not-a-real-test-secret"}')]),
            dict(prompt_extra=[dict(role="user",content='{"oracle_target":"hidden-target"}')]),
            dict(prompt_extra=[dict(role="user",content="Bearer example-credential")]),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for index, kwargs in enumerate(cases):
                with self.subTest(index=index):
                    path=Path(temp)/str(index)
                    standard,_=make_episode(path,**kwargs)
                    with self.assertRaises(ValueError):
                        approve(path,standard)

    def test_success_flag_does_not_override_independent_evidence_or_run_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            for kind in ("wrong_target","other_journal","wrong_run","changed_source"):
                with self.subTest(kind=kind):
                    path=Path(temp)/kind
                    standard,execution=make_episode(path)
                    if kind=="wrong_target": execution["evidence"][0]["target_id"]="stadium"
                    if kind=="other_journal": execution["trajectory_events_sha256"]="different"
                    if kind=="wrong_run": execution["run_id"]="other"
                    if kind=="changed_source": execution["source_unchanged"]=False
                    (path/"execution.json").write_text(json.dumps(execution))
                    with self.assertRaises(ValueError):
                        approve(path,standard)

    def test_evidence_and_standard_hashes_are_rechecked_on_export(self):
        with tempfile.TemporaryDirectory() as temp:
            for field in ("execution","standard"):
                path=Path(temp)/field
                standard,execution=make_episode(path)
                approve(path,standard)
                if field=="execution":
                    execution["evidence"][0]["target_id"]="stadium"
                    (path/"execution.json").write_text(json.dumps(execution))
                else:
                    standard["instruction"]="Another task"
                    (path/"standard.json").write_text(json.dumps(standard))
                result=audit_episode(path)
                self.assertFalse(result["approved"],result)
                self.assertEqual(result["samples"],[])

    def test_split_leakage_is_rejected_before_creating_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            paths=[]
            for index in (1,2):
                path=Path(temp)/str(index)
                standard,_=make_episode(path,name=f"task{index}",family=f"family{index}")
                approve(path,standard)
                paths.append(path)
            output=Path(temp)/"dataset"
            with self.assertRaisesRegex(ValueError,"Leakage"):
                export_dataset(paths,output,splits={"task1":"train","task2":"test"})
            self.assertFalse(output.exists())

    def test_vertical_path_constraint_rejects_a_horizontal_excursion(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"episode"
            standard, execution=make_episode(path)
            standard["trajectory_checks"]=[dict(kind="horizontal_radius",target="home",max_distance_m=.5)]
            execution["world"]={"home":{"center":[0,0,0]}}
            execution["telemetry"]=[dict(t=0,mission_id=1,position=[0,0,0]),
                                    dict(t=1,mission_id=1,position=[1,0,1]),
                                    dict(t=2,mission_id=1,position=[0,0,0])]
            # Unit fixture commits the task amendment before invoking the checker.
            digest=canonical_sha256(standard)
            manifest=json.loads((path/"manifest.json").read_text())
            manifest["metadata"]["task_standard_sha256"]=digest
            (path/"manifest.json").write_text(json.dumps(manifest))
            execution["task_standard_sha256"]=digest
            (path/"execution.json").write_text(json.dumps(execution))
            with self.assertRaisesRegex(ValueError,"horizontal radius"):
                approve(path,standard)

    def test_visual_tools_require_their_observed_causal_continuation(self):
        from uav_llm_demo.direct_vision import TOOLS, DirectVisionAgent

        def row(number, tool, arguments, image, feedback=None, state=None):
            raw = dict(arguments, reason="Reviewed decision")
            body = dict(choices=[dict(finish_reason="tool_calls", message=dict(role="assistant",
                tool_calls=[dict(id=str(number), type="function", function=dict(name=tool, arguments=json.dumps(raw)))]))])
            action = DirectVisionAgent._decision(body, image["frame_id"])
            return dict(decision_id=f"d{number}", response_sequence=number*10+2,
                input=dict(sequence=number*10, payload=dict(frames=[image])),
                request=dict(sequence=number*10+1, payload=dict(request=dict(model="test", tools=TOOLS))),
                response=dict(payload=dict(response=body, http_status=200, outcome="ok")),
                result=dict(sequence=number*10+3, payload=dict(status="applied", action=action,
                    feedback=feedback or {}, state=state or dict(position_grid=image["position_grid"], simulation_time=image["simulation_time"]))))

        start = frame()
        finish = dict(start, frame_id="after", simulation_time=7., position_grid=[2,2,3], yaw_deg=90, pitch_deg=-40)
        preview = row(1, "preview_move", dict(offset_m=[3,0,0]), start,
                      dict(candidate_id="candidate-1", endpoint_grid=[2,2,3]))
        submit = row(2, "submit_move", dict(candidate_id="candidate-1"), start)
        ground = row(4, "ground", dict(frame_id="after", box=[200,200,800,800], point=[500,500], confidence=.9), finish)
        motion = [dict(kind="motion_started", sequence=24, payload=dict(decision_id="d2", simulation_time=3.)),
                  dict(kind="motion_completed", sequence=25, payload=dict(decision_id="d2", simulation_time=6., position_grid=[2,2,3]))]
        standard = dict(visual_limits=dict(max_surface_error_m=.2, max_frame_age_seconds=1, max_distance_m=20))
        execution = dict(evidence=[], localizations=[dict(decision_id="d4", frame_id="after", rgb_sha256=finish["rgb_sha256"],
            pixel=[500/999,500/999], source="gazebo_rgbd", surface_error_m=.05, frame_age_seconds=.1, distance_m=10)])
        annotations = [dict(frame_id="after", rgb_sha256=finish["rgb_sha256"], reviewer="agent-assisted-test",
                            polygon=[[.4,.4],[.8,.4],[.8,.8],[.4,.8]])]
        def check(target, candidates, events):
            with patch("uav_llm_demo.demonstration_data._journal", return_value=({}, events)), \
                 patch("uav_llm_demo.demonstration_data.decision_candidates", return_value=candidates):
                return _visual_action(Path("."), target, target["result"]["payload"]["action"], standard, execution, annotations)

        for target in (preview, submit):
            proof = check(target, [preview,submit,ground], motion)
            self.assertEqual(proof["continuation_frame_id"], "after")
            self.assertEqual(proof["motion_completed_sequence"], 25)
        look = row(1, "look", dict(yaw_deg=90,pitch_deg=-40), start)
        self.assertEqual(check(look, [look,ground], [])["continuation_decision_id"], "d4")

        # Distinct failure boundaries: no actual flight, aborted flight, wrong
        # endpoint, no new observation, and a camera request not used downstream.
        for defect in ("unused_preview", "aborted", "wrong_endpoint", "stale_frame", "wrong_view"):
            with self.subTest(defect=defect):
                rows, events = deepcopy([preview,submit,ground]), deepcopy(motion)
                target = rows[0]
                if defect == "unused_preview": rows.pop(1)
                if defect == "aborted": events[-1]["kind"] = "motion_aborted"
                if defect == "wrong_endpoint": events[-1]["payload"]["position_grid"] = [9,2,3]
                if defect == "stale_frame": rows[-1]["input"]["payload"]["frames"][-1]["simulation_time"] = 1.
                if defect == "wrong_view":
                    target = look
                    rows[-1]["input"]["payload"]["frames"][-1]["yaw_deg"] = 0
                with self.assertRaises(ValueError): check(target,rows,events)
        stationary = row(3, "wait", {}, start, state=dict(velocity=[0,0,0]))
        with self.assertRaisesRegex(ValueError, "Stationary wait"):
            check(stationary,[preview,submit,stationary,ground],motion)

    def test_visual_requires_visible_interior_and_exports_portable_images(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"visual"
            standard,_=make_episode(path,visual=True)
            approve(path,standard)
            report=export_dataset([path],Path(temp)/"dataset")
            self.assertEqual(report["decisions"],1)
            row=json.loads((Path(temp)/"dataset/pilot.jsonl").read_text())
            self.assertEqual(row["prompt"][0]["content"][1]["type"],"image")
            self.assertFalse(Path(row["images"][0]).is_absolute())
            self.assertTrue((Path(temp)/"dataset"/row["images"][0]).is_file())
            self.assertEqual({ref["path"] for ref in report["files"]}, {"pilot.jsonl", row["images"][0]})
            # A close world-space estimate does not excuse a background pixel.
            bad=Path(temp)/"background"
            standard,execution=make_episode(bad,visual=True,background=True)
            with self.assertRaisesRegex(ValueError,"visible target"):
                approve(bad,standard)


if __name__=="__main__":
    unittest.main()
