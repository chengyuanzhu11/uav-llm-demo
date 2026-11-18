"""Task trace evidence at the real parser/tool/executor boundary, without remote calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.semantic_fixtures import fixture_requirements, fixture_audit, response
from tests.test_semantic_service import advance_until
from tests.test_semantic_updates import compact_campus, home, program, skill
from uav_llm_demo.agents import DeepSeekAgent, SemanticPlanError, task_interaction_trace
from uav_llm_demo.interaction_trajectory import InteractionRecorder
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.planning_tools import PlanningTools
from uav_llm_demo.semantic_benchmark import _BudgetedAgent
from uav_llm_demo.semantic_tasks import task_catalog


def events(directory):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


def trace(recorder, stage="plan", **kwargs):
    return task_interaction_trace(recorder, run_id="episode", request_id="task-1", stage=stage,
                                  context={"instruction": "返回基地。"}, **kwargs)


class TaskInteractionTests(unittest.TestCase):
    def setUp(self):
        guard = patch("requests.sessions.Session.request", side_effect=AssertionError("Remote API forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_expert_override_records_three_stages_and_actual_execution(self):
        scene, instruction = compact_campus(), "巡检图书馆后返航。"
        plan = program([skill("inspect", "inspect", "library"), home()])
        requirements = fixture_requirements(plan, task_catalog(scene), instruction).to_dict()
        agent = _BudgetedAgent(api_key="", base_url="http://127.0.0.1:8001/v1", replies=[plan],
            requirements=[requirements], audits=[fixture_audit(requirements, instruction)])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "episode"
            recorder = InteractionRecorder(directory, metadata={"source": "scripted_expert"})
            service = MissionService(scenario=scene, agent=agent, trajectory_recorder=recorder)
            try:
                service.submit_task(instruction, background=False)
                state = advance_until(service, lambda s: s["status"] == "completed")
            finally:
                service.close()
            result = recorder.close(outcome={"status": state["status"]})
            self.assertTrue(result["complete"], result["errors"])
            self.assertEqual(result["decisions"], 3)
            rows = events(directory)
            inputs = [row["payload"] for row in rows if row["kind"] == "decision_input"]
            self.assertEqual([row["stage"] for row in inputs], ["requirements", "audit", "plan"])
            self.assertTrue(all(row["modality"] == "text" and row["frames"] == [] for row in inputs))
            self.assertTrue(all(row["context"]["mission_id"] == service.semantic.mission_id for row in inputs))
            self.assertTrue(all(row["context"]["vehicle_id"] == service.semantic.vehicle_id for row in inputs))
            replies = [row["payload"] for row in rows if row["kind"] == "model_response"]
            self.assertTrue(all(row["source"] == "provider_override" and row["http_status"] is None for row in replies))
            self.assertEqual([r["action"] for r in state["semantic_mission"]["evidence"]], ["inspect", "return_home"])

    def test_each_tool_completion_has_exact_request_feedback_and_terminal(self):
        scene = compact_campus()
        catalog = task_catalog(scene)
        context = {**scene.to_dict(), "position": dict(zip("xyz", scene.base))}
        tools = PlanningTools(scene, context, catalog)
        replies = [dict(tool="submit_plan", arguments={"candidate_id": "unknown"}),
                   dict(tool="preview_plan", arguments={"plan": program([home()])}),
                   dict(tool="submit_plan", arguments={"candidate_id": "candidate-1"})]
        agent = DeepSeekAgent(api_key="", model="fixture", base_url="http://127.0.0.1:8001/v1",
                              task_response_format="json_schema")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "episode"
            recorder = InteractionRecorder(directory)
            with trace(recorder), patch("uav_llm_demo.agents._post_deepseek", side_effect=[response(r) for r in replies]) as post:
                accepted = agent.interpret_task(catalog, "返回基地。", planning=tools)
            result = recorder.close(outcome="planned")
            self.assertEqual(accepted.status, "ready")
            self.assertTrue(result["complete"], result["errors"])
            rows = events(directory)
            requests = [row["payload"] for row in rows if row["kind"] == "model_request"]
            self.assertEqual(len({row["decision_id"] for row in requests}), 3)
            for saved, sent in zip(requests, post.call_args_list):
                self.assertEqual(saved["request"], sent.kwargs["json"])
            self.assertIn("unknown_candidate", requests[1]["request"]["messages"][-1]["content"])
            terminals = [row["payload"]["status"] for row in rows if row["kind"] == "decision_result"]
            self.assertEqual(terminals, ["rejected", "applied", "applied"])
            self.assertEqual(sum(row["kind"] == "tool_feedback" for row in rows), 3)

    def test_rejected_raw_reply_is_saved_but_credentials_are_redacted(self):
        secret, nested = "known-private-key", "nested-private-secret"
        agent = DeepSeekAgent(api_key=secret, model="fixture", base_url="http://127.0.0.1:8001/v1")
        malformed = response({})
        malformed.json.return_value["choices"][0]["message"]["content"] = '{"password":"' + nested + '","steps":['
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "episode"
            recorder = InteractionRecorder(directory)
            with trace(recorder), patch("uav_llm_demo.agents._post_deepseek", return_value=malformed):
                with self.assertRaises(SemanticPlanError):
                    agent.interpret_task(task_catalog(compact_campus()), "返回基地。")
            result = recorder.close(outcome="failed")
            self.assertTrue(result["complete"], result["errors"])
            raw = (directory / "events.jsonl").read_text()
            self.assertNotIn(secret, raw)
            self.assertNotIn(nested, raw)
            rows = events(directory)
            self.assertIn('[redacted]', rows[2]["payload"]["response"]["choices"][0]["message"]["content"])
            self.assertEqual(rows[-1]["payload"]["status"], "rejected")

    def test_discard_and_recording_failure_do_not_accept_stale_data_or_break_planning(self):
        scene = compact_campus()
        agent = DeepSeekAgent(api_key="", model="fixture", base_url="http://127.0.0.1:8001/v1")
        with tempfile.TemporaryDirectory() as temp:
            for fail in (False, "write", "redact"):
                directory = Path(temp) / str(fail)
                recorder = InteractionRecorder(directory)
                original = recorder.record
                def write(kind, **payload):
                    if fail == "write" and kind == "model_response":
                        raise OSError("disk unavailable")
                    return original(kind, **payload)
                from uav_llm_demo import agents
                sanitizer = agents._redact_task_interaction
                def clean(value, key):
                    if fail == "redact":
                        raise RecursionError("cannot sanitize")
                    return sanitizer(value, key)
                with trace(recorder, cancelled=lambda: not fail), patch.object(recorder, "record", side_effect=write), \
                        patch("uav_llm_demo.agents._redact_task_interaction", side_effect=clean), \
                        patch("uav_llm_demo.agents._post_deepseek", return_value=response(program([home()]))):
                    result = agent.interpret_task(task_catalog(scene), "返回基地。")
                self.assertEqual(result.status, "ready")
                checked = recorder.close(outcome="cancelled" if not fail else "planned")
                self.assertEqual(checked["complete"], not fail, checked["errors"])
                if not fail:
                    self.assertEqual(events(directory)[-1]["payload"]["status"], "discarded")


if __name__ == "__main__":
    unittest.main()
