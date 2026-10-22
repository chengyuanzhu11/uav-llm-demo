"""Offline protocol repair with real campus geometry, task state, and flight control."""
from copy import deepcopy
from dataclasses import replace
import json
from math import atan2, pi
from threading import Event
import unittest
from unittest.mock import Mock, patch

from tests.test_semantic_motion_service import motion
from tests.test_semantic_service import advance_until, response
from tests.test_semantic_updates import compact_campus, home, program
from uav_llm_demo.agents import DeepSeekAgent, SemanticPlanError
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import distance, norm
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_tasks import task_catalog
from uav_llm_demo.spatial_motion import spatial_catalog


PRIVATE_CANDIDATE = "rejected-candidate-private-marker"
INSTRUCTION = "沿着东侧花园转一圈然后返航"


def boundary_program(target="east_garden", direction="clockwise"):
    return program([motion(dict(kind="boundary", direction=direction, laps=1), target), home()])


def rejected_program(target="east_garden"):
    candidate = boundary_program(target)
    candidate["summary"] = PRIVATE_CANDIDATE
    candidate["steps"][0]["motion"]["path"]["radius_m"] = 4
    return candidate


def message_text(call):
    return json.dumps(call.kwargs["json"]["messages"], ensure_ascii=False)


class SemanticRepairTests(unittest.TestCase):
    def setUp(self):
        # Both transport boundaries are guarded: no test may contact a real provider.
        for target in ("uav_llm_demo.agents._post_deepseek", "requests.sessions.Session.request"):
            guard = patch(target, side_effect=AssertionError("Remote API forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def service(self, *, actual=False):
        scene = replace(load_scenario("semantic_campus"), dynamic_objects=()) if actual else compact_campus()
        journal = Mock(dropped=0, errors=0)
        service = MissionService(scenario=scene, agent=DeepSeekAgent(api_key="offline-repair-test-key"),
                                 planner_latency_seconds=0, traffic_seed=17, journal=journal)
        self.addCleanup(service.close)
        return service, journal

    def wait_for_rejection(self, service):
        return advance_until(service, lambda _: service._semantic_schema_failures == 1, seconds=5)

    def assert_no_private_candidate(self, service, journal):
        public = json.dumps(service.state(), ensure_ascii=False)
        records = json.dumps([call.args[0] for call in journal.emit.call_args_list], ensure_ascii=False)
        for text in (public, records):
            self.assertNotIn(PRIVATE_CANDIDATE, text)
            self.assertNotIn("offline-repair-test-key", text)

    def test_provider_repair_contains_rejected_program_and_precise_diagnostic(self):
        scene = load_scenario("semantic_campus")
        agent = DeepSeekAgent(api_key="offline-repair-test-key")
        bad = rejected_program()
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=[response(bad), response(boundary_program())]) as post:
            with self.assertRaises(SemanticPlanError) as caught:
                agent.interpret_task(task_catalog(scene), INSTRUCTION)
            error = caught.exception
            self.assertEqual(json.loads(error.candidate)["steps"], bad["steps"])
            self.assertIn("steps[0].motion.path", error.diagnostic["path"])
            self.assertIn("radius_m", json.dumps(error.diagnostic))
            result = agent.interpret_task(task_catalog(scene), INSTRUCTION, error.detail,
                                         repair=dict(candidate=error.candidate, diagnostic=error.diagnostic))
        self.assertEqual(result.status, "ready")
        self.assertEqual(post.call_count, 2)
        first, second = map(message_text, post.call_args_list)
        self.assertNotIn(PRIVATE_CANDIDATE, first)
        self.assertIn(PRIVATE_CANDIDATE, second)
        self.assertIn(error.diagnostic["path"], second)
        self.assertIn(INSTRUCTION, second)
        self.assertIsNone(agent._previous_messages)

    def test_malformed_json_uses_bounded_correction_and_clears_prompt_history(self):
        agent = DeepSeekAgent(api_key="offline-repair-test-key")
        catalog = task_catalog(load_scenario("semantic_campus"))
        malformed = response({})
        malformed.json.return_value["choices"][0]["message"]["content"] = '{"status":"ready",\n"steps":['
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=[malformed, response(boundary_program())]) as post:
            with self.assertRaises(SemanticPlanError) as caught:
                agent.interpret_task(catalog, INSTRUCTION)
            error = caught.exception
            self.assertEqual(error.diagnostic["code"], "invalid_json")
            self.assertEqual(error.diagnostic["path"], "$")
            self.assertRegex(error.diagnostic["message"], r"line \d+, column \d+")
            self.assertIsNone(error.candidate)
            result = agent.interpret_task(catalog, INSTRUCTION, error.detail,
                                         repair=dict(candidate=None, diagnostic=error.diagnostic))
        self.assertEqual(result.status, "ready")
        self.assertEqual(post.call_count, 2)
        self.assertIn("invalid_json", message_text(post.call_args_list[1]))
        self.assertIsNone(agent._previous_messages)

    def test_rejected_candidates_redact_credential_fields_and_bound_response_size(self):
        agent = DeepSeekAgent(api_key="offline-repair-test-key")
        catalog = task_catalog(load_scenario("semantic_campus"))
        candidate = rejected_program()
        candidate["steps"][0]["motion"]["path"]["api_key"] = "private-candidate-token"
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(candidate)):
            with self.assertRaises(SemanticPlanError) as caught:
                agent.interpret_task(catalog, INSTRUCTION)
        error = caught.exception
        self.assertIsInstance(error.candidate, str)
        self.assertNotIn("private-candidate-token", error.candidate)
        self.assertNotIn("private-candidate-token", json.dumps(error.diagnostic))
        self.assertNotIn("private-candidate-token", error.detail)
        self.assertEqual(json.loads(error.candidate)["steps"][0]["target_id"], "east_garden")
        self.assertLessEqual(len(error.candidate), 32000)

        oversized = rejected_program()
        oversized["summary"] = "x" * 40000
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(oversized)):
            with self.assertRaises(SemanticPlanError) as caught:
                agent.interpret_task(catalog, INSTRUCTION)
        self.assertIsNone(caught.exception.candidate)
        self.assertLess(len(caught.exception.detail), 4000)
        self.assertLess(len(json.dumps(caught.exception.diagnostic)), 4000)

    def test_three_invalid_programs_have_generation_failure_and_allow_same_instruction(self):
        service, journal = self.service()
        bad = rejected_program("library")
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=[
                response(bad), response(bad), response(bad), response(program([home()]))]) as post:
            service.submit_task(INSTRUCTION, background=False)
            failed = advance_until(service, lambda s: s["semantic_mission"]["status"] == "generation_failed", 10)
            self.assertEqual(post.call_count, 3)
            self.assertFalse(failed["flight"]["running"])
            self.assertNotIn("请调整任务描述", failed["semantic_mission"]["message"])
            self.assertEqual(failed["uav"], failed["base"])
            self.assertEqual(failed["semantic_mission"]["evidence"], [])
            for _ in range(50):
                service.advance(.1)
            self.assertEqual(post.call_count, 3)
            self.assert_no_private_candidate(service, journal)
            service.submit_task(INSTRUCTION, background=False)
            final = advance_until(service, lambda s: s["status"] == "completed", 10)
            self.assertEqual(post.call_count, 4)
            self.assertNotIn(PRIVATE_CANDIDATE, message_text(post.call_args_list[-1]))
        self.assertEqual(final["semantic_mission"]["instruction"], INSTRUCTION)
        self.assertIsNone(service._semantic_repair)

    def test_new_instruction_and_reset_clear_pending_repair(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                service, journal = self.service()
                with patch("uav_llm_demo.agents._post_deepseek", side_effect=[
                        response(rejected_program("library")), response(program([home()]))]) as post:
                    service.submit_task(INSTRUCTION, background=False)
                    self.wait_for_rejection(service)
                    self.assertIsNotNone(service._semantic_repair)
                    self.assert_no_private_candidate(service, journal)
                    if reset:
                        service.reset()
                        self.assertIsNone(service._semantic_repair)
                    service.submit_task("取消剩余任务，返航。", background=False)
                    self.assertIsNone(service._semantic_repair)
                    advance_until(service, lambda s: s["status"] == "completed", 10)
                self.assertEqual(post.call_count, 2)
                self.assertNotIn(PRIVATE_CANDIDATE, message_text(post.call_args_list[-1]))
                self.assert_no_private_candidate(service, journal)

    def test_late_rejected_correction_cannot_contaminate_new_instruction_or_reset(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                entered, release = Event(), Event()
                service, journal = self.service()
                self.addCleanup(release.set)

                def delayed_rejection(*args, **kwargs):
                    entered.set()
                    if not release.wait(10):
                        raise AssertionError("Test did not release the delayed response")
                    return response(rejected_program("library"))

                replies = iter([response(rejected_program("library")), delayed_rejection,
                                response(program([home()]))])

                def respond(*args, **kwargs):
                    item = next(replies)
                    return item(*args, **kwargs) if item is delayed_rejection else item

                with patch("uav_llm_demo.agents._post_deepseek", side_effect=respond) as post:
                    service.submit_task(INSTRUCTION, background=False)
                    self.wait_for_rejection(service)
                    for _ in range(40):
                        service.advance(.1)
                        if entered.wait(.005):
                            break
                    self.assertTrue(entered.is_set())
                    self.assertIn(PRIVATE_CANDIDATE, message_text(post.call_args_list[1]))
                    if reset:
                        service.reset()
                    service.submit_task("结束任务并返航。", background=False)
                    self.assertIsNone(service._semantic_repair)
                    release.set()
                    final = advance_until(service, lambda s: s["status"] == "completed", 10)
                self.assertEqual(post.call_count, 3)
                self.assertNotIn(PRIVATE_CANDIDATE, message_text(post.call_args_list[2]))
                self.assertEqual(final["semantic_mission"]["instruction"], "结束任务并返航。")
                self.assertEqual(service._semantic_schema_failures, 0)
                self.assertIsNone(service._semantic_repair)
                self.assert_no_private_candidate(service, journal)

    def test_repaired_east_and_west_garden_missions_traverse_full_lap_and_land(self):
        cases = (("east_garden", "clockwise", INSTRUCTION, 1),
                 ("west_garden", "counterclockwise", "沿着西侧花园逆时针转一圈，然后返航。", -1))
        for target, direction, instruction, sign in cases:
            with self.subTest(target=target):
                service, journal = self.service(actual=True)
                accepted = boundary_program(target, direction)
                rejected = deepcopy(accepted)
                rejected["summary"] = PRIVATE_CANDIDATE
                rejected["steps"][0]["motion"]["path"]["radius_m"] = 4
                center = next(o["center_grid"] for o in spatial_catalog(service._environment.scenario)["objects"]
                              if o["id"] == target)
                angles, progress = [], []
                with patch("uav_llm_demo.agents._post_deepseek", side_effect=[response(rejected), response(accepted)]) as post:
                    service.submit_task(instruction, background=False)
                    for _ in range(5000):
                        if service._future is not None:
                            try:
                                service._future.result(timeout=15)
                            except SemanticPlanError:
                                pass
                        service.advance(.1)
                        state = service.state()
                        tracking = state["semantic_mission"].get("motion_progress")
                        if tracking and tracking["phase"] == "tracking":
                            position = service._vehicle.position
                            angles.append(atan2(position[1] - center[1], position[0] - center[0]))
                            progress.append(tracking["distance_m"])
                        if state["status"] == "completed":
                            break
                self.assertEqual(post.call_count, 2)
                self.assertIn(PRIVATE_CANDIDATE, message_text(post.call_args_list[1]))
                self.assertEqual(state["status"], "completed", service.semantic.message)
                self.assertGreater(len(angles), 100)
                winding = sign * sum((b - a + pi) % (2 * pi) - pi for a, b in zip(angles, angles[1:]))
                self.assertGreater(winding, 1.9 * pi)
                self.assertLess(winding, 2.1 * pi)
                self.assertEqual(progress, sorted(progress))
                evidence = state["semantic_mission"]["evidence"]
                self.assertEqual([item["action"] for item in evidence], ["motion", "return_home"])
                self.assertEqual(evidence[0]["source"], "runtime_trajectory_tracking")
                self.assertTrue(evidence[0]["result"]["completed"])
                self.assertAlmostEqual(evidence[0]["result"]["distance_m"], evidence[0]["result"]["total_length_m"])
                self.assertTrue(evidence[-1]["result"]["landed"])
                self.assertTrue(state["milestones"]["safe_success"])
                self.assertLessEqual(distance(service._vehicle.position, service._environment.base), .18)
                self.assertLessEqual(norm(service._vehicle.velocity), .02)
                self.assertIsNone(service._semantic_repair)
                self.assert_no_private_candidate(service, journal)


if __name__ == "__main__":
    unittest.main()
