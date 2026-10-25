"""Offline service regressions for complete task acceptance across replanning.

The model replies are fixtures; parsing, geometry, flight, observations, review
handling and final success decisions use the production service.
"""
from copy import deepcopy
from threading import Event
import unittest
from unittest.mock import patch

from tests.test_semantic_updates import (
    OfflineSemanticAgent, compact_campus, home, program, pump, repeat, skill,
)
from uav_llm_demo.mission_service import MissionService


def review(name="assess"):
    return dict(id=name, action="review", instruction="根据观测调整后续执行，保留原任务要求。", when=None)


class RecordingOfflineAgent(OfflineSemanticAgent):
    def __init__(self, replies):
        super().__init__(replies)
        self.repairs = []

    def interpret_task(self, catalog, instruction, feedback="", *, repair=None):
        self.repairs.append(deepcopy(repair))
        return super().interpret_task(catalog, instruction, feedback, repair=repair)


class TaskContractServiceTests(unittest.TestCase):
    def setUp(self):
        for target in ("uav_llm_demo.agents._post_deepseek", "requests.sessions.Session.request"):
            guard = patch(target, side_effect=AssertionError("Network is forbidden in contract tests"))
            guard.start()
            self.addCleanup(guard.stop)

    def service(self, *replies):
        service = MissionService(
            scenario=compact_campus(), agent=RecordingOfflineAgent(replies),
            planner_latency_seconds=0, traffic_seed=17,
        )
        self.addCleanup(service.close)
        return service

    def completed(self, service):
        state = pump(service, lambda item: item["status"] == "completed", seconds=240)
        self.assertTrue(state["milestones"]["task_completed"])
        self.assertTrue(state["milestones"]["safe_success"])
        self.assertTrue(state["semantic_mission"]["task_contract"]["satisfied"])
        return state

    @staticmethod
    def work_records(state):
        return [record for record in state["semantic_mission"]["evidence"]
                if record["action"] not in ("review", "return_home")]

    def test_review_cannot_drop_obligations_across_actions_and_places(self):
        for action, target in (("inspect", "conference"), ("thermal_scan", "laboratory")):
            with self.subTest(action=action, target=target):
                required = skill("required", action, target, .1)
                corrected = skill("renamed", action, target, .1)
                service = self.service(program([review(), required, home()]),
                                       program([home()]), program([corrected, home()]))
                service.submit_task("评估情况后执行指定地点任务，最后返航。", background=False)
                final = self.completed(service)
                records = self.work_records(final)
                self.assertEqual([(record["action"], record["target_id"]) for record in records],
                                 [(action, target)])
                self.assertEqual(len(service.agent.inputs), 3)
                self.assertIsNotNone(service.agent.repairs[2])
                contract_id = final["semantic_mission"]["task_contract"]["contract_id"]
                self.assertEqual([item[0]["mission_context"]["task_contract"]["contract_id"]
                                  for item in service.agent.inputs[1:]], [contract_id, contract_id])
                self.assertEqual(sum(record["action"] == "review"
                                     for record in service.semantic.evidence), 1)

    def test_review_repairs_reordered_work_and_reduced_repetition(self):
        first = skill("first", "inspect", "library", .1)
        second = skill("second", "thermal_scan", "laboratory", .1)
        cases = (
            ([first, second], [second, first], [("inspect", "library"), ("thermal_scan", "laboratory")]),
            ([repeat([first], count=3)], [repeat([first], count=2)], [("inspect", "library")] * 3),
        )
        for required, altered, expected in cases:
            with self.subTest(required=required):
                service = self.service(program([review(), *required, home()]),
                                       program([*altered, home()]), program([*required, home()]))
                service.submit_task("评估后按指定顺序与次数执行任务，最后返航。", background=False)
                final = self.completed(service)
                self.assertEqual([(record["action"], record["target_id"])
                                  for record in self.work_records(final)], expected)
                self.assertEqual(len(service.agent.inputs), 3)
                self.assertIsNotNone(service.agent.repairs[2])

    def test_legal_review_adds_auxiliary_work_without_repeating_completed_work(self):
        scan = skill("scan", "thermal_scan", "library", .1)
        inspect = skill("required", "inspect", "conference", .1)
        auxiliary = skill("extra", "capture", "library", .1)
        service = self.service(program([scan, review(), inspect, home()]),
                               program([auxiliary, inspect, home()]))
        service.submit_task("测量图书馆温度，根据结果调整，然后巡检会议中心并返航。", background=False)
        final = self.completed(service)
        self.assertEqual([(record["action"], record["target_id"])
                          for record in self.work_records(final)],
                         [("thermal_scan", "library"), ("capture", "library"), ("inspect", "conference")])
        self.assertEqual(len(service.agent.inputs), 2)
        self.assertEqual(service.agent.repairs, [None, None])
        observed = service.agent.inputs[1][0]["mission_context"]["evidence"]
        self.assertEqual(observed[0]["result"]["temperature_c"], 29)

    def test_review_result_can_enable_work_after_an_unconditional_action(self):
        enabled = dict(step_id="assess", field="reviewed", op="==", value=True)
        inspect = skill("inspect", "inspect", "library", .1)
        conditional_capture = skill("sample", "capture", "conference", .1, when=enabled)
        capture = skill("sample", "capture", "conference", .1)
        service = self.service(program([review(), inspect, conditional_capture, home()]),
                               program([inspect, capture, home()]))
        service.submit_task("完成评估后先巡检图书馆，再到会议中心采样，然后返航。", background=False)
        final = self.completed(service)
        self.assertEqual([(item["action"], item["target_id"]) for item in self.work_records(final)],
                         [("inspect", "library"), ("capture", "conference")])
        records = [item for item in final["semantic_mission"]["evidence"] if item["action"] == "review"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "validated_execution_review")
        self.assertEqual(len(service.agent.inputs), 2)
        self.assertEqual(service.agent.repairs, [None, None])
        self.assertEqual(service.agent.inputs[1][0]["mission_context"]["evidence"], [])

    def test_review_inside_loop_preserves_remaining_iterations_without_redo(self):
        inspect = skill("inspect", "inspect", "library", .1)
        service = self.service(program([repeat([inspect, review()], count=3), home()]),
                               program([repeat([inspect], count=1), home()]),
                               program([repeat([inspect], count=2), home()]))
        service.submit_task("图书馆连续检查三次，每次检查后评估，最终返航。", background=False)
        final = self.completed(service)
        records = self.work_records(final)
        self.assertEqual([record["action"] for record in records], ["inspect"] * 3)
        self.assertEqual(len({record["execution_id"] for record in records}), 3)
        self.assertEqual(len(service.agent.inputs), 3)
        self.assertIsNotNone(service.agent.repairs[2])

    def test_failed_reviews_never_complete_and_explicit_cancellation_rebuilds_contract(self):
        service = self.service(program([review(), skill("must", "inspect", "library", .1), home()]),
                               program([home()]), program([home()]), program([home()]), program([home()]))
        service.submit_task("评估后巡检图书馆再返回。", background=False)
        failed = pump(service, lambda item: item["semantic_mission"]["status"] == "generation_failed", seconds=15)
        self.assertFalse(failed["milestones"]["task_completed"])
        self.assertFalse(failed["milestones"]["safe_success"])
        self.assertFalse(failed["semantic_mission"]["task_contract"]["satisfied"])
        self.assertEqual(service.semantic.evidence, [])
        self.assertEqual(len(service.agent.inputs), 4)
        prior_contract_id = failed["semantic_mission"]["task_contract"]["contract_id"]
        self.assertEqual([item[0]["mission_context"]["task_contract"]["contract_id"]
                          for item in service.agent.inputs[1:]], [prior_contract_id] * 3)
        for _ in range(20):
            service.advance(.1)
        self.assertEqual(len(service.agent.inputs), 4)
        service.submit_task("取消尚未执行的所有工作，现在只需返航。", background=False)
        final = self.completed(service)
        self.assertEqual([record["action"] for record in final["semantic_mission"]["evidence"]], ["return_home"])
        self.assertEqual(len(service.agent.inputs), 5)
        self.assertIsNone(service.agent.repairs[-1])
        self.assertNotEqual(final["semantic_mission"]["task_contract"]["contract_id"], prior_contract_id)

    def test_autonomous_review_preserves_elapsed_clock_and_user_revision_resets_it(self):
        pending = skill("later", "inspect", "conference", 30)
        service = self.service(program([skill("scan", "thermal_scan", "library", .1), review(), pending, home()]),
                               program([pending, home()]), program([home()]))
        service.submit_task("先测图书馆温度，评估后检查会议中心三十秒，再返航。", background=False)
        reviewed = pump(service, lambda item: item["semantic_mission"]["revision"] == 2
                        and item["semantic_mission"]["status"] == "running")
        prior_elapsed = service.agent.inputs[1][0]["mission_context"]["telemetry"]["elapsed_seconds"]
        self.assertGreater(prior_elapsed, 1)
        self.assertGreaterEqual(service._semantic_telemetry()["elapsed_seconds"], prior_elapsed)
        prior_contract_id = service.agent.inputs[1][0]["mission_context"]["task_contract"]["contract_id"]
        self.assertEqual(reviewed["semantic_mission"]["task_contract"]["contract_id"], prior_contract_id)

        service.submit_task("取消会议中心检查，现在返航。", background=False)
        revised = pump(service, lambda item: item["semantic_mission"]["revision"] == 3)
        self.assertLess(service._semantic_telemetry()["elapsed_seconds"], 1)
        self.assertNotEqual(revised["semantic_mission"]["task_contract"]["contract_id"], prior_contract_id)
        final = self.completed(service)
        self.assertEqual([record["action"] for record in self.work_records(final)], ["thermal_scan"])

    def test_late_review_cannot_restore_cancelled_obligations_or_pollute_new_contract(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)

        def delayed():
            entered.set()
            if not release.wait(10):
                raise AssertionError("Delayed offline reply was not released")
            return program([home()])

        latest = skill("new", "capture", "conference", .1)
        service = self.service(program([review(), skill("old", "inspect", "library", .1), home()]),
                               delayed, program([latest, home()]))
        service.submit_task("评估后巡检图书馆。", background=False)
        for _ in range(100):
            service.advance(.05)
            if entered.wait(.005):
                break
        self.assertTrue(entered.is_set())
        service.submit_task("取消图书馆任务，改为会议中心采样并返航。", background=False)
        release.set()
        final = self.completed(service)
        self.assertEqual([(record["action"], record["target_id"])
                          for record in self.work_records(final)], [("capture", "conference")])
        self.assertEqual(len(service.agent.inputs), 3)
        self.assertIsNone(service.agent.repairs[-1])
        self.assertEqual(service._semantic_schema_failures, 0)
        self.assertEqual(final["semantic_mission"]["accepted_instruction"], "取消图书馆任务，改为会议中心采样并返航。")

    def test_program_exhaustion_with_missing_work_cannot_report_success(self):
        service = self.service(program([skill("first", "inspect", "library", .1),
                                       skill("missing", "capture", "conference", .1),
                                       skill("final", "inspect", "library", .1)]))
        service.submit_task("先巡检图书馆，再到会议中心采样，最后再次巡检图书馆并悬停。", background=False)
        pump(service, lambda item: item["semantic_mission"]["current_step_id"] == "first")
        # Inject a scheduler fault: the current action may finish, but its pending
        # continuation is lost. The first action ends at exactly the intended
        # terminal location and state, so an endpoint-only gate would succeed.
        service.semantic._stack.clear()
        final = pump(service, lambda item: item["semantic_mission"]["status"] != "running")
        self.assertFalse(final["milestones"]["task_completed"])
        self.assertFalse(final["milestones"]["safe_success"])
        self.assertFalse(final["semantic_mission"]["task_contract"]["satisfied"])
        self.assertNotEqual(final["status"], "completed")
        self.assertEqual([record["action"] for record in self.work_records(final)], ["inspect"])


if __name__ == "__main__":
    unittest.main()
