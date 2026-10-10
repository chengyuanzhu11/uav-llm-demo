import json
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from uav_llm_demo.agents import MockLLMAgent, RoutePlan, checked_plan, PlanValidationError
from uav_llm_demo.benchmark import run_scenario, summarize
from uav_llm_demo.evaluation import MissionEvidence
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import VehicleState
from uav_llm_demo.recovery import RecoveryMemory
from uav_llm_demo.replay import load_run, scenario_from_record
from uav_llm_demo.run_log import JournalWriter, RunTrace
from uav_llm_demo.scenarios import load_scenario


class RecoveryTests(unittest.TestCase):
    def test_provider_backoff_is_bounded_but_attempts_are_not(self):
        memory = RecoveryMemory()
        delays = [memory.provider_failed('temporary failure', (1, 2, 2), n) for n in range(12)]
        self.assertEqual(delays[:5], [2, 4, 8, 16, 30])
        self.assertEqual(delays[-1], 30)
        self.assertEqual(len(memory.failures), 8)
        memory.accepted()
        self.assertEqual(memory.provider_failed('temporary failure', (1, 2, 2), 20), 2)

    def test_repeated_geometry_retains_zero_based_segment_and_changes_feedback(self):
        memory = RecoveryMemory()
        plan = RoutePlan(((3, 4, 2), (6, 7, 2)), '', 'low', '')
        memory.route_failed(plan, ['Segment 0 is statically unsafe.'], (1, 2, 2), 0)
        first = memory.feedback('rejected')
        memory.route_failed(plan, ['Segment 0 is statically unsafe.'], (1, 2, 2), 3)
        self.assertNotEqual(first, memory.feedback('rejected'))
        self.assertEqual(memory.last_action, 'choose_different_corridor')
        self.assertEqual(memory.failures[-1]['rejected_segments'], [dict(start=(1, 2, 2), end=(3, 4, 2))])
        self.assertEqual(memory.repeat_count, 2)

    def test_checked_plan_retains_rejected_numeric_proposal(self):
        class Unsafe(MockLLMAgent):
            def plan(self, context, feedback=''):
                return RoutePlan(((-1, -1, -1),), '', 'low', '')
        context = load_scenario('open_field').to_dict()
        context.update(position=context['base'], goal=context['target'])
        with self.assertRaises(PlanValidationError) as caught:
            checked_plan(Unsafe(), context, '')
        self.assertEqual(caught.exception.plan.waypoints, ((-1, -1, -1),))
        self.assertTrue(caught.exception.errors)

    def test_evidence_requires_target_then_base_and_low_speed(self):
        evidence = MissionEvidence()
        base, target = (1, 2, 2), (6, 2, 2)
        evidence.observe(VehicleState(base), target, base, 0)
        self.assertFalse(evidence.snapshot()['task_completed'])
        evidence.observe(VehicleState(target, (1, 0, 0)), target, base, 1)
        self.assertFalse(evidence.snapshot()['target_reached'])
        evidence.observe(VehicleState(target), target, base, 2)
        self.assertEqual(evidence.snapshot()['remaining'], ['return_to_base'])
        evidence.observe(VehicleState(base), target, base, 3)
        self.assertTrue(evidence.snapshot()['task_completed'])
        self.assertFalse(evidence.snapshot(dynamic_events=1)['safe_success'])
        self.assertIsNone(evidence.snapshot(dynamic_measured=False)['safe_success'])


class TimeAndJournalTests(unittest.TestCase):
    def test_reads_are_isolated_and_do_not_advance_clock(self):
        service = MissionService(scenario_id='open_field', target_seed=7, traffic_seed=8)
        self.addCleanup(service.close)
        first = service.state()
        first['target']['x'] = -999
        second = service.state()
        self.assertNotEqual(first['target'], second['target'])
        before = second['simulation_time']
        with patch.object(service._traffic, 'sync', side_effect=AssertionError('Read mutated world')), \
                patch('uav_llm_demo.mission_service.scene_layout', side_effect=AssertionError('Rebuilt scene')):
            self.assertEqual(second, service.state())
        service.advance(.1)
        after = service.state()
        self.assertAlmostEqual(after['simulation_time'], before+.1)
        self.assertAlmostEqual(after['traffic_motion']['elapsed'], after['simulation_time'])

    def test_reset_has_new_run_identity_and_clears_failure_and_completion(self):
        service = MissionService(scenario_id='open_field', target_seed=7)
        self.addCleanup(service.close)
        old = service.trace.run_id
        service.recovery.provider_failed('fail', (1, 2, 2), 0)
        service.evidence.target_reached = 1
        state = service.reset()
        self.assertNotEqual(old, state['diagnostics']['run_id'])
        self.assertFalse(state['milestones']['target_reached'])
        self.assertFalse(state['flight']['recovery']['recent_failures'])
        self.assertEqual(state['simulation_time'], 0)

    def test_async_journal_redacts_and_survives_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'events.jsonl'
            writer = JournalWriter(path)
            try:
                a = RunTrace(writer, secrets=('test-secret-123456',))
                a.event('failure', 1, feedback='contains test-secret-123456', coordinates=[1, 2, 3])
                b = RunTrace(writer)
                b.event('run_started', 0)
                writer.flush()
                text = path.read_text()
                self.assertNotIn('test-secret-123456', text)
                events = [json.loads(line) for line in text.splitlines()]
                self.assertEqual([e['run_id'] for e in events], [a.run_id, b.run_id])
                self.assertEqual(events[0]['coordinates'], [1, 2, 3])
            finally:
                writer.close()


class BenchmarkProtocolTests(unittest.TestCase):
    def test_virtual_waits_for_local_planner_before_consuming_time_budget(self):
        class Slow(MockLLMAgent):
            def plan(self, context, feedback=''):
                time.sleep(.1)
                return super().plan(context, feedback)
        result = run_scenario('open_field', agent=Slow(), max_seconds=.2)
        self.assertEqual(result['metrics']['steps'], 1)
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['termination_reason'], 'simulation_budget')
        self.assertGreaterEqual(result['wall_seconds'], .1)
        self.assertAlmostEqual(result['simulation_seconds'], .2)

    def test_simulated_latency_is_explicit_and_can_consume_budget(self):
        result = run_scenario('open_field', max_seconds=.2, planner_latency_seconds=1)
        self.assertEqual(result['metrics']['steps'], 0)
        self.assertFalse(result['metrics']['success'])
        self.assertEqual(result['status'], 'timeout')

    def test_virtual_rejects_remote_api_without_calling_it(self):
        with patch('uav_llm_demo.agents._post_deepseek', side_effect=AssertionError('API used')):
            with self.assertRaises(ValueError):
                run_scenario('open_field', backend='deepseek', mode='virtual')

    def test_realtime_budget_uses_wall_clock_and_production_loop(self):
        result = run_scenario('open_field', mode='realtime', max_seconds=.15)
        self.assertGreaterEqual(result['wall_seconds'], .15)
        self.assertEqual(result['status'], 'timeout')
        self.assertGreater(result['diagnostics']['timing']['stages']['control_tick']['count'], 0)

    def test_invalid_budget_rejected(self):
        for value in [0, -1, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                run_scenario('open_field', max_seconds=value)

    def test_response_replay_and_seeded_runs_preserve_completion(self):
        base = load_scenario('open_field')
        scenario = replace(base, target=(base.base[0]+1, base.base[1], base.base[2]), dynamic_objects=())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'journal.jsonl'
            writer = JournalWriter(path)
            try:
                original = run_scenario('open_field', scenario=scenario, seed=7, max_seconds=30, journal=writer)
                writer.flush()
                header, agent = load_run(path, original['diagnostics']['run_id'])
                replayed = run_scenario('open_field', scenario=scenario_from_record(header['scenario']),
                                        agent=agent, seed=header['target_seed'], traffic_seed=header['traffic_seed'], max_seconds=30)
                self.assertEqual(original['status'], 'completed')
                self.assertEqual(replayed['status'], 'completed')
                self.assertTrue(replayed['milestones']['base_returned'])
                self.assertAlmostEqual(original['metrics']['path_length'], replayed['metrics']['path_length'], places=2)
                self.assertEqual(summarize([original, replayed])['successes'], 2)
            finally:
                writer.close()


class ComparisonTests(unittest.TestCase):
    def test_paired_comparison_exposes_recoveries_and_regressions(self):
        from copy import deepcopy
        from uav_llm_demo.benchmark import compare_results
        first = run_scenario('open_field', max_seconds=.1, seed=3)
        second = deepcopy(first)
        second['metrics']['success'] = True
        report = compare_results([first], [second])
        self.assertEqual(len(report['failure_to_success']), 1)
        self.assertEqual(len(compare_results([second], [first])['success_to_failure']), 1)
        second['protocol']['seed'] = 4
        with self.assertRaises(ValueError):
            compare_results([first], [second])

    def test_cost_is_explicit_and_missing_usage_is_not_assumed_zero(self):
        from uav_llm_demo.benchmark import estimate_cost
        usage = dict(prompt_cache_miss_tokens=1000000, prompt_cache_hit_tokens=1000000,
                     completion_tokens=1000000, usage_missing=1)
        cost = estimate_cost(usage, dict(input_miss=2, input_hit=1, output=4, currency='test-unit'))
        self.assertEqual(cost['observed_token_cost'], 7)
        self.assertFalse(cost['complete_usage'])


class LongRunningTests(unittest.TestCase):
    def test_trimmed_history_keeps_cumulative_metrics(self):
        from uav_llm_demo.evaluation import EvaluationLog
        log = EvaluationLog()
        for _ in range(250):
            log.record(dict(decision=dict(provider_error=True, safety_intervened=True)))
        self.assertEqual(len(log.events), 200)
        metrics = log.snapshot(final_status='active', trajectory=[], battery=100)['metrics']
        self.assertEqual(metrics['steps'], 250)
        self.assertEqual(metrics['agent_errors'], 250)
        self.assertEqual(metrics['safety_interventions'], 250)
        log.reset()
        self.assertEqual(log.total_events, 0)

    def test_unreported_request_keeps_cost_incomplete(self):
        from uav_llm_demo.benchmark import estimate_cost
        trace = RunTrace()
        trace.provider_started()
        cost = estimate_cost(trace.snapshot()['usage'], dict(input_miss=1, input_hit=1, output=1, currency='test-unit'))
        self.assertFalse(cost['complete_usage'])

    def test_reset_discards_ready_stale_plan_without_waiting_old_injected_delay(self):
        service = MissionService(scenario_id='open_field', planner_latency_seconds=100)
        self.addCleanup(service.close)
        service.start(background=False)
        service._future.result(timeout=2)
        service.reset()
        service.advance(.05)
        self.assertIsNone(service._future)
        self.assertEqual(service.state()['flight']['plan_id'], 0)


class RealtimeDeadlineTests(unittest.TestCase):
    def run_completion(self, late):
        class CompletionService(MissionService):
            def start(self, *, background=True):
                early = time.monotonic()
                time.sleep(.1)
                self._terminal_wall = time.monotonic() if late else early
                self._environment.status = 'completed'
                self._environment.phase = 'completed'
                self.evidence.target_reached = 0.01
                self.evidence.base_returned = 0.02
                self._publish_state()
                return self.state()
        with patch('uav_llm_demo.benchmark.MissionService', CompletionService):
            return run_scenario('open_field', mode='realtime', max_seconds=.05)

    def test_completion_after_deadline_is_timeout(self):
        result = self.run_completion(late=True)
        self.assertEqual(result['status'], 'timeout')
        self.assertFalse(result['metrics']['success'])
        self.assertFalse(result['metrics']['safe_success'])

    def test_completion_before_deadline_survives_late_poll(self):
        result = self.run_completion(late=False)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['metrics']['success'])
