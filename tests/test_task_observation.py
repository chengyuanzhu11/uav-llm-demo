"""Sensor uncertainty and freshness use the same interpreter/contract boundaries."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from tests.semantic_fixtures import fixture_requirements
from tests.test_semantic_tasks import parse, proposal, scene, step
from uav_llm_demo.semantic_tasks import SemanticExecutionError, SemanticMission, task_catalog
from uav_llm_demo.task_observation import TaskObservationConfig, TaskObservationWorld, TaskObserver, observation_validity
from uav_llm_demo.mission_service import MissionService
from tests.test_semantic_updates import OfflineSemanticAgent, compact_campus, pump


def mission(nodes, *, faults=(), max_age_seconds=120):
    scenario = scene()
    world = TaskObservationWorld.from_dict({point['id']: {'temperature_c': 72.25}
                                           for point in scenario.points_of_interest})
    config = TaskObservationConfig(faults=faults, max_age_seconds=max_age_seconds)
    value = SemanticMission(scenario, task_observer=TaskObserver(world, config))
    plan = parse(proposal(nodes))
    value.accept(plan, requirements=fixture_requirements(plan, task_catalog(scenario)))
    return value


def finish(value, now):
    return value.observe(value.current_goal, 0, max(.05, value.current_step['duration_seconds']), now)


class TaskObservationTests(unittest.TestCase):
    def test_seeded_world_sensor_boundary_and_configuration_validation(self):
        scenario = scene()
        worlds = [TaskObservationWorld.generate(scenario, seed) for seed in (3, 3, 4)]
        config = TaskObservationConfig(seed=9, noise_std_c=.1, latency_seconds=2)
        records = [TaskObserver(world, config).observe('thermal_scan', 'library', 10) for world in worlds]
        self.assertEqual(records[0], records[1])
        self.assertNotEqual(records[0]['result'], records[2]['result'])
        self.assertEqual(records[0]['observed_at'], 8)
        self.assertEqual(records[0]['delivered_at'], 10)
        self.assertEqual(len(records[0]['readings']), 2)
        self.assertNotIn(str(worlds[0]._targets['library']['temperature_c']), json.dumps(task_catalog(scenario)))
        for kwargs in ({'seed': True}, {'max_age_seconds': 0}, {'noise_std_c': float('nan')},
                       {'dropout_probability': 1.1}, {'faults': ({'status': 'unknown'},)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TaskObservationConfig(**kwargs)
        for kwargs in ({'dropout_probability': 1}, {'noise_std_c': 100, 'min_confidence': .9}):
            observer = TaskObserver(worlds[0], TaskObservationConfig(**kwargs))
            self.assertEqual(observer.observe('thermal_scan', 'library', 10)['status'], 'unknown')

    def test_invalid_readings_do_not_complete_work_and_fresh_retry_does(self):
        for action in ('thermal_scan', 'inspect', 'capture'):
            for status in ('unknown', 'conflict', 'stale'):
                with self.subTest(action=action, status=status):
                    value = mission([step('sense', action, 'library', .1), step('home', 'return_home', 'home')],
                                    faults=({'action': action, 'target_id': 'library', 'attempt': 1, 'status': status},))
                    execution = value.current_step['execution_id']
                    with self.assertRaises(SemanticExecutionError) as caught:
                        finish(value, 1)
                    self.assertEqual(caught.exception.code, 'observation_' + status)
                    self.assertEqual(value.evidence, [])
                    self.assertEqual(value.task_observations[-1]['status'], status)
                    self.assertEqual(value.task_observations[-1]['result'], {'simulated': True})
                    self.assertFalse(value.task_contract.verdict()['satisfied'])
                    self.assertTrue(value.retry_observation())
                    self.assertEqual(value.current_step['execution_id'], execution)
                    finish(value, 2)
                    self.assertEqual(value.evidence[0]['observation']['attempt'], 2)
                    self.assertEqual(value.current_step['action'], 'return_home')
                    finish(value, 3)
                    self.assertEqual(value.status, 'completed')
                    self.assertEqual(len(value.evidence), 2)

    def test_expired_condition_refreshes_same_sensor_without_repeating_obligation(self):
        condition = {'step_id': 'sense', 'field': 'temperature_c', 'op': '>', 'value': 60}
        branch = {'id': 'choice', 'action': 'branch', 'condition': condition,
                  'then': [step('record', 'capture', 'library', .1)], 'else': []}
        value = mission([step('sense', 'thermal_scan', 'library', .1),
                         step('travel', 'visit', 'laboratory'), branch, step('home', 'return_home', 'home')],
                        max_age_seconds=5,
                        faults=({'action': 'thermal_scan', 'target_id': 'library', 'attempt': 2, 'status': 'unknown'},))
        finish(value, 1)
        original = deepcopy(value.evidence[0])
        with self.assertRaises(SemanticExecutionError) as caught:
            finish(value, 10)
        self.assertEqual(caught.exception.observation_evidence_id, original['evidence_id'])
        self.assertEqual(value.evidence[0], original)
        matched = value.task_contract.snapshot()['matched_occurrences']
        waiting = value.task_contract.context()
        self.assertTrue(waiting['remaining_steps'])
        self.assertEqual(waiting['condition_wait']['code'], 'contract_observation_unavailable')
        self.assertTrue(value.request_observation_refresh(original['evidence_id']))
        self.assertEqual(value.current_step['target_id'], original['target_id'])
        self.assertNotEqual(value.current_step['execution_id'], original['execution_id'])
        with self.assertRaises(SemanticExecutionError):
            finish(value, 11)
        self.assertTrue(value.retry_observation())
        finish(value, 12)
        self.assertEqual(value.task_contract.snapshot()['matched_occurrences'], matched)
        self.assertEqual(value.current_step['id'], 'record')
        self.assertEqual(value.evidence[0], original)
        self.assertIsNone(observation_validity(value.evidence[-1], 12))
        self.assertEqual(observation_validity(original, 12), 'stale')
        decisions = value.to_dict()['condition_evaluations']
        self.assertEqual(len(decisions), 1)  # Unavailable attempts did not decide a branch.
        self.assertEqual(decisions[0]['requested_evidence_id'], original['evidence_id'])
        self.assertEqual(decisions[0]['resolved_evidence_id'], value.evidence[-1]['evidence_id'])
        self.assertEqual(decisions[0]['simulation_time'], 12)
        self.assertTrue(decisions[0]['outcome'])
        # Refreshing an explicitly referenced refresh must also update the root
        # alias, while the already selected conditional branch remains frozen.
        refresh_id = value.evidence[-1]['evidence_id']
        value.task_observer = TaskObserver(TaskObservationWorld.from_dict(
            {point['id']: {'temperature_c': 20} for point in value.scenario.points_of_interest}),
            TaskObservationConfig(max_age_seconds=5))
        self.assertTrue(value.request_observation_refresh(refresh_id))
        finish(value, 20)
        self.assertFalse(value._evaluate(condition))
        self.assertEqual(value.to_dict()['condition_evaluations'][:-1], decisions)
        self.assertEqual(value.to_dict()['condition_evaluations'][-1]['observed_value'], 20)
        self.assertEqual(value.current_step['id'], 'record')
        self.assertEqual(value._observation_aliases[original['evidence_id']]['evidence_id'], value.evidence[-1]['evidence_id'])
        self.assertEqual(value.task_contract.snapshot()['matched_occurrences'], matched)
        finish(value, 21)
        finish(value, 22)
        self.assertEqual(value.status, 'completed')
        self.assertTrue(value.task_contract.verdict()['satisfied'])
        self.assertEqual(sum(item['action'] == 'thermal_scan' for item in value.evidence), 3)
        history = value.to_dict()['condition_evaluations']
        next_plan = parse(proposal([step('home_again', 'return_home', 'home')]))
        requirements = fixture_requirements(next_plan, task_catalog(value.scenario))
        value.accept(next_plan, preserve_evidence=True, requirements=requirements)
        self.assertEqual(value.to_dict()['condition_evaluations'], history)
        value.accept(next_plan, preserve_evidence=False, requirements=requirements)
        self.assertEqual(value.to_dict()['condition_evaluations'], [])
        with self.subTest('multiple stale inputs at one decision boundary'):
            both = {'all': [condition, {'step_id': 'other', 'field': 'temperature_c', 'op': '>', 'value': 60}]}
            value = mission([step('sense', 'thermal_scan', 'library', .1),
                             step('other', 'thermal_scan', 'laboratory', .1),
                             step('wait', 'visit', 'conference'), {**branch, 'condition': both},
                             step('home', 'return_home', 'home')], max_age_seconds=5)
            finish(value, 1)
            finish(value, 2)
            for now in (10, 11):
                with self.assertRaises(SemanticExecutionError) as caught:
                    finish(value, now)
                self.assertEqual(caught.exception.code, 'observation_stale')
                self.assertTrue(value.request_observation_refresh(caught.exception.observation_evidence_id))
            finish(value, 12)
            self.assertEqual(value.current_step['id'], 'record')
            self.assertEqual(value.task_contract.snapshot()['matched_occurrences'], 3)
            finish(value, 13)
            finish(value, 14)
            self.assertEqual(value.status, 'completed')
        for count in (1, 2):
            with self.subTest('loop bound short-circuits stale until', count=count):
                loop = {'id': 'bounded', 'action': 'repeat', 'count': count, 'until': condition, 'when': None,
                        'steps': [step('sense', 'thermal_scan', 'library', .1), step('wait', 'visit', 'laboratory')]}
                value = mission([loop, step('home', 'return_home', 'home')], max_age_seconds=5)
                finish(value, 1)
                if count == 1:
                    finish(value, 10)
                    self.assertEqual(value.to_dict()['condition_evaluations'], [])
                else:
                    with self.assertRaises(SemanticExecutionError) as caught:
                        finish(value, 10)
                    remaining = value.task_contract.remaining_steps()
                    self.assertEqual(remaining[0]['action'], 'branch')
                    self.assertIn('not', remaining[0]['condition'])
                    self.assertEqual(remaining[0]['then'][0]['count'], 1)
                    self.assertTrue(value.request_observation_refresh(caught.exception.observation_evidence_id))
                    finish(value, 11)
                self.assertEqual(value.current_step['id'], 'home')
                finish(value, 12)
                self.assertEqual(value.status, 'completed')

    def test_delivered_context_is_isolated_and_never_contains_world_or_future_faults(self):
        value = mission([step('sense', 'thermal_scan', 'library', .1), step('home', 'return_home', 'home')],
                        faults=({'action': 'thermal_scan', 'target_id': 'library', 'attempt': 2, 'status': 'conflict'},))
        before = json.dumps(value.catalog_context())
        self.assertNotIn('72.25', before)
        self.assertNotIn('conflict', before)
        finish(value, 1)
        context = value.catalog_context()
        self.assertEqual(context['task_observations'][0]['result']['temperature_c'], 72.25)
        context['task_observations'][0]['result']['temperature_c'] = -100
        context['evidence'][0]['observation']['status'] = 'unknown'
        self.assertEqual(value.task_observations[0]['result']['temperature_c'], 72.25)
        self.assertEqual(value.evidence[0]['observation']['status'], 'valid')

    def test_service_returns_to_refresh_expired_branch_input_without_model_call(self):
        scenario = compact_campus()
        world = TaskObservationWorld.from_dict({point['id']: {'temperature_c': 72.25}
                                               for point in scenario.points_of_interest})
        branch = {'id': 'choice', 'action': 'branch',
                  'condition': {'step_id': 'sense', 'field': 'temperature_c', 'op': '>', 'value': 60},
                  'then': [step('record', 'capture', 'library', .1)], 'else': []}
        plan = proposal([step('sense', 'thermal_scan', 'library', .1),
                         step('wait', 'visit', 'laboratory', 10), branch, step('home', 'return_home', 'home')])
        service = MissionService(scenario=scenario, agent=OfflineSemanticAgent([plan]), task_world=world,
                                 task_observation_config=TaskObservationConfig(max_age_seconds=5))
        self.addCleanup(service.close)
        with patch('uav_llm_demo.agents._post_deepseek', side_effect=AssertionError('Network forbidden')):
            service.submit_task('测量图书馆温度，去实验楼等待再按温度条件采样并返航。', background=False)
            state = pump(service, lambda value: value['status'] == 'completed', seconds=180)
        evidence = state['semantic_mission']['evidence']
        scans = [item for item in evidence if item['action'] == 'thermal_scan']
        self.assertEqual(len(scans), 2)
        self.assertEqual(scans[1]['refresh_of'], scans[0]['evidence_id'])
        self.assertEqual(scans[1]['target_id'], 'library')
        self.assertEqual(len(service.agent.inputs), 1)
        self.assertTrue(state['semantic_mission']['task_contract']['satisfied'])
        self.assertEqual([item['action'] for item in evidence],
                         ['thermal_scan', 'visit', 'thermal_scan', 'capture', 'return_home'])
        with self.subTest('new instruction starts with an expired historical condition'):
            historical = deepcopy(branch)
            historical['condition'] = {'evidence_id': scans[0]['evidence_id'],
                                       'field': 'temperature_c', 'op': '>', 'value': 60}
            service.agent.replies.append(proposal([historical, step('home', 'return_home', 'home')]))
            service.submit_task('再次根据先前温度决定采样，随后返航。', background=False)
            updated = pump(service, lambda value: value['status'] == 'completed', seconds=120)
            self.assertTrue(updated['semantic_mission']['task_contract']['satisfied'])
            self.assertEqual(len(service.agent.inputs), 2)
            self.assertEqual(service._autonomous_reviews, 0)
        class ReviewedRefreshAgent(OfflineSemanticAgent):
            def __init__(self, *, reject_missing_condition):
                super().__init__([plan])
                self.reject_missing_condition = reject_missing_condition
                self.review_repairs = []

            def interpret_task(self, catalog, instruction, feedback='', *, repair=None, planning=None):
                if self.inputs:
                    self.review_repairs.append(deepcopy(repair))
                    remaining = deepcopy(branch)
                    remaining['condition'] = {
                        'evidence_id': catalog['mission_context']['evidence'][0]['evidence_id'],
                        'field': 'temperature_c', 'op': '>', 'value': 60}
                    steps = (remaining['then'] if self.reject_missing_condition and len(self.inputs) == 1
                             else [remaining])
                    self.replies.append(proposal([*steps, step('home', 'return_home', 'home')]))
                return super().interpret_task(catalog, instruction, feedback, repair=repair, planning=planning)

        for persistent in (False, True):
            with self.subTest('expired refresh escalates without losing its predicate', persistent=persistent):
                faults = tuple({'action': 'thermal_scan', 'target_id': 'library', 'attempt': attempt, 'status': 'unknown'}
                               for attempt in range(2, 12 if persistent else 5))
                agent = ReviewedRefreshAgent(reject_missing_condition=not persistent)
                recovering = MissionService(scenario=scenario, agent=agent, task_world=world,
                    task_observation_config=TaskObservationConfig(max_age_seconds=5, faults=faults))
                self.addCleanup(recovering.close)
                with patch('uav_llm_demo.agents._post_deepseek', side_effect=AssertionError('Network forbidden')):
                    recovering.submit_task('测量温度，等待后按原条件采样并返航。', background=False)
                    final = pump(recovering, lambda value: value['status'] == 'completed'
                                 or value['semantic_mission']['status'] == 'safety_wait', seconds=180)
                attempts = [item for item in recovering.semantic.task_observations if item['action'] == 'thermal_scan']
                self.assertEqual(len(attempts), 6 if persistent else 5)
                self.assertEqual(len(agent.inputs), 3)
                self.assertEqual(final['milestones']['task_completed'], not persistent)
                if persistent:
                    self.assertEqual(recovering._autonomous_reviews, 2)
                    for _ in range(100):
                        recovering.advance(.1)
                    self.assertEqual(len(agent.inputs), 3)
                    self.assertEqual(recovering.semantic.status, 'safety_wait')
                else:
                    self.assertEqual(recovering._autonomous_reviews, 1)
                    self.assertEqual(agent.review_repairs[-1]['diagnostic']['code'], 'contract_obligation_changed')
                    original = recovering.semantic.evidence[0]
                    self.assertFalse(recovering.semantic.task_contract.has_pending_observation_refresh(original['evidence_id']))
                    self.assertEqual([item['action'] for item in recovering.semantic.evidence],
                                     ['thermal_scan', 'visit', 'thermal_scan', 'capture', 'return_home'])
                    self.assertTrue(final['semantic_mission']['task_contract']['satisfied'])


if __name__ == '__main__':
    unittest.main()
