"""A shape must be verified from its own flown samples, never a return leg."""
import math
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from scripts.compare_qwen_sft import additional_motion_check, decision_diagnostics, summary
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.spatial_motion import spatial_catalog


class SFTComparisonTests(unittest.TestCase):
    def test_retry_counts_provider_successors_not_normal_tools_or_budget_blocks(self):
        events = []
        def add(kind, **payload):
            events.append(dict(sequence=len(events)+1, kind=kind, payload=dict(stage='plan', **payload)))
        add('decision_result', decision_id='geometry', status='applied', action=dict(tool='query_geometry'))
        add('model_request', decision_id='bad')
        add('model_response', decision_id='bad', source='provider', http_status=200, elapsed_ms=100)
        add('decision_result', decision_id='bad', status='rejected', action=dict(tool='preview_plan'))
        add('model_request', decision_id='fixed')
        add('model_response', decision_id='fixed', source='provider', http_status=200, elapsed_ms=200)
        add('decision_result', decision_id='fixed', status='applied', action=dict(tool='preview_plan'))
        add('model_request', decision_id='submit')
        add('model_response', decision_id='submit', source='provider', http_status=200, elapsed_ms=100)
        add('decision_result', decision_id='submit', status='error', action=dict(tool='submit_plan'))
        add('model_request', decision_id='budget')
        add('model_response', decision_id='budget', source='provider_override', http_status=None)
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            stats = decision_diagnostics(p)
        self.assertEqual(stats['retry_requests'], 1)
        self.assertEqual(stats['provider_calls'], 3)
        self.assertAlmostEqual(stats['provider_seconds'], .4)

    def test_three_arm_summary_keeps_invalid_runs_and_pairwise_changes(self):
        rows = [dict(arm=arm, case_id='task', seed=17, passed=passed,
                     valid_evaluation=valid, false_completion=False, model_calls=5,
                     retry_requests=retry, wall_seconds=duration, failure_category='passed' if passed else 'failed')
                for arm, passed, valid, retry, duration in [
                    ('base', False, True, 2, 20), ('sft', True, True, 0, 10), ('eight', False, False, 0, 5)]]
        stats = summary(rows, ('base', 'sft', 'eight'))
        self.assertEqual(stats['sft']['success_rate'], 1)
        self.assertEqual(stats['base']['mean_retry_requests'], 2)
        self.assertEqual(stats['eight']['invalid_evaluations'], 1)
        self.assertEqual(stats['paired'][0]['changes_from_reference'], {'sft':'improved', 'eight':'not_comparable'})
        self.assertEqual(summary(rows[:-1], ('base','sft','eight'))['paired'], [])

    def test_crossing_oracle_binds_only_the_completed_motion(self):
        obj = next(o for o in spatial_catalog(load_scenario('semantic_campus'))['objects'] if o['id'] == 'stadium')
        center = obj['center_grid']
        samples = []
        for i in range(401):
            t = i / 400 * 2 * math.pi
            samples.append(dict(action='motion', phase='tracking', mission_id='mission', vehicle_id='uav-1',
                execution_id='figure-eight', yaw=0, position=[center[0]+math.sin(t), center[1]+math.sin(2*t), center[2]+.8*i/400]))
        # This measured return leg must not spoil or complete the shape.
        samples.append(dict(action='return_home', phase='tracking', mission_id='mission', vehicle_id='uav-1',
            execution_id='return', yaw=0, position=[0,0,0]))
        case = dict(measured_motion_case='crossing_curve', measurement_target='stadium')
        evidence = dict(action='motion', target_id='stadium', mission_id='mission', vehicle_id='uav-1', execution_id='figure-eight')
        execution = dict(mission_id='mission', evidence=[evidence], telemetry=samples)
        measured = additional_motion_check(case, execution)
        self.assertTrue(measured['ok'], measured)
        self.assertEqual(measured['runs'][0]['samples'], 401)
        evidence['execution_id'] = 'a-different-motion'
        self.assertFalse(additional_motion_check(case, execution)['ok'])


if __name__ == '__main__':
    unittest.main()
