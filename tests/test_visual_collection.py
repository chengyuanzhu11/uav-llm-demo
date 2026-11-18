"""Batch resource ownership and fail-fast checks without models or simulation."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests
from scripts import collect_visual_trajectories as collection
from scripts import verify_gazebo as verification
from scripts.verify_gazebo import infrastructure_failure
from uav_llm_demo.runtime import RuntimeUnavailableError


class VisualCollectionTests(unittest.TestCase):
    def test_complete_task_failure_is_retained_but_infrastructure_failure_stops_next_simulator(self):
        cases = [dict(id=name,scene='visual_garden',instruction='Find the object',oracle_target='bin')
                 for name in ('first','next')]
        scenarios = [
            (200, 'ok', None, [], {}, None),
            (401, 'error', None, [], {}, 'vision_provider_failed'),
            (200, 'error', 'network_timeout', [], {}, 'vision_provider_failed'),
            (200, 'error', 'invalid function arguments', [], {}, None),
            (200, 'ok', None, [dict(event='camera_failed')], {}, 'camera_failed'),
            (200, 'ok', None, [], dict(connected=False), 'runtime_disconnected'),
            (200, 'ok', None, [], dict(connected=True,last_error='fixture'), 'runtime_error'),
            (200, 'ok', None, [dict(event='feedback_hold')], {}, 'runtime_telemetry_failed'),
            (200, 'ok', None, [dict(event='feedback_hold'),dict(event='feedback_restored')], {}, None),
            (200, 'ok', None, [dict(event='control_safety_blocked')], {}, None),
            (200, 'ok', None, [dict(event='semantic_execution_event',reason='missing_telemetry')], {},
             'runtime_telemetry_failed'),
        ]
        for status, outcome, error, journal, runtime_status, failure_code in scenarios:
            with self.subTest(status=status,error=error,journal=journal,runtime=runtime_status), TemporaryDirectory() as temporary:
                def episode(args, output, runtime, *, planner_timeout):
                    self.assertEqual(planner_timeout, 25)
                    response = dict(kind='model_response',payload=dict(http_status=status,outcome=outcome,error=error))
                    result = dict(passed=args.case_id=='next',runtime=runtime_status,
                                  trajectory=dict(validation=dict(complete=True)))
                    result['failure_code'] = infrastructure_failure(result,journal,[response])
                    result['infrastructure_failed'] = result['failure_code'] is not None
                    return result
                session = Mock(runtime=object(),manifest={},log_dir='fixture-logs')
                with patch.object(collection,'GazeboSession',return_value=session) as launch, \
                     patch.object(collection,'visual_check',side_effect=episode), redirect_stdout(io.StringIO()):
                    result = collection.collect(cases,Path(temporary)/'new',30)
                expected = 1 if failure_code else 2
                self.assertEqual(launch.call_count,expected)
                self.assertEqual(session.close.call_count,expected)
                self.assertEqual(result['attempted'],expected)
                self.assertEqual(result['complete'],expected)
                self.assertEqual(result['passed'],failure_code is None)
                self.assertEqual(result['episodes'][0]['failure_code'],failure_code)
                self.assertFalse(result['episodes'][0]['passed'])
                self.assertTrue((Path(temporary)/'new/first/result.json').exists())

    def test_upstream_terminal_is_not_a_complete_visual_trajectory(self):
        validation = dict(complete=False, decisions=0, errors=["Episode contains no decisions"],
                          outcome=dict(status="failed"))
        result = dict(passed=False, semantic=dict(status="generation_failed"), vision=dict(calls=0),
                      journal=dict(dropped=0, errors=0), runtime=dict(connected=True))
        self.assertTrue(verification.terminal_without_visual_decisions(result, validation))
        self.assertIsNone(infrastructure_failure(result))
        self.assertFalse(verification.terminal_without_visual_decisions(result, {**validation, "outcome": None}))
        self.assertFalse(verification.terminal_without_visual_decisions(result, {**validation, "errors": ["Missing outcome"]}))
        self.assertFalse(verification.terminal_without_visual_decisions({**result, "vision": dict(calls=1)}, validation))
        cases = [dict(id=name, scene="visual_garden", instruction="Find object", oracle_target="bin")
                 for name in ("first", "next")]
        episode = dict(result, terminal_complete=True, failure_stage="semantic_generation",
                       trajectory=dict(validation=validation), infrastructure_failed=False)
        with TemporaryDirectory() as temporary, patch.object(collection, "GazeboSession", return_value=Mock(manifest={},log_dir="logs")), \
                patch.object(collection, "visual_check", return_value=episode) as verify, redirect_stdout(io.StringIO()):
            summary = collection.collect(cases, Path(temporary)/"new", 120, planner_timeout=90)
        self.assertEqual(summary['planner_timeout_seconds'], 90)
        self.assertEqual(summary['wall_budget_seconds'], 120)
        self.assertTrue(all(call.kwargs['planner_timeout'] == 90 for call in verify.call_args_list))
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["terminal_complete"], 2)
        self.assertEqual(summary["complete"], 0)
        self.assertFalse(summary["passed"])

    def test_runtime_failure_reports_code_without_skipping_shutdown(self):
        for exception, code in ((RuntimeUnavailableError, 'runtime_unavailable'),
                                (RuntimeError, 'runtime_health_check_failed')):
            with self.subTest(exception=exception), TemporaryDirectory() as temporary:
                runtime = Mock(backend_name='gazebo')
                runtime.status.return_value = dict(connected=True,last_error=None)
                runtime.read_state.side_effect = exception('fixture telemetry failure')
                service = Mock(runtime=runtime)
                service.submit_task.side_effect = exception('fixture task start failure')
                args = SimpleNamespace(scene='visual_garden',instruction='Find the object',timeout=30)
                with patch.object(verification,'MissionService',return_value=service):
                    result = verification.visual_check(args,Path(temporary)/'result.json',runtime)
                self.assertFalse(result['passed'])
                self.assertTrue(result['infrastructure_failed'])
                self.assertEqual(result['failure_code'],code)
                runtime.read_state.assert_called_once()
                service.close.assert_called_once()
                self.assertEqual(result['journal'],dict(dropped=0,errors=0))

    def test_planner_timeout_is_injected_without_changing_production_default(self):
        from uav_llm_demo.agents import DeepSeekAgent
        for requested, wall_budget, expected in ((None, 120, 25), (90, 120, 90), (90, 30, 30)):
            with self.subTest(requested=requested), TemporaryDirectory() as temporary:
                runtime = Mock(backend_name='gazebo')
                runtime.status.return_value = dict(connected=True, last_error=None)
                service = Mock(runtime=runtime)
                service.submit_task.side_effect = RuntimeError('fixture stops before inference')
                args = SimpleNamespace(scene='visual_garden', instruction='Find the object', timeout=wall_budget)
                options = {} if requested is None else dict(planner_timeout=requested)
                with patch.object(verification, 'MissionService', return_value=service) as factory:
                    result = verification.visual_check(args, Path(temporary)/'result.json', runtime, **options)
                self.assertEqual(factory.call_args.kwargs['agent'].timeout_seconds, expected)
                self.assertEqual(result['planner_timeout_seconds'], expected)
                self.assertEqual(result['wall_budget_seconds'], wall_budget)
                service.close.assert_called_once()
        self.assertEqual(DeepSeekAgent().timeout_seconds, 25)
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(invalid=value), TemporaryDirectory() as temporary:
                output = Path(temporary)/'new'
                with self.assertRaises(ValueError):
                    collection.collect([], output, 120, planner_timeout=value)
                self.assertFalse(output.exists())
                args = SimpleNamespace(timeout=120)
                with patch.object(verification, 'MissionService') as factory, self.assertRaises(ValueError):
                    verification.visual_check(args, output, planner_timeout=value)
                factory.assert_not_called()

    def test_unavailable_model_stops_before_output_or_simulator_creation(self):
        with TemporaryDirectory() as temporary:
            output=Path(temporary)/'no-data'
            env=dict(VISION_STRATEGY='direct',VISION_BASE_URL='http://127.0.0.1:9/v1',VISION_MODEL='fixture',
                     LLM_BASE_URL='http://127.0.0.1:9/v1',LLM_MODEL='fixture')
            with patch.dict('os.environ',env,clear=True), patch.object(collection,'load_dotenv'), \
                 patch.object(collection.requests,'get',side_effect=requests.ConnectionError()), \
                 patch.object(collection,'GazeboSession') as launch, \
                 patch('sys.argv',['collect','--output',str(output)]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    collection.main()
            self.assertEqual(failure.exception.code,2)
            launch.assert_not_called()
            self.assertFalse(output.exists())
