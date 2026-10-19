"""Traffic stays independent of the planner and simulation cadence in fixed runs."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from tests.scenario_fixtures import empty_airspace
from uav_llm_demo.benchmark import compare_results, run_scenario
from uav_llm_demo.environment import UAVEnvironment
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.replay import load_run
from uav_llm_demo.run_log import JournalWriter
from uav_llm_demo.scenarios import DEFAULT_SCENARIO
from uav_llm_demo.traffic_motion import ContinuousTraffic, TRAFFIC_MODEL_VERSION
from uav_llm_demo.traffic_observation import TrafficObservationConfig


class FixedTrafficTests(unittest.TestCase):
    def test_goal_base_order_and_timestep_do_not_change_map_traffic(self):
        now = [0.0, 0.0]
        scenarios = [DEFAULT_SCENARIO, replace(DEFAULT_SCENARIO, base=(4, 4, 2), target=(30, 5, 6))]
        traffic = [ContinuousTraffic(scenarios[i], seed=19, policy='fixed', clock=lambda i=i: now[i])
                   for i in range(2)]
        worlds = [UAVEnvironment(scenario) for scenario in scenarios]
        # Reverse object iteration to detect shared random-stream coupling.
        worlds[1].dynamic_objects.reverse()
        self.assertFalse(hasattr(traffic[0], 'focus_route'))
        for elapsed in (0.0, 7.0, 19.0, 63.0, 120.0):
            while now[0] < elapsed:
                now[0] = min(elapsed, now[0] + .2)
                traffic[0].sync(worlds[0])
            now[:] = [elapsed, elapsed]
            for flow, world in zip(traffic, worlds):
                flow.sync(world)
            self.assertEqual(traffic[0].models, traffic[1].models)
            left = {bird.object_id: (bird.position, bird.velocity) for bird in worlds[0].dynamic_objects}
            right = {bird.object_id: (bird.position, bird.velocity) for bird in worlds[1].dynamic_objects}
            self.assertEqual(left, right)

    def test_invalid_policy_rejected_before_building_traffic(self):
        for value in ('', 'auto', 'route_demo', None, True):
            with self.assertRaisesRegex(ValueError, 'policy'):
                ContinuousTraffic(DEFAULT_SCENARIO, policy=value)
            with self.assertRaisesRegex(ValueError, 'policy'):
                run_scenario('urban_crossing', traffic_policy=value)


class TrafficProtocolTests(unittest.TestCase):
    def test_browser_and_benchmark_share_fixed_map_traffic_and_record_it(self):
        service = MissionService(scenario=empty_airspace(), target_seed=7, traffic_seed=8)
        try:
            self.assertEqual(service._traffic.policy, 'fixed')
            self.assertEqual(service.state()['traffic_motion']['policy'], 'fixed')
        finally:
            service.close()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.jsonl'
            writer = JournalWriter(path)
            try:
                result = run_scenario('test_airspace', scenario=empty_airspace(), max_seconds=.15,
                                      seed=7, traffic_seed=8, journal=writer)
                writer.flush()
                records = [json.loads(line) for line in path.read_text().splitlines()]
            finally:
                writer.close()
        self.assertEqual(result['protocol']['traffic_policy'], 'fixed')
        self.assertEqual(result['protocol']['traffic_model_version'], TRAFFIC_MODEL_VERSION)
        header = next(record for record in records if record['event'] == 'run_started')
        self.assertEqual(header['traffic_policy'], 'fixed')
        self.assertEqual(header['traffic_model_version'], TRAFFIC_MODEL_VERSION)
        self.assertEqual(result['protocol']['traffic_observation'], asdict(TrafficObservationConfig()))
        self.assertEqual(header['traffic_observation'], result['protocol']['traffic_observation'])
        self.assertEqual(header['traffic_prediction'], result['protocol']['traffic_prediction'])

    def test_explicit_fixed_policy_survives_reset(self):
        service = MissionService(scenario=empty_airspace(), target_seed=7, traffic_seed=8,
                                 traffic_policy='fixed')
        try:
            state = service.reset()
            self.assertEqual(service._traffic.policy, 'fixed')
            self.assertEqual(state['traffic_motion']['policy'], 'fixed')
            self.assertEqual(state['traffic_motion']['model_version'], TRAFFIC_MODEL_VERSION)
        finally:
            service.close()

    def test_replay_cli_preserves_sensor_config_and_marks_legacy_boundaries(self):
        config = TrafficObservationConfig(range_m=35, latency_seconds=.2, position_noise_m=.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            writer = JournalWriter(path)
            try:
                original = run_scenario('test_airspace', scenario=empty_airspace(), max_seconds=.15,
                                        seed=7, traffic_seed=8, observation_config=config, journal=writer)
                writer.flush()
            finally:
                writer.close()
            replay_script = Path(__file__).resolve().parents[1] / 'scripts/replay_run.py'
            original_records = [json.loads(line) for line in path.read_text().splitlines()]
            for variant in ('current', 'route_v1', 'fixed_v1', 'pre_policy'):
                records = deepcopy(original_records)
                header = next(record for record in records if record['event'] == 'run_started')
                if variant == 'pre_policy':
                    for key in ('traffic_policy', 'traffic_model_version', 'traffic_observation', 'traffic_prediction'):
                        del header[key]
                elif variant != 'current':
                    header['traffic_policy'] = 'route_demo' if variant == 'route_v1' else 'fixed'
                    header['traffic_model_version'] = 'independent-streams-v1'
                path.write_text('\n'.join(json.dumps(record) for record in records))
                output = Path(directory) / (variant + '.json')
                subprocess.run([sys.executable, str(replay_script), '--journal', str(path),
                                '--run-id', original['diagnostics']['run_id'], '--max-seconds', '.15',
                                '--output', str(output)], check=True, capture_output=True, text=True)
                result = json.loads(output.read_text())
                self.assertEqual(result['protocol']['traffic_policy'], 'fixed')
                source = result['replay_source']
                self.assertEqual(source['traffic_policy'],
                                 'route_demo' if variant in ('route_v1', 'pre_policy') else 'fixed')
                self.assertEqual(source['current_traffic_policy'], 'fixed')
                self.assertEqual(source['current_traffic_model_version'], TRAFFIC_MODEL_VERSION)
                self.assertEqual(source['traffic_policy_matches'], variant in ('current', 'fixed_v1'))
                self.assertEqual(source['traffic_model_version_matches'], variant == 'current')
                self.assertEqual(result['protocol']['traffic_observation'],
                                 asdict(TrafficObservationConfig()) if variant == 'pre_policy' else asdict(config))
                for field in ('observation_protocol_matches', 'prediction_version_matches'):
                    self.assertEqual(source[field], variant != 'pre_policy')

    def test_custom_sensor_config_is_recorded_in_benchmark(self):
        config = TrafficObservationConfig(range_m=35, latency_seconds=.2, occlusion=False,
                                          position_noise_m=.1, dropout_probability=.1)
        result = run_scenario('test_airspace', scenario=empty_airspace(), max_seconds=.15,
                              observation_config=config)
        self.assertEqual(result['protocol']['traffic_observation'], asdict(config))

    def test_pairs_reject_different_traffic_policy_or_model_version(self):
        original = dict(scenario_id='urban_crossing',
                        protocol=dict(mode='virtual', time_budget_seconds=120, control_dt=.05,
                                      planner_latency_seconds=0, planner_timeout_seconds=30,
                                      seed=7, traffic_seed=8, traffic_policy='fixed',
                                      traffic_model_version=TRAFFIC_MODEL_VERSION,
                                      traffic_observation=asdict(TrafficObservationConfig()),
                                      traffic_prediction='history_damped_acceleration'),
                        initial_scenario={}, metrics=dict(success=True, planner_calls=1), wall_seconds=1)
        self.assertEqual(compare_results([original], [deepcopy(original)])['paired_cases'], 1)
        for field, changed in (('traffic_policy', 'route_demo'), ('traffic_model_version', 'legacy-shared-stream'),
                               ('traffic_observation', asdict(TrafficObservationConfig(range_m=30)))):
            other = deepcopy(original)
            other['protocol'][field] = changed
            with self.assertRaisesRegex(ValueError, field):
                compare_results([original], [other])
        legacy = deepcopy(original)
        del legacy['protocol']['traffic_policy']
        del legacy['protocol']['traffic_model_version']
        del legacy['protocol']['traffic_observation']
        del legacy['protocol']['traffic_prediction']
        self.assertEqual(compare_results([legacy], [deepcopy(legacy)])['paired_cases'], 1)
        with self.assertRaisesRegex(ValueError, 'traffic_policy'):
            compare_results([legacy], [original])
        different_predictor = deepcopy(original)
        different_predictor['protocol']['traffic_prediction'] = 'constant_velocity_baseline'
        self.assertEqual(compare_results([original], [different_predictor])['paired_cases'], 1)

    def test_legacy_and_current_journal_policy_are_explicit_when_loading(self):
        records = [dict(run_id='sample', event='run_started', scenario=dict(mission_mode='navigation')),
                   dict(run_id='sample', event='plan_requested', request_id=1, simulation_seconds=0),
                   dict(run_id='sample', event='plan_accepted', request_id=1, simulation_seconds=.2,
                        waypoints=[[1, 1, 2]])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.jsonl'
            for policy, version in ((None, None), ('route_demo', 'independent-streams-v1'),
                                    ('fixed', 'independent-streams-v1'), ('fixed', TRAFFIC_MODEL_VERSION)):
                if policy is not None:
                    records[0].update(traffic_policy=policy, traffic_model_version=version,
                                      traffic_observation=asdict(TrafficObservationConfig(range_m=30)),
                                      traffic_prediction='history_damped_acceleration')
                path.write_text('\n'.join(json.dumps(record) for record in records))
                header, agent = load_run(path, 'sample')
                self.assertEqual(header['traffic_policy'], policy or 'route_demo')
                self.assertEqual(header['traffic_model_version'], version or 'legacy-shared-stream')
                self.assertEqual(header['traffic_observation'],
                                 asdict(TrafficObservationConfig(range_m=30)) if policy else 'legacy_ground_truth')
                self.assertEqual(header['traffic_prediction'],
                                 'history_damped_acceleration' if policy else 'legacy_constant_velocity')
                self.assertEqual(agent.plan({}).waypoints, ((1, 1, 2),))
            records[0]['traffic_policy'] = 'unknown'
            path.write_text('\n'.join(json.dumps(record) for record in records))
            with self.assertRaisesRegex(ValueError, 'traffic policy'):
                load_run(path, 'sample')

    def test_retired_demo_journal_cannot_silently_replay_with_delayed_departure(self):
        records = [dict(run_id='demo', event='run_started', scenario=dict(mission_mode='navigation')),
                   dict(run_id='demo', event='plan_requested', request_id=1, simulation_seconds=0),
                   dict(run_id='demo', event='plan_accepted', request_id=1, simulation_seconds=0,
                        waypoints=[[1, 1, 2]]),
                   dict(run_id='demo', event='demo_started', simulation_seconds=0,
                        demo=dict(id='fixed-crossing-v1', departure_time=0.0))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demo.jsonl'
            path.write_text('\n'.join(json.dumps(record) for record in records))
            with self.assertRaisesRegex(ValueError, 'zero-time planning and departure protocol'):
                load_run(path, 'demo')


if __name__ == '__main__':
    unittest.main()
