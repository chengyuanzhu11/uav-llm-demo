"""Replay structured planner proposals against the current controller, offline."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_llm_demo.benchmark import run_scenario
from uav_llm_demo.replay import load_run, scenario_from_record
from uav_llm_demo.traffic_observation import TrafficObservationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    header, agent = load_run(args.journal, args.run_id)
    scenario = scenario_from_record(header['scenario'])
    if header['runtime'] != 'html':
        parser.error('Offline replay supports HTML missions only.')
    observation = header['traffic_observation']
    config = TrafficObservationConfig(**observation) if isinstance(observation, dict) else None
    result = run_scenario(scenario.scenario_id, agent=agent, scenario=scenario,
                          seed=header['target_seed'], traffic_seed=header['traffic_seed'],
                          traffic_policy='fixed', observation_config=config, max_seconds=args.max_seconds)
    policy_matches = header['traffic_policy'] == result['protocol']['traffic_policy']
    traffic_matches = header['traffic_model_version'] == result['protocol']['traffic_model_version']
    observation_matches = observation == result['protocol']['traffic_observation']
    prediction_matches = header['traffic_prediction'] == result['protocol']['traffic_prediction']
    result['replay_source'] = dict(run_id=args.run_id, source_sha256=header['source_sha256'],
                                 traffic_policy=header['traffic_policy'],
                                 current_traffic_policy=result['protocol']['traffic_policy'],
                                 traffic_policy_matches=policy_matches,
                                 traffic_model_version=header['traffic_model_version'],
                                 current_traffic_model_version=result['protocol']['traffic_model_version'],
                                 traffic_model_version_matches=traffic_matches,
                                 traffic_observation=observation, observation_protocol_matches=observation_matches,
                                 traffic_prediction=header['traffic_prediction'], prediction_version_matches=prediction_matches,
                                 boundary=('Recorded proposals and simulation delays in the current world model; '
                                           'not original OS scheduling or recorded sensor samples. Historical traffic '
                                           'and perception are not reproduced when their protocol versions differ.'))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], termination_reason=result['termination_reason'],
                         source_version_matches=header['source_sha256']==result['protocol']['source_sha256'],
                         traffic_policy_matches=policy_matches,
                         traffic_model_version_matches=traffic_matches,
                         observation_protocol_matches=observation_matches, prediction_version_matches=prediction_matches)))


if __name__ == '__main__':
    main()
