"""Replay structured planner proposals against the current controller, offline."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_llm_demo.benchmark import run_scenario
from uav_llm_demo.replay import load_run, scenario_from_record


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
    result = run_scenario(scenario.scenario_id, agent=agent, scenario=scenario,
                          seed=header['target_seed'], traffic_seed=header['traffic_seed'],
                          max_seconds=args.max_seconds)
    result['replay_source'] = dict(run_id=args.run_id, source_sha256=header['source_sha256'],
                                 boundary='Recorded proposals and simulation delays; not original OS scheduling or sensor replay.')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], termination_reason=result['termination_reason'],
                         source_version_matches=header['source_sha256']==result['protocol']['source_sha256'])))


if __name__ == '__main__':
    main()
