"""Multi-model real-inference, fixed-step task execution with independent scoring.

No serving configuration is changed. All requests use explicitly selected local
endpoints; no expert answers or independent task oracles enter model inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_demonstrations import collect_case
from scripts.verify_tool_planning import measure
from uav_llm_demo.direct_vision import DirectVisionAgent
from uav_llm_demo.interaction_trajectory import restore_payload, validate_trajectory
from uav_llm_demo.provider_config import normalize_endpoint, is_loopback_endpoint, discover_models
from uav_llm_demo.run_log import source_fingerprint
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_benchmark import _BudgetedAgent, validate_requirement
from uav_llm_demo.spatial_motion import spatial_catalog


class EvaluationAgent(_BudgetedAgent):
    def __init__(self, *, inference_seed, **kwargs):
        super().__init__(**kwargs)
        self.inference_seed = inference_seed

    def _json_request_payload(self, messages, **kwargs):
        payload = super()._json_request_payload(messages, **kwargs)
        # One declared decoding policy for every arm, independent of web settings.
        return {**payload, 'temperature': 0.0, 'seed': self.inference_seed}


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('Comparison requires a nonempty independent task suite')
    seen = set()
    for case in data['cases']:
        identifier = case.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in seen or Path(identifier).name != identifier or identifier in {'.', '..'}:
            raise ValueError('Each task needs a unique safe ID')
        seen.add(identifier)
        if not case.get('instruction') or not case.get('family'):
            raise ValueError('Each task needs an instruction and family')
        if case.get('updates'):
            raise ValueError('This collector does not execute mid-mission updates')
        if case.get('scenario', 'semantic_campus') != 'semantic_campus':
            raise ValueError('This comparison uses the existing campus task execution path')
        if case.get('measured_motion_case') not in {None, 'crossing_curve', 'half_boundary'}:
            raise ValueError('Unknown independent measured-motion oracle')
        validate_requirement(case['oracle'])
        case['task_world'] = deepcopy(case.get('task_world', data.get('task_world', {})))
        if not isinstance(case['task_world'], dict):
            raise ValueError('Task world must be an independent mapping')
    return data


def additional_motion_check(case, execution):
    kind = case.get('measured_motion_case')
    if not kind:
        return None
    target = case.get('measurement_target', 'stadium')
    objects = spatial_catalog(load_scenario(case.get('scenario', 'semantic_campus')))['objects']
    obj = next(item for item in objects if item['id'] == target)
    # Only the completed motion's own measured tracking samples qualify. Return
    # and approach segments may never create an artificial closure or crossing.
    motions = [e for e in execution.get('evidence', []) if e.get('action') == 'motion'
               and e.get('target_id') == target and e.get('mission_id') == execution.get('mission_id')
               and e.get('mission_id') and e.get('vehicle_id') and e.get('execution_id')]
    runs = []
    for motion in motions:
        samples = [s for s in execution.get('telemetry', [])
                   if s.get('action') == 'motion' and s.get('phase') == 'tracking'
                   and s.get('mission_id') == motion.get('mission_id')
                   and s.get('vehicle_id') == motion.get('vehicle_id')
                   and s.get('execution_id') == motion.get('execution_id')]
        runs.append({'execution_id': motion.get('execution_id'), 'samples': len(samples),
                     **measure(kind, samples, obj['center_grid'], obj['footprint_grid'])})
    return {'kind': kind, 'target': target, 'ok': any(run['ok'] for run in runs), 'runs': runs}


def decision_diagnostics(directory):
    records = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines() if line]
    responses, rejected, tools = [], [], Counter()
    # A retry is an actual provider request following a failed decision in the
    # same stage, not an ordinary geometry -> preview -> submit tool dialogue.
    pending_failures, retry_decisions, retry_stages = set(), set(), Counter()
    for event in records:
        payload = restore_payload(directory, event.get('payload', {}))
        stage, decision_id = payload.get('stage'), payload.get('decision_id')
        if event['kind'] == 'model_request' and stage in pending_failures:
            retry_decisions.add(decision_id)
            pending_failures.remove(stage)
        elif event['kind'] == 'model_response':
            body = payload.get('response')
            body = body if isinstance(body, dict) else {}
            choices = body.get('choices')
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            source = payload.get('source')
            if source == 'provider' and decision_id in retry_decisions:
                retry_stages[stage] += 1
            responses.append({'sequence': event['sequence'], 'stage': stage,
                              'http_status': payload.get('http_status'), 'source': source,
                              'elapsed_ms': payload.get('elapsed_ms'), 'outcome': payload.get('outcome'),
                              'finish_reason': choice.get('finish_reason'),
                              'usage': body.get('usage', {})})
        elif event['kind'] == 'decision_result':
            action = payload.get('action') or {}
            if isinstance(action, dict) and action.get('tool'):
                tools[action['tool']] += 1
            if payload.get('status') in {'error', 'rejected'}:
                pending_failures.add(stage)
            if payload.get('status') in {'error', 'rejected', 'discarded', 'cancelled'}:
                rejected.append({'sequence': event['sequence'], 'stage': stage,
                                 'status': payload.get('status'), 'feedback': payload.get('feedback')})
    actual = [r for r in responses if r['source'] == 'provider']
    return {'responses': responses, 'rejected_decisions': rejected, 'tools': dict(tools),
            'provider_calls': len(actual), 'retry_requests': sum(retry_stages.values()),
            'retry_stage_calls': dict(retry_stages),
            'provider_seconds': sum((r['elapsed_ms'] or 0) / 1000 for r in actual)}


def failure_category(execution, diagnostics, *, complete, budget_exhausted, passed):
    if passed:
        return 'passed'
    if not complete:
        return 'recording_error'
    replies = diagnostics['responses']
    if any(r['source'] == 'provider' and (r['http_status'] is None or not 200 <= r['http_status'] < 300) for r in replies):
        return 'provider_transport'
    if budget_exhausted:
        return 'call_budget'
    if execution.get('termination_reason') in {'wall_budget', 'simulation_budget'}:
        return 'timeout'
    if execution.get('safety_violations') or execution.get('static_path_blocked'):
        return 'safety_or_execution'
    if any(r['finish_reason'] == 'length' for r in replies):
        return 'output_truncated'
    if execution.get('status') == 'completed':
        return 'independent_task_oracle'
    stages = [r['stage'] for r in diagnostics['rejected_decisions']]
    if stages:
        return {'requirements': 'requirements_generation', 'audit': 'requirements_audit',
                'plan': 'planning_or_validation', 'review': 'review_or_validation'}.get(stages[-1], 'execution')
    return 'execution'


def run_episode(case, directory, *, profile, seed, max_calls, wall_budget, simulation_budget, planner_timeout, runtime):
    # Only this evaluation process is affected; the web service and .env remain
    # untouched. Explicit vision settings prevent accidental external fallback.
    os.environ.update(VISION_API_KEY='', VISION_MODEL=profile['model'], VISION_BASE_URL=profile['base_url'])
    started = time.monotonic()
    agent = EvaluationAgent(api_key='', model=profile['model'], base_url=profile['base_url'],
        task_response_format='json_schema', timeout_seconds=planner_timeout,
        max_calls=max_calls, deadline=started + wall_budget, inference_seed=seed)
    vision = DirectVisionAgent({'api_key': '', 'base_url': profile['base_url'], 'vision_model': profile['model']})
    standard = {k: deepcopy(v) for k, v in case.items() if not k.startswith('offline_')}
    standard.setdefault('scenario', 'semantic_campus')
    execution = collect_case({'standard': standard, 'task_world': case.get('task_world', {})},
        directory, seed=seed, wall_budget=wall_budget, simulation_budget=simulation_budget,
        runtime_backend=runtime, agent=agent, vision_agent=vision)
    recording = validate_trajectory(directory)
    extra = additional_motion_check(case, execution)
    diagnostics = decision_diagnostics(directory)
    task_passed = execution['oracle_success'] and (extra is None or extra['ok'])
    recording_valid = recording['complete'] and execution.get('source_unchanged', False)
    valid = recording_valid
    transport_failed = any(r['source'] == 'provider' and
        (r['http_status'] is None or not 200 <= r['http_status'] < 300) for r in diagnostics['responses'])
    valid = valid and not transport_failed
    passed = task_passed and valid
    comparison = {'case_id': case['id'], 'model': profile['model'], 'passed': passed,
        'valid_evaluation': valid, 'task_passed': task_passed, 'recording': recording,
        'false_completion': execution.get('service_success', False) and not task_passed,
        'status': execution['status'], 'termination_reason': execution['termination_reason'],
        'failure_category': failure_category(execution, diagnostics, complete=recording_valid,
            budget_exhausted=agent.budget_exhausted, passed=passed),
        'additional_motion_check': extra, 'model_calls': agent.calls,
        'stage_calls': dict(agent.stage_calls), 'diagnostics': diagnostics,
        'retry_requests': diagnostics['retry_requests'], 'provider_seconds': diagnostics['provider_seconds'],
        'wall_seconds': execution.get('wall_seconds'), 'simulation_seconds': execution.get('simulation_seconds'),
        'safety_violations': execution.get('safety_violations', 0), 'source_sha256': execution.get('source_sha256'),
        'sft_relation': case.get('sft_relation'), 'episode': str(directory)}
    (directory / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return comparison


def summary(results, arms=('base', 'sft')):
    output = {}
    for arm in arms:
        rows = [row for row in results if row['arm'] == arm]
        durations = [r['wall_seconds'] for r in rows if isinstance(r.get('wall_seconds'), (int, float))]
        successful_durations = [r['wall_seconds'] for r in rows if r['passed'] and isinstance(r.get('wall_seconds'), (int, float))]
        retries = sum(row.get('retry_requests', 0) for row in rows)
        output[arm] = {'runs': len(rows), 'successes': sum(row['passed'] for row in rows),
            'success_rate': sum(row['passed'] for row in rows) / len(rows) if rows else None,
            'invalid_evaluations': sum(not row['valid_evaluation'] for row in rows),
            'false_completions': sum(row['false_completion'] for row in rows),
            'model_calls': sum(row['model_calls'] for row in rows),
            'retry_requests': retries, 'mean_retry_requests': retries / len(rows) if rows else None,
            'mean_wall_seconds': statistics.mean(durations) if durations else None,
            'median_wall_seconds': statistics.median(durations) if durations else None,
            'mean_success_wall_seconds': statistics.mean(successful_durations) if successful_durations else None,
            'provider_seconds': sum(row.get('provider_seconds', 0) for row in rows),
            'failure_categories': dict(Counter(row['failure_category'] for row in rows))}
    paired = {}
    for row in results:
        paired.setdefault((row['case_id'], row['seed']), {})[row['arm']] = row
    reference = arms[0]
    output['paired'] = [{'case_id': task, 'seed': seed,
        'outcomes': {arm: {k: row.get(k) for k in ('passed', 'valid_evaluation', 'failure_category',
            'retry_requests', 'model_calls', 'wall_seconds')} for arm, row in rows.items()},
        'changes_from_reference': {arm: (
            'not_comparable' if not rows[reference]['valid_evaluation'] or not rows[arm]['valid_evaluation'] else
            'improved' if not rows[reference]['passed'] and rows[arm]['passed'] else
            'regressed' if rows[reference]['passed'] and not rows[arm]['passed'] else 'unchanged')
            for arm in arms[1:]}}
        for (task, seed), rows in paired.items() if set(rows) == set(arms)]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'benchmarks/qwen_sft_regression.json')
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--sft-url', required=True)
    parser.add_argument('--sft-model', required=True)
    parser.add_argument('--profile', nargs=3, action='append', default=[], metavar=('NAME', 'URL', 'MODEL'),
                        help='Additional model arm; base and sft remain the reference profiles')
    parser.add_argument('--arm', action='append', help='Run only named arms; useful for sequential deployment on limited GPUs')
    parser.add_argument('--case', action='append', dest='case_ids')
    parser.add_argument('--seed', type=int, action='append')
    parser.add_argument('--max-calls', type=int, default=20)
    parser.add_argument('--wall-budget', type=float, default=300)
    parser.add_argument('--simulation-budget', type=float, default=1200)
    parser.add_argument('--planner-timeout', type=float, default=90)
    parser.add_argument('--runtime', choices=('html', 'gazebo'), default='html')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profiles = {arm: {'base_url': normalize_endpoint(getattr(args, arm + '_url')),
                     'model': getattr(args, arm + '_model'), 'api_key': '', 'provider': 'custom'} for arm in ('base', 'sft')}
    for name, url, model in args.profile:
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}', name) or name in profiles or not model.strip():
            parser.error('Additional profile requires a unique simple name and nonempty model')
        profiles[name] = {'base_url': normalize_endpoint(url), 'model': model, 'api_key': '', 'provider': 'custom'}
    if args.arm and (len(set(args.arm)) != len(args.arm) or set(args.arm) - set(profiles)):
        parser.error('Selected arms must be unique configured profile names')
    arms = tuple(name for name in profiles if not args.arm or name in args.arm)
    if any(not is_loopback_endpoint(p['base_url']) for p in profiles.values()):
        parser.error('All model endpoints must be explicit local services; external fallback is forbidden')
    if not 1 <= args.max_calls <= 30 or any(not math.isfinite(v) or v <= 0 for v in (args.wall_budget, args.simulation_budget, args.planner_timeout)):
        parser.error('Budgets must be positive and max-calls must be 1..30')
    data = load_cases(args.dataset)
    known = {c['id'] for c in data['cases']}
    if args.case_ids and (set(args.case_ids) - known or len(args.case_ids) != len(set(args.case_ids))):
        parser.error('Case IDs must exist and must not repeat')
    seeds = args.seed or [17]
    if len(seeds) != len(set(seeds)):
        parser.error('Seeds must not repeat')
    cases = [c for c in data['cases'] if not args.case_ids or c['id'] in args.case_ids]
    for name in arms:
        profile = profiles[name]
        if profile['model'] not in discover_models(profile)['models']:
            parser.error('Requested model is absent from the explicit local service: ' + profile['model'])
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    report = {'report_version': 2, 'complete': False, 'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        'source_sha256': source_fingerprint(), 'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol': {'profiles': profiles, 'selected_arms': arms, 'seeds': seeds, 'max_calls': args.max_calls,
            'wall_budget': args.wall_budget, 'simulation_budget': args.simulation_budget,
            'planner_timeout': args.planner_timeout, 'runtime': args.runtime, 'temperature': 0.0,
            'model_seed': 'same as episode seed', 'order': 'rotating arm order by task and seed',
            'retry_definition': 'Actual provider request following a rejected/error decision in the same stage; ordinary successful tool calls excluded',
            'html_clock': 'fixed simulation steps; waits for actual model inference',
            'scope': 'Engineering regression suite; not an independent hidden generalization benchmark'},
        'selected_runs': len(cases) * len(seeds) * len(arms), 'results': results}
    def save():
        report.update(complete=len(results) == report['selected_runs'], summary=summary(results, arms))
        temporary = args.output / 'comparison.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output / 'comparison.json')
    save()
    try:
        for seed_index, seed in enumerate(seeds):
            for index, case in enumerate(cases):
                rotation = (index + seed_index) % len(arms)
                order = arms[rotation:] + arms[:rotation]
                for arm in order:
                    episode = args.output / f'seed-{seed}' / case['id'] / arm
                    result = run_episode(case, episode, profile=profiles[arm], seed=seed,
                        max_calls=args.max_calls, wall_budget=args.wall_budget,
                        simulation_budget=args.simulation_budget, planner_timeout=args.planner_timeout, runtime=args.runtime)
                    results.append({**result, 'arm': arm, 'seed': seed})
                    save()
                    print(json.dumps({k: results[-1][k] for k in ('case_id', 'arm', 'passed', 'failure_category', 'model_calls', 'retry_requests', 'wall_seconds')}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        save()
        return 130
    return 0 if all(r['valid_evaluation'] for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
