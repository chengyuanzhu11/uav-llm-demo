"""Execute independently specified expert demonstrations; never start training.

Only the provider boundary is substituted by an explicitly disclosed scripted
expert. Production parsing, tools, contracts, planning and control run normally.
Task standards and evaluation truth are never included in policy observations.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_llm_demo.agents import DeepSeekAgent, PlannerProviderError
from uav_llm_demo.html_runtime import HtmlRuntime
from uav_llm_demo.interaction_trajectory import InteractionRecorder
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.run_log import source_fingerprint
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_benchmark import (TERMINAL_FAILURES, _Journal, _measurement,
    _sample, _world, evaluate_oracle, redact, validate_requirement)
from uav_llm_demo.task_observation import TaskObservationWorld

DEFAULT_STANDARDS = ROOT / 'benchmarks/demonstration_pilot.json'


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_standards(path=DEFAULT_STANDARDS):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('version') != 'demonstration-pilot-v1' or not data.get('cases'):
        raise ValueError('Unknown or empty demonstration standards')
    seen = set()
    for case in data['cases']:
        standard, demo = case['standard'], case['demonstration']
        identifier = standard.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Every demonstration needs a unique source task ID')
        if Path(identifier).name != identifier or identifier in {'.', '..'}:
            raise ValueError('Task ID must be a single safe path component')
        seen.add(identifier)
        if not standard.get('family') or not standard.get('intent') or not standard.get('instruction'):
            raise ValueError('Independent source intent, family and instruction are required')
        if standard.get('annotation_source') != 'agent_assisted' or standard.get('human_reviewed') is not False:
            raise ValueError('This seed collector must not claim human annotation')
        validate_requirement(standard['oracle'])
        for key in ('requirements', 'audit', 'plan', 'query_target'):
            if key not in demo:
                raise ValueError('Missing independent expert declaration: ' + key)
        if demo.get('repair') not in {None, 'missing_obligation', 'invalid_field', 'invalid_map_target'}:
            raise ValueError('Unknown generic candidate corruption')
    return data


def corrupt_candidate(plan, mode):
    """Return rejected context, not a positive supervision target."""
    changed = deepcopy(plan)
    if mode == 'missing_obligation':
        if len(changed['steps']) < 2:
            raise ValueError('Omission demonstration needs at least two obligations')
        changed['steps'].pop(0)
    elif mode == 'invalid_field':
        changed['steps'][0]['unsupported_candidate_field'] = True
    elif mode == 'invalid_map_target':
        changed['steps'][0]['target_id'] = 'unregistered_target'
    else:
        raise ValueError('Unknown corruption mode')
    return changed


class DemonstrationAgent(DeepSeekAgent):
    """Disclosed expert provider; not a trained or evaluated language model."""
    def __init__(self, case):
        super().__init__(api_key='', model='scripted_expert/task-standard-v1',
                         base_url='http://127.0.0.1:1/v1', task_response_format='json_schema')
        self.declaration = deepcopy(case['demonstration'])
        self.calls, self._plan_round = [], 0

    def _request_json_content(self, messages, result_kind, **kwargs):
        demo = self.declaration
        if result_kind == 'task requirements':
            value = demo['requirements']
        elif result_kind == 'requirements audit':
            value = demo['audit']
        elif result_kind == 'semantic task plan':
            current = self._plan_round
            self._plan_round += 1
            repair = bool(demo.get('repair'))
            if current == 0:
                value = dict(tool='query_geometry', arguments=dict(target_id=demo['query_target']))
            elif current == 1 and repair:
                value = dict(tool='preview_plan', arguments=dict(plan=corrupt_candidate(demo['plan'], demo['repair'])))
            elif current == (2 if repair else 1):
                if repair and self._observation(messages).get('ok') is not False:
                    raise PlannerProviderError('Expected corrupt candidate to be rejected.', retryable=False)
                value = dict(tool='preview_plan', arguments=dict(plan=demo['plan']))
            elif current == (3 if repair else 2):
                observed = self._observation(messages)
                if observed.get('ok') is not True or not observed.get('candidate_id'):
                    raise PlannerProviderError('Expert candidate failed production preview: '+
                        str(observed.get('diagnostic')), retryable=False)
                value = dict(tool='submit_plan', arguments=dict(candidate_id=observed['candidate_id']))
            else:
                raise PlannerProviderError('Expert exhausted; do not synthesize a label.', retryable=False)
        else:
            raise PlannerProviderError('Expert collection forbids remote route inference.', retryable=False)
        self.calls.append(dict(stage=result_kind, response=deepcopy(value), prompt_version=kwargs.get('prompt_version')))
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))

    @staticmethod
    def _observation(messages):
        prefix = 'Tool observation:\n'
        for message in reversed(messages):
            content = message.get('content', '')
            if message.get('role') == 'user' and isinstance(content, str) and content.startswith(prefix):
                return json.JSONDecoder().raw_decode(content[len(prefix):])[0]
        raise PlannerProviderError('Missing actual tool observation; cannot invent feedback.', retryable=False)


def collect_case(case, output, *, seed=17, simulation_budget=600, wall_budget=120,
                 runtime_backend='html', vision_agent=None, agent=None, scenario_override=None, scene_file=None):
    """Fresh isolated execution; does not modify a running web application."""
    if runtime_backend not in {'html', 'gazebo'}:
        raise ValueError('Only existing HTML and Gazebo runtimes are supported')
    output, standard = Path(output), deepcopy(case['standard'])
    selected = scenario_override if scenario_override is not None else load_scenario(standard['scenario'])
    if selected.scenario_id != standard['scenario']:
        raise ValueError('Selected scene differs from the source task standard')
    scenario = replace(selected, dynamic_objects=())
    scene_digest = None
    if standard.get('scene_sha256'):
        from uav_llm_demo.simulation_scene import load_scene
        scene_digest = stable_hash(load_scene(scenario.asset_scene))
        if scene_digest != standard['scene_sha256']:
            raise ValueError('Scene content differs from the frozen task standard')
    initial_source = source_fingerprint()
    agent = DemonstrationAgent(case) if agent is None else agent
    scripted = isinstance(agent, DemonstrationAgent)
    recorder = InteractionRecorder(output, metadata=dict(task_id=standard['id'],
        task_standard_sha256=stable_hash(standard), family=standard['family'], scenario=standard['scenario'],
        split=standard.get('frozen_split', 'pilot'), scene_asset_content_sha256=scene_digest, model=agent.model, annotation_source='agent_assisted',
        model_inference=not scripted, demonstration_source='scripted_expert' if scripted else 'model_rollout',
        runtime=runtime_backend, source_sha256=initial_source, seed=seed,
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), secrets=(agent.api_key,))
    defaults = TaskObservationWorld.demonstration(scenario)
    # The scenario's existing demonstration world remains the single default.
    world = TaskObservationWorld.from_dict({**defaults._targets, **case.get('task_world', {})})
    journal, samples, session = _Journal(), [], None
    runtime = HtmlRuntime()
    runtime.reset(scenario.base)
    if runtime_backend == 'gazebo':
        from uav_llm_demo.gazebo_session import GazeboSession
        session = GazeboSession(scenario, seed=seed, scene_file=scene_file)
        runtime = session.runtime
    service = MissionService(scenario=scenario, runtime=runtime, agent=agent, task_world=world,
        traffic_seed=seed, target_seed=seed, journal=journal, trajectory_recorder=recorder, vision_agent=vision_agent)
    start, reason, status = time.monotonic(), 'simulation_budget', 'timeout'
    result = dict(case_id=standard['id'], status='failed', termination_reason='collection_error', oracle_success=False)
    try:
        if session:
            session.start(timeout=min(180, wall_budget))
            service.advance(service.CONTROL_DT)
        service.submit_task(standard['instruction'], background=bool(session))
        while True:
            state = _measurement(service)
            sample = _sample(service)
            if sample is not None and (not samples or sample['t'] > samples[-1]['t']):
                samples.append(sample)
            semantic = state['semantic_mission']
            if state['status'] == 'completed':
                status, reason = 'completed', 'service_completed'
                break
            if semantic['status'] in TERMINAL_FAILURES or state['flight']['error']:
                status, reason = 'failed', semantic['status']
                break
            if time.monotonic() - start >= wall_budget:
                reason = 'wall_budget'
                break
            if service._time >= simulation_budget:
                break
            if session:
                time.sleep(.02)
            else:
                future = service._future
                if future is not None:
                    try:
                        future.result(timeout=max(.01, wall_budget-(time.monotonic()-start)))
                    except Exception:
                        pass  # Service classifies completed worker errors.
                service.advance(service.CONTROL_DT)
        semantic, scenario_world = _measurement(service)['semantic_mission'], _world(scenario)
        conditions = dict(condition_evaluations=semantic.get('condition_evaluations', []),
                          condition_evaluations_dropped=semantic.get('condition_evaluations_dropped', 0))
        oracle = evaluate_oracle(standard['oracle'], evidence=semantic['evidence'], telemetry=samples,
            world=scenario_world, mission_id=semantic['mission_id'], **conditions)
        violations = state['metrics'].get('collision_or_separation_events', 0)
        blocked = not bool(state['milestones'].get('static_path_unblocked'))
        success = status == 'completed' and oracle['satisfied'] and not violations and not blocked
        result = redact(dict(case_id=standard['id'], status=status, termination_reason=reason,
            instruction=standard['instruction'], oracle_success=success, requirements_satisfied=oracle['satisfied'],
            service_success=bool(state['milestones'].get('safe_success')), safe_execution=not violations and not blocked,
            safety_violations=violations, static_path_blocked=blocked, oracle=oracle, evidence=semantic['evidence'],
            telemetry=samples, world=scenario_world, mission_id=semantic['mission_id'], **conditions,
            source_sha256=initial_source, source_unchanged=initial_source == source_fingerprint(),
            runtime=runtime_backend, runtime_final=runtime.status(), simulator=session.manifest if session else None,
            model=agent.model, model_inference=not scripted, annotation_source='agent_assisted', events=journal.records,
            simulation_seconds=service._time, wall_seconds=time.monotonic()-start,
            expert_calls=agent.calls if scripted else [], metrics=state['metrics']), (agent.api_key,))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), source_sha256=initial_source,
                      source_unchanged=initial_source == source_fingerprint())
        result = redact(result, (agent.api_key,))
    finally:
        # Wait for the active provider before closing the asynchronous journal.
        # Provider timeout bounds this; never turn a late reply into a success.
        pending = [(service._future, getattr(agent, 'timeout_seconds', 30)),
                   (service.visual.future, getattr(service.visual.agent, 'timeout_seconds', 60))]
        service.close()
        for future, timeout in pending:
            if future is not None:
                try:
                    future.result(timeout=max(1, timeout) + 5)
                except Exception:
                    pass
        if session:
            session.close()
        recorder.close(outcome=dict(status=result['status'], passed=result['oracle_success'],
            failure_kind=None if result['oracle_success'] else result['termination_reason']),
            oracle=dict(task_standard_sha256=stable_hash(standard), result=result.get('oracle')))
    result.update(scene_asset_content_sha256=scene_digest, trajectory_events_sha256=recorder.manifest['events']['sha256'],
                  task_standard_sha256=stable_hash(standard), run_id=service.trace.run_id,
                  collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for filename, value in [('standard.json', standard), ('execution.json', result)]:
        (output/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standards', type=Path, default=DEFAULT_STANDARDS)
    parser.add_argument('--case', action='append', dest='case_ids')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--wall-budget', type=float, default=120)
    parser.add_argument('--simulation-budget', type=float, default=600)
    args = parser.parse_args()
    dataset = load_standards(args.standards)
    known = {c['standard']['id'] for c in dataset['cases']}
    if args.case_ids and (set(args.case_ids)-known or len(set(args.case_ids))!=len(args.case_ids)):
        parser.error('Case IDs must exist and must not repeat')
    if args.wall_budget <= 0 or args.simulation_budget <= 0:
        parser.error('Budgets must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    selected = [c for c in dataset['cases'] if not args.case_ids or c['standard']['id'] in args.case_ids]
    results = []
    for case in selected:
        identifier = case['standard']['id']
        result = collect_case(case, args.output/identifier, seed=args.seed,
                              wall_budget=args.wall_budget, simulation_budget=args.simulation_budget)
        results.append(dict(case_id=identifier, status=result['status'], passed=result['oracle_success'],
                            wall_seconds=result.get('wall_seconds'), reason=result.get('termination_reason')))
        report = dict(version=dataset['version'], selected=len(selected), complete=len(results)==len(selected),
                      executions_passed=sum(r['passed'] for r in results), results=results,
                      model_inference=False, training_started=False, quality_review_required=True)
        (args.output/'collection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f"{identifier}: execution={result['oracle_success']}; status={result['status']}", flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
