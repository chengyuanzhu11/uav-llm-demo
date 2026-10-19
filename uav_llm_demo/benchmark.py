from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeout
import math
import time

from .agents import build_agent
from .mission_service import MissionService
from .scenarios import list_scenarios, load_scenario
from .traffic_motion import TRAFFIC_POLICIES, TRAFFIC_MODEL_VERSION


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive.')
    return float(value)


def run_scenario(scenario_id, *, backend='mock', max_seconds=120.0,
                 control_dt=MissionService.CONTROL_DT, mode='virtual', seed=None,
                 planner_latency_seconds=0.0, planner_timeout_seconds=30.0,
                 agent=None, scenario=None, traffic_seed=None, traffic_policy="fixed",
                 observation_config=None, journal=None):
    """Virtual runs await local computation, then inject declared simulated latency.

    Real-time runs use the actual service loop and a wall-clock deadline. Remote
    providers are deliberately disallowed in accelerated virtual experiments.
    """
    if traffic_policy not in TRAFFIC_POLICIES:
        raise ValueError("Traffic policy must be fixed.")
    max_seconds = _positive(max_seconds, 'max_seconds')
    control_dt = _positive(control_dt, 'control_dt')
    planner_timeout_seconds = _positive(planner_timeout_seconds, 'planner_timeout_seconds')
    if control_dt > .1:
        raise ValueError('control_dt must be <= 0.1 seconds.')
    if mode not in {'virtual', 'realtime'}:
        raise ValueError('mode must be virtual or realtime.')
    if not math.isfinite(planner_latency_seconds) or planner_latency_seconds < 0:
        raise ValueError('planner_latency_seconds must be finite and nonnegative.')
    agent = agent or build_agent(backend)
    if mode == 'virtual' and agent.backend_name not in {'mock', 'replay'}:
        raise ValueError('Remote API evaluation requires mode=realtime; use local plans or recorded responses in virtual mode.')
    if mode == 'realtime' and (control_dt != MissionService.CONTROL_DT or planner_latency_seconds):
        raise ValueError('Realtime uses the production control period and actual provider latency.')
    template = scenario or load_scenario(scenario_id)
    if template.mission_mode != 'navigation':
        raise ValueError('Navigation benchmarks require Urban Crossing; semantic tasks use the task API.')
    seed = template.seed if seed is None else seed
    traffic_seed = seed + 1 if traffic_seed is None else traffic_seed
    service = MissionService(agent=agent, scenario_id=scenario_id, target_seed=seed,
                             traffic_seed=traffic_seed, traffic_policy=traffic_policy, observation_config=observation_config,
                             scenario=scenario, journal=journal,
                             planner_latency_seconds=planner_latency_seconds)
    started = time.monotonic()
    status, reason = 'timeout', 'wall_budget' if mode == 'realtime' else 'simulation_budget'
    initial = service._environment.scenario.to_dict()
    try:
        service.start(background=mode == 'realtime')
        while True:
            state = service.state()
            if mode == 'realtime' and time.monotonic()-started >= max_seconds:
                # A terminal state observed just after the poll deadline counts
                # only if its recorded transition occurred within the budget.
                if service._terminal_wall is None or service._terminal_wall > started+max_seconds:
                    break
            if state['flight']['error']:
                status, reason = 'failed', 'controller_or_planner_error'
                break
            if state['status'] == 'completed':
                status = 'completed' if state['milestones']['task_completed'] else 'failed'
                reason = 'verified_round_trip' if status == 'completed' else 'completion_evidence_missing'
                break
            elapsed = time.monotonic()-started if mode == 'realtime' else service._time
            if elapsed >= max_seconds:
                break
            if mode == 'realtime':
                time.sleep(min(.02, max_seconds-elapsed))
                continue
            future = service._future
            if future is not None:
                # Freeze virtual time while local CPU work finishes. Provider
                # latency, if studied, is injected explicitly on the sim clock.
                try:
                    future.result(timeout=planner_timeout_seconds)
                except FutureTimeout:
                    if not future.done():
                        reason = 'local_planner_timeout'
                        break
                except Exception:
                    pass  # Collected/classified by MissionService on the next step.
                if agent.backend_name == 'replay':
                    service._future_ready_at = max(service._future_ready_at,
                        service._future_context['simulation_time'] + agent.delivery_delay)
            service.advance(min(control_dt, max_seconds-service._time))
        # Capture a consistent endpoint before the background loop can advance again.
        with service._lock:
            service._shutdown.set()
            state = service.state()
            measured_seconds = time.monotonic()-started
            metrics = dict(state['metrics'])
            metrics['success'] = status == 'completed' and state['milestones']['task_completed']
            metrics['status'] = status
            if metrics['safe_success'] is not None:
                metrics['safe_success'] = bool(metrics['safe_success'] and metrics['success'])
            service.trace.event('benchmark_finished', service._time, status=status, reason=reason,
                                metrics=metrics, milestones=state['milestones'])
        return dict(scenario_id=scenario_id, target=state['target'], agent_backend=state['agent_backend'],
                    status=status, termination_reason=reason, error=state['flight']['error'],
                    protocol=dict(mode=mode, time_budget_seconds=max_seconds, control_dt=control_dt,
                                  planner_latency_seconds=planner_latency_seconds,
                                  planner_timeout_seconds=planner_timeout_seconds, seed=seed, traffic_seed=traffic_seed,
                                  traffic_policy=traffic_policy, traffic_model_version=TRAFFIC_MODEL_VERSION,
                                  traffic_observation=state["traffic_observation"],
                                  traffic_prediction=service.avoidance.snapshot()["prediction"],
                                  source_sha256=service._source_version),
                    initial_scenario=initial, simulation_seconds=service._time, wall_seconds=measured_seconds,
                    metrics=metrics, milestones=state['milestones'], diagnostics=service.trace.snapshot(),
                    local_replans=service.avoidance.replans,
                    min_sampled_dynamic_clearance_m=metrics['min_dynamic_clearance_m'])
    finally:
        service.close()


def summarize(results):
    """Descriptive run-level Wilson interval; runs must use independent seeds."""
    n = len(results)
    if not n:
        return dict(runs=0, success_rate=None, success_rate_95ci=None)
    successes = sum(r['metrics']['success'] for r in results)
    z, rate = 1.96, successes/n
    center = (rate+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(runs=n, successes=successes, timeouts=sum(r['status']=='timeout' for r in results),
                failures=sum(r['status']=='failed' for r in results), success_rate=rate,
                success_rate_95ci=[max(0,center-radius), min(1,center+radius)],
                mean_wall_seconds=sum(r['wall_seconds'] for r in results)/n,
                mean_planner_calls=sum(r['metrics']['planner_calls'] for r in results)/n,
                total_separation_events=sum(r['metrics']['collision_or_separation_events'] for r in results))


def run_benchmark(scenario_ids=None, *, backend='mock', max_seconds=120.0, mode='virtual',
                  seeds=None, planner_latency_seconds=0.0, planner_timeout_seconds=30.0,
                  traffic_policy="fixed", observation_config=None, journal=None):
    selected = scenario_ids or [item['id'] for item in list_scenarios() if item['mission_mode'] == 'navigation']
    return [run_scenario(scenario_id, backend=backend, max_seconds=max_seconds, mode=mode, seed=seed,
                         planner_latency_seconds=planner_latency_seconds,
                         planner_timeout_seconds=planner_timeout_seconds, traffic_policy=traffic_policy,
                         observation_config=observation_config, journal=journal)
            for scenario_id in selected for seed in (seeds if seeds is not None else [None])]


def estimate_cost(usage, prices):
    """Use caller-supplied per-million-token prices, never a hard-coded tariff."""
    rates = {}
    for name in ('input_miss', 'input_hit', 'output'):
        value = prices.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Prices must contain finite, nonnegative input_miss, input_hit and output rates per million tokens.')
        rates[name] = value
    currency = prices.get('currency')
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError('Prices must specify a currency label.')
    cost = (usage['prompt_cache_miss_tokens']*rates['input_miss']
            + usage['prompt_cache_hit_tokens']*rates['input_hit']
            + usage['completion_tokens']*rates['output'])/1_000_000
    return dict(observed_token_cost=cost, currency=currency,
                complete_usage=usage['usage_missing']==0 and usage.get('unreported_requests', 0)==0,
                note='Caller-supplied token rates; excludes unreported usage and non-token charges.')


def compare_results(before, after):
    """Paired regression report; never silently drop unmatched cases."""
    def indexed(results):
        output = {}
        for result in results:
            p = result['protocol']
            key = (result['scenario_id'], p['seed'], p['traffic_seed'])
            if key in output:
                raise ValueError('Duplicate case seeds; compare one paired repetition at a time.')
            output[key] = result
        return output
    a, b = indexed(before), indexed(after)
    if not a or a.keys() != b.keys():
        raise ValueError('Both files must contain the same nonempty set of scenario/seed cases.')
    recovered, regressed, both_success, both_failed = [], [], 0, 0
    for key in a:
        left, right = a[key], b[key]
        for field in ('mode', 'time_budget_seconds', 'control_dt', 'planner_latency_seconds', 'planner_timeout_seconds'):
            if left['protocol'][field] != right['protocol'][field]:
                raise ValueError('Mismatched evaluation protocol: '+field)
        for field, legacy in (('traffic_policy', 'route_demo'),
                              ('traffic_model_version', 'legacy-shared-stream'),
                              ('traffic_observation', 'legacy_ground_truth')):
            if left['protocol'].get(field, legacy) != right['protocol'].get(field, legacy):
                raise ValueError('Mismatched evaluation protocol: '+field)
        if left['initial_scenario'] != right['initial_scenario']:
            raise ValueError('Mismatched initial scenario for paired comparison.')
        old, new = left['metrics']['success'], right['metrics']['success']
        if not old and new:
            recovered.append(key)
        elif old and not new:
            regressed.append(key)
        elif old:
            both_success += 1
        else:
            both_failed += 1
    return dict(paired_cases=len(a), failure_to_success=recovered, success_to_failure=regressed,
                both_success=both_success, both_failed=both_failed,
                success_rate_change=(len(recovered)-len(regressed))/len(a),
                mean_wall_seconds_change=sum(b[k]['wall_seconds']-a[k]['wall_seconds'] for k in a)/len(a),
                mean_calls_change=sum(b[k]['metrics']['planner_calls']-a[k]['metrics']['planner_calls'] for k in a)/len(a))
