"""Causal traffic forecast benchmark; never opens a server or calls a provider.

Calibration seeds choose envelope quantiles; held-out seeds are only evaluated.
The predictor only receives current position/velocity/time. Future snapshots are
kept by this evaluator solely to score previously issued predictions.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_llm_demo.environment import UAVEnvironment
from uav_llm_demo.motion import SCALE, distance
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.traffic_motion import ContinuousTraffic, TRAFFIC_MODEL_VERSION
from uav_llm_demo.traffic_prediction import TrafficPredictor

CALIBRATION_SEEDS = (2, 4, 6, 8)
HELD_OUT_SEEDS = (9, 21, 37, 53)
HORIZONS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)


def quantile(values, probability):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(len(values) * probability) - 1)]


def snapshots(scenario, seed, duration):
    clock = [0.0]
    environment = UAVEnvironment(scenario)
    traffic = ContinuousTraffic(scenario, seed=seed, policy='fixed', clock=lambda: clock[0])
    result = []
    for step in range(round((duration + max(HORIZONS)) / 0.1) + 1):
        clock[0] = step / 10
        traffic.sync(environment)
        result.append({bird.object_id: dict(id=bird.object_id, position=tuple(bird.position),
                                           velocity=tuple(bird.velocity), observed_at=clock[0])
                       for bird in environment.dynamic_objects})
    return result


def evaluate(scenario, seeds, duration, decay_seconds, observation_age):
    errors = defaultdict(list)
    cv_errors = defaultdict(list)
    horizontal_errors, vertical_errors = defaultdict(list), defaultdict(list)
    cv_horizontal_errors, cv_vertical_errors = defaultdict(list), defaultdict(list)
    joint_covered, cold_joint_covered = defaultdict(list), defaultdict(list)
    horizontal_covered, vertical_covered = defaultdict(list), defaultdict(list)
    cold_horizontal_covered, cold_vertical_covered = defaultdict(list), defaultdict(list)
    covered = defaultdict(list)
    cold_covered = defaultdict(list)
    old_covered = defaultdict(list)
    latencies = []
    per_seed = []
    age_steps = round(observation_age * 10)
    for seed in seeds:
        frames = snapshots(scenario, seed, duration)
        predictor = TrafficPredictor(decay_seconds=decay_seconds, stale_seconds=2, calibration_profile=scenario.scenario_id)
        seed_errors, seed_cv = [], []
        for step in range(round(duration * 10) + 1):
            now = step / 10
            if step < age_steps:
                continue
            observations = list(frames[step - age_steps].values())
            predictor.update(observations, now)
            if step < 10 + age_steps or step % 5:
                continue
            for observation in observations:
                started = time.perf_counter()
                forecast = predictor.forecast(observation['id'], now, horizon=8, dt=0.5)
                latencies.append((time.perf_counter() - started) * 1000)
                cold = TrafficPredictor(decay_seconds=decay_seconds, stale_seconds=2, calibration_profile=scenario.scenario_id)
                cold.update([observation], now)
                cold_forecast = cold.forecast(observation['id'], now, horizon=8, dt=0.5)
                for index, horizon in enumerate(HORIZONS):
                    future = frames[step + round(horizon * 10)][observation['id']]['position']
                    error = distance(future, forecast.positions[index])
                    cv = tuple(p + v * (horizon + observation_age) for p, v in zip(observation['position'], observation['velocity']))
                    cv_error = distance(future, cv)
                    errors[horizon].append(error)
                    cv_errors[horizon].append(cv_error)
                    delta = tuple((future[axis] - forecast.positions[index][axis]) * SCALE[axis] for axis in range(3))
                    cv_delta = tuple((future[axis] - cv[axis]) * SCALE[axis] for axis in range(3))
                    xy_error, z_error = math.hypot(*delta[:2]), abs(delta[2])
                    cv_xy, cv_z = math.hypot(*cv_delta[:2]), abs(cv_delta[2])
                    horizontal_errors[horizon].append(xy_error)
                    vertical_errors[horizon].append(z_error)
                    cv_horizontal_errors[horizon].append(cv_xy)
                    cv_vertical_errors[horizon].append(cv_z)
                    xy_inside = xy_error <= forecast.uncertainty_horizontal_m[index] + 1e-9
                    z_inside = z_error <= forecast.uncertainty_vertical_m[index] + 1e-9
                    cold_xy_inside = cv_xy <= cold_forecast.uncertainty_horizontal_m[index] + 1e-9
                    cold_z_inside = cv_z <= cold_forecast.uncertainty_vertical_m[index] + 1e-9
                    horizontal_covered[horizon].append(xy_inside)
                    vertical_covered[horizon].append(z_inside)
                    cold_horizontal_covered[horizon].append(cold_xy_inside)
                    cold_vertical_covered[horizon].append(cold_z_inside)
                    joint_covered[horizon].append(xy_inside and z_inside)
                    cold_joint_covered[horizon].append(cold_xy_inside and cold_z_inside)
                    covered[horizon].append(error <= forecast.uncertainty_m[index])
                    cold_covered[horizon].append(cv_error <= cold_forecast.uncertainty_m[index])
                    total_horizon = horizon + observation_age
                    old_covered[horizon].append(cv_error <= 0.12 * total_horizon + 0.025 * total_horizon ** 2)
                    if horizon == 3:
                        seed_errors.append(error)
                        seed_cv.append(cv_error)
        per_seed.append(dict(seed=seed, samples_3s=len(seed_errors),
                             p95_3s_m=quantile(seed_errors, .95), cv_p95_3s_m=quantile(seed_cv, .95)))
    report = dict(seeds=list(seeds), duration_per_seed_seconds=duration, per_seed=per_seed,
                  observation_age_seconds=observation_age, traffic_model_version=TRAFFIC_MODEL_VERSION,
                  forecasts_timed=len(latencies), mean_forecast_ms=statistics.mean(latencies),
                  p95_forecast_ms=quantile(latencies, .95), horizons={})
    for horizon in HORIZONS:
        report['horizons'][str(horizon)] = dict(
            warm_samples=len(errors[horizon]), cold_samples=len(cv_errors[horizon]), p50_m=quantile(errors[horizon], .5),
            p95_m=quantile(errors[horizon], .95), p99_m=quantile(errors[horizon], .99),
            cv_p95_m=quantile(cv_errors[horizon], .95), cv_p99_m=quantile(cv_errors[horizon], .99),
            horizontal_p995_m=quantile(horizontal_errors[horizon], .995),
            vertical_p995_m=quantile(vertical_errors[horizon], .995),
            cv_horizontal_p995_m=quantile(cv_horizontal_errors[horizon], .995),
            cv_vertical_p995_m=quantile(cv_vertical_errors[horizon], .995),
            horizontal_coverage=statistics.mean(horizontal_covered[horizon]),
            vertical_coverage=statistics.mean(vertical_covered[horizon]),
            cold_horizontal_coverage=statistics.mean(cold_horizontal_covered[horizon]),
            cold_vertical_coverage=statistics.mean(cold_vertical_covered[horizon]),
            joint_coverage=statistics.mean(joint_covered[horizon]),
            cold_joint_coverage=statistics.mean(cold_joint_covered[horizon]),
            warm_coverage=sum(covered[horizon]) / len(covered[horizon]),
            cold_coverage=sum(cold_covered[horizon]) / len(cold_covered[horizon]),
            previous_cv_margin_coverage=sum(old_covered[horizon]) / len(old_covered[horizon]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--duration', type=float, default=120)
    parser.add_argument('--decay', type=float, default=TrafficPredictor.DECAY_SECONDS)
    parser.add_argument('--scenario', choices=('all', 'urban_crossing', 'semantic_campus'), default='all')
    parser.add_argument('--observation-age', type=float, default=0.1)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/traffic-prediction/held-out.json')
    args = parser.parse_args()
    if args.duration < 5 or not math.isfinite(args.duration):
        parser.error('duration must be at least five finite seconds')
    if not math.isfinite(args.observation_age) or not 0 <= args.observation_age <= 1.5 or abs(args.observation_age * 10 - round(args.observation_age * 10)) > 1e-8:
        parser.error('observation-age must be a multiple of 0.1 seconds from zero to 1.5')
    scenario_ids = ('urban_crossing', 'semantic_campus') if args.scenario == 'all' else (args.scenario,)
    results = {scenario_id: evaluate(load_scenario(scenario_id),
                                    CALIBRATION_SEEDS if args.calibrate else HELD_OUT_SEEDS,
                                    args.duration, args.decay, args.observation_age)
               for scenario_id in scenario_ids}
    report = dict(scenarios=results, calibration=args.calibrate,
                  method='history_damped_acceleration', decay_seconds=args.decay,
                  calibration_seeds=list(CALIBRATION_SEEDS), held_out_seeds=list(HELD_OUT_SEEDS),
                  observation_interval_seconds=0.1, observation_age_seconds=args.observation_age,
                  traffic_model_version=TRAFFIC_MODEL_VERSION, traffic_policy='fixed',
                  frozen_calibration_traffic_version=TrafficPredictor.CALIBRATION_TRAFFIC_VERSION,
                  information_boundary='Only delayed observed position, velocity and timestamp enter predictor.',
                  limitations='Empirical simulator coverage with noiseless measurements; no safety guarantee or real-sensor validation.')
    if args.calibrate:
        profiles = {}
        for scenario_id, result in results.items():
            profiles[scenario_id] = {}
            for key, metric in [('radial', 'p99_m'), ('cold_radial', 'cv_p99_m'),
                                ('horizontal', 'horizontal_p995_m'), ('vertical', 'vertical_p995_m'),
                                ('cold_horizontal', 'cv_horizontal_p995_m'), ('cold_vertical', 'cv_vertical_p995_m')]:
                previous = 0
                knots = [(0.0, 0.0)]
                for horizon in HORIZONS:
                    total = round(horizon + args.observation_age, 10)
                    if total == 0:
                        continue
                    error = result['horizons'][str(horizon)][metric]
                    reserve = (.02 if 'vertical' in key else .1) * total
                    previous = max(previous, math.ceil((error + reserve) * 100) / 100)
                    knots.append((total, previous))
                profiles[scenario_id][key] = knots
        profiles['combined'] = {key: [(knots[0][0], max(item[1] for item in knots))
                                      for knots in zip(*(profile[key] for profile in profiles.values()))]
                                 for key in next(iter(profiles.values()))}
        report['proposed_profiles'] = profiles
    else:
        checks = {
            'calibration_matches_traffic_version': TrafficPredictor.CALIBRATION_TRAFFIC_VERSION == TRAFFIC_MODEL_VERSION,
            'calibration_matches_decay': args.decay == TrafficPredictor.DECAY_SECONDS,
            'calibration_matches_observation_age': args.observation_age == .1,
        }
        for scenario_id, result in results.items():
            three = result['horizons']['3.0']
            checks[f'{scenario_id}:p95_3s_improved'] = three['p95_m'] < three['cv_p95_m']
            for horizon in ('1.0', '3.0', '6.0'):
                for state in ('warm', 'cold'):
                    checks[f'{scenario_id}:{state}_{horizon}s_coverage_at_least_95_percent'] = result['horizons'][horizon][f'{state}_coverage'] >= .95
            checks[f'{scenario_id}:joint_3s_at_least_99_percent'] = three['joint_coverage'] >= .99
            checks[f'{scenario_id}:joint_6s_at_least_99_percent'] = result['horizons']['6.0']['joint_coverage'] >= .99
            checks[f'{scenario_id}:cold_joint_3s_at_least_99_percent'] = three['cold_joint_coverage'] >= .99
            checks[f'{scenario_id}:p95_forecast_under_5ms'] = result['p95_forecast_ms'] < 5
        report['checks'] = checks
        report['ok'] = all(checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if args.calibrate or report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
