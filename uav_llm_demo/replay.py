"""Offline response replay: no credentials, network calls or live-state mutation."""
import json
from pathlib import Path

from .agents import RoutePlan, PlannerProviderError
from .scenarios import ScenarioDefinition, DynamicObjectSpec, UrbanParcel


def scenario_from_record(data):
    point = lambda item: tuple(float(item[k]) for k in 'xyz')
    band = data.get('preferred_altitude')
    return ScenarioDefinition(
        scenario_id=data['id'], name=data['name'], grid_size=data['grid_size'],
        min_altitude=data['altitude_range']['min'], max_altitude=data['altitude_range']['max'],
        base=point(data['base']), target=point(data['target']),
        buildings=tuple((b['x'], b['y'], b['height']) for b in data['buildings']),
        dynamic_objects=tuple(DynamicObjectSpec(b['id'], b['kind'], point(b['position']), point(b['velocity']), b['radius'])
                              for b in data['dynamic_objects']),
        seed=data['seed'], no_fly_zones=tuple(data.get('no_fly_zones', [])),
        preferred_altitude=(band['min'], band['max']) if band else None,
        urban_parcels=tuple(UrbanParcel(p['id'], p['use'], tuple(p['footprint']), p['podium_height'],
                                      p['facade'], tuple(p['tower']) if p['tower'] else None, p.get('profile','slab'))
                           for p in data.get('urban_parcels', [])),
        urban_street_centers=tuple(data.get('urban_street_centers', [])))


class ReplayAgent:
    backend_name = 'replay'

    def __init__(self, attempts):
        self.attempts = iter(attempts)
        self.delivery_delay = 0.0

    def plan(self, context, feedback=''):
        try:
            item = next(self.attempts)
        except StopIteration:
            raise PlannerProviderError('Recorded responses exhausted; no API request was made.', retryable=False) from None
        self.delivery_delay = item['delivery_delay']
        if item['event'] == 'provider_failed':
            raise PlannerProviderError(item['feedback'], retryable=item.get('retryable', True))
        if not item['waypoints']:
            raise ValueError('Recorded planner response did not contain a valid route.')
        return RoutePlan(tuple(tuple(p) for p in item['waypoints']), 'Recorded proposal', 'medium', 'Verify execution')


def load_run(path, run_id):
    records = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                event = json.loads(line)
                if event.get('run_id') == run_id:
                    records.append(event)
    starts = [r for r in records if r['event'] == 'run_started']
    if len(starts) != 1:
        raise ValueError('Run header missing or duplicated; include the journal file containing run_started.')
    if any(r['event'] in {'goal_changed', 'agent_changed', 'mission_paused'} for r in records):
        raise ValueError('This run contains manual changes; replay currently requires one uninterrupted mission.')
    requests = {r['request_id']: r for r in records if r['event'] == 'plan_requested'}
    attempts = []
    for item in records:
        if item['event'] not in {'provider_failed', 'plan_accepted', 'plan_rejected'}:
            continue
        request = requests.get(item['request_id'])
        if request is None:
            raise ValueError('Request record missing; journal may have rotated or dropped events.')
        attempts.append({**item, 'delivery_delay': max(0, item['simulation_seconds']-request['simulation_seconds'])})
    if not attempts:
        raise ValueError('No completed planning attempts in this run.')
    return starts[0], ReplayAgent(attempts)
