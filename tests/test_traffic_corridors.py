"""Bird traffic is a reproducible property of map geometry and world time."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from tests.scenario_fixtures import empty_airspace
from uav_llm_demo.environment import UAVEnvironment
from uav_llm_demo.scenarios import DEFAULT_SCENARIO, SCENARIOS
from uav_llm_demo.traffic_motion import ContinuousTraffic, TRAFFIC_MODEL_VERSION, TRAFFIC_POLICIES


class TrafficCorridorTests(unittest.TestCase):
    def test_constructor_needs_no_mission_vehicle_or_initial_bird_kinematics(self):
        source = DEFAULT_SCENARIO
        geometry = SimpleNamespace(scenario_id=source.scenario_id, grid_size=source.grid_size,
            buildings=source.buildings, no_fly_zones=source.no_fly_zones,
            dynamic_objects=tuple(SimpleNamespace(object_id=bird.object_id, radius=bird.radius)
                                  for bird in source.dynamic_objects))
        actual = ContinuousTraffic(geometry, seed=7, clock=lambda: 0)
        expected = ContinuousTraffic(source, seed=7, clock=lambda: 0)
        self.assertEqual(actual.models, expected.models)
        self.assertEqual(actual.corridors, expected.corridors)
        self.assertEqual(TRAFFIC_POLICIES, ("fixed",))
        self.assertEqual(TRAFFIC_MODEL_VERSION, "map-corridors-v2")
        self.assertFalse(hasattr(actual, "focus_route"))

    def test_targets_bases_timestep_and_bird_order_do_not_change_traffic(self):
        now = [0.0, 0.0]
        alternate = replace(DEFAULT_SCENARIO, base=(35, 35, 0), target=(2, 10, 9),
                            dynamic_objects=DEFAULT_SCENARIO.dynamic_objects[::-1])
        traffic = [ContinuousTraffic(scenario, seed=19, clock=lambda i=i: now[i])
                   for i, scenario in enumerate((DEFAULT_SCENARIO, alternate))]
        worlds = [UAVEnvironment(scenario) for scenario in (DEFAULT_SCENARIO, alternate)]
        self.assertEqual(traffic[0].corridors, traffic[1].corridors)
        for elapsed in (0, 7, 19, 63, 120, 180):
            while now[0] < elapsed:
                now[0] = min(elapsed, now[0] + .23)
                worlds[0].position = (3, 4, 5)
                worlds[0].goal = (38, 32, 3)
                traffic[0].sync(worlds[0])
            now[:] = [float(elapsed), float(elapsed)]
            for flow, world in zip(traffic, worlds):
                flow.sync(world)
            self.assertEqual(traffic[0].models, traffic[1].models)
            self.assertEqual({b.object_id: (b.position, b.velocity) for b in worlds[0].dynamic_objects},
                             {b.object_id: (b.position, b.velocity) for b in worlds[1].dynamic_objects})

    def test_urban_preserves_eight_birds_and_multiple_three_dimensional_bands(self):
        traffic = ContinuousTraffic(DEFAULT_SCENARIO, seed=7, clock=lambda: 0)
        self.assertEqual(len(traffic.models), 8)
        self.assertEqual(len({item['id'] for item in traffic.corridors.values()}), 8)
        self.assertEqual({bird.kind for bird in DEFAULT_SCENARIO.dynamic_objects}, {'bird'})
        bounds = [item['altitude_range_m'] for item in traffic.corridors.values()]
        self.assertEqual(min(item['min'] for item in bounds), 10)
        self.assertEqual(max(item['max'] for item in bounds), 30)
        for corridor in traffic.corridors.values():
            a, b = corridor['polyline']
            self.assertNotEqual(a[2], b[2])
            self.assertTrue(a[0] == b[0] == 9.5 or a[0] == b[0] == 29.5
                            or a[1] == b[1] == 9.5 or a[1] == b[1] == 29.5)
            self.assertLessEqual(corridor['peak_speed_mps'], 4)

    def test_campus_corridors_use_fixed_roads_and_publish_metric_altitudes(self):
        traffic = ContinuousTraffic(SCENARIOS['semantic_campus'], seed=7, clock=lambda: 0)
        self.assertEqual({item['id'] for item in traffic.corridors.values()},
                         {'campus-west', 'campus-north', 'campus-south'})
        for corridor in traffic.corridors.values():
            self.assertEqual(corridor['coordinate_unit'], 'grid')
            self.assertEqual(corridor['altitude_range_m']['min'],
                             min(p[2] for p in corridor['polyline']) * 5)

    def test_custom_map_uses_fixed_geometry_not_random_walk(self):
        scenario = empty_airspace()
        first = ContinuousTraffic(scenario, seed=3, clock=lambda: 0)
        second = ContinuousTraffic(scenario, seed=18, clock=lambda: 0)
        for key in first.models:
            self.assertEqual(first.corridors[key]['polyline'], second.corridors[key]['polyline'])
            self.assertEqual(first.models[key]['corridor_id'], second.models[key]['corridor_id'])
            self.assertNotEqual(first.corridors[key]['phase_seconds'], second.corridors[key]['phase_seconds'])

    def test_no_safe_corridor_has_an_explicit_error(self):
        blocked = replace(empty_airspace(), no_fly_zones=(
            dict(min=dict(x=-10, y=-10, z=0), max=dict(x=30, y=30, z=20)),))
        with self.assertRaisesRegex(ValueError, 'No statically safe bird corridor'):
            ContinuousTraffic(blocked, seed=0)


if __name__ == '__main__':
    unittest.main()
