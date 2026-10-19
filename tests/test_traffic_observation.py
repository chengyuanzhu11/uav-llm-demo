from dataclasses import FrozenInstanceError, replace
import unittest
import random
from unittest.mock import patch

from uav_llm_demo.environment import DynamicFlyingObject
from uav_llm_demo.safety import SafetyChecker, _intersects
from uav_llm_demo.traffic_observation import TrafficObservationConfig, TrafficObserver


def checker(buildings=(), zones=(), clearance=.6):
    return SafetyChecker(dict(grid_size=40, altitude_range=dict(min=0, max=15),
                              buildings=list(buildings), no_fly_zones=list(zones)), clearance=clearance)


def bird(object_id='bird', position=(3, 0, 2), velocity=(.5, 0, 0)):
    return DynamicFlyingObject(object_id, 'bird', position, velocity)


class TrafficObservationTests(unittest.TestCase):
    def test_config_validation_and_immutability(self):
        config = TrafficObservationConfig()
        with self.assertRaises(FrozenInstanceError):
            config.range_m = 30
        for field in ('range_m', 'sample_period_seconds', 'latency_seconds',
                      'position_noise_m', 'velocity_noise_mps', 'dropout_probability'):
            for value in (-1, float('nan'), float('inf'), True, None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(config, **{field: value})
        for field, value in (('range_m', 0), ('sample_period_seconds', 0),
                             ('dropout_probability', 1.01), ('occlusion', 1)):
            with self.assertRaises(ValueError):
                replace(config, **{field: value})
        with self.assertRaises(ValueError):
            TrafficObserver({})

    def test_delay_preserves_snapshot_and_sampling_does_not_invent_history(self):
        flow = TrafficObserver(seed=7)
        obstacle, geometry = bird(), checker()
        self.assertEqual(flow.sample([obstacle], (0, 0, 2), geometry, 0), [])
        obstacle.position = (5, 0, 2)
        self.assertEqual(flow.sample([obstacle], (0, 0, 2), geometry, .05), [])
        self.assertEqual(flow.latest(.05), [])
        delivered = flow.sample([obstacle], (0, 0, 2), geometry, .1)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0]['observed_at'], 0)
        self.assertEqual(delivered[0]['position'], (3, 0, 2))
        self.assertEqual(delivered[0]['velocity'], (.5, 0, 0))
        self.assertAlmostEqual(delivered[0]['radius_m'], 2.1)
        delivered[0]['position'] = (100, 100, 100)
        self.assertEqual(flow.latest(.1)[0]['position'], (3, 0, 2))
        later = flow.sample([obstacle], (0, 0, 2), geometry, 1)
        self.assertEqual([item['observed_at'] for item in later], [.1])
        self.assertEqual(flow.sample([], (0, 0, 2), geometry, 1.1)[0]['observed_at'], 1)

    def test_range_uses_anisotropic_meters_not_grid_distance(self):
        flow = TrafficObserver(TrafficObservationConfig(range_m=10, latency_seconds=0), seed=7)
        observations = flow.sample([bird('horizontal', (3, 0, 0)), bird('vertical', (0, 0, 3)),
                                    bird('boundary', (0, 0, 2))], (0, 0, 0), checker(), 0)
        self.assertEqual({item['id'] for item in observations}, {'horizontal', 'boundary'})

    def test_buildings_occlude_but_safety_padding_and_no_fly_zones_do_not(self):
        config = TrafficObservationConfig(latency_seconds=0)
        obstacle = bird(position=(4, 0, 2))
        geometry = checker(buildings=[dict(x=2, y=0, height=4)], clearance=3)
        flow = TrafficObserver(config)
        self.assertEqual(flow.sample([obstacle], (0, 0, 2), geometry, 0), [])
        obstacle.position = (4, .6, 2)
        self.assertEqual(len(flow.sample([obstacle], (0, .6, 2), geometry, .1)), 1)
        obstacle.position = (4, 0, 5)
        self.assertEqual(len(flow.sample([obstacle], (0, 0, 5), geometry, .2)), 1)
        flow = TrafficObserver(replace(config, occlusion=False))
        self.assertEqual(len(flow.sample([bird()], (0, 0, 2), geometry, 0)), 1)
        zone = dict(min=dict(x=1, y=-1, z=0), max=dict(x=2, y=1, z=4))
        flow = TrafficObserver(config)
        self.assertEqual(len(flow.sample([bird()], (0, 0, 2), checker(zones=[zone]), 0)), 1)

    def test_occlusion_broadphase_matches_full_slab_scan_and_reduces_narrowphase_calls(self):
        rng = random.Random(192)
        geometry = checker([dict(x=x * 3, y=y * 3, height=rng.uniform(.5, 12))
                            for x in range(1, 21) for y in range(1, 21)], clearance=1.2)
        buildings = TrafficObserver._physical_buildings(geometry)
        rays = [(tuple(rng.uniform(0, bound) for bound in (65, 65, 16)),
                 tuple(rng.uniform(0, bound) for bound in (65, 65, 16))) for _ in range(400)]
        # Degenerate, axis-aligned, exactly touching and almost-touching rays.
        lo, hi = buildings[0]
        rays.extend([(lo, lo), (lo, hi), ((0, lo[1], lo[2]), (hi[0], lo[1], lo[2])),
                     ((lo[0], 0, lo[2]), (lo[0], hi[1], lo[2])),
                     ((0, 0, 16), (65, 65, 16)), ((0, 0, 0), (0, 0, 15)),
                     ((0, lo[1], lo[2]), (lo[0] - 1e-13, lo[1], lo[2]))])
        expected, old_calls = [], 0
        for start, end in rays:
            hit = False
            for lower, upper in buildings:
                old_calls += 1
                if _intersects(start, end, lower, upper):
                    hit = True
                    break
            expected.append(hit)
        with patch('uav_llm_demo.traffic_observation._intersects', wraps=_intersects) as narrowphase:
            actual = [TrafficObserver._occluded(start, end, buildings) for start, end in rays]
        self.assertEqual(actual, expected)
        self.assertLess(narrowphase.call_count, old_calls * .25)
        self.assertIn(True, actual)
        self.assertIn(False, actual)

    def test_noise_and_dropout_reproduce_independently_of_object_iteration_order(self):
        config = TrafficObservationConfig(latency_seconds=0, position_noise_m=.2,
                                          velocity_noise_mps=.1, dropout_probability=.35)
        flows = [TrafficObserver(config, seed=12), TrafficObserver(config, seed=12)]
        birds = [bird('a'), bird('b', (4, 1, 2))]
        sequences = [[], []]
        for i in range(20):
            for index, flow in enumerate(flows):
                result = flow.sample(birds if index == 0 else reversed(birds), (0, 0, 2), checker(), i * .1)
                sequences[index].append(sorted(result, key=lambda item: item['id']))
        self.assertEqual(sequences[0], sequences[1])
        self.assertTrue(any(len(items) < 2 for items in sequences[0]))
        self.assertTrue(any(items for items in sequences[0]))
        self.assertTrue(any(item['position'] != birds[0].position for items in sequences[0] for item in items if item['id'] == 'a'))
        flows[0].reset()
        self.assertEqual(sorted(flows[0].sample(birds, (0, 0, 2), checker(), 0), key=lambda item: item['id']), sequences[0][0])

    def test_lost_observations_expire_and_clock_reversal_requires_reset(self):
        flow = TrafficObserver(TrafficObservationConfig(latency_seconds=0))
        geometry = checker()
        flow.sample([bird()], (0, 0, 2), geometry, 0)
        self.assertEqual(len(flow.latest(1.5)), 1)
        self.assertEqual(flow.sample([], (0, 0, 2), geometry, 1.5001), [])
        self.assertEqual(flow.latest(1.5001), [])
        with self.assertRaisesRegex(ValueError, 'reset'):
            flow.latest(0)
        with self.assertRaisesRegex(ValueError, 'reset'):
            flow.sample([bird()], (0, 0, 2), geometry, 0)
        flow.reset()
        self.assertEqual(len(flow.sample([bird()], (0, 0, 2), geometry, 0)), 1)
        for value in (None, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                flow.latest(value)

    def test_complete_loss_and_excessive_latency_do_not_deliver_stale_truth(self):
        geometry = checker()
        lost = TrafficObserver(TrafficObservationConfig(latency_seconds=0, dropout_probability=1))
        self.assertEqual(lost.sample([bird()], (0, 0, 2), geometry, 0), [])
        self.assertEqual(lost.latest(0), [])
        delayed = TrafficObserver(TrafficObservationConfig(latency_seconds=2))
        delayed.sample([bird()], (0, 0, 2), geometry, 0)
        self.assertEqual(delayed.sample([], (0, 0, 2), geometry, 2), [])
        self.assertEqual(delayed.latest(2), [])


if __name__ == '__main__':
    unittest.main()
