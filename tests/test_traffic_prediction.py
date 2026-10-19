import math
import unittest

from uav_llm_demo.motion import SCALE, distance
from uav_llm_demo.traffic_prediction import TrafficPredictor
from uav_llm_demo.traffic_motion import TRAFFIC_MODEL_VERSION


def observation(time, position=(0, 0, 0), velocity=(0, 0, 0), object_id='bird'):
    return dict(id=object_id, position=position, velocity=velocity, observed_at=time)


class TrafficPredictionTests(unittest.TestCase):
    def test_constant_velocity_and_stationary_tracks_remain_exact(self):
        for velocity in ((0, 0, 0), (1, -.5, .2)):
            predictor = TrafficPredictor()
            for index in range(11):
                now = index / 10
                predictor.update([observation(now, tuple(v * now for v in velocity), velocity)], now)
            forecast = predictor.forecast('bird', 1, horizon=6)
            self.assertEqual(len(forecast.positions), 41)
            for index, position in enumerate(forecast.positions):
                expected = tuple(v * (1 + index * .15) for v in velocity)
                self.assertLess(distance(position, expected), 1e-9)

    def test_history_improves_three_dimensional_accelerating_target(self):
        acceleration_m = (0.6, -.4, .5)
        predictor = TrafficPredictor()
        for index in range(11):
            now = index / 10
            position = tuple(.5 * a * now * now / scale for a, scale in zip(acceleration_m, SCALE))
            velocity = tuple(a * now / scale for a, scale in zip(acceleration_m, SCALE))
            predictor.update([observation(now, position, velocity)], now)
        forecast = predictor.forecast('bird', 1, horizon=1, dt=.1)
        actual = tuple(.5 * a * 4 / scale for a, scale in zip(acceleration_m, SCALE))
        cv = tuple(p + v for p, v in zip(position, velocity))
        self.assertEqual(forecast.model, 'history_damped_acceleration')
        self.assertLess(distance(actual, forecast.positions[-1]), distance(actual, cv) * .5)

    def test_acceleration_forecast_respects_known_speed_limit(self):
        predictor = TrafficPredictor(max_speed_mps=4)
        for index in range(6):
            now = index / 10
            # Metric speed grows from 1 to 2 m/s; future motion is capped at 4.
            predictor.update([observation(now, ((now + now * now) / 3, 0, 0),
                                                 ((1 + 2 * now) / 3, 0, 0))], now)
        forecast = predictor.forecast('bird', .5, horizon=6, dt=.15)
        for a, b in zip(forecast.positions, forecast.positions[1:]):
            self.assertLessEqual(distance(a, b) / .15, 4.000001)

    def test_axis_envelopes_preserve_vertical_clearance_and_unknown_profiles_are_conservative(self):
        forecasts = {}
        for profile in ('urban_crossing', 'semantic_campus', 'unknown-map'):
            predictor = TrafficPredictor(calibration_profile=profile)
            for index in range(11):
                now = index / 10
                predictor.update([observation(now, (now, 0, 2), (1, 0, 0))], now)
            forecasts[profile] = predictor.forecast('bird', 1.1, horizon=6)
        for forecast in forecasts.values():
            self.assertEqual(len(forecast.positions), len(forecast.uncertainty_horizontal_m))
            self.assertEqual(len(forecast.positions), len(forecast.uncertainty_vertical_m))
            for values in (forecast.uncertainty_horizontal_m, forecast.uncertainty_vertical_m):
                self.assertTrue(all(math.isfinite(value) and value >= 0 for value in values))
                self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        for field in ('uncertainty_horizontal_m', 'uncertainty_vertical_m'):
            conservative = getattr(forecasts['unknown-map'], field)
            for profile in ('urban_crossing', 'semantic_campus'):
                self.assertTrue(all(a >= b for a, b in zip(conservative, getattr(forecasts[profile], field))))

    def test_calibration_profiles_match_current_traffic_and_use_ordered_knots(self):
        self.assertEqual(TrafficPredictor.CALIBRATION_TRAFFIC_VERSION, TRAFFIC_MODEL_VERSION)
        for profile in TrafficPredictor.CALIBRATION_PROFILES.values():
            for knots in profile.values():
                self.assertEqual(knots[0], (0.0, 0.0))
                self.assertTrue(all(left[0] < right[0] and left[1] <= right[1]
                                    for left, right in zip(knots, knots[1:])))

    def test_unannounced_turn_requires_new_observations(self):
        predictor = TrafficPredictor(calibration_profile='urban_crossing')
        for index in range(11):
            now = index / 10
            predictor.update([observation(now, ((now + now * now) / 3, 0, 2),
                                                 ((1 + 2 * now) / 3, 0, 0))], now)
        before = predictor.forecast('bird', 1, horizon=3)
        # An unseen 1 rad/s turn starts now, with a bounded 3 m/s speed.
        # Past acceleration cannot reveal this change of intent. Its error may
        # exceed CV, so empirical envelopes must never be described as guarantees.
        actual = (2 / 3 + math.sin(3), 1 - math.cos(3), 2)
        cv = (2 / 3 + 3, 0, 2)
        self.assertGreater(distance(actual, before.positions[-1]), distance(actual, cv))
        new_position = (2 / 3 + math.sin(.1), 1 - math.cos(.1), 2)
        new_velocity = (math.cos(.1), math.sin(.1), 0)
        predictor.update([observation(1.1, new_position, new_velocity)], 1.1)
        after = predictor.forecast('bird', 1.1, horizon=3)
        self.assertLess(distance(after.positions[0], new_position), 1e-9)
        self.assertGreater(after.positions[1][1], new_position[1])

    def test_cold_start_uses_conservative_calibrated_cv(self):
        predictor = TrafficPredictor()
        predictor.update([observation(2, velocity=(0, 0, 1))], 2)
        forecast = predictor.forecast('bird', 2, horizon=3)
        self.assertEqual(forecast.model, 'observed_constant_velocity')
        self.assertEqual(forecast.positions[-1], (0, 0, 3))
        self.assertGreater(forecast.uncertainty_m[-1], .585)

    def test_duplicate_and_out_of_order_observations_do_not_poison_history(self):
        predictor = TrafficPredictor()
        for index in range(5):
            now = index / 10
            predictor.update([observation(now, (now, 0, 0), (1, 0, 0))], now)
        before = predictor.forecast('bird', .4)
        predictor.update([observation(.4, (999, 999, 999)), observation(.1, (999, 0, 0))], .4)
        self.assertEqual(before, predictor.forecast('bird', .4))

    def test_future_observations_and_generator_metadata_are_never_used(self):
        predictor = TrafficPredictor()
        item = observation(0, (1, 2, 3), (0, 1, 0))
        item['motion_model'] = {'segments': object()}
        predictor.update([item, observation(1, (99, 99, 99))], 0)
        self.assertEqual(predictor.forecast('bird', 0, horizon=1).positions[-1], (1, 3, 3))

    def test_delayed_observations_start_at_now_and_grow_uncertainty(self):
        predictor = TrafficPredictor()
        predictor.update([observation(1, (2, 3, 4), (1, 0, .5))], 1.2)
        forecast = predictor.forecast('bird', 1.2, horizon=.3, dt=.15)
        self.assertAlmostEqual(forecast.age_seconds, .2)
        self.assertLess(distance(forecast.positions[0], (2.2, 3, 4.1)), 1e-9)
        self.assertLess(distance(forecast.positions[-1], (2.5, 3, 4.25)), 1e-9)
        self.assertGreater(forecast.uncertainty_m[0], 0)
        self.assertGreater(forecast.uncertainty_m[-1], forecast.uncertainty_m[0])

    def test_occluded_track_extrapolates_then_expires(self):
        predictor = TrafficPredictor(stale_seconds=1.5)
        predictor.update([observation(0, velocity=(1, 0, 0))], 0)
        predictor.update([], 1)
        self.assertIsNotNone(predictor.forecast('bird', 1))
        self.assertEqual(predictor.object_ids, ('bird',))
        predictor.update(None, 1.51)
        self.assertEqual(predictor.object_ids, ())
        self.assertIsNone(predictor.forecast('bird', 1.51))
        self.assertIsNone(predictor.forecast('missing', 1.51))

    def test_reacquisition_after_expiry_has_no_stale_acceleration(self):
        predictor = TrafficPredictor()
        for index in range(5):
            now = index / 10
            predictor.update([observation(now, (now * now, 0, 0), (now * 2, 0, 0))], now)
        predictor.update([observation(3, (10, 0, 0), (0, 1, 0))], 3)
        forecast = predictor.forecast('bird', 3, horizon=1)
        self.assertEqual(forecast.model, 'observed_constant_velocity')
        self.assertEqual(forecast.positions[-1], (10, 1, 0))

    def test_invalid_values_are_rejected_without_erasing_valid_tracks(self):
        predictor = TrafficPredictor()
        predictor.update([observation(0, (1, 2, 3))], 0)
        predictor.update([observation(.1, (math.nan, 0, 0)), observation(.1, velocity=(0, math.inf, 0)),
                          observation(.1, position=(1, 2)), {}, None, observation(3)], .1)
        self.assertEqual(predictor.forecast('bird', .1).positions[0], (1, 2, 3))
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                predictor.update([], value)
        for args in (dict(dt=0), dict(dt=-1), dict(horizon=-1), dict(horizon=math.inf), dict(horizon=13), dict(dt=1e-10)):
            with self.assertRaises(ValueError):
                predictor.forecast('bird', .1, **args)

    def test_reset_requires_explicit_clock_reset_and_clears_history(self):
        predictor = TrafficPredictor()
        predictor.update([observation(5)], 5)
        with self.assertRaises(ValueError):
            predictor.update([], 0)
        with self.assertRaises(ValueError):
            predictor.forecast('bird', 4)
        predictor.reset()
        predictor.update([observation(0)], 0)
        self.assertEqual(predictor.forecast('bird', 0).observed_at, 0)

    def test_input_containers_are_copied_and_non_divisible_horizon_is_included(self):
        predictor = TrafficPredictor()
        position = [1, 2, 3]
        predictor.update([observation(0, position, {'x': 1, 'y': 0, 'z': 0})], 0)
        position[0] = 99
        forecast = predictor.forecast('bird', 0, horizon=.31, dt=.15)
        self.assertEqual(len(forecast.positions), 4)
        self.assertAlmostEqual(forecast.positions[-1][0], 1.31)
        self.assertEqual(len(forecast.positions), len(forecast.uncertainty_m))
        self.assertTrue(all(a <= b for a, b in zip(forecast.uncertainty_m, forecast.uncertainty_m[1:])))


if __name__ == '__main__':
    unittest.main()
