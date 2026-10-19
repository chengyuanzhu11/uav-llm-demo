import unittest
from uav_llm_demo.environment import DynamicFlyingObject
from uav_llm_demo.evaluation import EvaluationLog


class DynamicIncidentTest(unittest.TestCase):
    def test_counts_overlap_episodes_instead_of_repeated_frames(self):
        log = EvaluationLog()
        bird = DynamicFlyingObject("bird", "bird", (10, 10, 2), (0, 0, 0))
        for tick in range(4):
            log.observe_dynamic((10, 10, 2), [bird], tick * .05)
        self.assertEqual(log.dynamic_event_count, 1)
        log.observe_dynamic((15, 10, 2), [bird], .2)
        log.observe_dynamic((15, 10, 2), [bird], .25)
        log.observe_dynamic((10, 10, 2), [bird], .3)
        self.assertEqual(log.dynamic_event_count, 2)
        data = log.snapshot(final_status="active", trajectory=[], battery=100)
        self.assertEqual(data["metrics"]["collision_or_separation_events"], 2)
        self.assertLess(data["metrics"]["min_dynamic_clearance_m"], 0)
        self.assertEqual(data["dynamic_incidents"][0]["object_id"], "bird")
        log.reset()
        self.assertEqual(log.dynamic_event_count, 0)
        self.assertIsNone(log.minimum_dynamic_clearance)

    def test_swept_encounter_is_detected_between_clear_endpoints(self):
        log = EvaluationLog()
        bird = DynamicFlyingObject("bird", "bird", (8, 10, 2), (0, 0, 0))
        log.observe_dynamic((10, 10, 2), [bird], 0)
        bird.position = (12, 10, 2)
        log.observe_dynamic((10, 10, 2), [bird], .05)
        self.assertEqual(log.dynamic_event_count, 1)
        self.assertLess(log.minimum_dynamic_clearance, 0)
        log.record(dict(feedback="No collision detected", decision={}))
        self.assertEqual(log.dynamic_event_count, 1)


class DynamicIncidentDetailTest(unittest.TestCase):
    @staticmethod
    def bird(x, y=0, z=2):
        return DynamicFlyingObject("bird", "bird", (x, y, z), (0, 0, 0), radius=.7)

    def test_separation_violation_without_envelope_contact(self):
        log = EvaluationLog()
        bird = self.bird(0, .95)
        log.observe_dynamic((0, 0, 2), [bird], 0)
        self.assertEqual(log.dynamic_event_count, 1)
        self.assertEqual(log.contact_event_count, 0)
        self.assertEqual(log.near_miss_event_count, 0)
        self.assertAlmostEqual(log.minimum_dynamic_clearance, -.25)
        self.assertAlmostEqual(log.minimum_contact_clearance, .35)

    def test_event_updates_minimum_and_duration_until_exit(self):
        log = EvaluationLog()
        bird = self.bird(1)
        log.observe_dynamic((0, 0, 2), [bird], 0)
        bird.position = (.2, 0, 2)
        log.observe_dynamic((0, 0, 2), [bird], 1)
        separation = next(e for e in log.dynamic_incidents if e["kind"] == "dynamic_separation_violation")
        self.assertAlmostEqual(separation["minimum_center_distance_m"], .6)
        self.assertAlmostEqual(separation["clearance_m"], -2.5)
        self.assertEqual(separation["duration_seconds"], 1)
        self.assertTrue(separation["active"])
        bird.position = (2, 0, 2)
        log.observe_dynamic((0, 0, 2), [bird], 2)
        self.assertFalse(separation["active"])
        self.assertAlmostEqual(separation["end_elapsed_seconds"], 1 + 2.5 / 5.4, places=5)
        self.assertAlmostEqual(separation["max_relative_speed_mps"], 5.4)
        self.assertEqual(log.near_miss_event_count, 0)
        self.assertEqual(log.contact_event_count, 1)

    def test_swept_crossing_has_exact_entry_exit_and_no_phantom_active_frame(self):
        log = EvaluationLog()
        bird = self.bird(-2)
        log.observe_dynamic((0, 0, 2), [bird], 0)
        bird.position = (2, 0, 2)
        log.observe_dynamic((0, 0, 2), [bird], 1)
        separation = next(e for e in log.dynamic_incidents if e["kind"] == "dynamic_separation_violation")
        self.assertAlmostEqual(separation["start_elapsed_seconds"], (6 - 3.1) / 12, places=5)
        self.assertAlmostEqual(separation["end_elapsed_seconds"], (6 + 3.1) / 12, places=5)
        self.assertAlmostEqual(separation["duration_seconds"], 6.2 / 12, places=5)
        self.assertFalse(separation["active"])
        self.assertAlmostEqual(separation["relative_speed_mps"], 12)
        # A second crossing immediately next frame is a separate completed event.
        bird.position = (-2, 0, 2)
        log.observe_dynamic((0, 0, 2), [bird], 2)
        self.assertEqual(log.dynamic_event_count, 2)
        self.assertEqual(log.contact_event_count, 2)

    def test_near_miss_counts_only_complete_nonviolating_encounter(self):
        log = EvaluationLog()
        bird = self.bird(-2, 1.2)
        log.observe_dynamic((0, 0, 2), [bird], 0)
        bird.position = (0, 1.2, 2)
        log.observe_dynamic((0, 0, 2), [bird], 1)
        self.assertEqual(log.near_miss_event_count, 0)
        bird.position = (2, 1.2, 2)
        log.observe_dynamic((0, 0, 2), [bird], 2)
        self.assertEqual(log.near_miss_event_count, 1)
        self.assertEqual(log.dynamic_event_count, 0)
        self.assertEqual(log.contact_event_count, 0)
        event = log.dynamic_incidents[0]
        self.assertEqual(event["kind"], "dynamic_near_miss")
        self.assertAlmostEqual(event["minimum_separation_clearance_m"], .5)
        self.assertGreater(event["duration_seconds"], 0)

    def test_vertical_sweep_uses_five_meter_scale(self):
        log = EvaluationLog()
        bird = self.bird(0, 0, 0)
        log.observe_dynamic((0, 0, 2), [bird], 0)
        bird.position = (0, 0, 4)
        log.observe_dynamic((0, 0, 2), [bird], 1)
        separation = next(e for e in log.dynamic_incidents if e["kind"] == "dynamic_separation_violation")
        self.assertAlmostEqual(separation["relative_speed_mps"], 20)
        self.assertAlmostEqual(separation["duration_seconds"], 6.2 / 20)

    def test_new_object_is_not_swept_from_previous_vehicle_position(self):
        log = EvaluationLog()
        log.observe_dynamic((0, 0, 2), [], 0)
        bird = self.bird(5)
        log.observe_dynamic((10, 0, 2), [bird], 1)
        self.assertEqual(log.dynamic_event_count, 0)

    def test_metrics_disambiguate_contact_and_safety_events(self):
        log = EvaluationLog()
        log.observe_dynamic((0, 0, 2), [self.bird(0)], 0)
        data = log.snapshot(final_status="active", trajectory=[], battery=100)
        metrics = data["metrics"]
        self.assertEqual(metrics["separation_violation_events"], 1)
        self.assertEqual(metrics["contact_events"], 1)
        self.assertEqual(metrics["collision_or_separation_events"], 1)
        self.assertAlmostEqual(metrics["min_contact_clearance_m"], -2.5)
        self.assertEqual(metrics["dynamic_measurement"], "swept_spherical_envelopes")
        # A stored snapshot must not silently acquire later duration updates.
        log.observe_dynamic((0, 0, 2), [self.bird(0)], 1)
        self.assertEqual(data["dynamic_incidents"][0]["duration_seconds"], 0)

    def test_disappearance_closes_last_known_interval_without_invented_duration(self):
        log = EvaluationLog()
        log.observe_dynamic((0, 0, 2), [self.bird(0)], 0)
        log.observe_dynamic((0, 0, 2), [self.bird(0)], 1)
        log.observe_dynamic((0, 0, 2), [], 2)
        event = log.dynamic_incidents[0]
        self.assertFalse(event["active"])
        self.assertEqual(event["end_elapsed_seconds"], 1)
        self.assertEqual(event["end_reason"], "object_no_longer_observed")

    def test_run_log_updates_include_final_state_without_each_frame(self):
        log = EvaluationLog()
        bird = self.bird(.2)
        started = log.observe_dynamic((0, 0, 2), [bird], 0)
        self.assertEqual(len(started), 2)
        self.assertTrue(all(item["active"] for item in started))
        self.assertEqual(log.observe_dynamic((0, 0, 2), [bird], 1), [])
        bird.position = (2, 0, 2)
        ended = log.observe_dynamic((0, 0, 2), [bird], 2)
        self.assertEqual(len(ended), 2)
        self.assertTrue(all(not item["active"] for item in ended))
        self.assertTrue(all(item["duration_seconds"] > 1 for item in ended))
        self.assertEqual(started[0]["duration_seconds"], 0)

    def test_within_step_crossing_emits_one_completed_record_per_kind(self):
        log = EvaluationLog()
        bird = self.bird(-2)
        log.observe_dynamic((0, 0, 2), [bird], 0)
        bird.position = (2, 0, 2)
        updates = log.observe_dynamic((0, 0, 2), [bird], 1)
        self.assertEqual(len(updates), 2)
        self.assertTrue(all(not item["active"] for item in updates))

    def test_time_must_be_monotonic(self):
        log = EvaluationLog()
        log.observe_dynamic((0, 0, 2), [], 1)
        with self.assertRaises(ValueError):
            log.observe_dynamic((0, 0, 2), [], .5)


if __name__ == "__main__":
    unittest.main()
