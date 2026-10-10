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


if __name__ == "__main__":
    unittest.main()
