import unittest
from types import SimpleNamespace

from tests.test_avoidance import encounter_service
from uav_llm_demo.avoidance import Rollout
from uav_llm_demo.avoidance_evidence import AvoidanceEvidence
from uav_llm_demo.environment import DynamicFlyingObject


class AvoidanceEvidenceTest(unittest.TestCase):
    def test_true_swept_clearance_is_independent_of_forecast(self):
        evidence = AvoidanceEvidence()
        bird = DynamicFlyingObject("crossing", "bird", (-2, 0, 2), (0, 0, 0))
        evidence.measure((0, 0, 2), [bird], 0)
        rollout = Rollout(-100, 0, (1, 0, 2), ((0, 0, 2), (1, 0, 2)), (("crossing", -100),))
        evidence.record(.1, True, False, rollout, rollout, .3)
        self.assertIsNone(evidence.snapshot()["episode"]["min_separation_clearance_m"])
        bird.position = (2, 0, 2)
        evidence.measure((0, 0, 2), [bird], .1)
        # Both endpoints are 6 m away but the relative swept segment crosses zero.
        self.assertAlmostEqual(evidence.snapshot()["episode"]["min_separation_clearance_m"], -3.1)
        self.assertEqual(evidence.snapshot()["threats"][0]["predicted_clearance_m"], -100)
        evidence.record(.2, False, False, rollout, None, .3)
        self.assertEqual(evidence.snapshot()["episode"]["status"], "cleared")
        evidence.interrupt(.3)
        self.assertEqual(evidence.snapshot()["episode"]["status"], "cleared")

    def test_real_control_rollouts_causal_ids_and_episode_end_are_published(self):
        service = encounter_service()
        self.addCleanup(service.close)
        events = []
        service.trace.writer = SimpleNamespace(emit=events.append, dropped=0, errors=0)
        captured = None
        for _ in range(140):
            service.advance(.05)
            view = service.state()["flight"]["avoidance"]["visualization"]
            if view["active"] and view["selected_path"]:
                captured = view
        self.assertIsNotNone(captured)
        self.assertNotIn("visualization", service._context()["local_avoidance"])
        self.assertNotIn("visualization", service.avoidance.snapshot())
        self.assertEqual(len(captured["selected_path"]), 21)
        self.assertIn("incoming", captured["episode"]["object_ids"])
        self.assertGreater(captured["episode"]["min_separation_clearance_m"], 0)
        self.assertTrue(all(set(point) == set("xyz") for point in captured["selected_path"]))
        ended = [e for e in events if e["event"] == "avoidance_cleared"]
        self.assertTrue(ended)
        self.assertEqual(ended[-1]["episode"]["status"], "cleared")
        self.assertGreater(ended[-1]["episode"]["min_separation_clearance_m"], 0)
        self.assertEqual(service._planner_calls, 0)

    def test_runtime_fault_preserves_interrupted_episode_then_reset_clears(self):
        service = encounter_service()
        self.addCleanup(service.close)
        for _ in range(30):
            service.advance(.05)
            if service.avoidance.active:
                break
        self.assertTrue(service.avoidance.active)
        before = service.avoidance.evidence.sequence
        service._fail("Runtime unavailable")
        view = service.avoidance.evidence.snapshot()
        self.assertFalse(view["active"])
        self.assertEqual(view["episode"]["status"], "interrupted")
        self.assertGreater(view["sequence"], before)
        service.reset()
        view = service.state()["flight"]["avoidance"]["visualization"]
        self.assertEqual(view["sequence"], 0)
        self.assertIsNone(view["episode"])

    def test_start_and_interrupt_in_one_tick_preserve_event_states(self):
        evidence = AvoidanceEvidence()
        rollout = Rollout(-1, 0, (0, 0, 2), (), (("bird", -1),))
        evidence.record(.1, True, False, rollout, rollout, .3)
        evidence.interrupt(.1)
        started, stopped = evidence.events()
        self.assertEqual(started[0], "avoidance_started")
        self.assertEqual(started[1]["status"], "active")
        self.assertEqual(started[1]["object_ids"], ["bird"])
        self.assertIsNone(started[1]["ended_at"])
        self.assertEqual(stopped[1]["status"], "interrupted")

    def test_reset_and_close_finish_the_old_episode_in_its_own_run(self):
        for action in ("reset", "close"):
            with self.subTest(action=action):
                service = encounter_service()
                self.addCleanup(service.close)
                records = []
                service.trace.writer = SimpleNamespace(emit=records.append, dropped=0, errors=0)
                for _ in range(30):
                    service.advance(.05)
                    if service.avoidance.active:
                        break
                self.assertTrue(service.avoidance.active)
                old_run = service.trace.run_id
                getattr(service, action)()
                ends = [e for e in records if e["event"] == "avoidance_interrupted"]
                self.assertEqual(len(ends), 1)
                self.assertEqual(ends[0]["run_id"], old_run)
                self.assertEqual(ends[0]["episode"]["status"], "interrupted")
