import random
import unittest
from threading import RLock
from unittest.mock import Mock, patch

from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.safety import SafetyChecker, _intersects
from uav_llm_demo.scenarios import load_scenario


class HeartbeatTest(unittest.TestCase):
    def run_delayed_tick(self, backend):
        service = object.__new__(MissionService)
        service._lock = RLock()
        service._shutdown = Mock()
        service._shutdown.wait.side_effect = [False, True]
        service._running = True
        service.trace = Mock()
        service.trace.snapshot.return_value = {"timing": {}}
        service._time = 0.0
        service._publish_state = Mock()
        service.runtime = Mock(backend_name=backend)
        service.advance = Mock()
        service._fail = Mock()
        with patch('uav_llm_demo.mission_service.time.monotonic', side_effect=[0, 0.45]):
            service._loop()
        return service

    def test_html_delay_continues_with_bounded_step(self):
        with self.assertLogs('uav_llm_demo.mission_service', level='WARNING'):
            service = self.run_delayed_tick('html')
        service._fail.assert_not_called()
        service.advance.assert_called_once_with(0.1)

    def test_airsim_delay_retains_safety_stop(self):
        service = self.run_delayed_tick('airsim')
        service._fail.assert_called_once()
        service.advance.assert_not_called()

    def test_segment_optimization_matches_exhaustive_checks(self):
        checker = SafetyChecker(load_scenario('urban_crossing').to_dict())
        rng = random.Random(42)
        pairs = [(tuple(rng.uniform(-1, 40) if i < 2 else rng.uniform(1, 11)
                        for i in range(3)),
                  tuple(rng.uniform(-1, 40) if i < 2 else rng.uniform(1, 11)
                        for i in range(3))) for _ in range(1000)]
        for lo, hi, _ in checker.boxes[:10]:
            pairs.extend([(lo, lo), (hi, hi), (lo, hi)])
        for a, b in pairs:
            expected = (not checker.validate_waypoint(a) and not checker.validate_waypoint(b)
                        and not any(_intersects(a, b, lo, hi) for lo, hi, _ in checker.boxes))
            self.assertEqual(checker.segment_safe(a, b), expected, (a, b))
        for invalid in [None, (), (float('nan'), 2, 2), (True, 2, 2)]:
            self.assertFalse(checker.segment_safe(invalid, (1, 18, 2)))
