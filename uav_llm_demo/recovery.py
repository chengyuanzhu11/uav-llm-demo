"""Bounded failure memory; retries have no attempt limit or planner fallback."""
from collections import deque
from copy import deepcopy
import hashlib
import json
import re


class RecoveryMemory:
    def __init__(self):
        self.failures = deque(maxlen=8)
        self.provider_streak = 0
        self.repeat_count = 0
        self._last_signature = None
        self.last_action = None
        self.retry_delay = 0.0
        self.accepted_after_failures = 0
        self._unresolved_failure = False

    def provider_failed(self, message, position, now, *, invalid_response=False):
        self._unresolved_failure = True
        self.provider_streak += 1
        self.retry_delay = min(30.0, 2.0 ** min(self.provider_streak, 5))
        self.last_action = 'repair_response' if invalid_response else 'refresh_observation_and_retry_provider'
        self.failures.append(dict(kind='invalid_response' if invalid_response else 'provider',
                                  message=message, position=position, time=now))
        return self.retry_delay

    def route_failed(self, plan, errors, position, now):
        self._unresolved_failure = True
        self.provider_streak = 0
        points = list(plan.waypoints) if plan else []
        signature = hashlib.sha256(json.dumps([points, errors], sort_keys=True).encode()).hexdigest()
        self.repeat_count = self.repeat_count + 1 if signature == self._last_signature else 1
        self._last_signature = signature
        segments = []
        for error in errors:
            match = re.match(r'Segment (\d+)', error)
            if match and 0 <= (index := int(match[1])) < len(points):
                segments.append(dict(start=position if index == 0 else points[index - 1],
                                     end=points[index]))
        self.last_action = 'choose_different_corridor' if self.repeat_count >= 2 else 'repair_rejected_segments'
        self.retry_delay = min(10.0, 2.0 * self.repeat_count)
        self.failures.append(dict(kind='route', errors=errors[:12], rejected_segments=segments[:12],
                                  signature=signature, repeats=self.repeat_count, position=position, time=now))
        return self.retry_delay

    def stalled(self, position, remaining_distance, now):
        self._unresolved_failure = True
        self.last_action = 'refresh_observation_and_change_route'
        self.failures.append(dict(kind='no_progress', position=position,
                                  remaining_distance_m=remaining_distance, time=now))

    def accepted(self):
        if self._unresolved_failure:
            self.accepted_after_failures += 1
        self._unresolved_failure = False
        self.provider_streak = 0
        self.repeat_count = 0
        self.retry_delay = 0.0
        self.last_action = 'track_accepted_route'
        self._last_signature = None

    def feedback(self, current):
        if not self.failures:
            return current
        return (current + '\nRecovery memory (past failures, not permanent no-fly zones): '
                + json.dumps(list(self.failures)[-4:], separators=(',', ':'))
                + '\nNext action: ' + str(self.last_action)
                + '. Use the refreshed position and goal. Do not repeat rejected geometry. '
                  'Preserve already completed mission stages; do not claim execution success.')

    def snapshot(self):
        return dict(recent_failures=deepcopy(list(self.failures)), provider_streak=self.provider_streak,
                    repeated_route_failures=self.repeat_count, next_action=self.last_action,
                    retry_delay_seconds=self.retry_delay, accepted_after_failures=self.accepted_after_failures)
