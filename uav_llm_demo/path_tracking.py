"""Ordered, continuous tracking of task paths whose geometry is part of the task."""
from bisect import bisect_right
from math import isfinite

from .motion import SCALE, distance, norm, segment_deviation


class PathFollower:
    """Track only the next local path interval, including repeated closed loops.

    Coordinates use the shared grid; progress and speed use metres. A safety
    detour freezes progress, then returns to the last verified path position.
    The caller must use the same velocity controller for avoidance and flight.
    """
    CORRIDOR_M = 0.8
    REJOIN_M = 0.2
    END_DISTANCE_M = 0.18
    END_SPEED_MPS = 0.02

    def __init__(self, points, speed_mps):
        if isinstance(speed_mps, bool) or not isinstance(speed_mps, (int, float)) or not isfinite(speed_mps) or not 0 < speed_mps <= 4:
            raise ValueError("Task path speed must be finite and in (0, 4] m/s.")
        if not isinstance(points, (list, tuple)):
            raise ValueError("Task path points must be an ordered list or tuple.")
        converted = []
        for point in points:
            if (not isinstance(point, (list, tuple)) or len(point) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in point)):
                raise ValueError("Task path points must contain three finite coordinates.")
            point = tuple(float(v) for v in point)
            if not converted or distance(converted[-1], point) > 1e-8:
                converted.append(point)
        if len(converted) < 2:
            raise ValueError("Task path requires at least two distinct consecutive points.")
        self.points = tuple(converted)
        self.speed_mps = float(speed_mps)
        lengths = [distance(a, b) for a, b in zip(self.points, self.points[1:])]
        self._lengths = tuple(lengths)
        cumulative = [0.0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        self._cumulative = tuple(cumulative)
        self.total_length_m = cumulative[-1]
        self.progress_m = 0.0
        self.completed = False
        self.recovering = False
        self._previous_position = None

    @property
    def segment_index(self):
        return min(len(self._lengths) - 1,
                   bisect_right(self._cumulative, self.progress_m + 1e-9) - 1)

    @property
    def rejoin_point(self):
        return self.point_at(self.progress_m)

    @property
    def remaining_distance_m(self):
        return max(0.0, self.total_length_m - self.progress_m)

    @property
    def tangent(self):
        index = self.segment_index
        return tuple((self.points[index + 1][axis] - self.points[index][axis]) * SCALE[axis]
                     / self._lengths[index] for axis in range(3))

    def point_at(self, progress):
        progress = max(0.0, min(self.total_length_m, progress))
        index = min(len(self._lengths) - 1,
                    bisect_right(self._cumulative, progress) - 1)
        fraction = (progress - self._cumulative[index]) / self._lengths[index]
        return tuple(self.points[index][axis] + fraction *
                     (self.points[index + 1][axis] - self.points[index][axis]) for axis in range(3))

    def target(self, state):
        if self.completed:
            return self.points[-1]
        anchor = self.point_at(self.progress_m)
        if self.recovering or (self._previous_position is None and distance(state.position, anchor) > self.REJOIN_M):
            return anchor
        # The existing acceleration-limited position controller tracks a moving
        # reference without stopping at every sample. Limit chord cutting at
        # curves/corners; requested speed remains an upper bound.
        lookahead = min(self.remaining_distance_m,
                        max(0.6, self.speed_mps ** 2 / 2.0 + 0.4))
        while lookahead > 0.35:
            target = self.point_at(self.progress_m + lookahead)
            first = bisect_right(self._cumulative, self.progress_m)
            last = bisect_right(self._cumulative, self.progress_m + lookahead)
            if all(segment_deviation(point, anchor, target) <= 0.10
                   for point in self.points[first:last]):
                break
            lookahead *= 0.7
        return self.point_at(self.progress_m + lookahead)

    def update(self, state, dt, avoiding=False):
        if not isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("Path follower dt must be in (0, 0.2] seconds.")
        previous = self._previous_position
        self._previous_position = tuple(state.position)
        if self.completed:
            return False
        anchor = self.point_at(self.progress_m)
        if avoiding:
            self.recovering = True
            return False
        if self.recovering:
            if distance(state.position, anchor) <= self.REJOIN_M and norm(state.velocity) <= 0.25:
                self.recovering = False
            return False
        if previous is None:
            self.recovering = distance(state.position, anchor) > self.REJOIN_M
            return False
        moved = distance(previous, state.position)
        # Reject discontinuous teleports; neither a reset nor arriving at the
        # shared start/end of a closed path may be credited as completed laps.
        plausible_move = max(self.speed_mps, norm(state.velocity), 0.25) * dt + 0.15
        if moved > plausible_move:
            self.recovering = True
            return False
        before = self.progress_m
        if moved > 1e-8:
            index = self.segment_index
            advance_limit = min(self.total_length_m, before + moved * 1.25 + 1e-6)
            while index < len(self._lengths):
                start, end = self.points[index:index + 2]
                direction = tuple((end[axis] - start[axis]) * SCALE[axis] for axis in range(3))
                offset = tuple((state.position[axis] - start[axis]) * SCALE[axis] for axis in range(3))
                fraction = sum(a * b for a, b in zip(offset, direction)) / self._lengths[index] ** 2
                if segment_deviation(state.position, start, end) > self.CORRIDOR_M:
                    self.recovering = True
                    break
                # A right-angle turn can approach its vertex asymptotically:
                # verify the vertex only inside the same 20 cm arrival gate.
                reached_gate = ((1.0 - fraction) * self._lengths[index] <= self.END_DISTANCE_M
                                and distance(state.position, end) <= self.REJOIN_M)
                if reached_gate:
                    fraction = 1.0
                    # A physically reached vertex is verified; do not strand
                    # progress just before a right angle due to rounding.
                    advance_limit = max(advance_limit, self._cumulative[index + 1])
                candidate = self._cumulative[index] + max(0.0, min(1.0, fraction)) * self._lengths[index]
                self.progress_m = max(self.progress_m, min(candidate, advance_limit))
                if fraction < 1.0 or self._cumulative[index + 1] >= advance_limit - 1e-9:
                    break
                index += 1
        # Endpoint completion tolerates the same settling error as ordinary
        # navigation, but only after every preceding interval was visited.
        if (self.segment_index == len(self._lengths) - 1
                and self.remaining_distance_m <= self.END_DISTANCE_M
                and distance(state.position, self.points[-1]) <= self.END_DISTANCE_M
                and norm(state.velocity) <= self.END_SPEED_MPS and not self.recovering):
            self.progress_m = self.total_length_m
            self.completed = True
        return self.progress_m > before

    def snapshot(self):
        return dict(progress_m=round(self.progress_m, 3),
                    total_length_m=round(self.total_length_m, 3),
                    remaining_distance_m=round(self.remaining_distance_m, 3),
                    fraction=self.progress_m / self.total_length_m,
                    segment_index=self.segment_index, completed=self.completed,
                    recovering=self.recovering, speed_mps=self.speed_mps)
