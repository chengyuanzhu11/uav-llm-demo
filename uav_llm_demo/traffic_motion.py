"""Collision-checked, time-parameterized bird routes shared with the renderer."""
import math
import random
import time
from collections import deque

from .safety import SafetyChecker, metric_distance


def sample_motion(model, elapsed):
    segments = model["segments"]
    segment = next((item for item in segments if elapsed < item["start"] + item["duration"]), segments[-1])
    fraction = min(1.0, max(0.0, (elapsed - segment["start"]) / segment["duration"]))
    remaining = 1 - fraction
    controls = segment["controls"]
    position = tuple(remaining ** 3 * controls[0][axis] + 3 * remaining ** 2 * fraction * controls[1][axis]
                     + 3 * remaining * fraction ** 2 * controls[2][axis] + fraction ** 3 * controls[3][axis]
                     for axis in range(3))
    velocity = tuple(3 * (remaining ** 2 * (controls[1][axis] - controls[0][axis])
                         + 2 * remaining * fraction * (controls[2][axis] - controls[1][axis])
                         + fraction ** 2 * (controls[3][axis] - controls[2][axis])) / segment["duration"]
                     for axis in range(3))
    return position, velocity


def curve_safe(checker, controls):
    if any(checker.validate_waypoint(point) for point in controls):
        return False
    lower = tuple(min(point[axis] for point in controls) for axis in range(3))
    upper = tuple(max(point[axis] for point in controls) for axis in range(3))
    return not any(all(lower[axis] <= high[axis] and upper[axis] >= low[axis] for axis in range(3))
                   for low, high, _ in checker.boxes)


def curve_speed_bound(controls):
    model = {"segments": [dict(start=0, duration=1, controls=controls)]}
    peak = max(math.hypot(*(value * scale for value, scale in zip(sample_motion(model, index / 32)[1], (3, 3, 5))))
               for index in range(33))
    derivative_bound = max(math.hypot(*(6 * (controls[index + 2][axis] - 2 * controls[index + 1][axis]
                                            + controls[index][axis]) * scale
                                       for axis, scale in enumerate((3, 3, 5)))) for index in range(2))
    return peak + derivative_bound / 64


class ContinuousTraffic:
    def __init__(self, scenario, *, seed=None, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.rng = random.Random(seed)
        self.models = {}
        self.checkers = {}
        self.graphs = {}
        navigation_by_radius = {}
        for bird in scenario.dynamic_objects:
            if bird.radius not in navigation_by_radius:
                context = scenario.to_dict()
                context["altitude_range"] = dict(min=1.6, max=2.4)
                checker = SafetyChecker(context, clearance=bird.radius * 3 + 0.2)
                edge = bird.radius + 0.1
                checker.lower = (edge, edge, 1.6)
                checker.upper = (scenario.grid_size - 1 - edge, scenario.grid_size - 1 - edge, 2.4)
                nodes = {(east, south): (float(east), float(south), 2.0)
                         for east in range(1, scenario.grid_size - 1)
                         for south in range(1, scenario.grid_size - 1)
                         if not checker.validate_waypoint((east, south, 2.0))}
                if not nodes:
                    raise ValueError("No safe low-altitude spawn exists for dynamic traffic.")
                graph = {}
                for key, point in nodes.items():
                    graph[point] = [nodes[neighbor] for offset in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                                                  (1, 1), (-1, 1), (1, -1), (-1, -1))
                                    if (neighbor := (key[0] + offset[0], key[1] + offset[1])) in nodes
                                    and checker.segment_safe(point, nodes[neighbor])]
                navigation_by_radius[bird.radius] = (checker, graph)
            checker, graph = navigation_by_radius[bird.radius]
            self.checkers[bird.object_id] = checker
            self.graphs[bird.object_id] = graph
            start = self.rng.choice(list(graph))
            self.models[bird.object_id] = dict(segments=[], phase=self.rng.uniform(0, math.tau))
            self._append_route(bird.object_id, start, 0.0)

    def _append_route(self, object_id, start, started):
        checker, graph = self.checkers[object_id], self.graphs[object_id]
        origin = min(graph, key=lambda point: metric_distance(start, point))
        parents, queue = {origin: None}, deque([origin])
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)
        destinations = [point for point in parents if metric_distance(origin, point) >= 9]
        goal = self.rng.choice(destinations or list(parents))
        path, current = [], goal
        while current is not None:
            path.append(current)
            current = parents[current]
        path.reverse()
        compressed, index = [start], 0
        while index < len(path) - 1:
            candidate = len(path) - 1
            while not checker.segment_safe(compressed[-1], path[candidate]):
                candidate -= 1
            compressed.append(path[candidate])
            index = candidate
        if len(compressed) == 1:
            self.models[object_id]["segments"].append(dict(start=started, duration=2.0, controls=[start] * 4))
            return
        for index in range(1, len(compressed) - 1):
            candidate = (*compressed[index][:2], self.rng.uniform(1.7, 2.3))
            if checker.segment_safe(compressed[index - 1], candidate) and checker.segment_safe(candidate, compressed[index + 1]):
                compressed[index] = candidate
        tangents = [(0.0, 0.0, 0.0)] * len(compressed)
        for index in range(1, len(compressed) - 1):
            before, point, after = compressed[index - 1:index + 2]
            length = min(metric_distance(before, point), metric_distance(point, after)) * 0.18
            span = metric_distance(before, after)
            if span:
                tangents[index] = tuple((after[axis] - before[axis]) * length / span for axis in range(3))
        for _ in range(2):
            for index in range(len(compressed) - 1):
                if not curve_safe(checker, self._controls(compressed, tangents, index)):
                    tangents[index] = tangents[index + 1] = (0.0, 0.0, 0.0)
        curves = [self._controls(compressed, tangents, index) for index in range(len(compressed) - 1)]
        speed = self.rng.uniform(2.0, 4.0)
        duration = max(0.5, max(curve_speed_bound(curve) for curve in curves) / speed)
        for controls in curves:
            straight = controls[0] == controls[1] and controls[2] == controls[3]
            if not curve_safe(checker, controls) and not straight:
                raise ValueError("Unsafe traffic curve.")
            if not checker.segment_safe(controls[0], controls[-1]):
                raise ValueError("Unsafe traffic segment.")
            self.models[object_id]["segments"].append(dict(start=started, duration=duration, controls=controls))
            started += duration

    @staticmethod
    def _controls(points, tangents, index):
        start, end = points[index:index + 2]
        return [start, tuple(start[axis] + tangents[index][axis] for axis in range(3)),
                tuple(end[axis] - tangents[index + 1][axis] for axis in range(3)), end]

    def sync(self, environment):
        elapsed = max(0, self.clock() - self.started)
        for bird in environment.dynamic_objects:
            model = self.models[bird.object_id]
            segments = model["segments"]
            last = segments[-1]
            while last["start"] + last["duration"] < elapsed + 8:
                self._append_route(bird.object_id, tuple(last["controls"][-1]), last["start"] + last["duration"])
                last = segments[-1]
            model["segments"] = [item for item in segments if item["start"] + item["duration"] >= elapsed - 1]
            bird.position, bird.velocity = sample_motion(model, elapsed)
            bird.motion_prediction = tuple(p + v for p, v in zip(bird.position, bird.velocity))
        return {"elapsed": elapsed, "models": self.models}
