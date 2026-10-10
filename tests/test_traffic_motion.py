import unittest
import math
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import replace
from uav_llm_demo.environment import UAVEnvironment
from uav_llm_demo.safety import metric_distance

from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.scenarios import DEFAULT_SCENARIO
from uav_llm_demo.traffic_motion import ContinuousTraffic, sample_motion


class TrafficMotionTests(unittest.TestCase):
    def test_route_patrols_follow_height_and_do_not_teleport_on_replan(self):
        from uav_llm_demo.agents import MockLLMAgent
        from copy import deepcopy
        scenario = DEFAULT_SCENARIO
        now = [0.0]
        traffic = ContinuousTraffic(scenario, seed=9, clock=lambda: now[0])
        self.assertEqual(len(traffic.models), 8)
        self.assertEqual(len(traffic.patrols), 4)
        self.assertTrue(any(lane[0][2] > 2.4 for lane in traffic.patrols.values()))
        context = scenario.to_dict()
        context.update(position=context['base'], goal=context['target'])
        route = MockLLMAgent().plan(context).waypoints
        before = deepcopy(traffic.models)
        self.assertGreater(traffic.focus_route(scenario.base, route), 0)
        self.assertEqual(traffic.models, before)
        environment = UAVEnvironment(scenario)
        for elapsed in range(0, 80, 2):
            now[0] = elapsed
            traffic.sync(environment)
            for object_id in traffic.patrols:
                segments = traffic.models[object_id]['segments']
                for left, right in zip(segments, segments[1:]):
                    end = left['start'] + left['duration']
                    self.assertAlmostEqual(end, right['start'])
                    a = sample_motion({'segments':[left]}, end)
                    b = sample_motion({'segments':[right]}, right['start'])
                    self.assertLess(metric_distance(a[0], b[0]), 1e-8)
                    self.assertLess(math.dist(a[1], b[1]), 1e-8)
                for segment in segments:
                    self.assertTrue(traffic.checkers[object_id].segment_safe(
                        segment['controls'][0], segment['controls'][-1]))

    def test_idle_ticks_move_birds_but_reads_do_not_advance_world(self):
        service = MissionService(target_seed=12, traffic_seed=12)
        self.addCleanup(service.close)
        before = service.state()
        again = service.state()
        self.assertEqual(before, again)
        for _ in range(80):
            service.advance(.05)
        after = service.state()
        self.assertNotEqual(before["dynamic_objects"], after["dynamic_objects"])
        self.assertEqual(after["step_count"], before["step_count"])
        self.assertEqual(after["flight"]["planner_calls"], before["flight"]["planner_calls"])
        self.assertAlmostEqual(after["simulation_time"], after["traffic_motion"]["elapsed"])

    def test_renderer_trajectory_changes_over_time(self):
        now = [0.0]
        traffic = ContinuousTraffic(DEFAULT_SCENARIO, seed=4, clock=lambda: now[0])
        model = traffic.models[DEFAULT_SCENARIO.dynamic_objects[0].object_id]

        position, _ = sample_motion(model, 1.0)
        predicted, _ = sample_motion(model, 2.0)
        self.assertNotEqual(position, predicted)

    def test_reset_randomizes_motion_model(self):
        first = ContinuousTraffic(DEFAULT_SCENARIO, seed=None)
        second = ContinuousTraffic(DEFAULT_SCENARIO, seed=None)

        self.assertNotEqual(first.models, second.models)

    def test_trajectories_stay_inside_airspace_between_buildings(self):
        traffic = ContinuousTraffic(DEFAULT_SCENARIO, seed=8, clock=lambda: 0.0)
        service = MissionService()
        self.addCleanup(service.close)
        for elapsed in range(0, 120, 5):
            traffic.clock = lambda elapsed=elapsed: float(elapsed)
            traffic.sync(service._environment)
            for bird in service._environment.dynamic_objects:
                self.assertGreaterEqual(bird.position[0], 1.0)
                self.assertLessEqual(bird.position[0], DEFAULT_SCENARIO.grid_size - 2.0)
                self.assertGreaterEqual(bird.position[1], 1.0)
                self.assertLessEqual(bird.position[1], DEFAULT_SCENARIO.grid_size - 2.0)
                checker = traffic.checkers[bird.object_id]
                self.assertGreaterEqual(bird.position[2], checker.lower[2])
                self.assertLessEqual(bird.position[2], checker.upper[2])
                self.assertFalse(traffic.checkers[bird.object_id].validate_waypoint(bird.position))

    def test_seeded_long_routes_clear_buildings_and_join_continuously(self):
        for seed in range(6):
            now = [0.0]
            environment = UAVEnvironment()
            traffic = ContinuousTraffic(DEFAULT_SCENARIO, seed=seed, clock=lambda: now[0])
            for elapsed in range(0, 180, 3):
                now[0] = elapsed
                traffic.sync(environment)
                for bird in environment.dynamic_objects:
                    model = traffic.models[bird.object_id]
                    self.assertEqual(bird.motion_prediction, tuple(p + v for p, v in zip(bird.position, bird.velocity)))
                    for segment in model["segments"]:
                        for sample in range(21):
                            position, velocity = sample_motion({"segments": [segment]}, segment["start"] + segment["duration"] * sample / 20)
                            self.assertFalse(traffic.checkers[bird.object_id].validate_waypoint(position))
                            self.assertLessEqual(math.hypot(*(value * scale for value, scale in zip(velocity, (3, 3, 5)))), 4.001)
                    for before, after in zip(model["segments"], model["segments"][1:]):
                        end = before["start"] + before["duration"]
                        left = sample_motion({"segments": [before]}, end)
                        right = sample_motion({"segments": [after]}, after["start"])
                        self.assertAlmostEqual(end, after["start"])
                        self.assertLess(metric_distance(left[0], right[0]), 1e-8)
                        self.assertLess(math.dist(left[1], right[1]), 1e-8)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser sampler parity")
    def test_browser_sampler_matches_backend(self):
        traffic = ContinuousTraffic(DEFAULT_SCENARIO, seed=1)
        model = next(iter(traffic.models.values()))
        module_url = (Path(__file__).resolve().parents[1] / "uav_llm_demo/static/traffic.js").as_uri()
        script = f"import {{sampleTraffic}} from {json.dumps(module_url)}; console.log(JSON.stringify([0,1,5,10,10000].map(time => sampleTraffic({json.dumps(model)},time))));"
        results = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", script], text=True))
        for elapsed, result in zip([0, 1, 5, 10, 10000], results):
            position, velocity = sample_motion(model, elapsed)
            self.assertLess(math.dist(position, result["position"]), 1e-10)
            self.assertLess(math.dist(velocity, result["velocity"]), 1e-10)

    def test_birds_can_enter_base_and_target_when_static_airspace_is_safe(self):
        scenario = replace(DEFAULT_SCENARIO, buildings=(), urban_parcels=())
        traffic = ContinuousTraffic(scenario, seed=3)
        for bird in scenario.dynamic_objects:
            if bird.object_id not in traffic.graphs:
                continue
            for terminal in (scenario.base, scenario.target):
                overhead = (*terminal[:2], 2.0)
                self.assertFalse(traffic.checkers[bird.object_id].validate_waypoint(overhead))
                self.assertIn(overhead, traffic.graphs[bird.object_id])

    def test_target_changes_only_modify_route_aware_birds(self):
        first = ContinuousTraffic(DEFAULT_SCENARIO, seed=8, clock=lambda: 0)
        second = ContinuousTraffic(replace(DEFAULT_SCENARIO, target=(8, 10, 2)), seed=8, clock=lambda: 0)
        ambient = [bird.object_id for bird in DEFAULT_SCENARIO.dynamic_objects if bird.kind == "bird"]
        self.assertEqual({key:first.models[key] for key in ambient}, {key:second.models[key] for key in ambient})
        self.assertEqual({key:first.graphs[key] for key in ambient}, {key:second.graphs[key] for key in ambient})
        self.assertNotEqual(first.patrols, second.patrols)

    def test_isolated_safe_bird_waits_without_crossing_restricted_region(self):
        zone = dict(min=dict(x=2, y=0, z=0), max=dict(x=4, y=4, z=5))
        second_zone = dict(min=dict(x=0, y=2, z=0), max=dict(x=4, y=4, z=5))
        scenario = replace(DEFAULT_SCENARIO, grid_size=5, buildings=(), no_fly_zones=(zone, second_zone),
                           dynamic_objects=DEFAULT_SCENARIO.dynamic_objects[:1])
        traffic = ContinuousTraffic(scenario, seed=3)
        model = next(iter(traffic.models.values()))
        for segment in model["segments"]:
            for elapsed in (segment["start"], segment["start"] + segment["duration"] / 2):
                point, velocity = sample_motion(model, elapsed)
                self.assertFalse(traffic.checkers[scenario.dynamic_objects[0].object_id].validate_waypoint(point))
                self.assertEqual(velocity, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
