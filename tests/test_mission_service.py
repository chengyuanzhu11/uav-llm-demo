from tests.scenario_fixtures import empty_airspace
import time
import unittest
from threading import Event
from unittest.mock import Mock, patch

from uav_llm_demo.agents import DeepSeekAgent, MockLLMAgent, PlannerProviderError, RoutePlan
from uav_llm_demo.environment import BASE
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import SCALE, VehicleState, distance
from uav_llm_demo.runtime import RuntimeUnavailableError
from uav_llm_demo.safety import SafetyChecker
from uav_llm_demo.traffic_motion import ContinuousTraffic


class RecordingRuntime:
    def __init__(self, backend_name="html"):
        self.backend_name = backend_name
        self.vehicle = VehicleState(tuple(float(value) for value in BASE))
        self.drive_calls = 0
        self.velocity_commands = []
        self.closed = False
        self.reset_calls = []
        self.scene = None

    def configure_scene(self, scenario):
        self.scene = scenario

    def reset(self, base=BASE):
        self.reset_calls.append(tuple(base))
        self.vehicle = VehicleState(tuple(float(value) for value in base))
        return self.status()

    def read_state(self):
        return self.vehicle

    def drive(self, velocity, dt):
        previous = self.vehicle
        self.velocity_commands.append(tuple(velocity))
        self.drive_calls += 1
        position = tuple(
            previous.position[index]
            + (previous.velocity[index] + velocity[index]) * dt / (2 * SCALE[index])
            for index in range(3)
        )
        acceleration = tuple(
            (velocity[index] - previous.velocity[index]) / dt for index in range(3)
        )
        self.vehicle = VehicleState(position, tuple(velocity), acceleration)
        return self.vehicle

    def stop(self):
        self.vehicle = VehicleState(self.vehicle.position)

    def status(self):
        return {
            "backend": self.backend_name,
            "connected": True,
            "supports_wind": self.backend_name == "airsim",
            "source": self.backend_name,
        }

    def close(self):
        self.closed = True


class FailingRuntime(RecordingRuntime):
    def drive(self, velocity, dt):
        raise RuntimeUnavailableError("test execution failed")


def pump(service, predicate, limit=300):
    for _ in range(limit):
        time.sleep(0.001)
        service.advance(service.CONTROL_DT)
        state = service.state()
        if predicate(state):
            return state
    return service.state()


class MissionServiceTest(unittest.TestCase):
    def test_plan_origin_is_published_fixed_and_reset(self):
        service = MissionService()
        try:
            self.assertIsNone(service.state()["flight"]["plan_start"])
            start = service._environment.position
            service.plan(background=False)
            planned = pump(service, lambda item: item["flight"]["plan_id"] > 0)
            self.assertGreater(planned["flight"]["plan_id"], 0)
            expected = dict(zip("xyz", start))
            self.assertEqual(service.state()["flight"]["plan_start"], expected)
            self.assertTrue(service.state()["flight"]["waypoints"])
            service.waypoints.segment_start = (5.0, 6.0, 2.0)
            service._publish_state()
            self.assertEqual(service.state()["flight"]["plan_start"], expected)
            service._reset_control()
            self.assertIsNone(service.state()["flight"]["plan_start"])
        finally:
            service.close()

    def test_return_recovers_after_more_than_three_mixed_planning_failures(self):
        class FlakyUnsafeDeepSeek(MockLLMAgent):
            backend_name = "deepseek"

            def __init__(self):
                self.calls = 0

            def plan(self, context, feedback=""):
                self.calls += 1
                if self.calls <= 2:
                    raise PlannerProviderError("DeepSeek network request failed (connection).", retryable=True)
                if self.calls <= 4:
                    building = context["buildings"][0]
                    goal = context["goal"]
                    return RoutePlan(((building["x"], building["y"], 2.0),
                                      (goal["x"], goal["y"], goal["z"])),
                                     "Unsafe proposal", "low", "Reach base")
                return super().plan(context, feedback)

        agent = FlakyUnsafeDeepSeek()
        runtime = RecordingRuntime()
        service = MissionService(agent=agent, runtime=runtime)
        self.addCleanup(service.close)
        target = service._environment.target
        runtime.reset(target)
        service._vehicle = runtime.read_state()
        service._environment.position = target
        service._environment.phase = "returning_home"
        with patch.object(service.avoidance, "predictions", return_value=[]):
            service.start(background=False)
            state = pump(service, lambda item: item["flight"]["plan_id"] > 0 or bool(item["flight"]["error"]))
        self.assertEqual(agent.calls, 5)
        self.assertIsNone(state["flight"]["error"])
        self.assertEqual(state["flight"]["plan_id"], 1)
        self.assertEqual(state["phase"], "returning_home")
        self.assertTrue(state["flight"]["running"])
        self.assertEqual(state["agent_backend"], "deepseek")
        self.assertTrue(all(item["decision"]["agent_backend"] == "deepseek" for item in state["history"]))
        self.assertFalse(state["history"][0]["decision"]["safety_intervened"])
        self.assertTrue(state["history"][1]["decision"]["safety_intervened"])
        self.assertEqual(SafetyChecker(service._context()).validate_path(
            target, service.waypoints.points), [])
        self.assertEqual(state["evaluation"]["events"][-1]["agent_backend"], "deepseek")
        with patch.object(service.avoidance, "predictions", return_value=[]):
            for _ in range(20):
                service.advance(service.CONTROL_DT)
        self.assertGreater(distance(target, service._environment.position), 0.001)

    def test_unsafe_routes_keep_retrying_until_paused(self):
        class InvalidDeepSeek(MockLLMAgent):
            backend_name = "deepseek"

            def plan(self, context, feedback=""):
                raise ValueError("Unsafe route")

        service = MissionService(agent=InvalidDeepSeek(), runtime=RecordingRuntime())
        self.addCleanup(service.close)
        service.start(background=False)
        state = pump(service, lambda item: len(item["history"]) >= 5)
        self.assertIsNone(state["flight"]["error"])
        self.assertEqual(state["flight"]["plan_id"], 0)
        self.assertTrue(state["flight"]["running"])
        self.assertTrue(state["flight"]["planning"])
        self.assertTrue(state["flight"]["safety_armed"])
        self.assertEqual(service.waypoints.points, ())
        self.assertTrue(all(item["decision"]["agent_backend"] == "deepseek" for item in state["history"]))
        paused = service.stop_runtime()
        calls = paused["flight"]["planner_calls"]
        paused = pump(service, lambda item: False, limit=100)
        self.assertEqual(paused["flight"]["planner_calls"], calls)
        self.assertFalse(paused["flight"]["planning"])
        self.assertTrue(paused["flight"]["safety_armed"])

    def test_authentication_failure_is_specific_and_not_retried(self):
        service = MissionService(agent=DeepSeekAgent(api_key="dummy-test-credential"),
                                 scenario=empty_airspace())
        self.addCleanup(service.close)
        response = Mock(status_code=401, text="dummy-test-credential")
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response) as post:
            service.start(background=False)
            failed = pump(service, lambda state: bool(state["flight"]["error"]))
        self.assertEqual(post.call_count, 1)
        self.assertIn("HTTP 401", failed["flight"]["error"])
        self.assertNotIn("dummy-test-credential", str(failed))
        self.assertTrue(failed["history"][-1]["decision"]["provider_error"])
        self.assertFalse(failed["history"][-1]["decision"]["safety_intervened"])
        self.assertEqual(failed["metrics"]["agent_errors"], 1)
        self.assertEqual(failed["metrics"]["collision_or_separation_events"], 0)

    def test_transient_provider_failure_retries_with_backoff(self):
        service = MissionService(agent=DeepSeekAgent(api_key="dummy-test-credential"),
                                 scenario=empty_airspace())
        self.addCleanup(service.close)
        response = Mock(status_code=503, text="dummy-test-credential")
        submitted = []
        def fail_request(*args, **kwargs):
            submitted.append(service._time)
            return response
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=fail_request):
            service.start(background=False)
            waiting = pump(service, lambda state: len(state["history"]) >= 5, limit=800)
            calls = len(submitted)
            service.reset()
            reset = pump(service, lambda state: False, limit=100)
        self.assertEqual(calls, 5)
        self.assertTrue(all(b - a >= delay for a, b, delay in zip(submitted, submitted[1:], (2, 4, 8, 16))))
        self.assertIsNone(waiting["flight"]["error"])
        self.assertEqual(waiting["metrics"]["agent_errors"], 5)
        self.assertTrue(waiting["flight"]["planning"])
        self.assertTrue(waiting["flight"]["safety_armed"])
        self.assertEqual(len(submitted), calls)
        self.assertFalse(reset["flight"]["planning"])
        self.assertFalse(reset["flight"]["safety_armed"])

    def test_pause_and_reset_discard_late_results_during_continuous_retry(self):
        for action in ("pause", "reset"):
            with self.subTest(action=action):
                entered, release = Event(), Event()

                class RetryThenWaitingPlanner(MockLLMAgent):
                    def __init__(self):
                        self.calls = 0

                    def plan(self, context, feedback=""):
                        self.calls += 1
                        if self.calls <= 3:
                            raise PlannerProviderError("Transient network failure.", retryable=True)
                        entered.set()
                        release.wait(5)
                        return super().plan(context, feedback)

                service = MissionService(agent=RetryThenWaitingPlanner(), scenario=empty_airspace())
                self.addCleanup(service.close)
                self.addCleanup(release.set)
                service.start(background=False)
                pump(service, lambda state: entered.is_set())
                self.assertTrue(entered.is_set())
                if action == "pause":
                    state = service.stop_runtime()
                else:
                    state = service.reset()
                self.assertFalse(state["flight"]["planning"])
                self.assertFalse(state["flight"]["running"])
                calls = state["flight"]["planner_calls"]
                release.set()
                state = pump(service, lambda state: service._future is None)
                self.assertEqual(state["flight"]["planner_calls"], calls)
                self.assertEqual(state["flight"]["plan_id"], 0)
                self.assertEqual(service.waypoints.points, ())
                self.assertIsNone(state["flight"]["error"])

    def short_mission(self, backend="html", agent=None):
        service = MissionService(agent=agent, runtime=RecordingRuntime(backend))
        self.addCleanup(service.close)
        risk = patch.object(service.avoidance, "predictions", return_value=[])
        risk.start()
        self.addCleanup(risk.stop)
        service.update_goal({"x": 3, "y": 18, "z": 2})
        service.start(background=False)
        return service

    def test_round_trip_completes_only_at_base_for_both_runtimes(self):
        for backend in ("html", "airsim"):
            with self.subTest(runtime=backend):
                service = self.short_mission(backend)
                returning = pump(service, lambda state: state["phase"] == "returning_home")
                self.assertEqual(returning["phase"], "returning_home")
                self.assertEqual(returning["status"], "active")
                self.assertTrue(returning["flight"]["running"])
                self.assertEqual(returning["observation"]["goal"], returning["base"])
                self.assertIn("Now return safely to base", service._context()["task"])
                outbound_path = list(returning["continuous_path"])
                completed = pump(service, lambda state: state["status"] == "completed", limit=800)
                self.assertEqual(completed["phase"], "completed")
                self.assertLess(distance(tuple(completed["uav"].values()), BASE), 0.18)
                self.assertLessEqual(completed["flight"]["speed_mps"], 0.02)
                self.assertFalse(completed["flight"]["running"])
                self.assertEqual(completed["flight"]["planner_calls"], 2)
                self.assertEqual(completed["history"][0]["observation"]["goal"], completed["base"])
                self.assertEqual(completed["continuous_path"][:len(outbound_path)], outbound_path)
                self.assertEqual(completed["observation"]["goal_label"], "base")
                service.start(background=False)
                for _ in range(20):
                    service.advance(0.05)
                self.assertEqual(service.state()["flight"]["planner_calls"], 2)

    def test_pause_at_return_transition_then_resume_and_reset(self):
        service = self.short_mission()
        returning = pump(service, lambda state: state["phase"] == "returning_home")
        service.stop_runtime()
        for _ in range(30):
            service.advance(0.05)
        paused = service.state()
        self.assertEqual(paused["phase"], "returning_home")
        self.assertFalse(paused["flight"]["running"])
        self.assertEqual(paused["flight"]["planner_calls"], returning["flight"]["planner_calls"])
        service.start(background=False)
        finished = pump(service, lambda state: state["status"] == "completed", limit=800)
        self.assertEqual(finished["status"], "completed")
        reset = service.reset()
        self.assertEqual(reset["phase"], "fly_to_target")
        self.assertEqual(reset["observation"]["goal"], reset["target"])
        self.assertEqual(reset["flight"]["planner_calls"], 0)

    def test_failed_return_plan_does_not_mark_mission_successful(self):
        class RejectReturnPlanner(MockLLMAgent):
            def plan(self, context, feedback=""):
                if context["phase"] == "returning_home":
                    raise ValueError("No safe return route.")
                return super().plan(context, feedback)
        service = self.short_mission(agent=RejectReturnPlanner())
        waiting = pump(service, lambda state: sum("No safe return route" in item["feedback"]
                                                 for item in state["history"]) >= 5, limit=600)
        self.assertIn("No safe return route", waiting["history"][0]["feedback"])
        self.assertIsNone(waiting["flight"]["error"])
        self.assertEqual(waiting["phase"], "returning_home")
        self.assertNotEqual(waiting["status"], "completed")
        self.assertFalse(waiting["metrics"]["success"])
        self.assertTrue(waiting["flight"]["running"])
        self.assertTrue(waiting["flight"]["planning"])

    def test_avoidance_does_not_read_future_traffic_curve(self):
        service = MissionService(scenario=empty_airspace())
        self.addCleanup(service.close)
        service._traffic.clock = lambda: 0
        service._traffic.started = 0
        bird = service._environment.dynamic_objects[0]
        bird.position = (5, 10, 2)
        bird.velocity = (0, 1, 0)
        service._environment.realtime_traffic = True
        service._traffic.models[bird.object_id] = dict(segments=[dict(start=0, duration=1,
            controls=[(5, 10, 2), (5, 12, 2), (5, 10, 2), (5, 5, 2)])])
        service.avoidance.observe([dict(id=bird.object_id, position=bird.position,
            velocity=bird.velocity, observed_at=0, radius_m=2.1)], 0)
        tracks = service.avoidance.predictions(VehicleState((5, 8, 2)), 0)
        from uav_llm_demo.safety import SafetyChecker
        rollout = service.avoidance.rollout(VehicleState((5, 8, 2)), None, service.controller,
                                            SafetyChecker(service._context()), tracks)
        self.assertGreater(rollout.clearance, 0)
        self.assertEqual(tracks[0].positions[10], (5, 11.5, 2))
        service._traffic.models.clear()
        self.assertEqual(tracks, service.avoidance.predictions(VehicleState((5, 8, 2)), 0))

    def test_traffic_moves_while_planner_is_waiting(self):
        release = Event()
        class WaitingPlanner(MockLLMAgent):
            def plan(self, context, feedback=""):
                release.wait(3)
                return super().plan(context, feedback)
        service = MissionService(agent=WaitingPlanner())
        self.addCleanup(service.close)
        self.addCleanup(release.set)
        service._traffic = ContinuousTraffic(service._environment.scenario, seed=12, clock=lambda: service._time)
        service.start(background=False)
        before = service.state()
        for _ in range(30):
            service.advance(0.05)
        after = service.state()
        self.assertTrue(after["flight"]["planning"])
        self.assertEqual(before["uav"], after["uav"])
        self.assertNotEqual(before["dynamic_objects"][0]["position"], after["dynamic_objects"][0]["position"])
    def test_defaults_to_html_runtime(self):
        service = MissionService(agent=MockLLMAgent())
        self.addCleanup(service.close)

        state = service.state()

        self.assertEqual(state["runtime_backend"], "html")
        self.assertTrue(state["runtime"]["connected"])
        self.assertEqual(state["flight"]["planner_calls"], 0)
        self.assertEqual(state["flight"]["waypoints"], [])

    def test_planner_is_called_once_and_controller_moves_continuously(self):
        runtime = RecordingRuntime()
        service = MissionService(agent=MockLLMAgent(), runtime=runtime)
        self.addCleanup(service.close)

        service.start(background=False)
        planned = pump(service, lambda state: state["flight"]["plan_id"] == 1)
        self.assertEqual(planned["flight"]["planner_calls"], 1)
        self.assertGreater(len(planned["flight"]["waypoints"]), 0)

        before = tuple(planned["uav"].values())
        moved = pump(service, lambda state: state["flight"]["speed_mps"] > 0.1)
        after = tuple(moved["uav"].values())
        self.assertNotEqual(before, after)
        self.assertEqual(moved["flight"]["planner_calls"], 1)
        self.assertGreater(runtime.drive_calls, 0)
        self.assertTrue(any(abs(value) > 0 for value in runtime.velocity_commands[-1]))

    def test_pause_decelerates_without_resetting_progress(self):
        runtime = RecordingRuntime()
        service = MissionService(agent=MockLLMAgent(), runtime=runtime)
        self.addCleanup(service.close)
        service.start(background=False)
        running = pump(service, lambda state: state["flight"]["speed_mps"] > 0.1)
        before = running["uav"]
        calls = running["flight"]["planner_calls"]

        paused = service.stop_runtime()
        self.assertFalse(paused["flight"]["running"])
        settled = pump(service, lambda state: state["flight"]["speed_mps"] <= 0.02)

        self.assertEqual(settled["flight"]["planner_calls"], calls)
        self.assertNotEqual(settled["uav"], settled["base"])
        self.assertNotEqual(before, settled["uav"])

    def test_runtime_failure_stops_without_committing_a_control_tick(self):
        service = MissionService(agent=MockLLMAgent(), runtime=FailingRuntime())
        self.addCleanup(service.close)
        before = service.state()
        with patch.object(service.avoidance, "predictions", return_value=[]):
            service.start(background=False)
            failed = pump(service, lambda state: state["flight"]["mode"] == "error")

        self.assertEqual(failed["uav"], before["uav"])
        self.assertEqual(failed["step_count"], before["step_count"])
        self.assertIn("Runtime command", failed["flight"]["error"])

    def test_replanning_on_goal_change_keeps_current_position_and_history(self):
        service = MissionService(agent=MockLLMAgent())
        self.addCleanup(service.close)
        service.start(background=False)
        running = pump(service, lambda state: state["flight"]["plan_id"] == 1)
        position = running["uav"]

        changed = service.update_goal({"x": 17, "y": 2, "z": 5})

        self.assertEqual(changed["uav"], position)
        self.assertEqual(changed["flight"]["last_replan_reason"], "mission_goal_changed")
        self.assertEqual(changed["flight"]["planner_calls"], 1)

    def test_switch_runtime_resets_only_when_new_runtime_is_ready(self):
        created = RecordingRuntime("airsim")
        service = MissionService(
            agent=MockLLMAgent(),
            runtime_factory=lambda backend, config: created,
        )
        self.addCleanup(service.close)

        state = service.switch_runtime("airsim", {"host": "127.0.0.1", "port": 41451})

        self.assertEqual(created.reset_calls, [BASE])
        self.assertEqual(state["runtime_backend"], "airsim")
        self.assertEqual(state["uav"], state["base"])
        self.assertEqual(state["step_count"], 0)

    def test_failed_runtime_switch_preserves_current_runtime(self):
        def failing_factory(backend, config):
            raise RuntimeUnavailableError("AirSim is offline")

        service = MissionService(agent=MockLLMAgent(), runtime_factory=failing_factory)
        self.addCleanup(service.close)
        before = service.state()

        with self.assertRaises(RuntimeUnavailableError):
            service.switch_runtime("airsim", {})

        after = service.state()
        self.assertEqual(after["runtime_backend"], "html")
        self.assertEqual(after["uav"], before["uav"])
        self.assertEqual(after["step_count"], before["step_count"])


if __name__ == "__main__":
    unittest.main()
