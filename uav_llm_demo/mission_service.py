from __future__ import annotations

import logging
import math
import secrets
import os
import random
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event, RLock, Thread

from .agents import DeepSeekAgent, MockLLMAgent, PlannerProviderError, PlanValidationError, build_agent, checked_plan
from .avoidance import LocalAvoidance
from .environment import UAVEnvironment, point_to_dict
from .evaluation import EvaluationLog, MissionEvidence
from .recovery import RecoveryMemory
from .run_log import RunTrace, source_fingerprint
from .provider_usage import request_trace
from .html_runtime import HtmlRuntime
from .motion import TrajectoryController, WaypointManager, ReplanningManager, SCALE, norm, distance
from .runtime import RuntimeCapabilityError, RuntimeUnavailableError, SimulationRuntime
from .safety import SafetyChecker
from .scenarios import load_scenario
from .scene_layout import scene_layout
from .traffic_motion import ContinuousTraffic


class MissionService:
    CONTROL_DT = 0.05
    MIN_TARGET_HORIZONTAL_DISTANCE_M = 30.0
    MIN_URBAN_TARGET_HORIZONTAL_DISTANCE_M = 75.0

    def __init__(self, *, agent=None, runtime=None, runtime_factory=None, scenario_id="urban_crossing", target_seed=None, traffic_seed=None, journal=None, scenario=None, planner_latency_seconds=0.0):
        self._target_seed = target_seed if target_seed is not None else secrets.randbits(32)
        self._traffic_seed = traffic_seed if traffic_seed is not None else secrets.randbits(32)
        self._target_rng = random.Random(self._target_seed)
        self._time = 0.0
        self._planner_latency = planner_latency_seconds
        self._future_ready_at = 0.0
        self._journal = journal
        self._source_version = source_fingerprint()
        self._environment = UAVEnvironment(scenario or self._random_scenario(load_scenario(scenario_id)))
        # Backend selection is a browser action; environment variables only seed
        # the in-memory DeepSeek configuration and never select the active agent.
        self.agent = agent or build_agent("mock")
        self._deepseek_config = self._capture_deepseek_config(self.agent)
        self.runtime = runtime or HtmlRuntime()
        if runtime is None:
            self.runtime.reset(self._environment.base)
        self._runtime_factory = runtime_factory
        self._evaluation = EvaluationLog()
        self._lock = RLock()
        self._traffic = self._make_traffic(self._environment.scenario)
        self.controller = TrajectoryController()
        self.waypoints = WaypointManager()
        self.replanning = ReplanningManager()
        self.avoidance = LocalAvoidance()
        self._safety_armed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uav-planner")
        self._shutdown = Event()
        self._thread = None
        self._future = None
        self._generation = 0
        self._future_generation = 0
        self._sequence = 0
        self._time = 0.0
        self._running = False
        self._plan_requested = False
        self._reason = "mission_start"
        self._feedback = ""
        self._error = None
        self._retry_at = 0.0
        self._planner_calls = 0
        self._plan_id = 0
        self._last_dynamic_plan_at = -10.0
        self._next_reconnect_at = 0.0
        self._task = "Fly safely from the current position to the target area."
        self._vehicle = self.runtime.read_state() if self.runtime.backend_name == "html" else None
        self._runtime_status = self.runtime.status()
        self._scene_layout = scene_layout(self._environment.scenario)
        self._traffic_snapshot = None
        self._closed = False
        self._new_run()
        self._sync_traffic()
        self._publish_state()

    def _make_traffic(self, scenario):
        return ContinuousTraffic(scenario, seed=self._traffic_seed, clock=lambda: self._time)

    def _new_run(self):
        self._terminal_wall = None
        self.recovery = RecoveryMemory()
        self.evidence = MissionEvidence()
        self.trace = RunTrace(self._journal, secrets=(self._deepseek_config.get("api_key"),))
        self._last_telemetry_at = self._time
        self.trace.event("run_started", self._time, scenario=self._environment.scenario.to_dict(),
                         target_seed=self._target_seed, traffic_seed=self._traffic_seed,
                         source_sha256=self._source_version, runtime=self.runtime.backend_name,
                         planner=dict(backend=self.agent.backend_name, model=getattr(self.agent, "model", None),
                                      timeout_seconds=getattr(self.agent, "timeout_seconds", None),
                                      temperature=0.1 if self.agent.backend_name == "deepseek" else None),
                         control_dt=self.CONTROL_DT, max_speed_mps=self.controller.max_speed,
                         max_acceleration_mps2=self.controller.max_acceleration,
                         max_climb_speed_mps=self.controller.max_climb_speed,
                         max_descent_speed_mps=self.controller.max_descent_speed)

    def _milestones(self):
        return self.evidence.snapshot(dynamic_events=self._evaluation.dynamic_event_count,
                                      dynamic_measured=self.runtime.backend_name == "html")

    def _random_scenario(self, scenario, previous=None):
        context = scenario.to_dict()
        context["position"] = point_to_dict(scenario.base)
        checker = SafetyChecker(context)
        heights = ([scenario.target[2]] if scenario.preferred_altitude else
                   list(range(math.ceil(scenario.min_altitude), math.floor(min(scenario.max_altitude, 6)) + 1)))
        if not heights:
            heights = [scenario.target[2]]
        minimum_distance = (self.MIN_URBAN_TARGET_HORIZONTAL_DISTANCE_M if scenario.urban_parcels
                            else self.MIN_TARGET_HORIZONTAL_DISTANCE_M)
        # Keep urban targets outside the initial district shown around the base.
        middle = (scenario.grid_size - 1) / 2
        base_block = tuple(coordinate < middle for coordinate in scenario.base[:2])
        candidates = [point
                      for x in range(1, scenario.grid_size - 1)
                      for y in range(1, scenario.grid_size - 1)
                      for height in heights
                      if (point := (float(x), float(y), float(height))) != previous
                      and norm(tuple((point[i] - scenario.base[i]) * SCALE[i] for i in range(2)))
                          >= minimum_distance
                      and (not scenario.urban_parcels or tuple(coordinate < middle for coordinate in point[:2]) != base_block)
                      and not checker.validate_waypoint(point)]
        self._target_rng.shuffle(candidates)
        for target in candidates:
            context["goal"] = point_to_dict(target)
            try:
                # Reachability only: never load this local route as the mission plan.
                checked_plan(MockLLMAgent(), context, "")
            except ValueError:
                continue
            return replace(scenario, target=target)
        raise ValueError("No safe reachable random target exists within the scenario flight envelope.")

    def _sync_traffic(self):
        self._environment.realtime_traffic = self.runtime.backend_name == "html"
        if self._environment.realtime_traffic:
            self._traffic_snapshot = deepcopy(self._traffic.sync(self._environment))
        else:
            self._traffic_snapshot = None
        return self._traffic_snapshot

    def _context(self):
        env = self._environment
        context = env.get_observation().to_dict()
        context.update(
            grid_size=env.grid_size, buildings=[
                dict(x=x, y=y, height=h) for (x, y), h in env.buildings.items()
            ], base=point_to_dict(env.base), target=point_to_dict(env.target),
            no_fly_zones=deepcopy(list(env.scenario.no_fly_zones)),
            orientation=dict(env.kinematics["orientation"]),
            task=(self._task + " Outbound target reached. Now return safely to base; goal is the base position."
                  if env.phase == "returning_home" else self._task),
            preferred_altitude=env.scenario.to_dict()["preferred_altitude"],
            urban_parcels=[parcel.to_dict() for parcel in env.scenario.urban_parcels],
            local_avoidance=self.avoidance.snapshot(), simulation_time=self._time,
        )
        return context

    def state(self, feedback=None):
        # Published objects are never mutated; copying does not hold the control lock.
        with self._lock:
            snapshot = self._snapshot
        payload = deepcopy(snapshot)
        if feedback is not None:
            payload["feedback"] = feedback
        return payload

    def _publish_state(self):
        with self._lock:
            traffic = self._traffic_snapshot
            payload = self._environment.snapshot()
            payload["continuous_path"] = list(self._environment.continuous_path)
            runtime_status = self._runtime_status
            self._sequence += 1
            speed = norm(self._vehicle.velocity) if self._vehicle else 0.0
            mode = ("avoiding" if self.avoidance.active else "error" if self._error else "completed" if self._environment.status == "completed"
                    else "braking" if speed > 0.02 and (not self._running or self._plan_requested)
                    else "planning" if self._plan_requested
                    else "running" if self._running else "paused" if self._plan_id else "idle")
            payload.update(
                sequence=self._sequence, traffic_motion=traffic,
                scene_layout=self._scene_layout,
                agent_backend=self.agent.backend_name, agent_model=getattr(self.agent, "model", None),
                deepseek_key_configured=bool(self._deepseek_config.get("api_key")),
                runtime_backend=self.runtime.backend_name, runtime=runtime_status,
                flight=dict(stage=("landed" if self._environment.position[2] * SCALE[2] <= 0.18 and speed <= 0.02
                                   else "landing" if self._environment.phase == "returning_home" and self._environment.position[2] < self._environment.min_altitude
                                   else "taking_off" if self._environment.position[2] < self._environment.min_altitude
                                   else "cruise"), mode=mode, running=self._running, planning=self._plan_requested,
                            planner_calls=self._planner_calls, plan_id=self._plan_id,
                            waypoint_index=self.waypoints.index,
                            skipped_waypoints=self.waypoints.skipped,
                            plan_start=(point_to_dict(self.waypoints.plan_start)
                                        if self.waypoints.plan_start is not None else None),
                            waypoints=[point_to_dict(p) for p in self.waypoints.points],
                            last_replan_reason=self._reason, error=self._error,
                            avoidance=self.avoidance.snapshot(), safety_armed=self._safety_armed,
                            recovery=self.recovery.snapshot(),
                            retry_in_seconds=max(0.0, self._retry_at-self._time) if self._plan_requested else 0.0,
                            speed_mps=speed, max_speed_mps=self.controller.max_speed,
                            max_acceleration_mps2=self.controller.max_acceleration,
                            max_climb_speed_mps=self.controller.max_climb_speed,
                            max_descent_speed_mps=self.controller.max_descent_speed),
            )
            evaluation = self._evaluation.snapshot(final_status=self._environment.status,
                                                   trajectory=payload["path"], battery=self._environment.battery)
            evaluation["metrics"]["planner_calls"] = self._planner_calls
            evaluation["metrics"]["control_ticks"] = self._environment.step_count
            payload["evaluation"] = evaluation
            payload["metrics"] = evaluation["metrics"]
            payload["milestones"] = self._milestones()
            payload["diagnostics"] = self.trace.snapshot()
            payload["simulation_time"] = self._time
            payload["metrics"].update(success=self._environment.status == "completed" and payload["milestones"]["task_completed"],
                                      task_completed=payload["milestones"]["task_completed"],
                                      safe_success=payload["milestones"]["safe_success"])
            self._snapshot = payload

    def _ensure_loop(self):
        if self._thread is None:
            self._thread = Thread(target=self._loop, name="uav-controller", daemon=True)
            self._thread.start()

    def _loop(self):
        last = time.monotonic()
        wait_seconds = self.CONTROL_DT
        while not self._shutdown.wait(wait_seconds):
            tick_started = time.perf_counter()
            try:
                lock_started = time.perf_counter()
                with self._lock:
                    self.trace.duration("lock_wait", (time.perf_counter()-lock_started)*1000)
                    # Include lock contention in the measured control interval.
                    now = time.monotonic()
                    elapsed, last = now - last, now
                    self.trace.interval(elapsed, min(elapsed, 0.1))
                    if elapsed > 0.3 and (self._running or self.avoidance.active):
                        if self.runtime.backend_name != "html":
                            self._fail("Control heartbeat exceeded 0.3s; stopped for safety.")
                            self._publish_state()
                            continue
                        # HTML motion advances only through drive(); a delayed tick
                        # does not leave a physical vehicle flying on an old command.
                        self.trace.event("control_delayed", self._time, interval_seconds=elapsed,
                                         timing=self.trace.snapshot()["timing"])
                        logging.getLogger(__name__).warning(
                            "HTML control delayed %.3fs; continuing with a capped simulation step.", elapsed)
                    self.advance(min(elapsed, 0.1))
            except Exception:
                with self._lock:
                    self._fail("Controller or runtime failed; inspect the local runtime connection.")
                    self._publish_state()
            finally:
                wait_seconds = max(0.001, self.CONTROL_DT - (time.perf_counter()-tick_started))

    def start(self, *, background=True):
        with self._lock:
            if self._environment.status == "completed":
                return self.state()
            self._running = True
            self._safety_armed = True
            retry_failed = bool(self._error)
            self._terminal_wall = None
            self._error = None
            if self.waypoints.current is None or retry_failed:
                self._request_replan("mission_start")
            if background:
                self._ensure_loop()
            self.advance(self.CONTROL_DT)
            return self.state()

    def plan(self, *, background=True):
        with self._lock:
            self._running = False
            self._invalidate_planning()
            self._request_replan("manual_plan")
            if background:
                self._ensure_loop()
            self.advance(self.CONTROL_DT)
            return self.state()

    def _invalidate_planning(self):
        self._generation += 1
        if self._future is not None:
            self._plan_requested = True
        self._error = None

    def _request_replan(self, reason):
        if not self._plan_requested:
            self._feedback = ""
        self._reason = reason
        self._plan_requested = True
        self._retry_at = self._time
        self._environment.last_feedback = "Replanning: " + reason
        if any(word in reason for word in ("stuck", "stalled", "progress", "persistent")):
            self.recovery.stalled(self._environment.position,
                                  self.waypoints.remaining_distance(self._environment.position), self._time)
        self.trace.event("replan_requested", self._time, reason=reason, phase=self._environment.phase,
                         position=self._environment.position, milestones=self._milestones())

    def _submit_plan(self):
        context = deepcopy(self._context())
        planner = self.agent
        self._future_context = context
        self._future_backend = planner.backend_name
        self._future_model = getattr(planner, "model", None)
        self._future_generation = self._generation
        self._planner_calls += 1
        if self._reason.startswith("dynamic_obstacle:"):
            self._last_dynamic_plan_at = self._time
        trace, request_id = self.trace, self._planner_calls
        if planner.backend_name == "deepseek":
            trace.provider_started()
        feedback = self.recovery.feedback(self._feedback)
        trace.event("plan_requested", self._time, request_id=request_id, generation=self._generation,
                    backend=planner.backend_name, position=context["position"], goal=context["goal"],
                    recovery=self.recovery.snapshot())
        self._future_trace = trace
        self._future_ready_at = self._time + self._planner_latency
        def execute():
            with request_trace(trace.run_id, request_id, trace.usage_event), trace.stage("planner"):
                return checked_plan(planner, context, feedback)
        self._future = self._executor.submit(execute)

    def _collect_plan(self):
        if self._future is None or not self._future.done():
            return
        if self._future_generation != self._generation:
            self._future = None
            self._future_trace.event("plan_discarded", None, reason="generation_changed")
            return
        if self._time < self._future_ready_at:
            return
        future, self._future = self._future, None
        plan = None
        try:
            plan = future.result()
            errors = SafetyChecker(self._context()).validate_path(
                self._environment.position, plan.waypoints)
        except PlannerProviderError as exc:
            self._feedback = str(exc)
            self._record_plan(None, self._feedback, provider_error=True, retryable=exc.retryable)
            if not exc.retryable:
                self._fail(self._feedback, planning=True)
            else:
                self._retry_at = self._time + self.recovery.provider_failed(
                    self._feedback, self._environment.position, self._time,
                    invalid_response="invalid route plan" in self._feedback)
                self.trace.event("retry_scheduled", self._time, recovery=self.recovery.snapshot())
            return
        except PlanValidationError as exc:
            plan, errors = exc.plan, exc.errors
        except Exception as exc:
            # Only provider diagnostics constructed above and known validation errors are displayed.
            errors = [str(exc)] if isinstance(exc, ValueError) else ["Planner request failed; check provider configuration."]
            plan = None
        if errors:
            self._feedback = "Unsafe or invalid route: " + "; ".join(errors)[:1500]
            self._record_plan(plan, self._feedback, rejected=True)
            self._retry_at = self._time + self.recovery.route_failed(
                plan, errors, tuple(self._future_context["position"][k] for k in "xyz"), self._time)
            self.trace.event("retry_scheduled", self._time, recovery=self.recovery.snapshot())
            return
        self._accept_plan(plan, "Route validated; ready to track waypoints.")

    def _accept_plan(self, plan, feedback):
        self.waypoints.load(plan.waypoints, self._environment.position)
        self.replanning.accepted(self._context(), self._time)
        self._plan_requested = False
        self._feedback = ""
        self._plan_id += 1
        self.recovery.accepted()
        self._record_plan(plan, feedback)
        if self._environment.realtime_traffic:
            assigned = self._traffic.focus_route(self._environment.position, plan.waypoints)
            if assigned:
                self.trace.event("traffic_route_focus", self._time, assigned=assigned, plan_id=self._plan_id)
        self._environment.last_feedback = (
            "Route validated; tracking waypoints." if self._running else "Route validated; paused.")

    def _record_plan(self, plan, feedback, rejected=False, provider_error=False, retryable=None):
        decision = plan.to_dict() if plan else dict(waypoints=[], reason=feedback, risk="high", expected_result="Hold")
        actual_backend = self._future_backend
        actual_model = self._future_model
        decision.update(agent_backend=actual_backend, agent_model=actual_model,
                        safety_intervened=bool(rejected),
                        safety_intervention=feedback if rejected else None,
                        provider_error=provider_error)
        entry = dict(step=self._planner_calls, observation=self._future_context,
                     decision=decision, feedback=feedback, runtime=self.runtime.backend_name)
        self._environment.history.append(entry)
        self._environment.history = self._environment.history[-200:]
        self._evaluation.record({**entry, "agent_backend": actual_backend, "agent_model": actual_model})
        self.trace.event("plan_rejected" if rejected else "provider_failed" if provider_error else "plan_accepted",
                         self._time, request_id=self._planner_calls, phase=self._environment.phase,
                         waypoints=[list(p) for p in plan.waypoints] if plan else [],
                         # Free-form model prose and raw exceptions never enter the journal.
                         feedback=feedback if rejected or provider_error else "accepted", retryable=retryable,
                         position=self._environment.position, goal=self._future_context["goal"])

    def advance(self, dt):
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Simulation step must be finite and in (0, 0.1].")
        with self._lock, self.trace.stage("control_tick"):
            try:
                self._advance(dt)
            finally:
                if self._vehicle is not None:
                    self.evidence.observe(self._vehicle, self._environment.target, self._environment.base, self._time)
                if self._time - self._last_telemetry_at >= 1.0:
                    self._last_telemetry_at = self._time
                    self.trace.event("telemetry", self._time, position=self._environment.position,
                                     phase=self._environment.phase, milestones=self._milestones(),
                                     diagnostics=self.trace.snapshot())
                with self.trace.stage("snapshot"):
                    self._publish_state()

    def _advance(self, dt):
        with self._lock:
            self._time += dt
            with self.trace.stage("traffic"):
                self._sync_traffic()
            with self.trace.stage("collect_plan"):
                self._collect_plan()
            if self._vehicle is None:
                self._vehicle = self.runtime.read_state()
                self._commit_vehicle(self._vehicle, 0)
            measure_dynamic = self._safety_armed and self._environment.realtime_traffic
            speed = norm(self._vehicle.velocity)
            if (self._plan_requested and self._future is None and (speed <= 0.02 or self.avoidance.active)
                    and self._time >= self._retry_at and not self._error):
                self._submit_plan()
            target = self.waypoints.current if self._running and not self._plan_requested and not self._error else None
            with self.trace.stage("safety_context"):
                checker = SafetyChecker(self._context())
            tracks = self.avoidance.predictions(
                self._vehicle, self._environment.dynamic_objects if self._environment.realtime_traffic else [],
                self.controller.max_speed) if self._safety_armed else []
            with self.trace.stage("route_reconnection"):
                if target is not None and self._time >= self._next_reconnect_at:
                    self._next_reconnect_at = self._time + 0.25
                    def reachable(point, final):
                        result = self.avoidance.rollout(self._vehicle, point, self.controller, checker, tracks,
                            terminal=final and distance(self._vehicle.position, point) <= 6)
                        return result is not None and result.clearance >= self.avoidance.BUFFER
                    if self.waypoints.reconnect(self._vehicle, checker, reachable):
                        self.replanning.accepted(self._context(), self._time)
                        self._environment.last_feedback = "Safe forward route connection; obsolete transit waypoints skipped."
                    target = self.waypoints.current
            if target is not None and not self.avoidance.active:
                reason = self.replanning.should_replan(self._context(), self._vehicle, self.waypoints, checker, self._time)
                if reason:
                    self._request_replan(reason)
                    target = None
            was_avoiding = self.avoidance.active
            if self._safety_armed:
                blocked = self.avoidance.stalled(self.waypoints, self._vehicle, self._time)
                with self.trace.stage("avoidance"):
                    target = self.avoidance.choose(self._vehicle, target, self.controller, checker, tracks, self._time,
                        terminal=self.waypoints.index == len(self.waypoints.points) - 1)
                if was_avoiding and not self.avoidance.active:
                    # Intentional local detours are not tracking failures.
                    self.waypoints.segment_start = self._vehicle.position
                    self.replanning.accepted(self._context(), self._time)
                    self._environment.last_feedback = self.avoidance.message
                if (self.avoidance.active and self._running and not self._plan_requested
                        and blocked
                        and self._time - self._last_dynamic_plan_at >= 10):
                    self._request_replan("dynamic_obstacle:persistent_local_blockage")
                    self._feedback = self.avoidance.message + " Replan from the updated position around this traffic."
            velocity = self.controller.velocity_command(self._vehicle, target, dt)
            projected = tuple(self._vehicle.position[i] + (self._vehicle.velocity[i] + velocity[i]) * dt / (2 * SCALE[i]) for i in range(3))
            if not checker.segment_safe(self._vehicle.position, projected):
                self.evidence.static_blocked = True
                self._fail("Safety Checker blocked the next control segment.")
                return
            if target is not None or speed > 0.0001:
                try:
                    with self.trace.stage("runtime"):
                        vehicle = self.runtime.drive(velocity, dt)
                except Exception:
                    self._fail("Runtime command/telemetry failed; motion stopped.")
                    return
                self._commit_vehicle(vehicle, dt)
                self._runtime_status = self.runtime.status()
                final_waypoint = self.waypoints.index == len(self.waypoints.points) - 1
                next_segment_safe = (final_waypoint or self.waypoints.current is None
                                     or checker.segment_safe(vehicle.position, self.waypoints.points[self.waypoints.index + 1]))
                if (target is not None and not self.avoidance.active and not self._error
                        and next_segment_safe
                        and self.waypoints.update(vehicle) and self.waypoints.current is None):
                    mission_goal = self._environment.target if self._environment.phase == "fly_to_target" else self._environment.base
                    if distance(vehicle.position, mission_goal) > 0.18:
                        self._request_replan("goal_arrival_not_verified")
                        self._feedback = "Route endpoint reached but mission goal not verified. End exactly at the current goal coordinates."
                        self.recovery.stalled(vehicle.position, distance(vehicle.position, mission_goal), self._time)
                    elif self._environment.phase == "fly_to_target":
                        self.evidence.observe(vehicle, self._environment.target, self._environment.base, self._time)
                        self.trace.event("target_reached", self._time, position=vehicle.position)
                        self.recovery = RecoveryMemory()
                        self._environment.phase = "returning_home"
                        self.waypoints = WaypointManager()
                        self._request_replan("target_reached_return_home")
                        self._environment.last_feedback = "Target reached; planning return to base."
                    else:
                        self._running = False
                        self._safety_armed = False
                        self._environment.phase = "completed"
                        self._environment.status = "completed"
                        self._terminal_wall = time.monotonic()
                        self._environment.last_feedback = "Base reached. Round-trip mission completed."
                        self.evidence.observe(vehicle, self._environment.target, self._environment.base, self._time)
                        self.trace.event("mission_completed", self._time, milestones=self._milestones())
            if measure_dynamic:
                previous_count = self._evaluation.dynamic_event_count
                self._evaluation.observe_dynamic(self._vehicle.position, self._environment.dynamic_objects, self._time)
                added = self._evaluation.dynamic_event_count - previous_count
                if added:
                    self.trace.event("separation_violation", self._time,
                                     incidents=self._evaluation.dynamic_incidents[-added:])
            if norm(self._vehicle.velocity) <= 0.0001 and target is None and self.runtime.backend_name == "airsim":
                # The short velocity lease handles loss of the controller; hold when intentionally idle.
                if getattr(self, "_holding", False) is False:
                    self.runtime.stop()
                    self._holding = True
            else:
                self._holding = False

    def _commit_vehicle(self, vehicle, dt):
        self._vehicle = vehicle
        env = self._environment
        moved = distance(env.position, vehicle.position)
        env.position = vehicle.position
        env.battery = max(0.0, env.battery - dt * 0.015)
        if dt:
            env.step_count += 1
        point = point_to_dict(vehicle.position)
        env.last_trajectory = [point]
        env.last_trajectory_duration = dt
        if moved > 0.001:
            env.path.append(vehicle.position)
            env.continuous_path.append(point)
            env.path = env.path[-10000:]
            env.continuous_path = env.continuous_path[-10000:]
        env.kinematics = dict(
            position_grid=point, orientation=dict(roll=0.0, pitch=0.0, yaw=vehicle.yaw),
            linear_velocity=point_to_dict(vehicle.velocity),
            linear_acceleration=point_to_dict(vehicle.acceleration),
            angular_velocity=dict(x=0.0, y=0.0, z=0.0),
            angular_acceleration=dict(x=0.0, y=0.0, z=0.0),
            source=self.runtime.backend_name, velocity_frame="east_south_up_mps",
        )
        if env.battery <= 0:
            self._fail("Battery depleted.")

    def _fail(self, message, *, planning=False):
        self._running = False
        self._plan_requested = False
        self._generation += 1
        self._error = message
        self._terminal_wall = time.monotonic()
        self._environment.last_feedback = message
        self.trace.event("mission_error", self._time, reason=message, phase=self._environment.phase)
        if planning:
            # A failed remote plan must not disable local collision protection.
            return
        self._safety_armed = False
        self.avoidance = LocalAvoidance()
        try:
            self.runtime.stop()
            self._vehicle = self.runtime.read_state()
            self._environment.position = self._vehicle.position
        except Exception:
            pass

    def stop_runtime(self):
        with self._lock:
            self._running = False
            self._invalidate_planning()
            self._plan_requested = False
            self._environment.last_feedback = "Mission paused; decelerating to hold. Local collision protection remains active."
            self.trace.event("mission_paused", self._time, milestones=self._milestones())
            self._publish_state()
            return self.state()

    def _reset_control(self, *, traffic=None):
        self._generation += 1
        self._running = False
        self._plan_requested = False
        self._error = None
        self._plan_id = 0
        self._planner_calls = 0
        self._last_dynamic_plan_at = self._time - 10.0
        self.waypoints = WaypointManager()
        self.replanning = ReplanningManager()
        self.avoidance = LocalAvoidance()
        self._safety_armed = False
        self.trace.event("run_reset", self._time, milestones=self._milestones())
        self._time = 0.0
        self._retry_at = 0.0
        self._last_dynamic_plan_at = -10.0
        self._next_reconnect_at = 0.0
        self._traffic = traffic if traffic is not None else self._make_traffic(self._environment.scenario)
        self._traffic.started = 0.0
        self._scene_layout = scene_layout(self._environment.scenario)
        self._runtime_status = self.runtime.status()
        self._new_run()
        self._sync_traffic()
        self._vehicle = self.runtime.read_state()
        self._commit_vehicle(self._vehicle, 0)
        self._evaluation.reset()
        self._publish_state()

    def reset(self):
        with self._lock:
            scenario = self._random_scenario(load_scenario(self._environment.scenario_id),
                                             previous=self._environment.target)
            self._traffic_seed = secrets.randbits(32)
            traffic = self._make_traffic(scenario)
            self.runtime.stop()
            if self.runtime.backend_name == "airsim":
                self.runtime.configure_scene(scenario)
            self.runtime.reset(scenario.base)
            self._environment = UAVEnvironment(scenario)
            self._reset_control(traffic=traffic)
            return self.state()

    def switch_scenario(self, scenario_id):
        with self._lock:
            scenario = self._random_scenario(load_scenario(scenario_id), previous=self._environment.target)
            self._traffic_seed = secrets.randbits(32)
            traffic = self._make_traffic(scenario)
            self.runtime.stop()
            self._running = False
            if self.runtime.backend_name == "airsim":
                self.runtime.configure_scene(scenario)
            self.runtime.reset(scenario.base)
            self._environment = UAVEnvironment(scenario)
            self._reset_control(traffic=traffic)
            return self.state()

    def switch_runtime(self, backend, config=None):
        if backend not in {"html", "airsim"}:
            raise ValueError("Runtime backend must be 'html' or 'airsim'.")
        with self._lock:
            self.runtime.stop()
            self._running = False
            candidate = None
            try:
                if backend == "html":
                    candidate = HtmlRuntime()
                else:
                    if self._runtime_factory is None:
                        raise RuntimeUnavailableError("AirSim factory is not configured.")
                    candidate = self._runtime_factory(backend, dict(config or {}))
                    candidate.configure_scene(self._environment.scenario)
                candidate.reset(self._environment.base)
                if backend == "airsim" and "wind_ned" in (config or {}):
                    self._set_runtime_wind(candidate, config["wind_ned"])
            except Exception as exc:
                if candidate is not None:
                    self._close_quietly(candidate)
                self._publish_state()
                raise RuntimeUnavailableError("Runtime switch failed: " + str(exc)) from exc
            previous, self.runtime = self.runtime, candidate
            # Closing a previous AirSim instance releases control of the same vehicle.
            self._close_quietly(previous)
            if backend == "airsim" and previous.backend_name == "airsim":
                candidate.connect()
            self._environment.reset()
            self._reset_control()
            return self.state()

    def update_goal(self, target, task=None):
        with self._lock:
            checker = SafetyChecker(self._context())
            errors = checker.validate_waypoint(target)
            if errors:
                raise ValueError("; ".join(errors))
            if task is not None and (not isinstance(task, str) or len(task) > 2000):
                raise ValueError("Task must be a string of at most 2000 characters.")
            self._environment.target = tuple(float(target[k]) for k in "xyz") if isinstance(target, dict) else tuple(target)
            self._environment.scenario = replace(self._environment.scenario, target=self._environment.target)
            self.trace.event("goal_changed", self._time, target=self._environment.target)
            self.recovery = RecoveryMemory()
            self.evidence = MissionEvidence()
            self._environment.phase = "fly_to_target"
            self._environment.status = "active"
            if task is not None:
                self._task = task
            self._invalidate_planning()
            self._request_replan("mission_goal_changed")
            self._publish_state()
            return self.state()

    def set_wind(self, wind):
        with self._lock:
            if not self.runtime.status().get("supports_wind"):
                raise RuntimeCapabilityError("HTML runtime does not support wind.")
            self._set_runtime_wind(self.runtime, wind)
            self._runtime_status = self.runtime.status()
            self._publish_state()
            return self.state()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()
        with self._lock:
            self._generation += 1
            self._running = False
            self._safety_armed = False
            self.runtime.stop()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.trace.event("run_closed", self._time, milestones=self._milestones(), diagnostics=self.trace.snapshot())
        self._close_quietly(self.runtime)

    def configure_agent(self, config: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(config, Mapping):
            raise TypeError("Agent config must be an object.")
        backend = str(config.get("backend", "mock")).strip().lower()
        if backend not in {"mock", "deepseek"}:
            raise ValueError(
                "Agent backend must be 'mock' or 'deepseek'. "
                "The OpenAI backend is not implemented in this demo."
            )

        with self._lock:
            if backend == "deepseek":
                next_config = dict(self._deepseek_config)
                provided_key = self._optional_text(config.get("api_key"), "api_key")
                if provided_key:
                    next_config["api_key"] = provided_key
                provided_model = self._optional_text(config.get("model"), "model")
                if provided_model:
                    next_config["model"] = provided_model
                provided_base_url = self._optional_text(
                    config.get("base_url"), "base_url"
                )
                if provided_base_url:
                    next_config["base_url"] = provided_base_url

                api_key = next_config.get("api_key")
                if not api_key:
                    raise ValueError(
                        "A DeepSeek API key is required before selecting the API agent."
                    )
                candidate = DeepSeekAgent(
                    api_key=api_key,
                    model=next_config.get("model"),
                    base_url=next_config.get("base_url"),
                )
                self._deepseek_config = next_config
                self.agent = candidate
            else:
                self.agent = build_agent(backend)
            self._invalidate_planning()
            self.recovery = RecoveryMemory()
            key = self._deepseek_config.get("api_key")
            if key and key not in self.trace.secrets:
                self.trace.secrets += (key,)
            self.trace.event("agent_changed", self._time, backend=backend)
            self._environment.last_feedback = (
                f"Agent switched to {backend}; mission continues from current position."
            )
            self._publish_state()
            return self.state()

    @staticmethod
    def _optional_text(value: object, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string.")
        value = value.strip()
        return value or None

    @staticmethod
    def _capture_deepseek_config(agent) -> dict[str, str | None]:
        return {
            "api_key": getattr(agent, "api_key", None) or os.getenv("DEEPSEEK_API_KEY"),
            "model": getattr(agent, "model", None)
            or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "base_url": getattr(agent, "base_url", None)
            or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        }

    @staticmethod
    def _set_runtime_wind(runtime, wind: object) -> None:
        if not isinstance(wind, Mapping):
            raise ValueError("wind_ned must be an object.")
        north = float(wind.get("north", wind.get("x", 0.0)))
        east = float(wind.get("east", wind.get("y", 0.0)))
        down = float(wind.get("down", wind.get("z", 0.0)))
        runtime.set_wind(north, east, down)

    @staticmethod
    def _close_quietly(runtime: SimulationRuntime) -> None:
        try:
            runtime.close()
        except Exception:
            pass
