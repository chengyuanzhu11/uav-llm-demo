from __future__ import annotations

import logging
import json
import math
import secrets
import os
import random
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from threading import Event, RLock, Thread

from .agents import DeepSeekAgent, GeometricPlanner, MockLLMAgent, PlannerProviderError, SemanticPlanError, PlanValidationError, build_agent, checked_plan
from .agents import _redact_task_data, MAX_REPAIR_CANDIDATE_CHARS
from .task_contract import TaskContractError
from .provider_config import ProviderConfigError, ProviderProfiles, discover_models
from .avoidance import LocalAvoidance
from .environment import UAVEnvironment, point_to_dict
from .evaluation import EvaluationLog, MissionEvidence
from .recovery import RecoveryMemory
from .run_log import RunTrace, source_fingerprint
from .provider_usage import request_trace
from .html_runtime import HtmlRuntime
from .motion import (TrajectoryController, WaypointManager, ReplanningManager, SCALE, norm, distance,
                     MAX_TRACKING_DEVIATION_M, segment_deviation)
from .runtime import RuntimeCapabilityError, RuntimeUnavailableError, SimulationRuntime
from .safety import ALTITUDE_EPSILON, SafetyChecker
from .scenarios import load_scenario
from .scene_layout import scene_layout
from .traffic_motion import ContinuousTraffic, TRAFFIC_POLICIES, TRAFFIC_MODEL_VERSION
from .traffic_observation import TrafficObserver, TrafficObservationConfig
from .dynamic_risk import envelope_fields, risk_levels
from .semantic_tasks import SemanticConstraintError, SemanticExecutionError, SemanticMission, task_catalog, PROGRAM_LIMITS
from .semantic_navigation import apply_task_constraints, validate_task_routes, validate_motion_path, task_ground_sites
from .spatial_motion import compile_motion
from .path_tracking import PathFollower


class MissionService:
    CONTROL_DT = 0.05
    MIN_TARGET_HORIZONTAL_DISTANCE_M = 30.0
    MIN_URBAN_TARGET_HORIZONTAL_DISTANCE_M = 75.0

    def __init__(self, *, agent=None, runtime=None, runtime_factory=None, scenario_id="urban_crossing", target_seed=None, traffic_seed=None, journal=None, scenario=None, planner_latency_seconds=0.0, traffic_policy="fixed", observation_config=None):
        if traffic_policy not in TRAFFIC_POLICIES:
            raise ValueError("Traffic policy must be fixed.")
        self._traffic_policy = traffic_policy
        self._observation_config = (TrafficObservationConfig() if observation_config is None else observation_config)
        self._target_seed = target_seed if target_seed is not None else secrets.randbits(32)
        self._traffic_seed = traffic_seed if traffic_seed is not None else secrets.randbits(32)
        self._target_rng = random.Random(self._target_seed)
        self._time = 0.0
        self._planner_latency = planner_latency_seconds
        self._future_ready_at = 0.0
        self._journal = journal
        self._source_version = source_fingerprint()
        self._scenario_template = scenario or load_scenario(scenario_id)
        self._environment = UAVEnvironment(scenario or self._random_scenario(self._scenario_template))
        # Backend selection is a browser action; environment variables only seed
        # the in-memory DeepSeek configuration and never select the active agent.
        semantic = self._environment.scenario.mission_mode == "semantic"
        self.agent = agent or build_agent("deepseek" if semantic else "mock")
        if semantic and self.agent.backend_name != "deepseek":
            raise ValueError("园区任务需要 DeepSeek 理解自然语言，不支持 Mock 模式。")
        self._deepseek_config = self._capture_deepseek_config(self.agent)
        self._provider_profiles = ProviderProfiles(self._deepseek_config)
        self.runtime = runtime or HtmlRuntime()
        if semantic and self.runtime.backend_name != "html":
            raise ValueError("园区任务的观测技能目前仅支持 HTML 仿真。")
        if runtime is None:
            self.runtime.reset(self._environment.base)
        self._runtime_factory = runtime_factory
        self._evaluation = EvaluationLog()
        self._lock = RLock()
        self._traffic = self._make_traffic(self._environment.scenario)
        self.controller = TrajectoryController()
        self.waypoints = WaypointManager()
        self.replanning = ReplanningManager()
        self.avoidance = LocalAvoidance(calibration_profile=self._environment.scenario_id)
        self._traffic_observer = TrafficObserver(self._observation_config, seed=self._traffic_seed)
        self._safety_armed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uav-planner")
        self._shutdown = Event()
        self._thread = None
        self._future = None
        self._future_kind = "route"
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
        self.semantic = SemanticMission(self._environment.scenario) if semantic else None
        self._semantic_calls = 0
        self._ground_sites = []
        self._hold_target = None
        self._control_yaw = None
        self._semantic_schema_failures = 0
        self._semantic_repair = None
        self._autonomous_reviews = 0
        self._semantic_review_pending = False
        self._clear_motion()
        self._semantic_interpretation_pending = False
        self._semantic_revision_started_at = None
        self._new_run()
        if self.semantic is not None:
            self.semantic.mission_id = self.trace.run_id
        self._sync_traffic()
        self._publish_state()

    def _make_traffic(self, scenario):
        return ContinuousTraffic(scenario, seed=self._traffic_seed, clock=lambda: self._time,
                                 policy=self._traffic_policy)

    def _new_run(self):
        self._terminal_wall = None
        self._motion_samples = deque(maxlen=32)
        self._path_start_index = 0
        self.recovery = RecoveryMemory()
        self.evidence = MissionEvidence()
        self.trace = RunTrace(self._journal, secrets=(self._deepseek_config.get("api_key"),))
        self._last_telemetry_at = self._time
        self.trace.event("run_started", self._time, scenario=self._environment.scenario.to_dict(),
                         target_seed=self._target_seed, traffic_seed=self._traffic_seed,
                         traffic_policy=self._traffic_policy, traffic_model_version=TRAFFIC_MODEL_VERSION,
                         traffic_observation=asdict(self._observation_config),
                         traffic_prediction=self.avoidance.snapshot()["prediction"],
                         source_sha256=self._source_version, runtime=self.runtime.backend_name,
                         planner=dict(backend=self.agent.backend_name, model=getattr(self.agent, "model", None),
                                      timeout_seconds=getattr(self.agent, "timeout_seconds", None),
                                      temperature=0.1 if self.agent.backend_name == "deepseek" else None),
                         control_dt=self.CONTROL_DT, max_speed_mps=self.controller.max_speed,
                         max_acceleration_mps2=self.controller.max_acceleration,
                         max_climb_speed_mps=self.controller.max_climb_speed,
                         max_descent_speed_mps=self.controller.max_descent_speed)

    def _milestones(self):
        if self.semantic is not None:
            snapshot = self.semantic.to_dict()
            steps = [step for step in snapshot["steps"]
                     if step.get("revision") == snapshot["revision"] and step["status"] != "cancelled"]
            visits = [step for step in steps if step["action"] != "return_home"]
            contract = snapshot["task_contract"]
            completed = self.semantic.status == "completed" and bool(contract and contract["satisfied"])
            current = self.semantic.current_step
            reached = bool(visits) and bool(completed or (self.semantic.status == "running" and current
                                                         and current["action"] == "return_home"))
            events = snapshot["evidence"]
            dynamic_safe = self._evaluation.dynamic_event_count == 0
            returned = completed and distance(self._vehicle.position, self._environment.base) <= .18 and norm(self._vehicle.velocity) <= .02
            return dict(target_reached=reached, base_returned=returned,
                        target_reached_at=next((e["simulation_time"] for e in reversed(events)
                                               if e["action"] != "return_home"), None) if reached else None,
                        base_returned_at=events[-1]["simulation_time"] if returned and events else None,
                        stationary=bool(self._vehicle and norm(self._vehicle.velocity) <= .02),
                        task_completed=completed, static_path_unblocked=not self.evidence.static_blocked,
                        dynamic_separation_safe=dynamic_safe,
                        safe_success=completed and not self.evidence.static_blocked and dynamic_safe,
                        remaining=[step["id"] for step in steps if step["status"] not in {"completed", "skipped"}])
        return self.evidence.snapshot(dynamic_events=self._evaluation.dynamic_event_count,
                                      dynamic_measured=self.runtime.backend_name == "html")

    def _random_scenario(self, scenario, previous=None):
        if scenario.mission_mode == "semantic":
            return scenario
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

    def _observation(self):
        """Planner input uses delivered observations; renderer keeps world truth."""
        observation = self._environment.get_observation()
        dynamic = []
        for item in self._traffic_observer.latest(self._time):
            forecast = self.avoidance.predictor.forecast(item["id"], self._time, horizon=1, dt=1)
            if forecast is None:
                continue
            position = forecast.positions[0]
            dynamic.append(dict(
                id=item["id"], kind=item["kind"], radius=item["radius"],
                position=point_to_dict(position), velocity=point_to_dict(item["velocity"]),
                position_unit="grid", velocity_unit="grid/s",
                predicted_position=point_to_dict(forecast.positions[-1]),
                prediction_model=forecast.model, prediction_uncertainty_m=forecast.uncertainty_m[-1],
                prediction_uncertainty_horizontal_m=forecast.uncertainty_horizontal_m[-1],
                prediction_uncertainty_vertical_m=forecast.uncertainty_vertical_m[-1],
                observed_at=item["observed_at"], observation_age_seconds=forecast.age_seconds,
                **envelope_fields(SimpleNamespace(radius=item["radius"], radius_m=item["radius_m"])),
                **risk_levels(position, self._environment.position, item["velocity"],
                              self._vehicle.velocity if self._vehicle else (0, 0, 0), radius_m=item["radius_m"])))
        return replace(observation, dynamic_objects=tuple(dynamic),
                       nearest_dynamic_obstacle=min(dynamic, key=lambda item: item["separation_clearance_m"], default=None))

    def _context(self, *, semantic_constraints=True):
        env = self._environment
        preferred = env.scenario.preferred_altitude
        context = self._observation().to_dict()
        context.update(
            grid_size=env.grid_size, buildings=[
                dict(x=x, y=y, height=h) for (x, y), h in env.buildings.items()
            ], base=point_to_dict(env.base), target=point_to_dict(env.target),
            no_fly_zones=deepcopy(list(env.scenario.no_fly_zones)),
            orientation=dict(env.kinematics["orientation"]),
            task=(self._task + " Outbound target reached. Now return safely to base; goal is the base position."
                  if env.phase == "returning_home" else self._task),
            preferred_altitude=(dict(min=preferred[0], max=preferred[1]) if preferred else None),
            urban_parcels=[parcel.to_dict() for parcel in env.scenario.urban_parcels],
            local_avoidance=self.avoidance.snapshot(), simulation_time=self._time,
            ground_sites=deepcopy(self._ground_sites),
        )
        if semantic_constraints and self.semantic is not None and self.semantic.plan is not None:
            apply_task_constraints(context, env.scenario, self.semantic.plan.constraints)
        return context

    def state(self, feedback=None):
        # Published objects are never mutated; copying does not hold the control lock.
        with self._lock:
            snapshot = self._snapshot
        payload = deepcopy(snapshot)
        if feedback is not None:
            payload["feedback"] = feedback
        return payload

    def _sample_motion(self):
        """Publish measured HTML motion; never feed display history into control.

        Each sample owns its vectors and is immutable after insertion. Publishing
        at the same simulation time replaces the last sample, so API actions do
        not consume the bounded history needed between browser state updates.
        """
        if self.runtime.backend_name != "html" or self._vehicle is None:
            return []
        vehicle = self._vehicle
        sample = dict(
            t=self._time, position=list(vehicle.position), velocity=list(vehicle.velocity),
            path_index=self._path_start_index + len(self._environment.continuous_path) - 1,
            yaw=vehicle.yaw,
            landed=vehicle.position[2] * SCALE[2] <= 0.18 and norm(vehicle.velocity) <= 0.02,
            # Traffic position and velocity both use grid units, unlike the
            # vehicle velocity, which is measured in metres per second.
            traffic={bird.object_id: [*bird.position, *bird.velocity]
                     for bird in self._environment.dynamic_objects},
        )
        if self._motion_samples and self._motion_samples[-1]["t"] == self._time:
            self._motion_samples[-1] = sample
        else:
            self._motion_samples.append(sample)
        return list(self._motion_samples)

    def _publish_state(self):
        with self._lock:
            traffic = self._traffic_snapshot
            payload = self._environment.snapshot()
            payload["observation"] = self._observation().to_dict()
            payload["continuous_path"] = list(self._environment.continuous_path)
            payload["path_start_index"] = self._path_start_index
            payload["motion_samples"] = self._sample_motion()
            runtime_status = self._runtime_status
            self._sequence += 1
            speed = norm(self._vehicle.velocity) if self._vehicle else 0.0
            mode = ("avoiding" if self.avoidance.active else "error" if self._error else "completed" if self._environment.status == "completed"
                    else "braking" if speed > 0.02 and (not self._running or self._plan_requested)
                    else "planning" if self._plan_requested
                    else "running" if self._running else "paused" if self._plan_id else "idle")
            payload.update(
                sequence=self._sequence, traffic_motion=traffic,
                traffic_policy=self._traffic_policy, traffic_model_version=TRAFFIC_MODEL_VERSION,
                traffic_observation=asdict(self._observation_config),
                scene_layout=self._scene_layout,
                agent_backend=self.agent.backend_name, agent_model=getattr(self.agent, "model", None),
                agent_provider=self._provider_profiles.active,
                agent_base_url=self._deepseek_config.get("base_url"),
                deepseek_key_configured=bool(self._deepseek_config.get("api_key")),
                runtime_backend=self.runtime.backend_name, runtime=runtime_status,
                flight=dict(stage=("landed" if self._environment.position[2] * SCALE[2] <= 0.18 and speed <= 0.02
                                   else "landing" if self._environment.target[2] == 0 and self._vehicle.velocity[2] < 0 and self._environment.position[2] < self._environment.min_altitude - ALTITUDE_EPSILON
                                   else "taking_off" if self._environment.position[2] < self._environment.min_altitude - ALTITUDE_EPSILON
                                   else "cruise"), mode=mode, running=self._running, planning=self._plan_requested,
                            planner_calls=self._planner_calls, plan_id=self._plan_id,
                            waypoint_index=self.waypoints.index,
                            skipped_waypoints=self.waypoints.skipped,
                            plan_start=(point_to_dict(self.waypoints.plan_start)
                                        if self.waypoints.plan_start is not None else None),
                            waypoints=[point_to_dict(p) for p in self.waypoints.points],
                            last_replan_reason=self._reason, error=self._error,
                            avoidance=self.avoidance.snapshot(include_visualization=True), safety_armed=self._safety_armed,
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
            payload["semantic_mission"] = self.semantic.to_dict() if self.semantic is not None else None
            if payload["semantic_mission"] is not None:
                payload["semantic_mission"]["provider_calls"] = self._semantic_calls
                payload["semantic_mission"]["motion_progress"] = self._motion_progress()
                if self._motion is not None:
                    if self._motion_phase == "tracking":
                        payload["flight"]["waypoints"] = [point_to_dict(p) for p in self._motion.points]
                        payload["flight"]["waypoint_index"] = self._motion_follower.snapshot()["segment_index"]
                        payload["flight"]["plan_start"] = point_to_dict(self._motion.points[0])
                    else:
                        payload["flight"]["waypoints"] += [point_to_dict(p) for p in self._motion.points]
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
            if self.semantic is not None:
                if self.semantic.status in {"idle", "needs_clarification", "safety_wait", "generation_failed"}:
                    raise ValueError("请先在任务输入框提交园区任务。")
                if self.semantic.status == "error":
                    self.semantic.status = "interpreting" if self._semantic_interpretation_pending else "running"
            if self._environment.status == "completed":
                return self.state()
            self._running = True
            self._safety_armed = True
            retry_failed = bool(self._error)
            self._terminal_wall = None
            self._error = None
            if self._motion_phase == "tracking" and not retry_failed:
                self._plan_requested = False
            elif self.waypoints.current is None or retry_failed:
                if self._motion_phase == "tracking":
                    self._environment.target = self._motion_follower.rejoin_point
                    self._motion_phase = "transit"
                self._request_replan("mission_start")
            if background:
                self._ensure_loop()
            self.advance(self.CONTROL_DT)
            return self.state()

    def plan(self, *, background=True):
        with self._lock:
            if self.semantic is not None:
                raise ValueError("园区通过自然语言安排任务，飞行路线由本地规划器自动生成。")
            self._running = False
            self._invalidate_planning()
            self._request_replan("manual_plan")
            if background:
                self._ensure_loop()
            self.advance(self.CONTROL_DT)
            return self.state()

    def submit_task(self, instruction, *, background=True):
        """Submit language for interpretation; no provider call runs under the control lock."""
        with self._lock:
            if self.semantic is None:
                raise ValueError("请切换到园区任务地图后输入自然语言任务。")
            if not isinstance(instruction, str) or not 1 <= len(instruction.strip()) <= 2000:
                raise ValueError("请输入 1–2000 字的任务指令。")
            key = self._deepseek_config.get("api_key")
            if not key:
                raise ValueError("请先在 Key 设置中配置 DeepSeek API key。")
            if key in instruction:
                raise ValueError("请勿在任务指令中填写 API key。")
            updating = self.semantic.status != "idle"
            self._hold_target = self._control_yaw = None
            self._semantic_schema_failures = self._autonomous_reviews = 0
            self._semantic_repair = None
            self._semantic_review_pending = False
            self._invalidate_planning()
            if self._future is not None:
                self._future.cancel()
            self.semantic.observe(self._vehicle.position, norm(self._vehicle.velocity), 0, self._time,
                                  enabled=False)
            self._clear_motion()
            self.semantic.instruction = instruction.strip()
            self.semantic.status = "interpreting"
            self.semantic.message = "正在理解任务调整，保持安全等待。" if updating else "DeepSeek 正在理解任务及约束。"
            self._semantic_interpretation_pending = True
            self._environment.status = "active"
            self._terminal_wall = None
            self._feedback = ""
            self.recovery = RecoveryMemory()
            self.waypoints = WaypointManager()
            self.replanning = ReplanningManager()
            self._running = self._safety_armed = True
            self._request_replan("semantic_instruction")
            self.trace.event("semantic_update_submitted" if updating else "semantic_submitted", self._time,
                             revision=self.semantic.revision, generation=self._generation)
            self._environment.last_feedback = self.semantic.message
            if background:
                self._ensure_loop()
            self.advance(self.CONTROL_DT)
            return self.state()

    def _submit_semantic_plan(self):
        scenario = self._environment.scenario
        catalog = task_catalog(scenario)
        catalog["mission_context"] = self.semantic.catalog_context()
        catalog["mission_context"]["telemetry"] = self._semantic_telemetry()
        catalog["mission_context"]["position"] = point_to_dict(self._vehicle.position)
        catalog["mission_context"]["observations"] = self._observation().to_dict()
        catalog["mission_context"]["autonomous_reviews_used"] = self._autonomous_reviews
        catalog["mission_context"]["autonomous_review"] = self._semantic_review_pending
        context = deepcopy(self._context(semantic_constraints=False))
        agent, instruction, feedback = self.agent, self.semantic.instruction, self._feedback
        repair = deepcopy(self._semantic_repair)
        self._future_kind = "semantic"
        self._future_context = context
        self._future_backend, self._future_model = "deepseek", getattr(agent, "model", None)
        self._future_generation = self._generation
        self._future_ready_at = self._time + self._planner_latency
        self._semantic_calls += 1
        trace, request_id = self.trace, f"task-{self._semantic_calls}"
        self._future_trace = trace
        trace.provider_started()
        trace.event("semantic_requested", self._time, request_id=request_id, generation=self._generation)
        def execute():
            with request_trace(trace.run_id, request_id, trace.usage_event), trace.stage("semantic_interpretation"):
                plan = agent.interpret_task(catalog, instruction, feedback, repair=repair)
                return validate_task_routes(plan, scenario, context)
        self._future = self._executor.submit(execute)

    def _collect_semantic_plan(self, future):
        request_id = f"task-{self._semantic_calls}"
        try:
            plan = future.result()
            if plan.status == "ready":
                if self._semantic_review_pending:
                    self.semantic.task_contract.validate_revision(plan, self._semantic_telemetry(), after_review=True)
                # Local avoidance can move the aircraft while interpretation is
                # pending. Never switch to constraints excluding its latest pose.
                candidate = self._context(semantic_constraints=False)
                apply_task_constraints(candidate, self._environment.scenario, plan.constraints)
                candidate["ground_sites"] += task_ground_sites(plan, self._environment.scenario)
                checker = SafetyChecker(candidate)
                if checker.validate_waypoint(self._vehicle.position):
                    raise SemanticConstraintError("current_position_outside_constraints")
                origin = tuple(self._future_context["position"][axis] for axis in "xyz")
                if not checker.segment_safe(self._vehicle.position, origin):
                    # Recheck a moved departure point in the worker, keeping the
                    # already interpreted plan and avoiding another paid request.
                    context = deepcopy(self._context(semantic_constraints=False))
                    self._future_context = context
                    self._future = self._executor.submit(validate_task_routes, plan, self._environment.scenario, context)
                    self._future_ready_at = self._time
                    self.trace.event("semantic_routes_rechecking", self._time, request_id=request_id,
                                     position=point_to_dict(self._vehicle.position))
                    return
            telemetry = self._semantic_telemetry()
            if plan.status == "ready" and not self._semantic_review_pending:
                telemetry["elapsed_seconds"] = 0.0
            if self._semantic_review_pending and plan.status == "ready":
                if self.semantic.complete_review(self._vehicle.position, self._time, telemetry=telemetry):
                    self.trace.event("semantic_evidence", self._time, evidence=self.semantic.evidence[-1])
                self._semantic_feedback("review_accepted", "已根据执行观测更新后续任务。")
            self.semantic.accept(plan, preserve_evidence=self.semantic.plan is not None, telemetry=telemetry,
                                 autonomous=self._semantic_review_pending)
            if plan.status == "ready":
                for site in task_ground_sites(plan, self._environment.scenario):
                    if site not in self._ground_sites:
                        self._ground_sites.append(site)
        except (SemanticPlanError, TaskContractError) as exc:
            if isinstance(exc, TaskContractError):
                candidate = json.dumps(_redact_task_data(plan.to_dict(), getattr(self.agent, "api_key", None)),
                                       ensure_ascii=False, allow_nan=False)
                candidate = candidate if len(candidate) <= MAX_REPAIR_CANDIDATE_CHARS else None
                exc = SemanticPlanError(exc.diagnostic["message"], candidate=candidate, diagnostic=exc.diagnostic)
            self._semantic_schema_failures += 1
            self._semantic_repair = {"candidate": exc.candidate, "diagnostic": deepcopy(exc.diagnostic)}
            self._feedback = "Correct the locally rejected program: " + exc.detail
            self.trace.event("semantic_schema_rejected", self._time, request_id=request_id,
                             reason=exc.detail, diagnostic=exc.diagnostic, correction=self._semantic_schema_failures)
            if self._semantic_schema_failures <= 2:
                self.semantic.message = "任务程序校验未通过，正在依据具体原因修正。"
                self._retry_at = self._time + 1.0
            else:
                self._hold_semantic_task("任务程序生成失败，自动修正仍未通过；可重试原指令。",
                                         status="generation_failed")
            return
        except PlannerProviderError as exc:
            self._feedback = str(exc)
            self.semantic.message = self._feedback
            self.trace.event("semantic_provider_failed", self._time, request_id=request_id,
                             feedback=self._feedback, retryable=exc.retryable)
            if not exc.retryable:
                self._fail(self._feedback, planning=True)
            else:
                self._retry_at = self._time + self.recovery.provider_failed(
                    self._feedback, self._environment.position, self._time,
                    invalid_response="invalid" in self._feedback)
                self.trace.event("retry_scheduled", self._time, recovery=self.recovery.snapshot())
            return
        except SemanticConstraintError as exc:
            # Only known scene conflicts are user constraints; never expose raw exceptions.
            self._hold_semantic_task(exc.user_message)
            self.trace.event("semantic_rejected", self._time, request_id=request_id, reason=exc.code)
            return
        except SemanticExecutionError as exc:
            self._hold_semantic_task(exc.user_message, status="safety_wait")
            return
        except Exception as exc:
            self.trace.event("semantic_rejected", self._time, request_id=request_id,
                             reason="internal_task_error", error_type=type(exc).__name__)
            self._fail("任务解析或路径预检发生内部错误，请调整指令后重试。", planning=True)
            return
        self._plan_requested = False
        self._semantic_repair = None
        self._semantic_schema_failures = 0
        self._feedback = ""
        self.recovery.accepted()
        self.trace.event("semantic_accepted", self._time, request_id=request_id, status=plan.status,
                         steps=plan.to_dict()["steps"], constraints=plan.constraints,
                         revision=self.semantic.revision, task_contract=self.semantic.task_contract.snapshot()
                         if self.semantic.task_contract else None)
        if plan.status != "ready":
            self._hold_semantic_task(self.semantic.message)
            return
        self._semantic_interpretation_pending = False
        if not self._semantic_review_pending:
            self._semantic_revision_started_at = self._time
        self._semantic_review_pending = False
        if self.semantic.status == "completed":
            self.semantic.goal_state = {"target_id": None, "position": point_to_dict(self._vehicle.position),
                                        "state": "landed" if self._vehicle.position[2] * SCALE[2] <= .05 else "airborne",
                                        "satisfied": True}
            self.semantic.message = "任务无需执行动作，已保持当前位置。"
            self._finish_semantic_task()
        else:
            self._activate_semantic_step()

    def _semantic_telemetry(self):
        elapsed = 0.0 if self._semantic_revision_started_at is None else self._time - self._semantic_revision_started_at
        return {"battery_percent": self._environment.battery, "elapsed_seconds": max(0.0, elapsed)}

    def _hold_semantic_task(self, message, *, status="needs_clarification"):
        """Stop task progress without disabling collision protection in flight."""
        self._running = self._plan_requested = False
        self._semantic_repair = None
        self.semantic.status = status
        self.semantic.message = message
        self._clear_motion()
        self._control_yaw = None
        self._hold_target = self._vehicle.position if self._safety_armed else None
        self.waypoints = WaypointManager()
        self._environment.last_feedback = message

    def _semantic_feedback(self, kind, message):
        event = {"kind": kind, "message": message, "position": point_to_dict(self._vehicle.position),
                 "simulation_time": self._time}
        self.semantic.execution_feedback = [*self.semantic.execution_feedback[-15:], event]
        self.trace.event("semantic_execution_feedback", self._time, feedback=event)

    def _request_semantic_review(self, instruction):
        if self._autonomous_reviews >= PROGRAM_LIMITS["max_autonomous_reviews"]:
            self._hold_semantic_task("本次任务已达到自主评估预算，保持安全等待，请提交后续任务。", status="safety_wait")
            return
        self._autonomous_reviews += 1
        self._semantic_review_pending = True
        self._invalidate_planning()
        self._clear_motion()
        self.waypoints = WaypointManager()
        self.semantic.status = "interpreting"
        self.semantic.message = "正在根据执行观测评估后续任务。"
        self._semantic_interpretation_pending = True
        self._semantic_schema_failures = 0
        self._semantic_repair = None
        self._request_replan("semantic_execution_review")
        self._feedback = ("Execution review: " + instruction +
                          " Use measured mission_context evidence and observations. Preserve every outstanding "
                          "obligation in mission_context.task_contract.remaining_steps, including order, repetitions, "
                          "conditions, geometry, dwell and final state, and retain the contract constraints.")
        self._semantic_feedback("review_requested", self.semantic.message)

    def _activate_semantic_step(self):
        step, goal = self.semantic.current_step, self.semantic.current_goal
        if step is None:
            return
        if step["action"] == "review":
            self._request_semantic_review(step["instruction"])
            return
        self._clear_motion()
        self._control_yaw = (math.radians(step["control"]["yaw_deg"])
                             if step["action"] == "control" and step["control"]["yaw_deg"] is not None else None)
        # The worker resolves a motion program and plans a safe transit to its entry.
        self._environment.target = tuple(goal) if goal is not None else self._vehicle.position
        self._environment.phase = "returning_home" if step["action"] == "return_home" else "fly_to_target"
        self._task = f"Execute {step['action']} at {step['target_id']} for uav-1."
        self.waypoints = WaypointManager()
        self.replanning = ReplanningManager()
        self.recovery = RecoveryMemory()
        self._request_replan("semantic_step:" + step["id"])

    def _clear_motion(self):
        self._motion = self._motion_follower = self._motion_controller = None
        self._motion_phase = self._motion_execution_id = None
        self._motion_duration = 0.0
        self._motion_heading_samples = 0
        self._motion_max_heading_error_deg = 0.0

    def _motion_progress(self):
        if self._motion_follower is None:
            return None
        follower = self._motion_follower
        return dict(execution_id=self._motion_execution_id, phase=self._motion_phase,
                    distance_m=round(follower.progress_m, 3),
                    total_length_m=round(follower.total_length_m, 3),
                    progress=min(1.0, follower.progress_m / max(follower.total_length_m, 1e-9)),
                    recovering=follower.snapshot().get("recovering", False))

    def _motion_yaw(self, state):
        heading = self._motion.heading
        if heading["mode"] == "fixed":
            return math.radians(heading["yaw_deg"])
        if heading["mode"] == "face_point":
            point = heading["point"]
            dx, dy = point[0] - state.position[0], point[1] - state.position[1]
            return math.atan2(dx, -dy) if math.hypot(dx, dy) > 1e-9 else state.yaw
        return None

    def _record_motion_heading(self):
        desired = self._motion_yaw(self._vehicle)
        if desired is not None:
            error = abs((self._vehicle.yaw - desired + math.pi) % (2 * math.pi) - math.pi)
            self._motion_heading_samples += 1
            self._motion_max_heading_error_deg = max(self._motion_max_heading_error_deg, math.degrees(error))

    def _collect_motion_route(self, future):
        try:
            plan, motion = future.result()
            candidate = self._context()
            candidate["goal"] = point_to_dict(motion.points[0])
            # Full path/corridor validation already ran in the worker. Generation
            # guards bind it to this map and task; only the moved departure needs rechecking.
            if SafetyChecker(candidate).validate_path(self._vehicle.position, plan.waypoints):
                # Motion may have changed while the worker ran (local avoidance).
                # Replan only the entry transfer; never discard task path progress.
                self._retry_at = self._time + .25
                return
        except (ValueError, SemanticConstraintError):
            self._hold_semantic_task("指定运动轨迹无法安全执行，请调整轨迹位置、高度或范围。",
                                     status="safety_wait")
            self.trace.event("motion_rejected", self._time, reason="unsafe_motion_path")
            return
        self._motion = motion
        self._motion_follower = PathFollower(motion.points, motion.speed_mps)
        self._motion_controller = TrajectoryController(
            max_speed=motion.speed_mps, max_acceleration=self.controller.max_acceleration,
            max_climb_speed=self.controller.max_climb_speed,
            max_descent_speed=self.controller.max_descent_speed)
        self._motion_phase = "transit"
        self._motion_execution_id = self.semantic.current_step["execution_id"]
        self._environment.target = motion.points[0]
        self._future_context["goal"] = point_to_dict(motion.points[0])
        self._accept_plan(plan, "Motion entry transit validated.")
        self.trace.event("motion_prepared", self._time, execution_id=self._motion_execution_id,
                         points=[list(p) for p in motion.points], length_m=motion.length_m,
                         speed_mps=motion.speed_mps, heading=motion.heading)

    def _advance_semantic_task(self, dt):
        step = self.semantic.current_step
        if self.semantic.status != "running" or step is None:
            return
        ready = (self._running and not self._plan_requested and not self._error
                 and not self.avoidance.active and bool(self.waypoints.points)
                 and self.waypoints.current is None)
        if step["action"] == "control" and self._control_yaw is not None:
            error = (self._vehicle.yaw - self._control_yaw + math.pi) % (2 * math.pi) - math.pi
            ready = ready and abs(error) <= math.radians(2)
        is_motion = step["action"] == "motion"
        if is_motion:
            if self._motion is None:
                return
            if self._motion_phase == "transit" and ready:
                entry = self.waypoints.points[-1]
                if distance(self._vehicle.position, entry) > .18:
                    self._environment.target = entry
                    self._request_replan("motion_entry_position_lost")
                    return
                desired = self._motion_yaw(self._vehicle)
                yaw_error = (abs((self._vehicle.yaw - desired + math.pi) % (2 * math.pi) - math.pi)
                             if desired is not None else 0.0)
                if norm(self._vehicle.velocity) > .02 or yaw_error > math.radians(2):
                    return
                self._motion_phase = "tracking"
                self._record_motion_heading()
                self._environment.target = self._motion.points[-1]
                self._plan_id += 1
                self.trace.event("motion_tracking", self._time, execution_id=self._motion_execution_id)
            if self._motion_phase == "tracking" and self._running:
                self._motion_duration += dt
            if not (self._motion_phase == "tracking" and self._motion_follower.completed
                    and self._running and not self._plan_requested and not self._error
                    and not self.avoidance.active):
                return
        else:
            goal = self.semantic.current_goal
            if ready and distance(self._vehicle.position, goal) > .18:
                self._request_replan("semantic_observation_position_lost")
                ready = False
        before = len(self.semantic.evidence)
        execution_error = None
        try:
            if is_motion:
                changed = self.semantic.complete_motion(self._vehicle.position, self._time, dict(
                    execution_id=self._motion_execution_id, completed=True,
                    covered_distance_m=self._motion_follower.progress_m,
                    total_length_m=self._motion_follower.total_length_m,
                    duration_seconds=self._motion_duration,
                    heading_samples=self._motion_heading_samples,
                    max_heading_error_deg=self._motion_max_heading_error_deg), telemetry=self._semantic_telemetry())
            else:
                changed = self.semantic.observe(self._vehicle.position, norm(self._vehicle.velocity), dt,
                                                self._time, enabled=ready, telemetry=self._semantic_telemetry(),
                                                measured_yaw_deg=math.degrees(self._vehicle.yaw))
        except SemanticExecutionError as exc:
            execution_error = exc
            changed = False
        for evidence in self.semantic.evidence[before:]:
            self.trace.event("semantic_evidence", self._time, evidence=evidence)
        if execution_error is not None:
            self._hold_semantic_task(execution_error.user_message, status="safety_wait")
            self.trace.event("semantic_execution_waiting", self._time, reason=execution_error.code)
            return
        if not changed:
            return
        if self.semantic.status == "completed":
            self._finish_semantic_task()
        elif self.semantic.status == "safety_wait":
            self._hold_semantic_task(self.semantic.message, status="safety_wait")
        else:
            self._activate_semantic_step()

    def _finish_semantic_task(self):
        acceptance = self.semantic.task_contract.verdict() if self.semantic.task_contract else None
        if not acceptance or not acceptance["satisfied"]:
            self._hold_semantic_task(SemanticExecutionError("task_contract_failed").user_message, status="safety_wait")
            self.trace.event("semantic_acceptance_failed", self._time, task_contract=acceptance)
            return
        self._running = False
        landed = self.semantic.goal_state["state"] == "landed"
        self._safety_armed = not landed
        self._hold_target = None if landed else self._vehicle.position
        self._environment.phase = self._environment.status = "completed"
        self._terminal_wall = time.monotonic()
        self._environment.last_feedback = self.semantic.message
        self._semantic_feedback("goal_satisfied", self.semantic.message)
        self.trace.event("mission_completed", self._time, milestones=self._milestones())

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
        if self.semantic is not None and self.semantic.status == "interpreting":
            self._submit_semantic_plan()
            return
        context = deepcopy(self._context())
        planner = GeometricPlanner() if self.semantic is not None else self.agent
        motion_step = (deepcopy(self.semantic.current_step) if self.semantic is not None
                       and self.semantic.current_step is not None
                       and self.semantic.current_step["action"] == "motion" and self._motion is None else None)
        motion_constraints = deepcopy(self.semantic.plan.constraints) if motion_step else None
        scenario = self._environment.scenario
        self._future_kind = "motion" if motion_step else "route"
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
                if motion_step:
                    motion = validate_motion_path(compile_motion(motion_step, scenario,
                        tuple(context["position"][k] for k in "xyz"), motion_constraints), context)
                    transfer = deepcopy(context)
                    transfer["goal"] = point_to_dict(motion.points[0])
                    return checked_plan(planner, transfer, feedback), motion
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
        if self._future_kind == "semantic":
            self._collect_semantic_plan(future)
            return
        if self._future_kind == "motion":
            self._collect_motion_route(future)
            return
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
                if self._vehicle is not None and self._environment.realtime_traffic:
                    self.avoidance.evidence.measure(self._vehicle.position, self._environment.dynamic_objects, self._time)
                for kind, episode in self.avoidance.evidence.events():
                    self.trace.event(kind, self._time, episode=episode)
                if self._vehicle is not None and self.semantic is None:
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
            with self.trace.stage("traffic_observation"):
                checker = SafetyChecker(self._context())
                observations = self._traffic_observer.sample(
                    self._environment.dynamic_objects if self._environment.realtime_traffic else [],
                    self._vehicle.position, checker, self._time)
                self.avoidance.observe(observations, self._time)
            measure_dynamic = self._safety_armed and self._environment.realtime_traffic
            speed = norm(self._vehicle.velocity)
            if (self._plan_requested and self._future is None and (speed <= 0.02 or self.avoidance.active)
                    and self._time >= self._retry_at and not self._error):
                self._submit_plan()
            tracking = self._motion_phase == "tracking"
            controller = self._motion_controller if tracking else self.controller
            active_motion = self._running and not self._plan_requested and not self._error
            target = (self._motion_follower.target(self._vehicle) if tracking else self.waypoints.current) if active_motion else None
            if not active_motion and self._hold_target is not None and not self._error and self._safety_armed:
                target = self._hold_target
            with self.trace.stage("traffic_prediction"):
                tracks = self.avoidance.predictions(
                    self._vehicle, self._time,
                    self.controller.max_speed) if self._safety_armed else []
            if (tracking and active_motion and self._motion_follower.recovering
                    and not checker.segment_safe(self._vehicle.position, self._motion_follower.rejoin_point)):
                self._environment.target = self._motion_follower.rejoin_point
                self._motion_phase = "transit"
                self._request_replan("motion_rejoin_blocked")
                target = None
            with self.trace.stage("route_reconnection"):
                if active_motion and target is not None and not tracking and self._time >= self._next_reconnect_at:
                    self._next_reconnect_at = self._time + 0.25
                    def reachable(point, final):
                        result = self.avoidance.rollout(self._vehicle, point, self.controller, checker, tracks,
                            terminal=final and distance(self._vehicle.position, point) <= 6, capture=True)
                        return (result is not None and result.clearance >= self.avoidance.BUFFER
                                and all(segment_deviation(position, self._vehicle.position, point)
                                        <= MAX_TRACKING_DEVIATION_M for position in result.path))
                    if self.waypoints.reconnect(self._vehicle, checker, reachable):
                        self.replanning.accepted(self._context(), self._time)
                        self._environment.last_feedback = "Safe forward route connection; obsolete transit waypoints skipped."
                    target = self.waypoints.current
            if active_motion and target is not None and not tracking and not self.avoidance.active:
                reason = self.replanning.should_replan(self._context(), self._vehicle, self.waypoints, checker, self._time)
                if reason:
                    self._request_replan(reason)
                    target = None
            was_avoiding = self.avoidance.active
            if self._safety_armed:
                progress_goal = ((self._motion_execution_id, self._motion.points[-1]) if tracking else
                                 self.waypoints.points[-1] if self.waypoints.current is not None else None)
                remaining = (self._motion_follower.remaining_distance_m if tracking else
                             self.waypoints.remaining_distance(self._vehicle.position))
                blocked = self.avoidance.stalled(progress_goal, remaining, self._time)
                with self.trace.stage("avoidance"):
                    target = self.avoidance.choose(self._vehicle, target, controller, checker, tracks, self._time,
                        terminal=(self._motion_follower.remaining_distance_m < .5 if tracking
                                  else self.waypoints.index == len(self.waypoints.points) - 1))
                if was_avoiding and not self.avoidance.active:
                    # Intentional local detours are not tracking failures.
                    if not tracking:
                        self.waypoints.segment_start = self._vehicle.position
                        self.replanning.accepted(self._context(), self._time)
                    self._environment.last_feedback = self.avoidance.message
                if (self.avoidance.active and self._running and not self._plan_requested
                        and blocked
                        and self._time - self._last_dynamic_plan_at >= 10):
                    if tracking:
                        self._environment.target = self._motion_follower.rejoin_point
                        self._motion_phase = "transit"
                    self._request_replan("dynamic_obstacle:persistent_local_blockage")
                    self._feedback = self.avoidance.message + " Replan from the updated position around this traffic."
            velocity = controller.velocity_command(self._vehicle, target, dt)
            projected = tuple(self._vehicle.position[i] + (self._vehicle.velocity[i] + velocity[i]) * dt / (2 * SCALE[i]) for i in range(3))
            if not checker.segment_safe(self._vehicle.position, projected):
                self.evidence.static_blocked = True
                self._fail("Safety Checker blocked the next control segment.")
                return
            aligning_motion = (self._motion is not None and self._motion_phase == "transit"
                               and bool(self.waypoints.points) and self.waypoints.current is None)
            # A pause brakes translation; it must not silently rotate an
            # independently commanded heading to the residual velocity vector.
            yaw = (self._motion_yaw(self._vehicle) if self._motion is not None and (tracking or aligning_motion)
                   else self._control_yaw)
            if target is not None or speed > 0.0001 or (active_motion and yaw is not None):
                try:
                    with self.trace.stage("runtime"):
                        vehicle = (self.runtime.drive(velocity, dt, yaw=yaw) if yaw is not None
                                   else self.runtime.drive(velocity, dt))
                except Exception:
                    self._fail("Runtime command/telemetry failed; motion stopped.")
                    return
                self._commit_vehicle(vehicle, dt)
                self._runtime_status = self.runtime.status()
                if tracking and active_motion and not self._plan_requested:
                    self._record_motion_heading()
                    self._motion_follower.update(vehicle, dt, avoiding=self.avoidance.active or was_avoiding)
                final_waypoint = self.waypoints.index == len(self.waypoints.points) - 1
                next_segment_safe = (final_waypoint or self.waypoints.current is None
                                     or checker.segment_safe(vehicle.position, self.waypoints.points[self.waypoints.index + 1]))
                if (active_motion and target is not None and not tracking and not self.avoidance.active and not self._error
                        and next_segment_safe
                        and self.waypoints.update(vehicle) and self.waypoints.current is None):
                    mission_goal = self._environment.target if self._environment.phase == "fly_to_target" else self._environment.base
                    if distance(vehicle.position, mission_goal) > 0.18:
                        self._request_replan("goal_arrival_not_verified")
                        self._feedback = "Route endpoint reached but mission goal not verified. End exactly at the current goal coordinates."
                        self.recovery.stalled(vehicle.position, distance(vehicle.position, mission_goal), self._time)
                    elif self.semantic is not None:
                        # Skill observation/dwell below owns completion and the next subgoal.
                        pass
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
            if self.semantic is not None:
                self._advance_semantic_task(dt)
            if measure_dynamic:
                incidents = self._evaluation.observe_dynamic(
                    self._vehicle.position, self._environment.dynamic_objects, self._time)
                for incident in incidents:
                    self.trace.event(incident["kind"], self._time, incident=incident)
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
            self._path_start_index += max(0, len(env.continuous_path) - 10000)
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
        self._semantic_repair = None
        self._plan_requested = False
        self._generation += 1
        self._error = message
        self._terminal_wall = time.monotonic()
        self._environment.last_feedback = message
        if self.semantic is not None:
            self.semantic.status = "error"
            self.semantic.message = message
        self.trace.event("mission_error", self._time, reason=message, phase=self._environment.phase)
        if planning:
            # A failed remote plan must not disable local collision protection.
            return
        self._safety_armed = False
        evidence = self.avoidance.evidence
        evidence.interrupt(self._time)
        self.avoidance = LocalAvoidance(calibration_profile=self._environment.scenario_id)
        self.avoidance.evidence = evidence
        try:
            self.runtime.stop()
            self._vehicle = self.runtime.read_state()
            self._environment.position = self._vehicle.position
        except Exception:
            pass

    def stop_runtime(self):
        with self._lock:
            if self.semantic is not None:
                self.semantic.observe(self._environment.position, 0, 0, self._time, enabled=False)
            self._running = False
            self._invalidate_planning()
            self._plan_requested = False
            self._environment.last_feedback = "Mission paused; decelerating to hold. Local collision protection remains active."
            self.trace.event("mission_paused", self._time, milestones=self._milestones())
            self._publish_state()
            return self.state()

    def _reset_control(self, *, traffic=None):
        self._semantic_repair = None
        self.avoidance.evidence.interrupt(self._time)
        for kind, episode in self.avoidance.evidence.events():
            self.trace.event(kind, self._time, episode=episode)
        self._generation += 1
        self._semantic_interpretation_pending = False
        self._semantic_revision_started_at = None
        self._clear_motion()
        self._running = False
        self._plan_requested = False
        self._error = None
        self._plan_id = 0
        self._planner_calls = 0
        self._last_dynamic_plan_at = self._time - 10.0
        self.waypoints = WaypointManager()
        self.replanning = ReplanningManager()
        self.avoidance = LocalAvoidance(calibration_profile=self._environment.scenario_id)
        self._traffic_observer = TrafficObserver(self._observation_config, seed=self._traffic_seed)
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
        # A new run begins at measured telemetry, including landing tolerance;
        # do not count a synthetic base-to-current-position segment as flight.
        self._environment.path = [self._vehicle.position]
        self._environment.continuous_path = [point_to_dict(self._vehicle.position)]
        self._path_start_index = 0
        self._evaluation.reset()
        self.semantic = (SemanticMission(self._environment.scenario, mission_id=self.trace.run_id)
                         if self._environment.scenario.mission_mode == "semantic" else None)
        self._semantic_calls = 0
        self._ground_sites = []
        self._hold_target = None
        self._control_yaw = None
        self._semantic_schema_failures = 0
        self._autonomous_reviews = 0
        self._semantic_review_pending = False
        self._publish_state()

    def reset(self):
        with self._lock:
            scenario = self._random_scenario(self._scenario_template,
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
            template = load_scenario(scenario_id)
            scenario = self._random_scenario(template, previous=self._environment.target)
            if scenario.mission_mode == "semantic" and self.runtime.backend_name != "html":
                raise ValueError("请先切换到 HTML；园区观测技能目前仅支持仿真。")
            self._traffic_seed = secrets.randbits(32)
            traffic = self._make_traffic(scenario)
            self.runtime.stop()
            self._running = False
            if self.runtime.backend_name == "airsim":
                self.runtime.configure_scene(scenario)
            self.runtime.reset(scenario.base)
            self._environment = UAVEnvironment(scenario)
            self._scenario_template = template
            if scenario.mission_mode == "semantic":
                self.agent = DeepSeekAgent(**self._deepseek_config)
            self._reset_control(traffic=traffic)
            return self.state()

    def switch_runtime(self, backend, config=None):
        if backend not in {"html", "airsim"}:
            raise ValueError("Runtime backend must be 'html' or 'airsim'.")
        with self._lock:
            if self.semantic is not None and backend != "html":
                raise ValueError("园区任务的观测技能目前仅支持 HTML 仿真。")
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
                candidate.acquire_control()
            self._environment.reset()
            self._reset_control()
            return self.state()

    def update_goal(self, target, task=None):
        with self._lock:
            if self.semantic is not None:
                raise ValueError("园区的目标由任务步骤确定，请通过任务输入框安排任务。")
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
            self.avoidance.evidence.interrupt(self._time)
            for kind, episode in self.avoidance.evidence.events():
                self.trace.event(kind, self._time, episode=episode)
            self._generation += 1
            self._running = False
            self._safety_armed = False
            self.runtime.stop()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.trace.event("run_closed", self._time, milestones=self._milestones(), diagnostics=self.trace.snapshot(),
                         dynamic_incidents=self._evaluation.dynamic_incidents)
        self._close_quietly(self.runtime)

    def provider_config(self):
        with self._lock:
            return self._provider_profiles.public(self.agent.backend_name)

    def provider_models(self, config):
        if not isinstance(config, Mapping):
            raise ProviderConfigError("configuration_invalid", "请提交有效的 API 配置。")
        with self._lock:
            candidate = self._provider_profiles.prepare(config)
        return discover_models(candidate)

    def configure_agent(self, config: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(config, Mapping):
            raise TypeError("Agent config must be an object.")
        backend = str(config.get("backend", "mock")).strip().lower()
        if backend not in {"mock", "deepseek"}:
            raise ValueError(
                "Agent backend must be 'mock' or 'deepseek'. "
                "The OpenAI backend is not implemented in this demo."
            )

        if type(config.get("verify", False)) is not bool:
            raise ProviderConfigError("configuration_invalid", "配置校验开关格式无效。")
        with self._lock:
            if self.semantic is not None and backend != "deepseek":
                raise ValueError("园区任务需要 DeepSeek 理解自然语言，不支持 Mock 模式。")
            revision = self._provider_profiles.revision
            if backend == "deepseek":
                profile = self._provider_profiles.prepare(config)
        if backend == "deepseek":
            # Remote metadata lookup must never hold the flight/control lock.
            # No live configuration is replaced until the complete check passes.
            if config.get("verify"):
                discovery = discover_models(profile)
                if profile["model"] not in discovery["models"]:
                    raise ProviderConfigError("model_unavailable", "所选模型不在当前服务商列表中，请重新读取并选择模型。")
            candidate = DeepSeekAgent(api_key=profile["api_key"], model=profile["model"] or None,
                                      base_url=profile["base_url"])
            next_config = dict(api_key=candidate.api_key, model=candidate.model, base_url=candidate.base_url)
        with self._lock:
            if revision != self._provider_profiles.revision:
                raise ProviderConfigError("provider_changed", "配置已在另一次操作中更新，请重新打开设置。")
            if self.semantic is not None and backend != "deepseek":
                raise ValueError("园区任务需要 DeepSeek 理解自然语言，不支持 Mock 模式。")
            was_requested = self._plan_requested
            if backend == "deepseek":
                self._provider_profiles.commit({**profile, **next_config})
                self._deepseek_config = next_config
                self.agent = candidate
            else:
                self.agent = build_agent(backend)
                self._provider_profiles.revision += 1
            self._invalidate_planning()
            if self.semantic is not None:
                # Editing a key must not resume an explicitly paused paid request.
                self._plan_requested = was_requested
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
    def _capture_deepseek_config(agent) -> dict[str, str | None]:
        return {
            "api_key": getattr(agent, "api_key", None) or os.getenv("DEEPSEEK_API_KEY"),
            "model": getattr(agent, "model", None)
            or os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
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
