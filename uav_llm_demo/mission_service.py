from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from threading import RLock

from .agents import AgentDecision, DeepSeekAgent, build_agent
from .evaluation import EvaluationLog
from .environment import (
    DIRECTION_DELTAS,
    UAVEnvironment,
    action_between,
    conflicts_with_dynamic,
    format_point,
    is_safe_cell,
    shortest_next_step,
    shortest_path_distance,
)
from .html_runtime import HtmlRuntime
from .scenarios import load_scenario
from .runtime import (
    RuntimeCapabilityError,
    RuntimeUnavailableError,
    SimulationRuntime,
)


RuntimeFactory = Callable[[str, Mapping[str, object]], SimulationRuntime]


class MissionService:
    def __init__(
        self,
        *,
        agent=None,
        runtime: SimulationRuntime | None = None,
        runtime_factory: RuntimeFactory | None = None,
        scenario_id: str = "urban_crossing",
    ) -> None:
        self._environment = UAVEnvironment(load_scenario(scenario_id))
        self.agent = agent or build_agent(os.getenv("UAV_LLM_BACKEND", "mock"))
        self.runtime = runtime or HtmlRuntime()
        self._runtime_factory = runtime_factory
        self._evaluation = EvaluationLog()
        self._lock = RLock()

    def state(self, feedback: str | None = None) -> dict[str, object]:
        with self._lock:
            payload = self._environment.snapshot(feedback=feedback)
            runtime_status = self.runtime.status()
            payload["agent_backend"] = self.agent.backend_name
            payload["agent_model"] = getattr(self.agent, "model", None)
            payload["runtime_backend"] = self.runtime.backend_name
            payload["runtime"] = runtime_status
            telemetry = runtime_status.get("telemetry") or {}
            actual_position = telemetry.get("position_grid")
            if actual_position:
                payload["uav"] = actual_position
            evaluation = self._evaluation.snapshot(
                final_status=self._environment.status,
                trajectory=payload["path"],
                battery=self._environment.battery,
            )
            payload["evaluation"] = evaluation
            payload["metrics"] = evaluation["metrics"]
            return payload

    def step(self) -> dict[str, object]:
        with self._lock:
            if self._environment.status != "active":
                return self.state("Mission is not active. Use Reset to start again.")

            observation = self._environment.get_observation()
            try:
                decision = self.agent.decide(observation)
            except Exception as exc:
                decision = AgentDecision(
                    action="hover",
                    reason=f"Agent backend error: {exc}",
                    risk="high",
                    expected_result=(
                        "UAV will hover because no valid LLM decision was available."
                    ),
                    skill="hover",
                    waypoint={
                        "x": float(observation.position[0]),
                        "y": float(observation.position[1]),
                        "z": float(observation.position[2]),
                    },
                )
            decision = validated_decision(self._environment, observation, decision)
            candidate = deepcopy(self._environment)
            feedback = candidate.apply_action(decision.action)

            try:
                result = self.runtime.execute(candidate.last_trajectory)
            except RuntimeUnavailableError:
                raise
            except Exception as exc:
                raise RuntimeUnavailableError(str(exc)) from exc

            candidate.set_executed_trajectory(
                list(result.trajectory),
                result.trajectory_duration,
                result.kinematics,
            )
            feedback = (
                f"{feedback} {self.runtime.backend_name.upper()} runtime executed "
                "a smooth continuous trajectory."
            )
            candidate.last_feedback = feedback
            candidate.record_decision(observation, decision.to_dict(), feedback)
            self._evaluation.record(
                {
                    "step": observation.step,
                    "scenario_id": observation.scenario_id,
                    "observation": observation.to_dict(),
                    "decision": decision.to_dict(),
                    "feedback": feedback,
                    "runtime": self.runtime.backend_name,
                }
            )
            self._environment = candidate
            return self.state(feedback)

    def reset(self) -> dict[str, object]:
        with self._lock:
            try:
                self.runtime.reset(self._environment.base)
            except RuntimeUnavailableError:
                raise
            except Exception as exc:
                raise RuntimeUnavailableError(str(exc)) from exc
            self._environment.reset()
            self._evaluation.reset()
            return self.state()

    def switch_scenario(self, scenario_id: str) -> dict[str, object]:
        scenario = load_scenario(scenario_id)
        with self._lock:
            try:
                self.runtime.reset(scenario.base)
            except RuntimeUnavailableError:
                raise
            except Exception as exc:
                raise RuntimeUnavailableError(str(exc)) from exc
            self._environment = UAVEnvironment(scenario)
            self._evaluation.reset()
            self._environment.last_feedback = (
                f"Scenario switched to {scenario.scenario_id}; mission reset at base."
            )
            return self.state()

    def switch_runtime(
        self,
        backend: str,
        config: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        normalized = backend.strip().lower()
        if normalized not in {"html", "airsim"}:
            raise ValueError("Runtime backend must be 'html' or 'airsim'.")

        options = dict(config or {})
        with self._lock:
            candidate: SimulationRuntime | None = None
            try:
                if normalized == "html":
                    candidate = HtmlRuntime()
                else:
                    if self._runtime_factory is None:
                        raise RuntimeUnavailableError(
                            "AirSim runtime factory is not configured."
                        )
                    candidate = self._runtime_factory(normalized, options)
                candidate.reset(self._environment.base)
                if normalized == "airsim" and "wind_ned" in options:
                    self._set_runtime_wind(candidate, options["wind_ned"])
            except RuntimeUnavailableError:
                if candidate is not None:
                    self._close_quietly(candidate)
                raise
            except Exception as exc:
                if candidate is not None:
                    self._close_quietly(candidate)
                raise RuntimeUnavailableError(str(exc)) from exc

            previous = self.runtime
            self.runtime = candidate
            self._environment.reset()
            self._evaluation.reset()
            if previous is not candidate:
                self._close_quietly(previous)
            self._environment.last_feedback = (
                f"Runtime switched to {normalized}; mission reset at base."
            )
            return self.state()

    def stop_runtime(self) -> dict[str, object]:
        runtime = self.runtime
        try:
            runtime.stop()
        except RuntimeUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeUnavailableError(str(exc)) from exc
        with self._lock:
            if runtime is self.runtime:
                self._environment.last_feedback = (
                    f"{runtime.backend_name.upper()} runtime stopped."
                )
            return self.state()

    def set_wind(self, wind: Mapping[str, object]) -> dict[str, object]:
        with self._lock:
            if not self.runtime.status().get("supports_wind"):
                raise RuntimeCapabilityError(
                    f"{self.runtime.backend_name.upper()} runtime does not support wind."
                )
            self._set_runtime_wind(self.runtime, wind)
            self._environment.last_feedback = "AirSim wind vector updated."
            return self.state()

    def configure_agent(self, config: Mapping[str, object]) -> dict[str, object]:
        backend = str(config.get("backend", "mock")).strip().lower()
        with self._lock:
            if backend == "deepseek":
                self.agent = DeepSeekAgent(
                    api_key=config.get("api_key"),
                    model=config.get("model"),
                    base_url=config.get("base_url"),
                )
            else:
                self.agent = build_agent(backend)
            return self.state()

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


def validated_decision(
    env: UAVEnvironment,
    observation,
    decision: AgentDecision,
) -> AgentDecision:
    if decision.reason.startswith("Agent backend error:"):
        return decision

    if observation.phase == "returning_home":
        if decision.action == "return_home":
            return decision
        return _override_decision(
            observation,
            decision,
            "return_home",
            "mission is already in returning_home phase",
        )

    if observation.battery <= 20:
        if decision.action == "return_home":
            return decision
        return _override_decision(
            observation,
            decision,
            "return_home",
            "battery is below the return-home threshold",
        )

    if _action_progresses_toward_goal(observation, decision.action):
        return decision

    planner_action = _planner_action(observation)
    if planner_action == decision.action:
        return decision

    return _override_decision(
        observation,
        decision,
        planner_action,
        "proposed action does not reduce safe path distance to the current goal",
    )


def _action_progresses_toward_goal(observation, action: str) -> bool:
    if action not in DIRECTION_DELTAS:
        return False

    dx, dy, dz = DIRECTION_DELTAS[action]
    proposed = (
        observation.position[0] + dx,
        observation.position[1] + dy,
        observation.position[2] + dz,
    )
    dynamic_positions = _predicted_dynamic_positions(observation)
    if not is_safe_cell(
        proposed,
        observation.buildings,
        observation.grid_size,
        observation.min_altitude,
        observation.max_altitude,
    ):
        return False
    if conflicts_with_dynamic(proposed, dynamic_positions):
        return False

    current_distance = shortest_path_distance(
        observation.position,
        observation.goal,
        observation.buildings,
        dynamic_positions,
        observation.grid_size,
        observation.min_altitude,
        observation.max_altitude,
    )
    proposed_distance = shortest_path_distance(
        proposed,
        observation.goal,
        observation.buildings,
        dynamic_positions,
        observation.grid_size,
        observation.min_altitude,
        observation.max_altitude,
    )
    return (
        current_distance is not None
        and proposed_distance is not None
        and proposed_distance < current_distance
    )


def _planner_action(observation) -> str:
    next_step = shortest_next_step(
        observation.position,
        observation.goal,
        observation.buildings,
        _predicted_dynamic_positions(observation),
        observation.grid_size,
        observation.min_altitude,
        observation.max_altitude,
    )
    if next_step is None:
        return "hover"
    return action_between(observation.position, next_step)


def _override_decision(
    observation,
    decision: AgentDecision,
    action: str,
    reason: str,
) -> AgentDecision:
    if action == "return_home":
        expected_result = (
            f"Return-home primitive will move one safe step toward {format_point(observation.base)}."
        )
    elif action == "hover":
        expected_result = (
            "UAV will hold position because no safe progress action is available."
        )
    else:
        dx, dy, dz = DIRECTION_DELTAS[action]
        next_point = (
            observation.position[0] + dx,
            observation.position[1] + dy,
            observation.position[2] + dz,
        )
        expected_result = f"Executor will move UAV to {format_point(next_point)}."

    waypoint = None
    if action == "return_home":
        waypoint = {
            "x": float(observation.base[0]),
            "y": float(observation.base[1]),
            "z": float(observation.base[2]),
        }
    elif action in DIRECTION_DELTAS:
        dx, dy, dz = DIRECTION_DELTAS[action]
        waypoint = {
            "x": float(observation.position[0] + dx),
            "y": float(observation.position[1] + dy),
            "z": float(observation.position[2] + dz),
        }
    else:
        waypoint = {
            "x": float(observation.position[0]),
            "y": float(observation.position[1]),
            "z": float(observation.position[2]),
        }

    return AgentDecision(
        action=action,
        reason=(
            f"Executor arbitration: DeepSeek proposed '{decision.action}', but {reason}. "
            f"Overriding to '{action}'. Original reason: {decision.reason}"
        ),
        risk="medium" if decision.risk == "low" else decision.risk,
        expected_result=expected_result,
        skill="return_home" if action == "return_home" else "hover" if action == "hover" else "navigate_to",
        waypoint=waypoint,
        proposed_action=decision.proposed_action or decision.action,
        safety_intervention=reason,
    )


def _predicted_dynamic_positions(
    observation,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            float(item["predicted_position"]["x"]),
            float(item["predicted_position"]["y"]),
            float(item["predicted_position"]["z"]),
        )
        for item in observation.dynamic_objects
    )
