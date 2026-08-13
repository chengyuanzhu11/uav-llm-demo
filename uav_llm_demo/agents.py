from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import requests

from .environment import (
    ACTIONS,
    Observation,
    action_between,
    format_point,
    point_to_dict,
    shortest_next_step,
)


@dataclass(frozen=True)
class AgentDecision:
    action: str
    reason: str
    risk: str
    expected_result: str
    skill: str = "navigate_to"
    waypoint: dict[str, float] | None = None
    proposed_action: str | None = None
    safety_intervention: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["safety_intervened"] = self.safety_intervention is not None
        return data


class BaseLLMAgent:
    backend_name = "base"

    def decide(self, observation: Observation) -> AgentDecision:
        raise NotImplementedError

    def build_prompt(self, observation: Observation) -> str:
        contract = (
            "You are a high-level UAV mission decision agent. Do not control motors, "
            "attitude, or low-level flight stabilization. Return one JSON object with "
            "action, skill, waypoint, reason, risk, and expected_result. skill must be "
            "navigate_to, return_home, or hover. waypoint must be an {x,y,z} object "
            "when navigation is possible. The executor may reject unsafe commands; "
            "never hide uncertainty or invent a successful result.\n"
        )
        return contract + (
            "你是一个 UAV 高层决策 Agent。你不直接控制电机，只为底层飞控执行器选择一个高层动作。\n"
            f"只能从这些动作中选择一个：{', '.join(observation.valid_actions)}。\n"
            "必须按屏幕网格坐标理解方向：原点在左上角，x 增大表示向东/向右，"
            "y 增大表示向南/向下，y 减小表示向北/向上，z 增大表示上升。\n"
            "判断目标方向时必须先看 Goal delta：dx>0 是 east/right，dx<0 是 west/left，"
            "dy>0 是 south/down，dy<0 是 north/up，dz>0 是 above。\n"
            "不要把 y 更小的目标解释成 south；那是 north/up。\n"
            "动态飞行物的 predicted next position 是下一动作周期的位置，必须保持安全间隔。\n"
            "如果任务阶段是 returning_home，优先选择 return_home，而不是手动选择 move_north/move_south。\n"
            "只返回一个 JSON object，不要 Markdown，不要额外解释，不要输出隐藏推理过程。\n"
            "字段必须是：action, reason, risk, expected_result。\n"
            "risk 只能是 low、medium 或 high。\n\n"
            f"当前观测：\n{observation.to_text()}"
        )


class MockLLMAgent(BaseLLMAgent):
    backend_name = "mock"

    def decide(self, observation: Observation) -> AgentDecision:
        if observation.status != "active":
            return AgentDecision(
                action="hover",
                reason="任务已经结束，保持悬停等待人工复位。",
                risk="low",
                expected_result="UAV 保持当前位置不再推进任务。",
                skill="hover",
                waypoint=point_to_dict(observation.position),
            )

        if observation.battery <= 20 and observation.phase != "returning_home":
            return AgentDecision(
                action="return_home",
                reason="电量已经接近安全阈值，高层 Agent 优先触发返航原语。",
                risk=self._risk_level(observation),
                expected_result="飞控执行返航原语，沿安全路径向基地移动一步。",
                skill="return_home",
                waypoint=point_to_dict(observation.base),
            )

        if observation.phase == "returning_home":
            return AgentDecision(
                action="return_home",
                reason="目标已经到达，当前任务阶段要求返回基地。",
                risk=self._risk_level(observation),
                expected_result="返航原语会选择下一格安全航点，逐步接近基地。",
                skill="return_home",
                waypoint=point_to_dict(observation.base),
            )

        next_step = shortest_next_step(
            observation.position,
            observation.goal,
            observation.buildings,
            _predicted_dynamic_positions(observation),
        )
        if next_step is None:
            return AgentDecision(
                action="hover",
                reason="当前观测下没有可达的安全路径，Agent 选择悬停等待新观测。",
                risk="high",
                expected_result="UAV 保持位置，避免进入未知或阻塞区域。",
                skill="hover",
                waypoint=point_to_dict(observation.position),
            )

        action = action_between(observation.position, next_step)
        if action not in ACTIONS:
            action = "hover"

        return AgentDecision(
            action=action,
            reason=(
                f"观测显示目标在 {format_point(observation.goal)}，"
                f"下一安全航点 {format_point(next_step)} 可以降低目标距离并避开障碍。"
            ),
            risk=self._risk_level(observation),
            expected_result=f"执行 {action} 后，UAV 将移动到 {format_point(next_step)}。",
            skill="navigate_to",
            waypoint=point_to_dict(next_step),
        )

    def _risk_level(self, observation: Observation) -> str:
        dynamic = observation.nearest_dynamic_obstacle
        if dynamic and dynamic["threat_level"] in {"critical", "high"}:
            return "high"
        if dynamic and dynamic["threat_level"] == "medium":
            return "medium"

        static = observation.nearest_static_obstacle
        if static and int(static["distance"]) <= 1:
            clearance = observation.position[2] - int(static["height"])
            if clearance <= 1:
                return "high"
        if static and int(static["distance"]) <= 3:
            return "medium"
        return "low"


def _predicted_dynamic_positions(
    observation: Observation,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            float(item["predicted_position"]["x"]),
            float(item["predicted_position"]["y"]),
            float(item["predicted_position"]["z"]),
        )
        for item in observation.dynamic_objects
    )


class DeepSeekAgent(BaseLLMAgent):
    backend_name = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 25,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def decide(self, observation: Observation) -> AgentDecision:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required when UAV_LLM_BACKEND=deepseek.")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是 UAV 高层决策 Agent。输出必须是合法 JSON object，"
                            "不能包含 Markdown 或额外文本。"
                        ),
                    },
                    {"role": "user", "content": self.build_prompt(observation)},
                ],
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
                "temperature": 0.1,
                "max_tokens": 280,
                "stream": False,
            },
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"DeepSeek API request failed with HTTP {response.status_code}: {response.text[:300]}")

        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return decision_from_json_text(content)


class OpenAICompatibleAgent(BaseLLMAgent):
    backend_name = "openai_compatible"

    def decide(self, observation: Observation) -> AgentDecision:
        raise RuntimeError("OpenAI backend is not implemented in this demo. Use mock or deepseek.")


def build_agent(backend: str) -> BaseLLMAgent:
    normalized = backend.strip().lower()
    if normalized == "mock":
        return MockLLMAgent()
    if normalized == "openai":
        return OpenAICompatibleAgent()
    if normalized == "deepseek":
        return DeepSeekAgent()
    return MockLLMAgent()


def decision_from_json_text(content: str) -> AgentDecision:
    parsed = json.loads(strip_json_fence(content))
    action = str(parsed.get("action", "hover")).strip()
    if action not in ACTIONS:
        action = "hover"

    risk = str(parsed.get("risk", "medium")).strip().lower()
    if risk not in {"low", "medium", "high"}:
        risk = "medium"

    skill = str(parsed.get("skill", "")).strip().lower()
    if skill not in {"navigate_to", "return_home", "hover"}:
        skill = "return_home" if action == "return_home" else "hover" if action == "hover" else "navigate_to"

    raw_waypoint = parsed.get("waypoint")
    waypoint = None
    if isinstance(raw_waypoint, dict) and {"x", "y", "z"}.issubset(raw_waypoint):
        try:
            waypoint = {
                "x": float(raw_waypoint["x"]),
                "y": float(raw_waypoint["y"]),
                "z": float(raw_waypoint["z"]),
            }
        except (TypeError, ValueError):
            waypoint = None

    return AgentDecision(
        action=action,
        reason=str(parsed.get("reason", "DeepSeek returned a structured UAV action.")),
        risk=risk,
        expected_result=str(parsed.get("expected_result", "The flight executor will validate and execute the action.")),
        skill=skill,
        waypoint=waypoint,
    )


def strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
