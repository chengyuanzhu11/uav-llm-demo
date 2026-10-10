from __future__ import annotations

import heapq
import json
import json as json_module
import math
import os
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import requests

from .safety import MAX_WAYPOINTS, Point, SafetyChecker, _point, metric_distance
from .provider_usage import prefix_metrics, record_request


PLANNER_SYSTEM_PROMPT = (
    "Plan a complete UAV route from position to goal, not motor controls or discrete actions. "
    "Return only JSON with exactly waypoints, reason, risk, expected_result. "
    "waypoints is 1 to 64 {x,y,z} numeric objects in GRID coordinates, not metres. "
    "XY scale is 3 metres/cell; Z scale is 5 metres/level. x increases east, "
    "y increases south, z increases upward. Bounds: 0 <= x,y <= grid_size-1, "
    "altitude_range.min <= z <= altitude_range.max. "
    "building_volumes are closed axis-aligned GRID boxes from min to max. "
    "When buildings are provided instead, they are centered at x,y, footprint 0.78 cells "
    "square, from z=0 to height. Keep 0.6 metres clearance from building and no-fly AABBs "
    "on EVERY continuous segment, including the segment from position to first waypoint. "
    "Final waypoint must be within 0.2 metres of goal. risk must be low, medium or high. "
    "No Markdown or extra fields. Do not claim execution success. Dynamic objects are "
    "snapshots; the executor handles near-term motion safety separately. "
    "When preferred_altitude is set, stay in that GRID altitude band and route between "
    "buildings. Only climb above it if a low route is unreachable. For endpoints outside "
    "the band use safe vertical entry/exit transitions where possible. "
    "Scene contains map constraints, Mission contains task goals, State contains the current "
    "observation. Feedback describes a rejected plan; correct it without hiding uncertainty."
)


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RoutePlan:
    waypoints: tuple[Point, ...]
    reason: str
    risk: str
    expected_result: str

    def to_dict(self) -> dict:
        return {
            "waypoints": [dict(zip(("x", "y", "z"), point)) for point in self.waypoints],
            "reason": self.reason,
            "risk": self.risk,
            "expected_result": self.expected_result,
        }


class PlanValidationError(ValueError):
    """Retain a parsed proposal and geometric rejection evidence for recovery."""
    def __init__(self, plan, errors):
        super().__init__("; ".join(errors))
        self.plan = plan
        self.errors = errors


class PlannerProviderError(RuntimeError):
    """A fixed, credential-free provider diagnostic for the mission trace."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def _post_deepseek(url, *, headers, json: dict, timeout):
    if os.name != "nt":
        return requests.post(url, headers=headers, json=json, timeout=timeout)

    script = Path(__file__).resolve().parents[1] / "scripts" / "deepseek_request.ps1"
    payload = {"url": url, "api_key": headers["Authorization"].removeprefix("Bearer "),
               "body": json_module.dumps(json, ensure_ascii=False), "timeout": timeout}
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(script)],
            input=json_module.dumps(payload, ensure_ascii=False), text=True,
            encoding="utf-8", capture_output=True, timeout=timeout + 5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode or not result.stdout:
            raise requests.ConnectionError()
        reply = json_module.loads(result.stdout)
        if reply.get("error") == "timeout":
            raise requests.Timeout()
        if reply.get("error") or not isinstance(reply.get("status"), int):
            raise requests.ConnectionError()
        response = requests.Response()
        response.status_code = reply["status"]
        response._content = reply.get("body", "").encode("utf-8")
        return response
    except subprocess.TimeoutExpired:
        raise requests.Timeout() from None
    except requests.RequestException:
        raise
    except (OSError, ValueError, KeyError, AttributeError):
        raise requests.ConnectionError() from None


def plan_from_json_text(content: str) -> RoutePlan:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field.")
            result[key] = value
        return result

    try:
        parsed = json.loads(content, object_pairs_hook=unique_object)
        if not isinstance(parsed, dict) or set(parsed) != {"waypoints", "reason", "risk", "expected_result"}:
            raise ValueError
        points = parsed["waypoints"]
        if not isinstance(points, list) or not 1 <= len(points) <= MAX_WAYPOINTS:
            raise ValueError
        if any(not isinstance(parsed[key], str) or not parsed[key].strip()
               for key in ("reason", "risk", "expected_result")):
            raise ValueError
        if parsed["risk"] not in {"low", "medium", "high"}:
            raise ValueError
        return RoutePlan(tuple(_point(point) for point in points), parsed["reason"],
                         parsed["risk"], parsed["expected_result"])
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("Invalid route plan JSON or schema.") from None


class BaseLLMAgent:
    backend_name = "base"

    def plan(self, context: dict, feedback: str = "") -> RoutePlan:
        raise NotImplementedError

    def build_prompt(self, context: dict, feedback: str = "") -> str:
        prompt_context = dict(context)
        parcels = prompt_context.pop("urban_parcels", [])
        if parcels:
            volumes = []
            for parcel in parcels:
                left, right, north, south = parcel["footprint"]
                volumes.append(dict(
                    id=parcel["id"] + "-podium",
                    min=dict(x=left - .39, y=north - .39, z=0),
                    max=dict(x=right + .39, y=south + .39, z=parcel["podium_height"]),
                ))
                if parcel["tower"]:
                    left, right, north, south, height = parcel["tower"]
                    volumes.append(dict(
                        id=parcel["id"] + "-tower",
                        min=dict(x=left - .39, y=north - .39, z=0),
                        max=dict(x=right + .39, y=south + .39, z=height),
                    ))
            prompt_context.pop("buildings", None)
            prompt_context["building_volumes"] = volumes
        scene = {key: prompt_context.pop(key) for key in (
            "scenario_id", "grid_size", "altitude_range", "preferred_altitude",
            "buildings", "building_volumes", "no_fly_zones") if key in prompt_context}
        for key in ("buildings", "building_volumes", "no_fly_zones"):
            if key in scene:
                scene[key] = sorted(scene[key], key=_canonical_json)
        mission = {key: prompt_context.pop(key) for key in (
            "task", "base", "target", "goal", "goal_label") if key in prompt_context}
        # Keep changing observations and retry feedback after every reusable prefix.
        return ("Scene:\n" + _canonical_json(scene) + "\nMission:\n" + _canonical_json(mission)
                + "\nState:\n" + _canonical_json(prompt_context) + "\nFeedback:\n" + feedback)

    def build_messages(self, context: dict, feedback: str = "") -> list[dict]:
        return [{"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": self.build_prompt(context, feedback)}]


class MockLLMAgent(BaseLLMAgent):
    backend_name = "mock"

    def plan(self, context: dict, feedback: str = "") -> RoutePlan:
        checker = SafetyChecker(context)
        start, goal = _point(context["position"]), _point(context["goal"])
        if checker.validate_waypoint(start) or checker.validate_waypoint(goal):
            raise ValueError("Start or goal is unsafe.")
        route = preferred_route(context)
        reason = "Low-altitude route between buildings."
        if route is None:
            route = [goal] if checker.segment_safe(start, goal) else self._search(checker, start, goal)
            reason = ("Overflight permitted: no safe low-altitude route or endpoint transition on the planning lattice."
                      if context.get("preferred_altitude") else "Static collision-checked route to goal.")
        if checker.validate_path(start, route):
            raise ValueError("No complete safe route within the waypoint limit.")
        return RoutePlan(tuple(route), reason, "low",
                         "Reach goal subject to execution-time safety checks.")

    @staticmethod
    def _search(checker, start, goal):
        # Include exact endpoints and envelope bounds so fractional starts never snap.
        axes = [sorted({start[i], goal[i], checker.lower[i], checker.upper[i],
                        *(float(n) for n in range(math.ceil(checker.lower[i]), math.floor(checker.upper[i]) + 1))})
                for i in range(3)]
        origin = tuple(axes[i].index(start[i]) for i in range(3))

        def point(node):
            return tuple(axes[i][node[i]] for i in range(3))

        queue = [(metric_distance(start, goal), 0.0, origin)]
        costs, parents = {origin: 0.0}, {}
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            current = point(node)
            if checker.segment_safe(current, goal):
                path = [current]
                while node != origin:
                    node = parents[node]
                    path.append(point(node))
                path.reverse()
                if path[-1] != goal:
                    path.append(goal)
                compressed, index = [], 0
                while index < len(path) - 1:
                    next_index = len(path) - 1
                    while not checker.segment_safe(path[index], path[next_index]):
                        next_index -= 1
                    compressed.append(path[next_index])
                    index = next_index
                return compressed
            for axis in range(3):
                for step in (-1, 1):
                    neighbor = list(node)
                    neighbor[axis] += step
                    if not 0 <= neighbor[axis] < len(axes[axis]):
                        continue
                    neighbor = tuple(neighbor)
                    candidate = point(neighbor)
                    new_cost = cost + metric_distance(current, candidate)
                    if new_cost >= costs.get(neighbor, math.inf):
                        continue
                    if not checker.segment_safe(current, candidate):
                        continue
                    costs[neighbor], parents[neighbor] = new_cost, node
                    heapq.heappush(queue, (new_cost + metric_distance(candidate, goal), new_cost, neighbor))
        raise ValueError("No safe route found on the planning lattice.")


def preferred_route(context):
    band = context.get("preferred_altitude")
    if not band:
        return None
    checker = SafetyChecker(context)
    start, goal = _point(context["position"]), _point(context["goal"])
    low, high = max(checker.lower[2], band["min"]), min(checker.upper[2], band["max"])
    entry = (*start[:2], min(high, max(low, start[2])))
    exit_point = (*goal[:2], min(high, max(low, goal[2])))
    if not checker.segment_safe(start, entry) or not checker.segment_safe(exit_point, goal):
        return None
    low_context = deepcopy(context)
    low_context["altitude_range"] = dict(min=low, max=high)
    low_checker = SafetyChecker(low_context)
    try:
        middle = ([exit_point] if low_checker.segment_safe(entry, exit_point)
                  else MockLLMAgent._search(low_checker, entry, exit_point))
    except ValueError:
        return None
    return ([entry] if entry != start else []) + middle + ([goal] if exit_point != goal else [])


def validate_flight_preference(context, waypoints):
    band = context.get("preferred_altitude")
    if not band:
        return []
    start, goal = _point(context["position"]), _point(context["goal"])
    previous = start
    violates = False
    for index, point in enumerate(waypoints):
        inside = lambda altitude: band["min"] <= altitude <= band["max"]
        transition = ((index == 0 and previous[:2] == point[:2] and inside(point[2]))
                      or (index == len(waypoints) - 1 and point == goal
                          and previous[:2] == point[:2] and inside(previous[2])))
        if not (inside(previous[2]) and inside(point[2])) and not transition:
            violates = True
        previous = point
    if violates and preferred_route(context) is not None:
        return ["A safe route exists between buildings at preferred_altitude. Replan within that band; "
                "only vertical endpoint transitions may leave it."]
    return []


def checked_plan(planner, context, feedback):
    plan = planner.plan(context, feedback)
    errors = SafetyChecker(context).validate_path(_point(context["position"]), plan.waypoints)
    if not errors:
        errors = validate_flight_preference(context, plan.waypoints)
    if errors:
        raise PlanValidationError(plan, errors)
    band = context.get("preferred_altitude")
    if band and planner.backend_name != "mock":
        policy = ("Low-altitude route with safe endpoint transitions." if preferred_route(context) is not None
                  else "Overflight permitted: no safe low-altitude route or endpoint transition on the planning lattice.")
        plan = replace(plan, reason=plan.reason + " " + policy)
    return plan


class DeepSeekAgent(BaseLLMAgent):
    backend_name = "deepseek"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout_seconds: int = 25) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._previous_messages = None

    def plan(self, context: dict, feedback: str = "") -> RoutePlan:
        if not self.api_key:
            raise PlannerProviderError("DeepSeek API key is missing; set it in Agent settings.", retryable=False)
        messages = self.build_messages(context, feedback)
        metrics = prefix_metrics(messages, self._previous_messages)
        self._previous_messages = messages
        started = time.monotonic()

        def log_usage(body=None, status=None):
            record_request(body, http_status=status, elapsed_ms=round((time.monotonic() - started) * 1000),
                           model=self.model.replace(self.api_key, "[redacted]"), metrics=metrics)

        try:
            response = _post_deepseek(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model,
                      "messages": messages,
                      "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"},
                      "temperature": 0.1, "max_tokens": 2000, "stream": False},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            log_usage()
            if isinstance(exc, requests.exceptions.Timeout):
                kind = "timeout"
            elif isinstance(exc, requests.exceptions.SSLError):
                kind = "TLS"
            elif isinstance(exc, requests.exceptions.ProxyError):
                kind = "proxy"
            else:
                kind = "connection"
            raise PlannerProviderError(f"DeepSeek network request failed ({kind}).", retryable=True) from None
        if not 200 <= response.status_code < 300:
            log_usage(status=response.status_code)
            status = int(response.status_code)
            causes = {
                400: "invalid request format", 401: "API key rejected; replace it in Agent settings",
                402: "insufficient account balance", 403: "access denied",
                404: "model or endpoint unavailable", 422: "invalid request parameters",
                429: "rate limited",
            }
            cause = causes.get(status, "provider unavailable" if status >= 500 else "request rejected")
            raise PlannerProviderError(f"DeepSeek request failed with HTTP {status} ({cause}).",
                                       retryable=status == 429 or status >= 500)
        try:
            body = response.json()
        except (ValueError, RecursionError):
            log_usage(status=response.status_code)
            raise PlannerProviderError("DeepSeek returned an invalid route plan.", retryable=True) from None
        log_usage(body, response.status_code)
        try:
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or self.api_key in content:
                raise ValueError
            result = plan_from_json_text(content)
            if self.api_key in json.dumps(result.to_dict(), ensure_ascii=False):
                raise ValueError
            return result
        except (KeyError, IndexError, TypeError, ValueError, RecursionError):
            raise PlannerProviderError("DeepSeek returned an invalid route plan.", retryable=True) from None


def build_agent(backend: str) -> BaseLLMAgent:
    normalized = backend.strip().lower()
    if normalized == "mock":
        return MockLLMAgent()
    if normalized == "deepseek":
        return DeepSeekAgent()
    raise ValueError("Unsupported planner backend; choose mock or deepseek.")
