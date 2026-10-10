"""Focused modules loaded independently of the concurrently migrated app entrypoint."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


PACKAGE = "_planner_safety_test_modules"
ROOT = Path(__file__).resolve().parents[1] / "uav_llm_demo"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


safety = load_module("safety")
agents = load_module("agents")


def xyz(x, y, z):
    return dict(x=x, y=y, z=z)


def context():
    return dict(position=xyz(0, 2, 2), orientation=dict(yaw=0), goal=xyz(6, 2, 2),
                base=xyz(0, 2, 2), target=xyz(6, 2, 2), buildings=[], grid_size=7,
                altitude_range=dict(min=1, max=5), no_fly_zones=[], dynamic_objects=[],
                task="Navigate to target")


def plan_data():
    return dict(waypoints=[xyz(6, 2, 2)], reason="Route proposed", risk="low",
                expected_result="Reach goal after validation")


class SafetyTests(unittest.TestCase):
    def test_metric_units(self):
        self.assertEqual(safety.metric_distance((0, 0, 0), (1, 0, 0)), 3)
        self.assertEqual(safety.metric_distance((0, 0, 0), (0, 0, 1)), 5)
        self.assertAlmostEqual(safety.metric_distance((0, 0, 0), (1, 1, 1)), math.sqrt(43))

    def test_invalid_coordinates_and_bounds(self):
        checker = safety.SafetyChecker(context())
        for point in (None, [], [1, 2], [True, 2, 2], ["1", 2, 2], [float("nan"), 2, 2],
                      [float("inf"), 2, 2], [10**400, 2, 2], [-.001, 2, 2],
                      [6.001, 2, 2], [2, 2, .999], [2, 2, 5.001], {"x": 1}):
            with self.subTest(point_type=type(point).__name__):
                self.assertTrue(checker.validate_waypoint(point))
                self.assertFalse(checker.segment_safe((0, 2, 2), point))

    def test_building_full_segment_and_roof_margin(self):
        ctx = context()
        ctx["buildings"] = [dict(x=3, y=2, height=3)]
        checker = safety.SafetyChecker(ctx)
        self.assertFalse(checker.segment_safe((0, 2, 2), (6, 2, 2)))
        self.assertTrue(checker.validate_path((0, 2, 2), [(6, 2, 2)]))
        self.assertFalse(checker.segment_safe((0, 2, 3.12), (6, 2, 3.12)))
        self.assertTrue(checker.segment_safe((0, 2, 3.121), (6, 2, 3.121)))
        self.assertTrue(checker.segment_safe((0, 2.591, 2), (6, 2.591, 2)))
        self.assertFalse(checker.segment_safe((0, 2.589, 2), (6, 2.589, 2)))

    def test_tiny_no_fly_box_continuous_and_reversed(self):
        ctx = context()
        ctx["no_fly_zones"] = [dict(min=xyz(3.12345, 2, 2), max=xyz(3.12346, 2, 2))]
        checker = safety.SafetyChecker(ctx)
        for a, b in (((0, 2, 2), (6, 2, 2)), ((6, 2, 2), (0, 2, 2))):
            self.assertFalse(checker.segment_safe(a, b))
        self.assertTrue(checker.segment_safe((0, 3, 2), (6, 3, 2)))

    def test_zero_length_and_vertical(self):
        ctx = context()
        ctx["buildings"] = [dict(x=3, y=2, height=3)]
        checker = safety.SafetyChecker(ctx)
        self.assertTrue(checker.segment_safe((0, 2, 2), (0, 2, 2)))
        self.assertFalse(checker.segment_safe((3, 2, 2), (3, 2, 2)))
        self.assertFalse(checker.segment_safe((3, 2, 5), (3, 2, 2)))

    def test_goal_tolerance_in_metres(self):
        checker = safety.SafetyChecker(context())
        self.assertEqual(checker.validate_path((0, 2, 2), [(5.95, 2, 2)]), [])
        self.assertTrue(checker.validate_path((0, 2, 2), [(5.9, 2, 2)]))
        self.assertTrue(checker.validate_path((0, 2, 2), [(6, 2, 2.05)]))
        self.assertEqual(checker.validate_path((0, 2, 2), [(5, 2, 2)], require_goal=False), [])

    def test_invalid_paths_and_start(self):
        checker = safety.SafetyChecker(context())
        for route in (None, [], [(6, 2, 2)] * 65, [(True, 2, 2)]):
            self.assertTrue(checker.validate_path((0, 2, 2), route))
        self.assertTrue(checker.validate_path((False, 2, 2), [(6, 2, 2)]))

    def test_dynamic_opt_in_and_predicted_snapshot(self):
        ctx = context()
        ctx["dynamic_objects"] = [dict(position=xyz(3, 5, 2), predicted_position=xyz(3, 2, 2), radius=.7)]
        checker = safety.SafetyChecker(ctx)
        self.assertEqual(checker.validate_path((0, 2, 2), [(6, 2, 2)]), [])
        self.assertTrue(checker.segment_safe((0, 2, 2), (6, 2, 2)))
        self.assertTrue(checker.validate_path((0, 2, 2), [(6, 2, 2)], check_dynamic=True))
        ctx["dynamic_objects"][0]["predicted_position"] = xyz(3, 5, 2)
        self.assertEqual(checker.validate_path((0, 2, 2), [(6, 2, 2)], check_dynamic=True), [])

    def test_bad_obstacle_configuration_fails_closed(self):
        ctx = context()
        ctx["no_fly_zones"] = [dict(min=xyz(4, 0, 0), max=xyz(3, 1, 1))]
        with self.assertRaises(ValueError):
            safety.SafetyChecker(ctx)


class PlannerTests(unittest.TestCase):
    def test_direct_route_and_roundtrip(self):
        plan = agents.MockLLMAgent().plan(context())
        self.assertEqual(plan.waypoints, ((6.0, 2.0, 2.0),))
        self.assertEqual(agents.plan_from_json_text(json.dumps(plan.to_dict())), plan)

    def test_fractional_detour_and_compression(self):
        ctx = context()
        ctx["position"] = xyz(.23, 2.1, 2.4)
        ctx["goal"] = xyz(5.83, 2.1, 2.4)
        ctx["buildings"] = [dict(x=3, y=2, height=5)]
        plan = agents.MockLLMAgent().plan(ctx)
        self.assertGreater(len(plan.waypoints), 1)
        self.assertEqual(safety.SafetyChecker(ctx).validate_path(ctx["position"], plan.waypoints), [])
        self.assertEqual(plan.waypoints[-1], (5.83, 2.1, 2.4))
        self.assertLess(len(plan.waypoints), 8)

    def test_no_fly_detour(self):
        ctx = context()
        ctx["no_fly_zones"] = [dict(min=xyz(2, 0, 1), max=xyz(4, 3, 5))]
        plan = agents.MockLLMAgent().plan(ctx)
        self.assertEqual(safety.SafetyChecker(ctx).validate_path(ctx["position"], plan.waypoints), [])
        self.assertTrue(any(p[1] > 3 for p in plan.waypoints))

    def test_no_route_and_unsafe_start_rejected(self):
        ctx = context()
        ctx["no_fly_zones"] = [dict(min=xyz(2, 0, 0), max=xyz(4, 6, 5))]
        with self.assertRaises(ValueError):
            agents.MockLLMAgent().plan(ctx)
        ctx["position"] = xyz(3, 2, 2)
        with self.assertRaises(ValueError):
            agents.MockLLMAgent().plan(ctx)

    def test_already_at_goal(self):
        ctx = context()
        ctx["position"] = ctx["goal"].copy()
        self.assertEqual(agents.MockLLMAgent().plan(ctx).waypoints, ((6., 2., 2.),))

    def test_strict_parser(self):
        invalid = ["{}", "[]", "null", "```json\n{}\n```", "{", json.dumps(plan_data()) + " extra"]
        for key, value in (("risk", "LOW"), ("reason", ""), ("expected_result", None),
                           ("waypoints", []), ("waypoints", [[1, 2, 3]] * 65),
                           ("waypoints", [[True, 2, 3]]), ("waypoints", [["1", 2, 3]]),
                           ("waypoints", [[float("nan"), 2, 3]]),
                           ("waypoints", [[float("inf"), 2, 3]]), ("action", "move_east")):
            data = plan_data()
            data[key] = value
            invalid.append(json.dumps(data))
        invalid.append(json.dumps(plan_data()).replace('"risk": "low"', '"risk": "low", "risk": "high"'))
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    agents.plan_from_json_text(text)
        data = plan_data()
        data["waypoints"] = [[1, 2, 3]] * 64
        self.assertEqual(len(agents.plan_from_json_text(json.dumps(data)).waypoints), 64)

    def test_backend_contract(self):
        self.assertIsInstance(agents.build_agent(" MOCK "), agents.MockLLMAgent)
        for backend in ("openai", "unknown", ""):
            with self.assertRaises(ValueError):
                agents.build_agent(backend)
        self.assertFalse(hasattr(agents, "AgentDecision"))
        self.assertFalse(hasattr(agents.BaseLLMAgent, "decide"))

    def test_prompt_context_and_feedback(self):
        prompt = "\n".join(message["content"] for message in
                           agents.MockLLMAgent().build_messages(context(), "Segment 0 rejected"))
        for token in ("GRID", "0.78", "0.6", "0.2", "orientation", "Segment 0 rejected"):
            self.assertIn(token, prompt)


class DeepSeekTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows native transport")
    def test_native_transport_sends_key_only_via_stdin(self):
        marker = "dummy-test-credential"
        result = Mock(returncode=0, stdout='{"status":401}', stderr="")
        with patch.object(agents.subprocess, "run", return_value=result) as run:
            response = agents._post_deepseek(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {marker}"}, json={"model": "test"}, timeout=20)
        self.assertEqual(response.status_code, 401)
        self.assertNotIn(marker, " ".join(run.call_args.args[0]))
        self.assertEqual(json.loads(run.call_args.kwargs["input"])["api_key"], marker)
        self.assertTrue(run.call_args.kwargs["capture_output"])

    @unittest.skipUnless(sys.platform == "win32", "Windows native transport")
    def test_native_transport_failure_is_sanitized(self):
        agent = agents.DeepSeekAgent(api_key="dummy-test-credential")
        for output, error in (('{"error":"connection"}', "connection"),
                              ('{"error":"timeout"}', "timeout"),
                              ('not JSON', "connection")):
            with self.subTest(output=output):
                result = Mock(returncode=0, stdout=output, stderr="secret diagnostic")
                with patch.object(agents.subprocess, "run", return_value=result):
                    with self.assertRaises(agents.PlannerProviderError) as caught:
                        agent.plan(context())
                self.assertIn(error, str(caught.exception))
                self.assertNotIn("secret diagnostic", str(caught.exception))

    def test_request_and_no_unsafe_plan_override(self):
        ctx = context()
        ctx["buildings"] = [dict(x=3, y=2, height=5)]
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps(plan_data())}}]}
        with patch.object(agents, "_post_deepseek", return_value=response) as post:
            plan = agents.DeepSeekAgent(api_key="dummy-test-credential").plan(ctx, "replan")
        self.assertTrue(safety.SafetyChecker(ctx).validate_path(ctx["position"], plan.waypoints))
        options = post.call_args.kwargs
        self.assertEqual(options["timeout"], 25)
        for field, value in (("temperature", .1), ("max_tokens", 2000),
                             ("thinking", {"type": "disabled"}),
                             ("response_format", {"type": "json_object"})):
            self.assertEqual(options["json"][field], value)

    def test_failure_messages_sanitized(self):
        marker = "dummy-test-credential"
        agent = agents.DeepSeekAgent(api_key=marker)
        response = Mock(status_code=401, text=marker)
        with patch.object(agents, "_post_deepseek", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                agent.plan(context())
        self.assertNotIn(marker, str(caught.exception))
        with patch.object(agents, "_post_deepseek", side_effect=agents.requests.ConnectionError(marker)):
            with self.assertRaisesRegex(RuntimeError, "network request failed") as caught:
                agent.plan(context())
        self.assertNotIn(marker, str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_http_error_categories_do_not_expose_response_body(self):
        agent = agents.DeepSeekAgent(api_key="dummy-test-credential")
        for status, retryable in ((400, False), (401, False), (402, False),
                                  (422, False), (429, True), (500, True), (503, True)):
            with self.subTest(status=status):
                response = Mock(status_code=status, text="dummy-test-credential")
                with patch.object(agents, "_post_deepseek", return_value=response):
                    with self.assertRaises(agents.PlannerProviderError) as caught:
                        agent.plan(context())
                self.assertEqual(caught.exception.retryable, retryable)
                self.assertIn(f"HTTP {status}", str(caught.exception))
                self.assertNotIn("dummy-test-credential", str(caught.exception))

    def test_network_failure_categories_do_not_expose_exception_text(self):
        agent = agents.DeepSeekAgent(api_key="dummy-test-credential")
        for failure, category in ((agents.requests.Timeout, "timeout"),
                                  (agents.requests.exceptions.SSLError, "TLS"),
                                  (agents.requests.exceptions.ProxyError, "proxy"),
                                  (agents.requests.ConnectionError, "connection")):
            with self.subTest(failure=failure.__name__):
                with patch.object(agents, "_post_deepseek", side_effect=failure("dummy-test-credential")):
                    with self.assertRaises(agents.PlannerProviderError) as caught:
                        agent.plan(context())
                self.assertTrue(caught.exception.retryable)
                self.assertIn(category, str(caught.exception))
                self.assertNotIn("dummy-test-credential", str(caught.exception))

    def test_bad_provider_payload_and_key_echo_rejected(self):
        marker = "dummy-test-credential"
        data = plan_data()
        data["reason"] = marker
        for payload in ({}, {"choices": []}, {"choices": [{"message": {"content": "not JSON"}}]},
                        {"choices": [{"message": {"content": json.dumps(data)}}]}):
            response = Mock(status_code=200)
            response.json.return_value = payload
            with patch.object(agents, "_post_deepseek", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "invalid route plan") as caught:
                    agents.DeepSeekAgent(api_key=marker).plan(context())
            self.assertNotIn(marker, str(caught.exception))

    def test_env_initialization_and_missing_key(self):
        with patch.dict(agents.os.environ, {"DEEPSEEK_API_KEY": "dummy", "DEEPSEEK_MODEL": "test-model",
                                           "DEEPSEEK_BASE_URL": "https://example.invalid/"}, clear=True):
            agent = agents.DeepSeekAgent()
            self.assertEqual((agent.api_key, agent.model, agent.base_url),
                             ("dummy", "test-model", "https://example.invalid"))
        with patch.dict(agents.os.environ, {}, clear=True), patch.object(agents, "_post_deepseek") as post:
            with self.assertRaises(RuntimeError):
                agents.DeepSeekAgent().plan(context())
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
