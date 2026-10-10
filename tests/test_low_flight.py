import json
import unittest
from unittest.mock import Mock, patch

from uav_llm_demo.agents import MockLLMAgent, DeepSeekAgent, checked_plan, preferred_route
from uav_llm_demo.safety import SafetyChecker


def context():
    return dict(grid_size=7, position=dict(x=0, y=3, z=2), goal=dict(x=6, y=3, z=2),
                altitude_range=dict(min=2, max=10), preferred_altitude=dict(min=2, max=2.4),
                buildings=[dict(x=3, y=3, height=4)], no_fly_zones=[])


class LowFlightTests(unittest.TestCase):
    def test_mock_routes_around_buildings_below_roofs(self):
        ctx = context()
        plan = checked_plan(MockLLMAgent(), ctx, "")
        self.assertTrue(all(2 <= point[2] <= 2.4 for point in plan.waypoints))
        self.assertFalse(SafetyChecker(ctx).validate_path((0, 3, 2), plan.waypoints))
        self.assertGreater(len(plan.waypoints), 1)

    def test_enclosed_low_route_allows_overflight(self):
        ctx = context()
        ctx["buildings"] = [dict(x=3, y=south, height=4) for south in range(7)]
        self.assertIsNone(preferred_route(ctx))
        plan = checked_plan(MockLLMAgent(), ctx, "")
        self.assertTrue(any(point[2] > 4 for point in plan.waypoints))
        self.assertIn("Overflight permitted", plan.reason)

    def test_high_start_and_goal_have_safe_vertical_transitions(self):
        ctx = context()
        ctx["position"]["z"] = 6
        ctx["goal"]["z"] = 5
        plan = checked_plan(MockLLMAgent(), ctx, "")
        self.assertEqual(plan.waypoints[0], (0, 3, 2.4))
        self.assertEqual(plan.waypoints[-1], (6, 3, 5))
        self.assertTrue(all(2 <= point[2] <= 2.4 for point in plan.waypoints[:-1]))

    def test_deepseek_unnecessary_climb_rejected_with_feedback(self):
        payload = dict(waypoints=[dict(x=0, y=3, z=6), dict(x=6, y=3, z=6), dict(x=6, y=3, z=2)],
                       reason="Overflight", risk="low", expected_result="Reach target")
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps(payload)}}]}
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response):
            with self.assertRaisesRegex(ValueError, "safe route exists between buildings"):
                checked_plan(DeepSeekAgent(api_key="test-only"), context(), "")

    def test_empty_field_keeps_unrestricted_plan(self):
        ctx = context()
        ctx.pop("preferred_altitude")
        ctx["buildings"] = []
        ctx["goal"]["z"] = 5
        self.assertEqual(checked_plan(MockLLMAgent(), ctx, "").waypoints, ((6, 3, 5),))

    def test_deepseek_low_route_accepted_without_extra_network_validation(self):
        ctx = context()
        expected = MockLLMAgent().plan(ctx)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps(expected.to_dict())}}]}
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response) as request:
            actual = checked_plan(DeepSeekAgent(api_key="test-only"), ctx, "")
        self.assertEqual(actual.waypoints, expected.waypoints)
        self.assertEqual(request.call_count, 1)
