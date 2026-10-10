from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from uav_llm_demo.agents import DeepSeekAgent, MockLLMAgent, PlannerProviderError
from uav_llm_demo.provider_usage import LOGGER, prefix_metrics, record_request
from uav_llm_demo.scenarios import SCENARIOS


def context():
    result = SCENARIOS["urban_crossing"].to_dict()
    result.update(position=result["base"], goal=result["target"], task="Reach target", step=0,
                  dynamic_objects=[dict(id="bird-1", position=dict(x=9, y=9, z=2))])
    return result


class PromptCacheTests(unittest.TestCase):
    def test_moving_state_and_retry_feedback_cannot_break_scene_and_mission_prefix(self):
        before = context()
        after = deepcopy(before)
        after.update(position=dict(x=1.5, y=17, z=2), step=20, battery=99)
        after["dynamic_objects"][0]["position"]["x"] = 10
        planner = MockLLMAgent()
        a = planner.build_messages(before)
        b = planner.build_messages(after, "Segment 2 unsafe")
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1]["content"].split("\nState:\n")[0], b[1]["content"].split("\nState:\n")[0])
        self.assertNotEqual(a[1]["content"], b[1]["content"])
        self.assertEqual([message["role"] for message in b], ["system", "user"])
        self.assertGreater(prefix_metrics(b, a)["common_prefix_content_chars"], len(a[0]["content"]))

    def test_goal_and_map_edits_invalidate_only_their_actual_prefix_sections(self):
        planner = MockLLMAgent()
        ctx = context()
        first = planner.build_prompt(ctx)
        ctx["goal"] = ctx["base"]
        second = planner.build_prompt(ctx)
        self.assertEqual(first.split("\nMission:\n")[0], second.split("\nMission:\n")[0])
        self.assertNotEqual(first, second)
        ctx["urban_parcels"][0]["podium_height"] += 1
        third = planner.build_prompt(ctx)
        self.assertNotEqual(second.split("\nMission:\n")[0], third.split("\nMission:\n")[0])

    def test_static_serialization_ignores_mapping_and_obstacle_insertion_order(self):
        ctx = context()
        original = deepcopy(ctx)
        reversed_context = dict(reversed(list(deepcopy(ctx).items())))
        reversed_context["urban_parcels"].reverse()
        self.assertEqual(MockLLMAgent().build_messages(ctx), MockLLMAgent().build_messages(reversed_context))
        self.assertEqual(ctx, original)

    def test_usage_is_logged_even_when_response_plan_is_invalid(self):
        response = Mock(status_code=200)
        response.json.return_value = dict(usage=dict(prompt_tokens=1000, prompt_cache_hit_tokens=800,
            prompt_cache_miss_tokens=200, completion_tokens=50, total_tokens=1050),
            choices=[dict(message=dict(content="not valid JSON"))])
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response), \
                self.assertLogs(LOGGER, level="INFO") as captured:
            with self.assertRaises(PlannerProviderError):
                DeepSeekAgent(api_key="test-credential").plan(context())
        event = json.loads(captured.records[0].getMessage())
        self.assertEqual(event["cache_hit_rate"], 0.8)
        self.assertTrue(event["usage_consistent"])
        self.assertNotIn("test-credential", captured.output[0])
        self.assertNotIn("not valid JSON", captured.output[0])

    def test_usage_allowlist_and_missing_counts_never_invent_hits_or_leak_body(self):
        for usage in ({}, {"prompt_tokens": "credential", "prompt_cache_hit_tokens": True,
                           "prompt_cache_miss_tokens": -1, "other": "credential"},
                      {"prompt_tokens": 100, "prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 40}):
            with self.assertLogs(LOGGER, level="INFO") as captured:
                event = record_request(dict(usage=usage, content="credential"), http_status=200,
                                       elapsed_ms=10, model="test", metrics={})
            self.assertFalse(event["usage_consistent"])
            self.assertIsNone(event["cache_hit_rate"])
            self.assertNotIn("credential", captured.output[0])


if __name__ == "__main__":
    unittest.main()
