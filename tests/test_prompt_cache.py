from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from uav_llm_demo.agents import DeepSeekAgent, MockLLMAgent, PlannerProviderError
from uav_llm_demo.environment import nearest_building
from uav_llm_demo.provider_usage import LOGGER, prefix_metrics, record_request
from uav_llm_demo.safety import SafetyChecker
from uav_llm_demo.scenarios import SCENARIOS


def context():
    result = SCENARIOS["urban_crossing"].to_dict()
    result.update(position=result["base"], goal=result["target"], task="Reach target", step=0,
                  dynamic_objects=[dict(id="bird-1", position=dict(x=9, y=9, z=2))])
    return result


class PromptCacheTests(unittest.TestCase):
    @staticmethod
    def sections(prompt):
        scene, remainder = prompt.removeprefix("Scene:\n").split("\nMission:\n", 1)
        _, remainder = remainder.split("\nState:\n", 1)
        state, feedback = remainder.split("\nFeedback:\n", 1)
        return json.loads(scene), json.loads(state), feedback

    def test_known_failed_descent_has_grid_metres_and_actual_clearance_evidence(self):
        ctx = context()
        ctx["position"] = dict(x=3.200829, y=17.082845, z=9.522685)
        ctx["nearest_static_obstacle"] = nearest_building(
            tuple(ctx["position"].values()), SCENARIOS["urban_crossing"].building_map)
        original = deepcopy(ctx)
        scene, state, feedback = self.sections(MockLLMAgent().build_prompt(
            ctx, "Waypoint 0 intersects building safety envelope; Segment 0 is statically unsafe."))
        self.assertEqual(ctx, original)
        self.assertEqual(scene["units"]["coordinates"], "grid")
        self.assertEqual(scene["units"]["metres_per_grid"], dict(x=3, y=3, z=5))
        for volume in scene["building_volumes"]:
            self.assertEqual(volume["roof_height_m"], volume["max"]["z"] * 5)
        nearest = state["nearest_static_obstacle"]
        self.assertNotIn("height", nearest)
        self.assertEqual(nearest["height_grid_levels"], 4)
        self.assertEqual(nearest["roof_height_m"], 20)
        column = state["descent_column"]
        self.assertEqual(column["clearance_m"], .6)
        self.assertEqual(column["safety_expanded_envelopes"], [dict(
            kind="building", min_z_grid=-.12, max_z_grid=4.12, max_altitude_m=20.6)])
        self.assertFalse(SafetyChecker(ctx).segment_safe(
            ctx["position"], dict(x=3.200829, y=17.082845, z=2.2)))
        self.assertIn("Segment 0 is statically unsafe", feedback)

    def test_descent_evidence_uses_all_actual_envelopes_including_no_fly_zones(self):
        ctx = context()
        ctx.pop("urban_parcels")
        ctx["position"] = dict(x=3.2, y=17.08, z=9.5)
        ctx["buildings"] = [dict(x=3, y=17, height=4), dict(x=4, y=17, height=12)]
        # An unrelated nearest-obstacle summary must not determine the column.
        ctx["nearest_static_obstacle"] = dict(position=dict(x=4, y=17), height=12)
        ctx["no_fly_zones"] = [dict(min=dict(x=3.1, y=17, z=6),
                                    max=dict(x=3.3, y=17.2, z=7))]
        scene, state, _ = self.sections(MockLLMAgent().build_prompt(ctx))
        self.assertEqual({item["roof_height_m"] for item in scene["buildings"]}, {20, 60})
        column = state["descent_column"]["safety_expanded_envelopes"]
        self.assertEqual({item["kind"] for item in column}, {"building", "no-fly zone"})
        self.assertEqual({item["max_z_grid"] for item in column}, {4.12, 7.12})
        checker = SafetyChecker(ctx)
        for item in column:
            self.assertTrue(any(kind == item["kind"] and lo[2] == item["min_z_grid"]
                                and hi[2] == item["max_z_grid"]
                                for lo, hi, kind in checker.boxes))
        ctx["position"] = dict(x=6, y=17, z=9.5)
        _, state, _ = self.sections(MockLLMAgent().build_prompt(ctx))
        self.assertEqual(state["descent_column"]["safety_expanded_envelopes"], [])

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
        self.assertEqual(event["prompt_version"], "static-prefix-v1")
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

    def test_explicit_prompt_version_reaches_event_and_log(self):
        with self.assertLogs(LOGGER, level="INFO") as captured:
            event = record_request({}, http_status=200, elapsed_ms=10, model="test", metrics={},
                                   prompt_version="semantic-tasks-v1")
        self.assertEqual(event["prompt_version"], "semantic-tasks-v1")
        self.assertEqual(json.loads(captured.records[0].getMessage())["prompt_version"], "semantic-tasks-v1")


if __name__ == "__main__":
    unittest.main()
