"""Tool dialogue uses production geometry and contracts, without a remote API."""
from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from tests.semantic_fixtures import response
from tests.test_semantic_updates import program, home
from tests.test_semantic_motion_service import motion
from uav_llm_demo.agents import DeepSeekAgent, PlannerProviderError, TaskPlanningError
from uav_llm_demo.planning_tools import PlanningTools, MAX_PLANNING_ROUNDS
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_tasks import task_catalog
from uav_llm_demo.task_requirements import requirements_from_json_text


class PlanningToolTests(unittest.TestCase):
    def setUp(self):
        self.scene = load_scenario("semantic_campus")
        self.catalog = task_catalog(self.scene)
        self.context = {**self.scene.to_dict(), "position": dict(zip("xyz", self.scene.base))}
        self.events = []
        self.instruction = "在田径场上空飞一个八字，飞行过程中逐渐升高4米，最后返航。"
        # Authored independently of the executable candidates below.
        source = dict(status="ready", summary="保留形状、范围、升高量和返航。", constraints={}, steps=[
            dict(id="shape", action="motion", target_id="stadium", motion=dict(within_target=True,
                path=dict(kind="curve", x="sin(t)", y="sin(2*t)", z="4*t/(2*pi)",
                    t_start=0, t_end=2*math.pi, altitude_delta_m=4,
                    flexible=dict(scale_groups=[["x"], ["y"]], offset_axes=["z"])))),
            dict(id="end", action="return_home", target_id="home")],
            coverage=[dict(quote=self.instruction, requirement_ids=["shape", "end"], constraint_keys=[])])
        self.requirements = requirements_from_json_text(json.dumps(source), self.catalog, self.instruction)
        self.catalog["task_requirements"] = source
        self.tools = PlanningTools(self.scene, self.context, self.catalog, requirements=self.requirements,
                                   event=lambda **value: self.events.append(value))

    def candidate(self, size=6):
        node = motion(dict(kind="curve", x=f"{size}*sin(t)", y="3*sin(2*t)", z="1+4*t/(2*pi)",
                           t_start=0, t_end=2*math.pi))
        node["motion"]["within_target"] = True
        return program([node, home()])

    def test_query_preview_correct_submit_and_never_execute_a_trial(self):
        bad, good = self.candidate(20), self.candidate()
        wrong_rise = self.candidate()
        wrong_rise["steps"][0]["motion"]["path"]["z"] = "1+0.5*t"
        original = deepcopy(bad)
        replies = [dict(tool="query_geometry", arguments=dict(target_id="stadium")),
                   dict(tool="preview_plan", arguments=dict(plan=bad)),
                   dict(tool="preview_plan", arguments=dict(plan=wrong_rise)),
                   dict(tool="preview_plan", arguments=dict(plan=good)),
                   dict(tool="submit_plan", arguments=dict(candidate_id="candidate-1"))]
        agent = DeepSeekAgent(api_key="offline-tool-key")
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=[response(v) for v in replies]) as post:
            accepted = agent.interpret_task(self.catalog, self.instruction, planning=self.tools)
        self.assertEqual(bad, original)
        self.assertEqual(accepted.steps[0]["motion"]["path"]["x"], "6*sin(t)")
        self.assertEqual([e["ok"] for e in self.events], [True, False, False, True, True])
        correction_prompt = post.call_args_list[2].kwargs["json"]["messages"][-1]["content"]
        self.assertIn("target footprint", correction_prompt)
        height_feedback = self.events[2]["diagnostic"]["mismatches"]
        delta = next(item for item in height_feedback if item["path"].endswith("altitude_delta_m"))
        self.assertEqual(delta["expected"], 4)
        self.assertAlmostEqual(delta["actual"], math.pi)
        self.assertIn("nearby_obstacles", post.call_args_list[1].kwargs["json"]["messages"][-1]["content"])
        self.assertIsNone(agent._previous_messages)
        self.assertEqual(self.context["position"], dict(zip("xyz", self.scene.base)))

    def test_preview_cannot_drop_obligations_or_submit_an_unknown_candidate(self):
        result, accepted = self.tools.call(dict(tool="preview_plan", arguments=dict(plan=program([home()]))))
        self.assertFalse(result["ok"])
        self.assertIsNone(accepted)
        result, accepted = self.tools.call(dict(tool="submit_plan", arguments=dict(candidate_id="candidate-1")))
        self.assertEqual(result["diagnostic"]["code"], "unknown_candidate")
        self.assertIsNone(accepted)

    def test_collision_feedback_contains_location_and_expanded_obstacle(self):
        tools = PlanningTools(self.scene, self.context, self.catalog)
        bad = self.candidate(20)
        bad["steps"][0]["motion"].pop("within_target")
        bad["steps"][0]["motion"]["path"].update(y="10*sin(2*t)", z=0)
        result, _ = tools.call(dict(tool="preview_plan", arguments=dict(plan=bad)))
        self.assertFalse(result["ok"])
        self.assertIn("segment_grid", result["diagnostic"])
        self.assertIsNotNone(result["diagnostic"]["obstacle_envelope"])
        self.assertEqual(result["diagnostic"]["node_id"], "flight")

    def test_repeated_queries_are_bounded_and_superseded_sessions_stop_before_calling(self):
        agent = DeepSeekAgent(api_key="offline-tool-key")
        query = dict(tool="query_geometry", arguments=dict(target_id="stadium"))
        with patch("uav_llm_demo.agents._post_deepseek", return_value=response(query)) as post:
            with self.assertRaises(PlannerProviderError) as stopped:
                agent.interpret_task(self.catalog, self.instruction, planning=self.tools)
            self.assertEqual(post.call_count, MAX_PLANNING_ROUNDS)
            self.assertEqual(stopped.exception.diagnostic['rounds'],MAX_PLANNING_ROUNDS)
            self.assertEqual(stopped.exception.diagnostic['code'],'planning_budget_exhausted')
            cancelled = PlanningTools(self.scene, self.context, self.catalog, cancelled=lambda: True)
            with self.assertRaises(PlannerProviderError):
                agent.interpret_task(self.catalog, self.instruction, planning=cancelled)
            self.assertEqual(post.call_count, MAX_PLANNING_ROUNDS)
        expired = PlanningTools(self.scene, self.context, self.catalog)
        def late_reply(*args, **kwargs):
            expired.deadline = 0
            return response(query)
        with patch("uav_llm_demo.agents._post_deepseek", side_effect=late_reply) as post:
            with self.assertRaises(TaskPlanningError) as stopped:
                agent.interpret_task(self.catalog, self.instruction, planning=expired)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(stopped.exception.diagnostic['exhausted_limit'], 'wall_time')
        clarification = dict(status="needs_clarification", summary="请选择未指定的高度。", steps=[], constraints={})
        tools = PlanningTools(self.scene, self.context, self.catalog, requirements=self.requirements)
        with patch("uav_llm_demo.agents._post_deepseek",
                   side_effect=[response(query)]*(MAX_PLANNING_ROUNDS-1)+[response(clarification)]) as post:
            with self.assertRaises(PlannerProviderError):
                agent.interpret_task(self.catalog, self.instruction, planning=tools)
            self.assertEqual(post.call_count, MAX_PLANNING_ROUNDS)


if __name__ == "__main__":
    unittest.main()
