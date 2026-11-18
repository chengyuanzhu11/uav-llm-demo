"""File expert uses the production visual protocol; it never infers through an API."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests.test_interaction_trajectory import frame

spec = importlib.util.spec_from_file_location("visual_demonstration", Path(__file__).resolve().parents[1]/"scripts/collect_visual_demonstration.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VisualDemonstrationTests(unittest.TestCase):
    def test_frozen_curriculum_has_real_disjoint_scenes_and_executable_expert_cases(self):
        from prepare_demonstration_curriculum import prepare
        from build_gazebo_scene import build
        from uav_llm_demo.demonstration_data import canonical_sha256
        from uav_llm_demo.scenarios import load_asset_scenario, load_scenario
        from uav_llm_demo.semantic_tasks import task_catalog, semantic_plan_from_json_text
        from uav_llm_demo.task_requirements import requirements_from_json_text, audit_requirements_from_json_text
        from copy import deepcopy
        source = json.loads((module.ROOT/"benchmarks/demonstration_curriculum_v2.json").read_text())
        cases = prepare(source)
        self.assertEqual(len(cases), 42)
        self.assertEqual({c["standard"]["frozen_split"] for c in cases}, {"train","validation","test"})
        for case in cases:
            catalog = task_catalog(load_asset_scenario(case["scene_file"]))
            demo = case["demonstration"]
            requirements = requirements_from_json_text(json.dumps(demo["requirements"]),catalog,case["standard"]["instruction"],for_extraction=True)
            audit_requirements_from_json_text(json.dumps(demo["audit"]),requirements,case["standard"]["instruction"])
            semantic_plan_from_json_text(json.dumps(demo["plan"]),catalog)
        with tempfile.TemporaryDirectory() as temp:
            for layout in source["layouts"]:
                name = layout["scene"]["id"]
                self.assertTrue(build(name,temp,scene_file=module.ROOT/"simulation/scenes"/(name+".json")).is_file())
        self.assertEqual(load_scenario("visual_garden").scenario_id,"visual_garden")
        repeated = deepcopy(source)
        repeated["layouts"][-1]["scene"] = deepcopy(repeated["layouts"][0]["scene"])
        repeated["layouts"][-1]["scene"]["id"] = "renamed_same_geometry"
        repeated["layouts"][-1]["scene_sha256"] = canonical_sha256(repeated["layouts"][-1]["scene"])
        with self.assertRaisesRegex(ValueError, "geometry"):
            prepare(repeated)

    def test_native_ground_from_current_image_is_recorded_with_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            agent = module.FileExpertVision(Path(temp)/"review")
            current = frame()
            current.update(vertical_fov_deg=60)
            reply = dict(tool="ground", arguments=dict(frame_id=current["frame_id"], box=[299,299,700,700], point=[500,500], confidence=.9,
                reason="Visible solid target interior."), annotation=dict(reviewer="agent_assisted_test",
                polygon=[[.3,.3],[.7,.3],[.7,.7],[.3,.7]]))
            module.write_json(agent.directory/"response-001.json", reply)
            records = []
            with patch("requests.sessions.Session.request", side_effect=AssertionError("No API permitted")):
                action = agent.decide({"visual_goal":{"description":"test target"}}, [dict(current,frame_id="previous"),current], interaction_observer=records.append)
            self.assertEqual(action["tool"], "ground")
            self.assertAlmostEqual(action["arguments"]["pixel"][0], 500/999)
            self.assertEqual(records[0]["source"], "provider_override")
            self.assertIsNone(records[0]["http_status"])
            self.assertEqual(records[0]["outcome"], "ok")
            self.assertEqual(agent.annotations[0]["rgb_sha256"], current["rgb_sha256"])
            saved = json.loads((agent.directory/"input-001.json").read_text())
            self.assertNotIn("depth_m", json.dumps(saved))
            self.assertNotIn("oracle", json.dumps(saved))
            self.assertTrue((agent.directory/saved["frames"][-1]["path"]).is_file())
            exposed = [p["image_url"]["url"] for p in saved["request"]["messages"][1]["content"] if p["type"] == "image_url"]
            self.assertEqual(exposed,[saved["frames"][-1]["path"]])


if __name__ == "__main__":
    unittest.main()
