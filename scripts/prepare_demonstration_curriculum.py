"""Freeze independently authored task cases and disjoint layouts; never train or run a model."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_llm_demo.demonstration_data import _standard, canonical_sha256
from uav_llm_demo.task_requirements import audit_requirements_projection


def geometry_fingerprint(scene):
    """Names and ordering cannot turn the same layout into a fresh split."""
    value = {key: deepcopy(item) for key,item in scene.items() if key not in {"id", "name", "objects"}}
    value["objects"] = sorted(({key:item for key,item in obj.items() if key != "id"}
                               for obj in scene["objects"]), key=canonical_sha256)
    return canonical_sha256(value)


def prepare(manifest):
    if manifest.get("version") != "visual-decision-curriculum-v2":
        raise ValueError("Unsupported frozen curriculum")
    if manifest.get("annotation_source") != "agent_assisted" or manifest.get("human_reviewed") is not False:
        raise ValueError("Keep the source annotation provenance explicit")
    reserved = set(manifest["source_exposure"]["reserved_scenarios"])
    legacy_geometry = {geometry_fingerprint(json.loads((ROOT/"simulation/scenes"/(name+".json")).read_text()))
                       for name in reserved if (ROOT/"simulation/scenes"/(name+".json")).is_file()}
    seen_geometry, seen_families, scene_ids, cases = {}, {}, set(), []
    groups = {group["split"]:group for group in manifest["groups"]}
    if set(groups) != {"train", "validation", "test"} or len(groups) != len(manifest["groups"]):
        raise ValueError("Exactly one group definition per frozen split is required")
    for layout in manifest["layouts"]:
        scene, split = layout["scene"], layout["split"]
        if split not in groups or scene["id"] in reserved or scene["id"] in scene_ids:
            raise ValueError("Duplicate or previously exposed scene identity")
        scene_ids.add(scene["id"])
        if canonical_sha256(scene) != layout["scene_sha256"]:
            raise ValueError("Frozen scene content changed")
        geometry = geometry_fingerprint(scene)
        if geometry in legacy_geometry or geometry in seen_geometry:
            raise ValueError("Frozen layouts must differ in geometry, not only their IDs")
        seen_geometry[geometry] = split
        group = groups[split]
        if group["family"] in seen_families and seen_families[group["family"]] != split:
            raise ValueError("Task family crosses frozen splits")
        seen_families[group["family"]] = split
        if group["sequence"] not in (["visual", "return_home"], ["visual", "dwell", "return_home"],
                                      ["dwell", "visual", "return_home"]):
            raise ValueError("Unknown independently authored task composition")
        for target in manifest["targets"]:
            if not any(obj["id"] == target["object_id"] for obj in scene["objects"]):
                raise ValueError("Task standard target is absent from frozen world")
            for seed in group["seeds"]:
                task_id = f"v2_{split}_{scene['id']}_{target['object_id']}_{seed}"
                visual = f"到花园从机载图像寻找{target['description']}，在18米三维距离内确认并定位目标"
                parts = {"visual":visual, "dwell":"在花园上空10米悬停3秒", "return_home":"返航并降落"}
                instruction = "，然后".join(parts[step] for step in group["sequence"]) + "。"
                requirements = []
                for step in group["sequence"]:
                    if step == "dwell":
                        requirements.append(dict(type="pose", target="garden", state="airborne", altitude_m=10, dwell_seconds=3))
                    elif step == "return_home":
                        requirements.append(dict(type="evidence", action="return_home", target="home", count=1))
                requirements.append(dict(type="goal", target="home", state="landed"))
                standard = dict(id=task_id, scenario=scene["id"], family=group["family"], instruction=instruction,
                    intent=[parts[step] for step in group["sequence"]], annotation_source="agent_assisted", human_reviewed=False,
                    evaluation_kind="visual_rgbd", oracle=dict(type="sequence", items=requirements),
                    visual_requirement=dict(description=target["description"], before_dwell=group["sequence"][0]=="visual" and "dwell" in group["sequence"],
                                            after_dwell=group["sequence"][0]=="dwell"),
                    visual_limits=dict(max_surface_error_m=.3, max_frame_age_seconds=45, max_distance_m=18,
                                       max_acquisition_distance_m=80, max_motion_endpoint_error_m=1),
                    frozen_split=split, scene_sha256=layout["scene_sha256"], scene_geometry_sha256=geometry,
                    trajectory_group=task_id, curriculum_version=manifest["version"])
                _standard(standard)
                nodes = {
                    "visual":dict(id="observe",action="visual",target_id="garden",visual=dict(description=target["description"],max_distance_m=18)),
                    "dwell":dict(id="dwell",action="control",target_id="garden",control=dict(state="airborne",altitude_m=10,duration_seconds=3)),
                    "return_home":dict(id="return_base",action="return_home",target_id="home")}
                requirements = dict(status="ready",summary="完成独立标准中的有序义务。",
                    steps=[deepcopy(nodes[step]) for step in group["sequence"]],constraints={},
                    coverage=[dict(quote=instruction,requirement_ids=[nodes[step]["id"] for step in group["sequence"]],constraint_keys=[])])
                plan = dict(status="ready",summary="通过真实图像确认目标并执行有序任务。",
                    steps=[dict(nodes[step],when=None,**(dict(duration_seconds=0) if step=="return_home" else {})) for step in group["sequence"]],
                    constraints=dict(max_altitude_m=None,avoid_regions=[]))
                audit = dict(status="approved",summary="目标、距离、顺序和结束状态均逐项保留。",issues=[],
                    assertions=[dict(quote=instruction,requirement_id=key,expected_json=json.dumps(value,ensure_ascii=False))
                                for key,value in audit_requirements_projection(requirements).items()])
                demonstration = dict(requirements=requirements,audit=audit,plan=plan,query_target="garden",
                    repair="missing_obligation" if split=="train" and seed in (29,61) else None)
                cases.append(dict(standard=standard, seed=seed, oracle_target=target["object_id"],
                    scene_file=str(ROOT/"simulation/scenes"/(scene["id"]+".json")), demonstration=demonstration,
                    status="pending_layout_validation_and_collection"))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"benchmarks/demonstration_curriculum_v2.json")
    parser.add_argument("--output", type=Path, help="Optional NEW directory for frozen cases and scene files")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = prepare(manifest)
    result = dict(version=manifest["version"], manifest_sha256=canonical_sha256(manifest),
                  planned_episodes=dict(Counter(case["standard"]["frozen_split"] for case in cases)),
                  collected_episodes=0, approved_episodes=0, training_started=False,
                  status="Task standards and layouts frozen; native layout validation and collection pending")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"scenes").mkdir()
        for layout in manifest["layouts"]:
            (args.output/"scenes"/(layout["scene"]["id"]+".json")).write_text(json.dumps(layout["scene"],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (args.output/"cases").mkdir()
        for case in cases:
            (args.output/"cases"/(case["standard"]["id"]+".json")).write_text(json.dumps(case,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        for split in ("train", "validation", "test"):
            (args.output/(split+".json")).write_text(json.dumps([case for case in cases if case["standard"]["frozen_split"]==split],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (args.output/"splits.json").write_text(json.dumps({case["standard"]["id"]:case["standard"]["frozen_split"] for case in cases},indent=2)+"\n",encoding="utf-8")
        (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (args.output/"status.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
