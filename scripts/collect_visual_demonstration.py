"""Collect one disclosed file-assisted expert episode from live Gazebo RGB-D.

Provide a separately authored task-standard case, then inspect each input-N.json
and current PNG before writing response-N.json. Response native-tool arguments
must include their reason. Pixel responses also require annotation.polygon
(normalized visible-interior coordinates) and annotation.reviewer. The hidden
--oracle-target is used only after execution, never in expert policy inputs.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import CancelledError
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from collect_demonstrations import collect_case
from verify_gazebo import target_error
from uav_llm_demo.direct_vision import DirectVisionAgent, TOOLS, PROMPT_VERSION
from uav_llm_demo.demonstration_data import _inside, _visual_action, decision_candidates
from uav_llm_demo.scenarios import load_scenario, load_asset_scenario
from uav_llm_demo.simulation_scene import load_scene, scene_boxes


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


class FileExpertVision(DirectVisionAgent):
    """Explicit expert provider without a network client or hidden target state."""
    def __init__(self, directory, *, timeout=45):
        if not 0 < timeout <= 45:
            raise ValueError("Expert input timeout must be positive and at most 45 seconds")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model = "scripted_expert/visible-image-annotation-v1"
        self.base_url, self.api_key, self.configured = "http://127.0.0.1:1/v1", None, True
        self.timeout_seconds, self.interaction_observer = timeout, None
        self.calls, self.annotations = 0, []

    def decide(self, context, frames, *, additional_request=lambda: None,
               cancelled=lambda: False, interaction_observer=None):
        self.calls += 1
        index, started = self.calls, time.monotonic()
        messages = self._messages(context, frames)
        request = dict(model=self.model, messages=messages, tools=deepcopy(TOOLS),
                       tool_choice="auto", parallel_tool_calls=False, temperature=.1,
                       max_tokens=1800, stream=False)
        review_request = deepcopy(request)
        images = []
        for frame in frames:
            filename = f"input-{index:03d}-frame-{len(images):02d}.png"
            (self.directory / filename).write_bytes(base64.b64decode(frame["rgb"].split(",", 1)[1], validate=True))
            images.append(dict(frame_id=frame["frame_id"], rgb_sha256=frame["rgb_sha256"], path=filename,
                               is_current=frame is frames[-1]))
        image_paths = {frame["rgb"]: saved["path"] for frame, saved in zip(frames, images)}
        for part in review_request["messages"][1]["content"]:
            if part.get("type") == "image_url":
                part["image_url"]["url"] = image_paths[part["image_url"]["url"]]
        write_json(self.directory / f"input-{index:03d}.json", dict(
            index=index, request=review_request, frames=images,
            response_file=f"response-{index:03d}.json", provenance="agent_assisted_expert"))
        print(json.dumps(dict(waiting=index, input=str(self.directory/f"input-{index:03d}.json"),
                              image=str(self.directory/images[-1]["path"])), ensure_ascii=False), flush=True)
        body, outcome, error = None, "error", None
        try:
            destination = self.directory / f"response-{index:03d}.json"
            while not destination.exists():
                if cancelled():
                    outcome = "cancelled"
                    raise CancelledError("Expert decision superseded")
                if time.monotonic() - started >= self.timeout_seconds:
                    raise TimeoutError("No expert input within the camera-decision time budget")
                time.sleep(.1)
            value = json.loads(destination.read_text(encoding="utf-8"))
            if set(value) - {"tool", "arguments", "annotation"}:
                raise ValueError("Expert response has unknown fields")
            body = dict(choices=[dict(finish_reason="tool_calls", message=dict(
                role="assistant", content=None, tool_calls=[dict(id=f"expert-{index}", type="function",
                    function=dict(name=value["tool"], arguments=json.dumps(value["arguments"],
                        ensure_ascii=False, allow_nan=False)))]))])
            action = self._decision(body, frames[-1]["frame_id"])
            if action["tool"] in {"ground", "probe"}:
                annotation = value.get("annotation", {})
                polygon, reviewer = annotation.get("polygon"), annotation.get("reviewer")
                if not isinstance(reviewer, str) or not reviewer.strip() or not _inside(action["arguments"]["pixel"], polygon):
                    raise ValueError("Pixel decision needs its visible-interior polygon and reviewer")
                region = dict(frame_id=frames[-1]["frame_id"], rgb_sha256=frames[-1]["rgb_sha256"],
                    polygon=deepcopy(polygon), reviewer=reviewer, annotation_source="agent_assisted")
                # Probe may query an occluder rather than the task target. Its
                # independent surface ID is evaluation-only, never policy input.
                if action["tool"] == "probe":
                    region["object_id"] = annotation.get("object_id")
                self.annotations.append(region)
            if cancelled():
                outcome = "cancelled"
                raise CancelledError("Expert decision superseded")
            outcome = "ok"
            return action
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            if interaction_observer is not None:
                interaction_observer(dict(model=self.model, prompt_version=PROMPT_VERSION,
                    request=request, response=body, http_status=None, source="provider_override",
                    elapsed_ms=round((time.monotonic()-started)*1000), outcome=outcome, error=error))


def score_visual(episode, result, oracle_target, annotations, *, scene_file=None):
    """Evaluate saved measurements; target geometry never reaches FileExpertVision."""
    standard = json.loads((episode / "standard.json").read_text(encoding="utf-8"))
    scene = load_asset_scenario(scene_file) if scene_file else load_scenario(standard["scenario"])
    parts = None if scene.campus_layout else scene_boxes(load_scene(scene.asset_scene))
    localizations = []
    candidates = decision_candidates(episode)
    for candidate in candidates:
        terminal = candidate.get("result", {}).get("payload", {})
        action = terminal.get("action", {})
        if terminal.get("status") not in {"applied", "completed"} or action.get("tool") not in {"ground", "probe"}:
            continue
        feedback = terminal.get("feedback", {}).get("measured")
        if not isinstance(feedback, dict):
            continue
        frame_id = action["arguments"]["frame_id"]
        frame = next(f for f in candidate["input"]["payload"]["frames"] if f["frame_id"] == frame_id)
        region = next((item for item in annotations if item.get("frame_id") == frame_id
                       and item.get("rgb_sha256") == frame["rgb_sha256"]
                       and _inside(action["arguments"]["pixel"], item.get("polygon"))), {})
        measured_object = region.get("object_id") if action["tool"] == "probe" else oracle_target
        if not measured_object:
            continue
        localizations.append(dict(decision_id=candidate["decision_id"], tool=action["tool"], frame_id=frame_id,
            rgb_sha256=frame["rgb_sha256"], pixel=action["arguments"]["pixel"],
            source=frame["source"], object_id=measured_object, point_grid=feedback["point_grid"],
            surface_error_m=None if scene.campus_layout else target_error(feedback["point_grid"], parts, measured_object),
            frame_age_seconds=terminal["state"]["simulation_time"]-frame["simulation_time"],
            distance_m=max(feedback["range_m"], feedback.get("current_range_m", 0)),
            response_sequence=candidate["response_sequence"]))
    if scene.campus_layout and localizations:
        registry = ROOT/".sim/worlds/models"/scene.scenario_id/"target-surfaces.json"
        queries = [dict(point_grid=loc["point_grid"], targets=[loc["object_id"]]) for loc in localizations]
        response = subprocess.run(["node", str(ROOT/"scripts/score_visual_geometry.cjs"), str(registry)],
            input=json.dumps(queries),text=True,capture_output=True,check=True,timeout=30)
        scores = json.loads(response.stdout)
        if len(scores) != len(localizations):
            raise ValueError("Surface evaluation omitted a recorded localization")
        for loc, score in zip(localizations, scores):
            loc["surface_error_m"] = score["distance_m"]
        result["target_surfaces_sha256"] = hashlib.sha256(registry.read_bytes()).hexdigest()
    final = result.get("runtime_final", {})
    result.update(localizations=localizations, landed=final.get("landed") is True,
        disarmed=final.get("armed") is False, oracle_target=oracle_target,
        vision_models=sorted({c["request"]["payload"]["request"]["model"] for c in candidates
                              if c.get("request") and c.get("input", {}).get("payload", {}).get("modality") != "text"}),
        annotation_source="agent_assisted", human_reviewed=False)
    # Final confirmation is the executor's two-view record. Intermediate
    # acquisitions/probes can be farther away and remain separately reviewed.
    limits = standard["visual_limits"]
    final_views = [view for item in result.get("evidence", [])
                   for view in item.get("result", {}).get("views", [])]
    confirmed = [loc for loc in localizations if loc.get("tool") == "ground" and any(
        view.get("frame_id") == loc["frame_id"] and view.get("rgb_sha256") == loc["rgb_sha256"]
        for view in final_views)]
    result["visual_target_verified"] = (len({loc["frame_id"] for loc in confirmed}) >= 2 and all(
        loc["source"] == "gazebo_rgbd" and loc["surface_error_m"] <= limits["max_surface_error_m"]
        and 0 <= loc["frame_age_seconds"] <= limits["max_frame_age_seconds"]
        and loc["distance_m"] <= limits["max_distance_m"] for loc in confirmed))
    # Read-only eligibility diagnostics: explicit response selection and review
    # are still required before export; failed actions remain repair context.
    eligibility = []
    for candidate in candidates:
        if not candidate.get("input") or candidate["input"]["payload"].get("modality") == "text":
            continue
        terminal = (candidate.get("result") or {}).get("payload", {})
        row = dict(decision_id=candidate["decision_id"], response_sequence=candidate["response_sequence"],
                   tool=terminal.get("action", {}).get("tool"), eligible=False)
        try:
            if terminal.get("status") not in {"applied", "completed"}:
                raise ValueError("Failed action retained as repair context")
            row["verification"] = _visual_action(episode, candidate, terminal["action"], standard, result, annotations)
            row["eligible"] = True
        except (ValueError, KeyError, TypeError) as exc:
            row["reason"] = str(exc)
        eligibility.append(row)
    result["visual_decision_review"] = eligibility
    result["passed"] = (result.get("oracle_success") is True and result.get("service_success") is True
        and result.get("safe_execution") is True and result["landed"] and result["disarmed"]
        and result["visual_target_verified"] and result.get("visual_collector_unchanged", True))
    (episode / "execution.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    write_json(episode / "visual_annotations.json", annotations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="One separately authored standard+demonstration JSON")
    parser.add_argument("--scene-file", type=Path, help="Explicit frozen scene JSON from the curriculum")
    parser.add_argument("--oracle-target", required=True, help="Evaluation-only scene object ID, never a policy input")
    parser.add_argument("--output", type=Path, required=True, help="New collection root")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wall-budget", type=float, default=600)
    parser.add_argument("--response-timeout", type=float, default=45)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    standard = case["standard"]
    if standard.get("evaluation_kind") != "visual_rgbd" or standard.get("annotation_source") != "agent_assisted":
        parser.error("Declare an agent-assisted visual_rgbd task standard before collecting")
    if "visible_regions" in standard:
        parser.error("Observed-frame annotations belong after collection, outside the source task standard")
    args.output.mkdir(parents=True, exist_ok=False)
    vision = FileExpertVision(args.output/"annotation", timeout=args.response_timeout)
    episode = args.output/"episode"
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = collect_case(case, episode, seed=args.seed, simulation_budget=600,
        wall_budget=args.wall_budget, runtime_backend="gazebo", vision_agent=vision,
        scenario_override=load_asset_scenario(args.scene_file) if args.scene_file else None, scene_file=args.scene_file)
    result.update(visual_collector_sha256=source_sha,
        visual_collector_unchanged=source_sha == hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result = score_visual(episode, result, args.oracle_target, vision.annotations, scene_file=args.scene_file)
    print(json.dumps({key: result.get(key) for key in ("status", "passed", "visual_target_verified",
          "landed", "disarmed", "wall_seconds")}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
