"""Actual Gazebo/PX4 acceptance; never substitutes integrated or recorded flight.

Start scripts/start_gazebo.sh first. --control-only checks flight and RGB-D
calibration without paid APIs. The default runs configured language and vision
providers, then scores their target location against scene geometry kept outside
the planner. Each run writes one result and its execution journal.
"""
import argparse
import base64
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from uav_llm_demo import _build_runtime
from uav_llm_demo.agents import DeepSeekAgent
from uav_llm_demo.provider_config import load_planner_config
from uav_llm_demo.gazebo_runtime import GazeboRuntime, GazeboRuntimeConfig
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import TrajectoryController, distance, norm
from uav_llm_demo.onboard_observation import accept_frame, camera_request, ground_pixel
from uav_llm_demo.run_log import JournalWriter, source_fingerprint
from uav_llm_demo.runtime import RuntimeUnavailableError
from uav_llm_demo.interaction_trajectory import InteractionRecorder, validate_trajectory
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.safety import SafetyChecker
from uav_llm_demo.simulation_scene import load_scene, scene_boxes


def save_image(frame, output):
    output.write_bytes(base64.b64decode(frame["rgb"].split(",", 1)[1]))


def control_check(output, scene_id="visual_garden", runtime=None):
    runtime = runtime or GazeboRuntime(GazeboRuntimeConfig(world=scene_id))
    scene = load_scenario(scene_id)
    checker = SafetyChecker(scene.to_dict())
    started = time.monotonic()
    samples = []
    result = {"mode": "actual_px4_control", "passed": False}
    try:
        runtime.configure_scene(scene)
        runtime.reset(scene.base)
        request = camera_request(scene.base, 0, -25, runtime.simulation_time(), 1, "ground_clearance")
        metadata, _ = runtime.capture_camera(request)
        result["ground_camera_height_m"] = metadata["position_grid"][2]*5
        if result["ground_camera_height_m"] <= 0:
            raise RuntimeError("Camera optical centre is below the ground plane")
        control = TrajectoryController(response_seconds=runtime.control_response_seconds)
        for target in ((*scene.base[:2], 1.4), scene.base):
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                state = runtime.read_state()
                velocity = control.velocity_command(state, target, .05,
                                                    landing_radius=checker.landing_radius(target))
                acceleration = tuple((velocity[i]-state.velocity[i])/.05 for i in range(3))
                runtime.drive(velocity, .05, acceleration=acceleration, target=target, sample_time=state.sample_time)
                samples.append({"t": runtime.simulation_time(), "position": state.position,
                                "speed": norm(state.velocity), "landed": runtime.status()["landed"],
                                "command_velocity": velocity, "command_acceleration": acceleration})
                if distance(state.position, target) <= .18 and norm(state.velocity) <= .02:
                    if target[2] or runtime.status()["landed"] and not runtime.armed:
                        break
                time.sleep(.05)
            else:
                raise RuntimeError("PX4 did not reach the measured flight/landing state within 120 seconds")
            if target[2]:
                # Straight-down empty launch area gives an independent ground plane
                # calibration check. Camera extrinsics include payload height/tilt.
                request = camera_request(state.position, 0, -90, runtime.simulation_time(), 1, "calibration")
                metadata, packet = runtime.capture_camera(request)
                frame = accept_frame(metadata, packet)
                points = [ground_pixel(frame, [u, v])["point_grid"]
                          for u, v in ((.35, .35), (.5, .5), (.65, .65))]
                error = max(abs(point[2] * 5) for point in points)
                result.update(ground_plane_error_m=error, pose_time_error_s=metadata["pose_time_error_s"],
                              calibration=metadata, ground_points=points)
                save_image(frame, output.with_suffix(".png"))
                # Even a failed calibration must finish the commanded landing.
                if error > .1:
                    result["error"] = "Measured RGB-D reprojection does not agree with the ground plane"
                visibility = []
                result["camera_visibility"] = visibility
                for yaw in (0, 45, 90, 180, -90):
                    request = camera_request(state.position, yaw, -25, runtime.simulation_time(), 1, "visibility")
                    metadata, packet = runtime.capture_camera(request)
                    view = accept_frame(metadata, packet)
                    w, h = view["width"], view["height"]
                    depths = [view["depth_m"][y*w+x] for y in range(int(h*.15),int(h*.85),4)
                              for x in range(int(w*.15),int(w*.85),4)]
                    fraction = sum(0 < depth < 1 for depth in depths) / len(depths)
                    visibility.append(dict(yaw_deg=yaw, pitch_deg=-25, measured_yaw_deg=metadata['yaw_deg'],
                                           measured_pitch_deg=metadata['pitch_deg'], near_occlusion_fraction=fraction))
                    if fraction > .01:
                        result["error"] = "Airframe obscures the central oblique camera view"
                        save_image(view, output.with_name(output.stem+"-occluded.png"))
        altitude=max(s["position"][2]*5 for s in samples)
        result.update(passed=altitude<=7.5 and "error" not in result,
                      final=runtime.status(), max_altitude_m=altitude)
        if altitude>7.5:
            result["error"]="Flight overshoot exceeds the 0.5 m tracking allowance"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        runtime.close()
        result.update(wall_seconds=time.monotonic()-started, telemetry=samples)
    return result


def target_error(point, parts, object_id):
    """Distance to independent object geometry; zero means inside/on the object.

    This is a target-binding score, not an estimate of millimetre localization
    accuracy. RGB-D calibration is checked separately against the ground plane.
    """
    parts = [part for part in parts
             if part["name"] == object_id or part["name"].startswith(object_id+"_")]
    metres = [point[i]*scale for i, scale in enumerate((3, 3, 5))]
    def error(part):
        q=[abs(metres[i]-part["position"][axis]*scale) for i,(axis,scale) in enumerate(zip("xyz",(3,3,5)))]
        half=[part['size'][axis]/2 for axis in 'xyz']
        if part.get('shape')=='cylinder':
            return math.hypot(max(0,math.hypot(*q[:2])-half[0]),max(0,q[2]-half[2]))
        if part.get('shape')=='ellipsoid':
            if sum((p/r)**2 for p,r in zip(q,half))<=1:
                return 0.
            lo,hi=0.,math.sqrt(sum(p*p for p in q))*max(half)
            for _ in range(50):
                mid=(lo+hi)/2
                if sum((r*p/(mid+r*r))**2 for p,r in zip(q,half))>1:lo=mid
                else:hi=mid
            return math.sqrt(sum((p-r*r*p/(hi+r*r))**2 for p,r in zip(q,half)))
        return math.sqrt(sum(max(0,p-r)**2 for p,r in zip(q,half)))
    if not parts:
        raise ValueError('Unknown independent evaluation target')
    return min(error(part) for part in parts)



def visual_acceptance(views, scenario, targets, final_position, runtime, *, maximum_distance_m=None):
    """Score the observed surface, requested distance and final pose independently."""
    targets = [targets] if isinstance(targets, str) else list(targets)
    if not targets or any(not isinstance(t, str) or not t for t in targets):
        raise ValueError("Evaluation needs explicit target IDs")
    if maximum_distance_m is not None and (not math.isfinite(maximum_distance_m) or maximum_distance_m <= 0):
        raise ValueError("Evaluation distance must be finite and positive")
    if scenario.campus_layout:
        registry = ROOT/'.sim/worlds/models'/scenario.scenario_id/'target-surfaces.json'
        queries = [dict(point_grid=view['point_grid'], targets=targets) for view in views]
        response = subprocess.run(['node', str(ROOT/'scripts/score_visual_geometry.cjs'), str(registry)],
            input=json.dumps(queries), text=True, capture_output=True, check=True, timeout=30)
        scores = json.loads(response.stdout)
        geometry_kind = "rendered_triangle_surface"
    else:
        parts = scene_boxes(load_scene(scenario.asset_scene))
        scores = [min((dict(distance_m=target_error(view['point_grid'], parts, target), target_id=target)
                       for target in targets), key=lambda item: item['distance_m']) for view in views]
        geometry_kind = "scene_solid"
    errors = [item['distance_m'] for item in scores]
    selected = {item['target_id'] for item in scores}
    ranges = [view.get('range_m') for view in views]
    distance_ok = maximum_distance_m is None or all(
        isinstance(value, (int,float)) and math.isfinite(value) and 0 < value <= maximum_distance_m
        for value in ranges)
    home_error = math.sqrt(sum(((final_position[i]-scenario.base[i])*scale)**2
                              for i,scale in enumerate((3,3,5))))
    checks = dict(target_surface=bool(errors) and max(errors) < .25,
        same_target=len(selected)==1,
        fresh_views=len({v.get('frame_id') for v in views if v.get('frame_id')})>=2,
        native_depth=bool(views) and all(v.get('source')=='gazebo_rgbd' for v in views),
        requested_distance=distance_ok,
        home_position=home_error <= 1.0,
        landed=runtime.get('landed') is True and runtime.get('armed') is False)
    return dict(passed=all(checks.values()), checks=checks, geometry_kind=geometry_kind,
        target_geometry_errors_m=errors, matched_targets=[s['target_id'] for s in scores],
        confirmation_ranges_m=ranges, maximum_distance_m=maximum_distance_m,
        home_error_m=home_error, final_position_grid=list(final_position))


def _events(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def infrastructure_failure(result, journal_events=(), model_events=()):
    """Classify failed episodes from status fields and machine events, never prose."""
    if result.get("passed"):
        return None
    if result.get("failure_code"):
        return result["failure_code"]
    runtime = result.get("runtime", {})
    if runtime.get("connected") is False:
        return "runtime_disconnected"
    if runtime.get("last_error"):
        return "runtime_error"
    feedback_lost, last_request = False, None
    for event in journal_events:
        kind = event.get("event")
        if kind == "camera_failed":
            return "camera_failed"
        if kind == "semantic_execution_event" and event.get("reason") == "missing_telemetry":
            return "runtime_telemetry_failed"
        if kind == "feedback_hold":
            feedback_lost = True
        elif kind == "feedback_restored":
            feedback_lost = False
        elif kind == "provider_request":
            last_request = event.get("request", {})
    if feedback_lost:
        return "runtime_telemetry_failed"
    last_response = None
    for event in model_events:
        if event.get("kind") == "model_response":
            last_response = event["payload"]
    if last_response is not None:
        if (last_response.get("outcome") == "error" and
                (last_response.get("http_status") != 200 or last_response.get("error") in
                 {"network_timeout", "network_TLS", "network_proxy", "network_connection"})):
            return "vision_provider_failed"
    elif last_request is not None and last_request.get("http_status") != 200:
        return "provider_failed"
    return None


def terminal_without_visual_decisions(result, validation):
    """A recorded upstream termination is not a visual training trajectory."""
    return (result.get("semantic", {}).get("status") in
            {"generation_failed", "needs_clarification", "safety_wait"}
            and result.get("vision", {}).get("calls") == 0
            and isinstance(validation.get("outcome"), dict)
            and validation["outcome"].get("status") == "failed"
            and validation.get("decisions") == 0
            and validation.get("errors") == ["Episode contains no decisions"]
            and not any(result.get("journal", {}).values()))


def visual_check(args, output, runtime=None, *, planner_timeout=25):
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, planner_timeout)):
        raise ValueError("Episode and planner timeouts must be finite positive seconds")
    planner_timeout = min(planner_timeout, args.timeout)
    trajectory_dir = getattr(args, "trajectory_dir", None)
    if trajectory_dir and os.getenv("VISION_STRATEGY", "two_stage") != "direct":
        raise ValueError("Complete model-input collection currently requires VISION_STRATEGY=direct.")
    recorder = None
    if trajectory_dir:
        recorder = InteractionRecorder(trajectory_dir, metadata=dict(
            case_id=getattr(args, "case_id", output.stem), scene=args.scene,
            seed=getattr(args, "seed", None), split=getattr(args, "split", "pilot"),
            instruction=args.instruction, source_sha256=source_fingerprint(), strategy="direct",
            wall_budget_seconds=args.timeout, planner_timeout_seconds=planner_timeout,
            model=os.getenv("VISION_MODEL"), model_revision=os.getenv("VISION_MODEL_REVISION"),
            simulator=getattr(args, "simulator_manifest", {})),
            secrets=tuple(os.getenv(k) for k in ("VISION_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY")))
    journal, service = None, None
    started = time.monotonic()
    result = {"mode": "actual_px4_live_models", "passed": False,
              "wall_budget_seconds": args.timeout, "planner_timeout_seconds": planner_timeout}
    last_frame = None
    try:
        journal = JournalWriter(output.with_suffix(".events.jsonl"))
        planner = DeepSeekAgent(**load_planner_config(), timeout_seconds=planner_timeout)
        service = MissionService(scenario_id=args.scene, runtime=runtime, runtime_factory=_build_runtime,
                                 agent=planner, journal=journal, trajectory_recorder=recorder)
        if runtime is None:
            service.switch_runtime("gazebo", dict(world=args.scene))
        else:
            service.advance(service.CONTROL_DT)
        service._ensure_loop()
        service.submit_task(args.instruction)
        while time.monotonic()-started < args.timeout:
            state = service.state()
            with service._lock:
                frame = service.visual.frame
                if frame and frame["frame_id"] != last_frame:
                    save_image(frame, output.with_suffix(".png"))
                    last_frame = frame["frame_id"]
            semantic = state["semantic_mission"]
            if (state["status"] == "completed" or state["flight"]["error"] or
                    semantic["status"] in {"generation_failed", "needs_clarification", "safety_wait"}):
                break
            time.sleep(.2)
        else:
            result["error"] = "End-to-end wall-clock budget exhausted"
        scenario = load_scenario(args.scene)
        views = [view for e in semantic["evidence"] if e["action"] == "visual"
                 for view in e["result"].get("views", [])]
        runtime = service.runtime.status()
        final_position = tuple(state['observation']['position'][axis] for axis in 'xyz')
        acceptance = visual_acceptance(views, scenario, args.oracle_target, final_position, runtime,
            maximum_distance_m=getattr(args, 'oracle_maximum_distance_m', None))
        result.update(instruction=args.instruction, milestones=state["milestones"],
                      runtime=runtime, semantic=semantic, vision=state["onboard"],
                      target_geometry_errors_m=acceptance['target_geometry_errors_m'],
                      independent_acceptance=acceptance, diagnostics=state["diagnostics"],
                      simulation_seconds=service._time)
        result["passed"] = bool(state["milestones"]["safe_success"] and acceptance['passed'])
        if not result["passed"] and not result.get("error"):
            result["error"] = (state["flight"]["error"]
                               or ("Independent target and safe landing checks did not pass"
                                   if state["milestones"]["safe_success"] else semantic.get("message"))
                               or "Independent target and safe landing checks did not pass")
    except Exception as exc:
        # Provider exceptions are already sanitized by the project's transport.
        result["error"] = str(exc)
        if isinstance(exc, RuntimeUnavailableError):
            result["failure_code"] = "runtime_unavailable"
    finally:
        if service is not None:
            # Check before close() intentionally disconnects the runtime. The
            # mission loop may already have caught a telemetry exception.
            if not result.get("passed"):
                try:
                    result["runtime"] = service.runtime.status()
                    service.runtime.read_state()
                    if service.runtime.backend_name == "gazebo":
                        service.runtime.simulation_time()
                except RuntimeUnavailableError:
                    result.setdefault("failure_code", "runtime_unavailable")
                except Exception:
                    result.setdefault("failure_code", "runtime_health_check_failed")
            try:
                service.close()
            except Exception as exc:
                result.update(passed=False, error="Shutdown failed: " + type(exc).__name__,
                              failure_code="shutdown_failed")
            finally:
                # Finalization waits for bounded in-flight requests after stopping
                # control, so late replies cannot arrive after the journal closes.
                if recorder is not None:
                    service.visual.executor.shutdown(wait=True, cancel_futures=True)
                    service._executor.shutdown(wait=True, cancel_futures=True)
        if journal is not None:
            journal.flush()
            result["journal"] = dict(dropped=journal.dropped, errors=journal.errors)
            journal.close()
        else:
            result["journal"] = dict(dropped=0, errors=1)
        if any(result["journal"].values()):
            result.update(passed=False, error="Execution journal is incomplete", failure_code="journal_incomplete")
        result["wall_seconds"] = time.monotonic()-started
        if recorder is not None:
            recorder.close(outcome=dict(status="completed" if result.get("passed") else "failed",
                error=result.get("error"), wall_seconds=result["wall_seconds"],
                simulation_seconds=result.get("simulation_seconds"), journal=result["journal"]),
                oracle=dict(target_id=args.oracle_target, passed=result.get("passed", False),
                    independent_acceptance=result.get("independent_acceptance"),
                    target_geometry_errors_m=result.get("target_geometry_errors_m"),
                    milestones=result.get("milestones"), runtime=result.get("runtime")))
            result["trajectory"] = dict(path=str(Path(trajectory_dir).resolve()),
                                        validation=validate_trajectory(trajectory_dir))
            validation = result["trajectory"]["validation"]
            upstream_terminal = terminal_without_visual_decisions(result, validation)
            result["terminal_complete"] = bool(validation["complete"] or upstream_terminal)
            if not result["terminal_complete"] or any(result["journal"].values()):
                result["passed"] = False
                result["error"] = "Interaction trajectory or execution journal is incomplete"
                result.setdefault("failure_code", "trajectory_incomplete")
        try:
            result["failure_code"] = infrastructure_failure(result,
                _events(output.with_suffix(".events.jsonl")) if journal is not None else (),
                _events(Path(trajectory_dir)/"events.jsonl") if recorder is not None else ())
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            result["failure_code"] = "failure_evidence_unreadable"
        result["infrastructure_failed"] = result["failure_code"] is not None
        if result["infrastructure_failed"]:
            result["failure_stage"] = "infrastructure"
        elif not result.get("passed"):
            result["failure_stage"] = (
                "semantic_generation" if result.get("semantic", {}).get("status") in
                {"generation_failed", "needs_clarification"} else
                "visual_execution" if result.get("vision", {}).get("calls", 0) else "semantic_execution")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--scene", default="visual_garden", choices=("visual_garden", "semantic_campus"))
    parser.add_argument("--managed", action="store_true", help="Own and close a fresh headless Gazebo/PX4 session")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-dir", help="New directory for complete direct-policy interaction data")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--planner-timeout", type=float, default=25,
                        help="Per-request planner timeout; capped by episode timeout; web defaults are unchanged")
    parser.add_argument("--instruction", default="到花园寻找绿色垃圾桶，接近到8米内从机载图像确认并定位后返航降落。")
    parser.add_argument("--oracle-target", default="bin", help="Evaluation only; never sent to either model")
    parser.add_argument("--oracle-maximum-distance-m", type=float, help="Evaluation-only confirmation range from source task")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.planner_timeout)):
        parser.error("Episode and planner timeouts must be finite positive seconds")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Choose a new result path; existing evidence is not overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT/".env")
    session = None
    started = time.monotonic()
    try:
        if args.managed:
            from uav_llm_demo.gazebo_session import GazeboSession
            session = GazeboSession(load_scenario(args.scene))
            session.start(timeout=min(180, args.timeout))
            args.simulator_manifest = session.manifest
            args.seed = session.seed
        runtime = session.runtime if session else None
        result = control_check(output, args.scene, runtime) if args.control_only else visual_check(args, output, runtime, planner_timeout=args.planner_timeout)
        if session:
            result.update(simulator=session.manifest, simulator_logs=str(session.log_dir))
    except Exception as exc:
        result = dict(mode="actual_px4_control" if args.control_only else "actual_px4_live_models",
                      passed=False, error=str(exc), wall_seconds=time.monotonic()-started,
                      simulator_logs=str(session.log_dir) if session else None)
    finally:
        if session:
            session.close()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("mode", "passed", "wall_seconds", "error") if k in result}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
