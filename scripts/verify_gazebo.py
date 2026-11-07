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
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from uav_llm_demo import _build_runtime
from uav_llm_demo.gazebo_runtime import GazeboRuntime, GazeboRuntimeConfig
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.motion import TrajectoryController, distance, norm
from uav_llm_demo.onboard_observation import accept_frame, camera_request, ground_pixel
from uav_llm_demo.run_log import JournalWriter
from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.safety import SafetyChecker
from uav_llm_demo.simulation_scene import load_scene, scene_boxes, campus_collision_geometry


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


def visual_check(args, output, runtime=None):
    journal = JournalWriter(output.with_suffix(".events.jsonl"))
    service = MissionService(scenario_id=args.scene, runtime=runtime, runtime_factory=_build_runtime, journal=journal)
    started = time.monotonic()
    result = {"mode": "actual_px4_live_models", "passed": False}
    last_frame = None
    try:
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
        parts = (campus_collision_geometry(scenario.campus_layout) if scenario.campus_layout
                 else scene_boxes(load_scene(scenario.asset_scene)))
        views = [view for e in semantic["evidence"] if e["action"] == "visual"
                 for view in e["result"].get("views", [])]
        errors = [target_error(view["point_grid"], parts, args.oracle_target) for view in views]
        runtime = service.runtime.status()
        result.update(instruction=args.instruction, milestones=state["milestones"],
                      runtime=runtime, semantic=semantic, vision=state["onboard"],
                      target_geometry_errors_m=errors, diagnostics=state["diagnostics"],
                      simulation_seconds=service._time)
        result["passed"] = (state["milestones"]["safe_success"] and runtime["landed"] and not runtime["armed"]
                            and len(errors) >= 2 and max(errors) < .6
                            and all(view["source"] == "gazebo_rgbd" for view in views))
    except Exception as exc:
        # Provider exceptions are already sanitized by the project's transport.
        result["error"] = str(exc)
    finally:
        service.close()
        journal.close()
        result["wall_seconds"] = time.monotonic()-started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--scene", default="visual_garden", choices=("visual_garden", "semantic_campus"))
    parser.add_argument("--managed", action="store_true", help="Own and close a fresh headless Gazebo/PX4 session")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--instruction", default="到花园寻找绿色垃圾桶，接近到8米内从机载图像确认并定位后返航降落。")
    parser.add_argument("--oracle-target", default="bin", help="Evaluation only; never sent to either model")
    args = parser.parse_args()
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
        runtime = session.runtime if session else None
        result = control_check(output, args.scene, runtime) if args.control_only else visual_check(args, output, runtime)
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
