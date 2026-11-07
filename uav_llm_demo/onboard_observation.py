"""Synchronized camera packets and pixel/depth grounding, independent of UI cameras.

The browser renders the requested measured pose. Only pixels and calibration cross
into the vision planner; scene object identities never cross this boundary.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import math
import struct
import uuid

WIDTH, HEIGHT = 320, 240
FOV_DEG = 75.0
NEAR_M, FAR_M = .1, 250.0


def finite(value, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("Expected a finite value within camera limits.")
    return float(value)


def camera_request(position, yaw_deg, pitch_deg, now, revision, execution_id, *, fov_deg=FOV_DEG):
    fov_deg = finite(fov_deg, 30, FOV_DEG)
    focal = HEIGHT / (2 * math.tan(math.radians(fov_deg / 2)))
    return dict(frame_id=uuid.uuid4().hex, revision=revision, execution_id=execution_id,
                simulation_time=now, position_grid=list(position),
                yaw_deg=finite(yaw_deg, -180, 180), pitch_deg=finite(pitch_deg, -90, 30),
                width=WIDTH, height=HEIGHT, fx=focal, fy=focal, cx=WIDTH / 2, cy=HEIGHT / 2,
                near_m=NEAR_M, far_m=FAR_M, vertical_fov_deg=fov_deg,
                depth_convention="optical_axis_metres", source="html_onboard_rgbd")


def accept_frame(request, packet):
    if not isinstance(packet, dict) or set(packet) != {"frame_id", "rgb", "depth_m"} or packet["frame_id"] != request["frame_id"]:
        raise ValueError("Camera frame does not match the pending measured pose.")
    rgb = packet["rgb"]
    if not isinstance(rgb, str) or not rgb.startswith("data:image/png;base64,") or len(rgb) > 2_000_000:
        raise ValueError("Camera requires a bounded PNG image.")
    try:
        data = base64.b64decode(rgb.split(",", 1)[1], validate=True)
        if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR" or struct.unpack(">II", data[16:24]) != (request["width"], request["height"]):
            raise ValueError
    except (ValueError, struct.error):
        raise ValueError("Invalid camera PNG dimensions or encoding.") from None
    depth = packet["depth_m"]
    if not isinstance(depth, list) or len(depth) != request["width"] * request["height"]:
        raise ValueError("Depth must have the same resolution as RGB.")
    depth = tuple(finite(value, 0, FAR_M) for value in depth)
    result = deepcopy(request)
    result.update(rgb=rgb, depth_m=depth, rgb_sha256=hashlib.sha256(data).hexdigest(),
                  depth_sha256=hashlib.sha256(struct.pack(f"<{len(depth)}f", *depth)).hexdigest())
    return result


def frame_metadata(frame):
    return {key: deepcopy(value) for key, value in frame.items() if key not in {"rgb", "depth_m"}}


def copy_frame(frame):
    """Copy small metadata; immutable pixels can be shared with inference."""
    return dict(frame_metadata(frame), rgb=frame["rgb"], depth_m=frame["depth_m"])


def ground_pixel(frame, pixel):
    """Back-project a normalized pixel using that frame's optical-axis depth."""
    if not isinstance(pixel, list) or len(pixel) != 2:
        raise ValueError("pixel must be [u,v] in normalized image coordinates.")
    u, v = (finite(value, 0, 1) for value in pixel)
    width, height = frame["width"], frame["height"]
    x, y = min(width - 1, int(u * width)), min(height - 1, int(v * height))
    d = frame["depth_m"][y * width + x]
    if d < NEAR_M or d >= FAR_M - .1:
        raise ValueError("The selected pixel has no usable depth; observe another surface.")
    yaw, pitch = math.radians(frame["yaw_deg"]), math.radians(frame["pitch_deg"])
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    right = (-math.sin(yaw), math.cos(yaw), 0)
    up = (-math.sin(pitch) * math.cos(yaw), -math.sin(pitch) * math.sin(yaw), math.cos(pitch))
    if "camera_basis" in frame:
        forward, right, up = frame["camera_basis"]
    dx, dy = (x + .5 - frame["cx"]) / frame["fx"], (frame["cy"] - y - .5) / frame["fy"]
    offset = [d * (forward[i] + dx * right[i] + dy * up[i]) for i in range(3)]
    return dict(pixel=[u, v], point_grid=[frame["position_grid"][i] + offset[i] / (3, 3, 5)[i] for i in range(3)],
                range_m=math.sqrt(sum(value * value for value in offset)), frame_id=frame["frame_id"],
                rgb_sha256=frame["rgb_sha256"], depth_sha256=frame["depth_sha256"])


def validate_visual_evidence(node, evidence):
    """Geometry and provenance are deterministic; visual identity remains a model judgement."""
    from .safety import _point, metric_distance
    result = evidence.get("result", {})
    views = result.get("views", [])
    if (result.get("observed") is not True or result.get("description") != node["visual"]["description"]
            or not isinstance(views, list) or len(views) != 2
            or len({item.get("frame_id") for item in views}) != 2):
        raise ValueError("Visual completion requires two distinct, bound observations.")
    now = evidence["simulation_time"]
    for view in views:
        if (view.get("revision") != evidence["revision"] or view.get("execution_id") != evidence["execution_id"]
                or view.get("source") not in {"html_onboard_rgbd", "gazebo_rgbd"}
                or not all(isinstance(view.get(k), str) and len(view[k]) == 64 for k in ("rgb_sha256", "depth_sha256"))):
            raise ValueError("Visual evidence is not bound to the current execution and RGB/depth pair.")
        finite(view.get("confidence"), .8, 1)
        finite(view.get("range_m"), 0, node["visual"]["max_distance_m"] + .1)
        observed_at = finite(view.get("simulation_time"), 0, now)
        if now - observed_at > 90:
            raise ValueError("Visual confirmation has expired.")
        if metric_distance(_point(evidence["position"]), _point(view["point_grid"])) > node["visual"]["max_distance_m"] + .1:
            raise ValueError("Measured position is outside the required viewing distance.")
    if metric_distance(_point(views[0]["point_grid"]), _point(views[1]["point_grid"])) > 3:
        raise ValueError("Visual observations do not agree spatially.")
