from __future__ import annotations

import json
import math
import time

import cosysairsim as airsim


def vector_dict(vector) -> dict[str, float]:
    return {
        "x": float(vector.x_val),
        "y": float(vector.y_val),
        "z": float(vector.z_val),
    }


client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
if not client.ping():
    raise SystemExit("AirSim RPC ping failed.")

vehicle_name = "Drone1"
client.enableApiControl(True, vehicle_name=vehicle_name)
client.armDisarm(True, vehicle_name=vehicle_name)
client.simSetWind(airsim.Vector3r(1.5, 0.5, 0.0))

state = client.getMultirotorState(vehicle_name=vehicle_name)
kinematics = state.kinematics_estimated
before = vector_dict(kinematics.position)

client.takeoffAsync(timeout_sec=15, vehicle_name=vehicle_name).join()
client.moveToPositionAsync(
    4.0,
    2.0,
    -4.0,
    2.0,
    timeout_sec=15,
    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
    yaw_mode=airsim.YawMode(False, 0.0),
    vehicle_name=vehicle_name,
).join()
time.sleep(0.5)

state = client.getMultirotorState(vehicle_name=vehicle_name)
kinematics = state.kinematics_estimated
after = vector_dict(kinematics.position)
distance = math.sqrt(
    sum((after[axis] - before[axis]) ** 2 for axis in ("x", "y", "z"))
)

client.hoverAsync(vehicle_name=vehicle_name).join()

if distance < 1.0:
    raise SystemExit(f"AirSim vehicle did not move far enough: {distance:.3f} m")

print(
    json.dumps(
        {
            "rpc": "ok",
            "vehicle": vehicle_name,
            "wind_ned_mps": {"north": 1.5, "east": 0.5, "down": 0.0},
            "position_before_ned": before,
            "position_after_ned": after,
            "distance_m": distance,
            "orientation": {
                "x": float(kinematics.orientation.x_val),
                "y": float(kinematics.orientation.y_val),
                "z": float(kinematics.orientation.z_val),
                "w": float(kinematics.orientation.w_val),
            },
            "linear_velocity_ned": vector_dict(kinematics.linear_velocity),
            "angular_velocity_body": vector_dict(kinematics.angular_velocity),
            "linear_acceleration_ned": vector_dict(
                kinematics.linear_acceleration
            ),
            "angular_acceleration_body": vector_dict(
                kinematics.angular_acceleration
            ),
        },
        indent=2,
    )
)
