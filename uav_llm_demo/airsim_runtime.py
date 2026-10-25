from __future__ import annotations

import asyncio
import importlib
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from math import atan2, asin, copysign, degrees, isfinite, pi
from queue import Queue
from threading import Thread
from collections.abc import Callable
from typing import Any

from .environment import BASE, point_to_dict
from .runtime import RuntimeUnavailableError
from .motion import VehicleState
from .scene_layout import AIRSIM_ORIGIN, scene_layout, color_texture
from .airsim_connection import default_airsim_host, validate_airsim_host, texture_directory

class AirSimRuntimeError(RuntimeUnavailableError):
    pass


class AirSimRpcWorker:
    def __init__(self) -> None:
        self._queue: Queue[object] = Queue()
        self._closed = False
        self._stop = object()
        self._thread = Thread(
            target=self._run,
            name="airsim-rpc-worker",
            daemon=True,
        )
        self._thread.start()

    def call(self, callback: Callable[[], Any], timeout: float | None = None) -> Any:
        if self._closed:
            raise AirSimRuntimeError("AirSim RPC worker is closed.")
        result: Future[Any] = Future()
        self._queue.put((callback, result))
        return result.result(timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._stop)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                task = self._queue.get()
                if task is self._stop:
                    return
                callback, result = task
                try:
                    result.set_result(callback())
                except BaseException as exc:
                    result.set_exception(exc)
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


@dataclass(frozen=True)
class AirSimRuntimeConfig:
    host: str = field(default_factory=default_airsim_host)
    port: int = 41451
    vehicle_name: str = "Drone1"
    velocity_mps: float = 4.0
    meters_per_cell: float = 3.0
    meters_per_altitude_level: float = 5.0
    command_timeout: float = 8.0

    def __post_init__(self):
        object.__setattr__(self, "host", validate_airsim_host(self.host))


class AirSimRuntime:
    backend_name = "airsim"

    def __init__(self, config: AirSimRuntimeConfig | None = None) -> None:
        self.config = config or AirSimRuntimeConfig()
        self.api = None
        self.client = None
        self.connected = False
        self.armed = False
        self._owns_control = False
        self.wind_ned = (0.0, 0.0, 0.0)
        self.last_error: str | None = None
        self.last_telemetry: dict[str, object] | None = None
        self.client_package: str | None = None
        self._worker = AirSimRpcWorker()
        self._closed = False
        self._scene = None
        self._origin = (0.0, 0.0, 0.0)

    def configure_scene(self, scenario) -> None:
        if not self.connected:
            self.connect()
        self._worker.call(lambda: self._configure_scene(scenario))

    def _configure_scene(self, scenario) -> None:
        self._require_connection()
        if (self.config.meters_per_cell, self.config.meters_per_altitude_level) != (3.0, 5.0):
            raise AirSimRuntimeError("Shared map uses 3 metres per cell and 5 metres per altitude level.")
        if "Cube" not in self.client.simListAssets():
            raise AirSimRuntimeError("Blocks must provide the Cube asset for the shared map.")
        layout = scene_layout(scenario)
        folder = texture_directory()
        materials = {color: color_texture(color, folder) for color in {item["color"] for item in layout}}
        self._origin = AIRSIM_ORIGIN
        # Only replace objects owned by this demo; retain the packaged Blocks map.
        for name in self.client.simListSceneObjects("UAVDemo_.*"):
            if not self.client.simDestroyObject(name):
                raise AirSimRuntimeError(f"Cannot replace shared scene object {name}.")
        for item in layout:
            point = as_grid_tuple(item["position"])
            position = self._grid_to_ned(point)
            size = item["size"]
            name = self.client.simSpawnObject(
                "UAVDemo_" + item["name"], "Cube",
                self.api.Pose(self.api.Vector3r(*position), self.api.Quaternionr()),
                self.api.Vector3r(size["y"], size["x"], size["z"]),
                physics_enabled=False,
            )
            if not name:
                raise AirSimRuntimeError(f"Cannot spawn shared scene object {item['name']}.")
            if not self.client.simSetObjectMaterialFromTexture(name, materials[item["color"]]):
                raise AirSimRuntimeError(f"Cannot apply shared map material to {name}.")
        self._scene = scenario

    def _grid_to_ned(self, point):
        local = grid_to_ned(point, self.config.meters_per_cell,
                            self.config.meters_per_altitude_level)
        return tuple(local[i] + self._origin[i] for i in range(3))

    def connect(self) -> dict[str, object]:
        return self._worker.call(self._connect)

    def _connect(self) -> dict[str, object]:
        self.api, self.client_package = import_airsim_client()
        try:
            self.client = self.api.MultirotorClient(
                ip=self.config.host,
                port=self.config.port,
                timeout_value=2,
            )
            if not self.client.ping():
                raise AirSimRuntimeError("AirSim RPC server did not answer ping().")
            self.connected = True
            self.last_error = None
            self.last_telemetry = self._read_telemetry()
            return self.status()
        except Exception as exc:
            self.connected = False
            self.armed = False
            self.last_error = str(exc)
            raise AirSimRuntimeError(
                f"Unable to connect to AirSim at {self.config.host}:{self.config.port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._closed:
            return
        try:
            self._worker.call(self._disconnect)
        finally:
            self._worker.close()
            self._closed = True

    def _disconnect(self) -> None:
        if self.client:
            if self.connected and self._owns_control and hasattr(self.client, "enableApiControl"):
                try:
                    self.client.enableApiControl(
                        False,
                        vehicle_name=self.config.vehicle_name,
                    )
                except Exception as exc:
                    self.last_error = f"Unable to release AirSim API control: {exc}"
            rpc_client = getattr(self.client, "client", None)
            if rpc_client and hasattr(rpc_client, "close"):
                rpc_client.close()
        self.connected = False
        self.armed = False
        self._owns_control = False

    def acquire_control(self):
        self._worker.call(self._acquire_control)

    def _acquire_control(self):
        self._require_connection()
        self.client.enableApiControl(True, vehicle_name=self.config.vehicle_name)
        self._owns_control = True
        self.client.armDisarm(True, vehicle_name=self.config.vehicle_name)
        self.armed = True

    def reset(
        self,
        target_grid: tuple[float, float, float] = BASE,
    ) -> dict[str, object]:
        if not self.connected:
            self.connect()
        return self._worker.call(lambda: self._reset_to_base(target_grid))

    def _reset_to_base(
        self,
        target_grid: tuple[float, float, float],
    ) -> dict[str, object]:
        self._reset()
        if self._scene is not None:
            position = self._grid_to_ned(target_grid)
            self.client.simSetVehiclePose(
                self.api.Pose(self.api.Vector3r(*position), self.api.Quaternionr()),
                True, vehicle_name=self.config.vehicle_name,
            )
        if target_grid[2] > 0:
            self._ensure_airborne(target_grid)
        self.last_telemetry = self._read_telemetry()
        if self._scene is not None and target_grid[2] == 0:
            # Teleporting after reset can retain a falling velocity for a few
            # physics frames. Do not hand that transient to the safety checker.
            deadline = time.monotonic() + 3.0
            while (self.last_telemetry["landed_state"] != str(self.api.LandedState.Landed)
                   or any(abs(value) > 0.1 for value in self.last_telemetry["linear_velocity"].values())):
                if time.monotonic() >= deadline:
                    raise AirSimRuntimeError("Blocks did not settle on the launch surface after reset.")
                time.sleep(0.05)
                self.last_telemetry = self._read_telemetry()
        return self.status()

    def _reset(self) -> None:
        self._require_connection()
        self.client.reset()
        time.sleep(0.3)
        self._acquire_control()

    def ensure_airborne(self, target_grid: tuple[float, float, float] = BASE) -> None:
        self._worker.call(lambda: self._ensure_airborne(target_grid))

    def _ensure_airborne(
        self,
        target_grid: tuple[float, float, float],
    ) -> None:
        self._require_connection()
        state = self.client.getMultirotorState(vehicle_name=self.config.vehicle_name)
        landed_state = getattr(state, "landed_state", None)
        landed_enum = getattr(getattr(self.api, "LandedState", None), "Landed", None)
        if landed_state == landed_enum:
            self.client.takeoffAsync(
                timeout_sec=12,
                vehicle_name=self.config.vehicle_name,
            ).join()

        target = self._grid_to_ned(target_grid)
        self.client.moveToPositionAsync(
            target[0],
            target[1],
            target[2],
            self.config.velocity_mps,
            timeout_sec=self.config.command_timeout,
            drivetrain=self.api.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=self.api.YawMode(False, 0.0),
            vehicle_name=self.config.vehicle_name,
        ).join()

    def set_wind(self, north: float, east: float, down: float) -> dict[str, object]:
        return self._worker.call(lambda: self._set_wind(north, east, down))

    def _set_wind(
        self,
        north: float,
        east: float,
        down: float,
    ) -> dict[str, object]:
        self._require_connection()
        self.client.simSetWind(self.api.Vector3r(north, east, down))
        self.wind_ned = (float(north), float(east), float(down))
        return self.status()

    def read_state(self):
        return self._worker.call(self._vehicle_state)

    def _vehicle_state(self):
        telemetry = self._read_telemetry()
        velocity = telemetry["linear_velocity"]
        acceleration = telemetry["linear_acceleration"]
        vertical_velocity = -velocity["z"]
        if telemetry["ground_contact_adjusted"]:
            vertical_velocity = max(0.0, vertical_velocity)
        return VehicleState(
            as_grid_tuple(telemetry["position_grid"]),
            (velocity["y"], -velocity["x"], vertical_velocity),
            (acceleration["y"], -acceleration["x"], -acceleration["z"]),
            telemetry["orientation"]["yaw"],
        )

    def drive(self, velocity, dt, *, yaw=None):
        if yaw is not None and (isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not isfinite(yaw)):
            raise ValueError("Yaw must be a finite angle in radians.")
        # Project heading already matches NED yaw: north=0, east=+90 degrees.
        # Keep legacy north-facing behavior when no independent heading is set.
        yaw_degrees = 0.0 if yaw is None else degrees((yaw + pi) % (2 * pi) - pi)

        def send():
            self._require_connection()
            # Grid-axis m/s -> world NED m/s. A short lease limits stale commands.
            self.client.moveByVelocityAsync(
                -velocity[1], velocity[0], -velocity[2], 0.3,
                drivetrain=self.api.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=self.api.YawMode(False, yaw_degrees),
                vehicle_name=self.config.vehicle_name,
            )
            return self._vehicle_state()
        return self._worker.call(send)

    def read_telemetry(self) -> dict[str, object]:
        return self._worker.call(self._read_telemetry)

    def _read_telemetry(self) -> dict[str, object]:
        self._require_connection()
        state = self.client.getMultirotorState(vehicle_name=self.config.vehicle_name)
        kinematics = state.kinematics_estimated
        position = vector_tuple(kinematics.position)
        orientation = quaternion_to_euler(kinematics.orientation)
        grid = ned_to_grid(tuple(position[i] - self._origin[i] for i in range(3)),
                           self.config.meters_per_cell, self.config.meters_per_altitude_level)
        # SimpleFlight reports Flying as soon as a velocity command is sent,
        # even while the body still touches the pad. Resolve Unreal's bounded
        # contact penetration only inside the known launch/landing column.
        contact_adjusted = (self._scene is not None and self._scene.base[2] == 0
                            and sum(((grid[i] - self._scene.base[i]) * self.config.meters_per_cell) ** 2
                                    for i in range(2)) <= 0.75 ** 2
                            and -0.1 <= grid[2] * self.config.meters_per_altitude_level < 0)
        if contact_adjusted:
            grid = (grid[0], grid[1], 0.0)
        telemetry = {
            "position_ned": vector_to_dict(position),
            "ground_contact_adjusted": contact_adjusted,
            "position_grid": point_to_dict(grid),
            "orientation": {
                "roll": orientation[0],
                "pitch": orientation[1],
                "yaw": orientation[2],
            },
            "linear_velocity": vector_to_dict(vector_tuple(kinematics.linear_velocity)),
            "angular_velocity": vector_to_dict(vector_tuple(kinematics.angular_velocity)),
            "linear_acceleration": vector_to_dict(
                vector_tuple(kinematics.linear_acceleration)
            ),
            "angular_acceleration": vector_to_dict(
                vector_tuple(kinematics.angular_acceleration)
            ),
            "timestamp": getattr(state, "timestamp", 0),
            "landed_state": str(getattr(state, "landed_state", "unknown")),
            "source": "airsim",
        }
        self.last_telemetry = telemetry
        return telemetry

    def stop(self):
        def halt():
            self._require_connection()
            self.client.cancelLastTask(vehicle_name=self.config.vehicle_name)
            self.client.hoverAsync(vehicle_name=self.config.vehicle_name)
        self._worker.call(halt)

    def close(self):
        self.disconnect()

    def status(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "scene_id": self._scene.scenario_id if self._scene else None,
            "scene_origin_ned": vector_to_dict(self._origin),
            "connected": self.connected,
            "supports_wind": True,
            "source": self.backend_name,
            "armed": self.armed,
            "host": self.config.host,
            "port": self.config.port,
            "vehicle_name": self.config.vehicle_name,
            "velocity_mps": self.config.velocity_mps,
            "meters_per_cell": self.config.meters_per_cell,
            "meters_per_altitude_level": self.config.meters_per_altitude_level,
            "wind_ned": vector_to_dict(self.wind_ned),
            "client_package": self.client_package,
            "last_error": self.last_error,
            "telemetry": self.last_telemetry,
        }

    def _require_connection(self) -> None:
        if not self.connected or self.client is None or self.api is None:
            raise AirSimRuntimeError("AirSim runtime is not connected.")


def import_airsim_client() -> tuple[Any, str]:
    errors = []
    for module_name in ("cosysairsim", "airsim"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        required = ("MultirotorClient", "Vector3r", "DrivetrainType", "YawMode")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            errors.append(
                f"{module_name}: incompatible package, missing {', '.join(missing)}"
            )
            continue
        return module, module_name
    raise AirSimRuntimeError(
        "No compatible AirSim Python client is installed. Install the "
        "Cosys-AirSim PythonClient or Microsoft airsim==1.8.1 in the Flask "
        "environment. " + "; ".join(errors)
    )


def grid_to_ned(
    point: tuple[float, float, float],
    meters_per_cell: float = 3.0,
    meters_per_altitude_level: float = 5.0,
) -> tuple[float, float, float]:
    x, y, z = point
    north = (BASE[1] - y) * meters_per_cell
    east = (x - BASE[0]) * meters_per_cell
    down = -z * meters_per_altitude_level
    return north, east, down


def ned_to_grid(
    point: tuple[float, float, float],
    meters_per_cell: float = 3.0,
    meters_per_altitude_level: float = 5.0,
) -> tuple[float, float, float]:
    north, east, down = point
    x = BASE[0] + east / meters_per_cell
    y = BASE[1] - north / meters_per_cell
    z = -down / meters_per_altitude_level
    return x, y, z


def quaternion_to_euler(quaternion: Any) -> tuple[float, float, float]:
    x = float(quaternion.x_val)
    y = float(quaternion.y_val)
    z = float(quaternion.z_val)
    w = float(quaternion.w_val)

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = copysign(pi / 2, sinp) if abs(sinp) >= 1 else asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def vector_tuple(vector: Any) -> tuple[float, float, float]:
    return float(vector.x_val), float(vector.y_val), float(vector.z_val)


def vector_to_dict(
    vector: tuple[float, float, float],
) -> dict[str, float]:
    return {"x": vector[0], "y": vector[1], "z": vector[2]}


def as_grid_tuple(
    point: dict[str, float] | tuple[float, float, float],
) -> tuple[float, float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"]), float(point["z"])
    return float(point[0]), float(point[1]), float(point[2])
