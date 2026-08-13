from __future__ import annotations

import asyncio
import importlib
import time
from concurrent.futures import Future
from dataclasses import dataclass
from math import atan2, asin, copysign, pi, sqrt
from queue import Queue
from threading import Event, Thread
from collections.abc import Callable
from typing import Any, Iterable

from .environment import BASE, minimum_jerk_trajectory, point_to_dict
from .runtime import RuntimeExecution, RuntimeUnavailableError

SHORT_SEGMENT_DISTANCE_METERS = 6.0
SHORT_SEGMENT_VELOCITY_CAP_MPS = 1.5
SETTLED_HORIZONTAL_TOLERANCE_METERS = 0.25
SETTLED_VERTICAL_TOLERANCE_METERS = 0.5
SETTLED_VELOCITY_TOLERANCE_MPS = 0.2
HOVER_COMMAND_MIN_VELOCITY_MPS = 0.3


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
    host: str = "127.0.0.1"
    port: int = 41451
    vehicle_name: str = "Drone1"
    velocity_mps: float = 4.0
    meters_per_cell: float = 3.0
    meters_per_altitude_level: float = 5.0
    command_timeout: float = 8.0


class AirSimRuntime:
    backend_name = "airsim"

    def __init__(self, config: AirSimRuntimeConfig | None = None) -> None:
        self.config = config or AirSimRuntimeConfig()
        self.api = None
        self.client = None
        self.connected = False
        self.armed = False
        self.wind_ned = (0.0, 0.0, 0.0)
        self.last_error: str | None = None
        self.last_telemetry: dict[str, object] | None = None
        self.client_package: str | None = None
        self._worker = AirSimRpcWorker()
        self._closed = False
        self._executing = Event()
        self._stop_requested = Event()

    def connect(self) -> dict[str, object]:
        return self._worker.call(self._connect)

    def _connect(self) -> dict[str, object]:
        self.api, self.client_package = import_airsim_client()
        try:
            self.client = self.api.MultirotorClient(
                ip=self.config.host,
                port=self.config.port,
            )
            if not self.client.ping():
                raise AirSimRuntimeError("AirSim RPC server did not answer ping().")
            self.client.enableApiControl(True, vehicle_name=self.config.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.config.vehicle_name)
            self.connected = True
            self.armed = True
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
            if self.connected and hasattr(self.client, "enableApiControl"):
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
        self._ensure_airborne(target_grid)
        self.last_telemetry = self._read_telemetry()
        return self.status()

    def _reset(self) -> None:
        self._require_connection()
        self.client.reset()
        time.sleep(0.3)
        self.client.enableApiControl(True, vehicle_name=self.config.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.config.vehicle_name)
        self.armed = True

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

        target = grid_to_ned(
            target_grid,
            self.config.meters_per_cell,
            self.config.meters_per_altitude_level,
        )
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

    def fly_trajectory(
        self,
        grid_trajectory: Iterable[dict[str, float] | tuple[float, float, float]],
    ) -> dict[str, object]:
        grid_points = [as_grid_tuple(point) for point in grid_trajectory]
        return self._worker.call(lambda: self._fly_trajectory(grid_points))

    def execute(
        self,
        grid_trajectory: Iterable[
            dict[str, float] | tuple[float, float, float]
        ],
    ) -> RuntimeExecution:
        self._stop_requested.clear()
        self._executing.set()
        try:
            result = self.fly_trajectory(grid_trajectory)
            return RuntimeExecution(
                trajectory=tuple(dict(point) for point in result["trajectory"]),
                trajectory_duration=float(result["trajectory_duration"]),
                kinematics=dict(result["kinematics"]),
            )
        finally:
            self._executing.clear()

    def _fly_trajectory(
        self,
        grid_points: list[tuple[float, float, float]],
    ) -> dict[str, object]:
        self._require_connection()
        if len(grid_points) < 2:
            telemetry = self._read_telemetry()
            return {
                "trajectory": [telemetry["position_grid"]],
                "trajectory_duration": 0.0,
                "kinematics": telemetry,
            }

        initial_telemetry = self._read_telemetry()
        actual_start = as_grid_tuple(initial_telemetry["position_grid"])
        execution_grid_points = rebase_grid_path(
            grid_points,
            actual_start,
            samples=len(grid_points),
        )
        ned_points = [
            grid_to_ned(
                point,
                self.config.meters_per_cell,
                self.config.meters_per_altitude_level,
            )
            for point in execution_grid_points
        ]
        path = [self.api.Vector3r(*point) for point in ned_points[1:]]
        distance_m = path_length(ned_points)
        segment_velocity = choose_segment_velocity(
            distance_m,
            self.config.velocity_mps,
        )
        timeout = calculate_path_timeout(
            distance_m,
            segment_velocity,
            self.config.command_timeout,
        )

        self.client.moveOnPathAsync(
            path,
            segment_velocity,
            timeout_sec=timeout,
            drivetrain=self.api.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=self.api.YawMode(False, 0.0),
            lookahead=-1,
            adaptive_lookahead=0,
            vehicle_name=self.config.vehicle_name,
        )

        samples: list[dict[str, float]] = [initial_telemetry["position_grid"]]
        started = time.monotonic()
        deadline = started + timeout
        target_ned = ned_points[-1]
        settled = False
        while time.monotonic() < deadline:
            self._cancel_if_stopped()
            telemetry = self._read_telemetry()
            samples.append(telemetry["position_grid"])
            if telemetry_has_settled(
                telemetry,
                target_ned,
            ):
                settled = True
                break
            time.sleep(0.05)

        if not settled:
            self.client.cancelLastTask(vehicle_name=self.config.vehicle_name)
            self.client.hoverAsync(vehicle_name=self.config.vehicle_name).join()
            final_telemetry = self._read_telemetry()
            self.last_telemetry = final_telemetry
            raise AirSimRuntimeError(
                "AirSim path did not settle before the command timeout; "
                "the logical mission state was not advanced."
            )

        final_telemetry = self._read_telemetry()
        samples.append(final_telemetry["position_grid"])
        self.last_telemetry = final_telemetry
        return {
            "trajectory": deduplicate_points(samples),
            "trajectory_duration": max(0.05, time.monotonic() - started),
            "kinematics": final_telemetry,
        }

    def read_telemetry(self) -> dict[str, object]:
        return self._worker.call(self._read_telemetry)

    def _read_telemetry(self) -> dict[str, object]:
        self._require_connection()
        state = self.client.getMultirotorState(vehicle_name=self.config.vehicle_name)
        kinematics = state.kinematics_estimated
        position = vector_tuple(kinematics.position)
        orientation = quaternion_to_euler(kinematics.orientation)
        telemetry = {
            "position_ned": vector_to_dict(position),
            "position_grid": point_to_dict(
                ned_to_grid(
                    position,
                    self.config.meters_per_cell,
                    self.config.meters_per_altitude_level,
                )
            ),
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

    def hover(self) -> None:
        self._worker.call(self._hover)

    def stop(self) -> None:
        self._stop_requested.set()
        if not self._executing.is_set():
            self.hover()

    def close(self) -> None:
        self.disconnect()

    def _cancel_if_stopped(self) -> None:
        if not self._stop_requested.is_set():
            return
        self.client.cancelLastTask(vehicle_name=self.config.vehicle_name)
        self.client.hoverAsync(vehicle_name=self.config.vehicle_name).join()
        self.last_telemetry = self._read_telemetry()
        raise AirSimRuntimeError(
            "AirSim trajectory was stopped; the logical mission state was not advanced."
        )

    def _hover(self) -> None:
        self._require_connection()
        telemetry = self._read_telemetry()
        if euclidean_3d(
            telemetry["linear_velocity"],
            (0.0, 0.0, 0.0),
        ) <= HOVER_COMMAND_MIN_VELOCITY_MPS:
            return
        self.client.hoverAsync(vehicle_name=self.config.vehicle_name).join()

    def status(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
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


def smooth_grid_path(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    samples: int = 31,
) -> list[tuple[float, float, float]]:
    return minimum_jerk_trajectory(start, end, samples)


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


def euclidean_3d(
    a: dict[str, float] | tuple[float, float, float],
    b: dict[str, float] | tuple[float, float, float],
) -> float:
    av = (
        (a["x"], a["y"], a["z"])
        if isinstance(a, dict)
        else a
    )
    bv = (
        (b["x"], b["y"], b["z"])
        if isinstance(b, dict)
        else b
    )
    return sqrt(sum((float(av[index]) - float(bv[index])) ** 2 for index in range(3)))


def within_position_tolerance(
    position: dict[str, float] | tuple[float, float, float],
    target: dict[str, float] | tuple[float, float, float],
    tolerance: float = 0.75,
) -> bool:
    return euclidean_3d(position, target) <= tolerance


def path_length(points: list[tuple[float, float, float]]) -> float:
    return sum(euclidean_3d(points[index - 1], points[index]) for index in range(1, len(points)))


def calculate_path_timeout(
    distance_m: float,
    velocity_mps: float,
    maximum_timeout: float,
) -> float:
    expected_duration = max(0.35, distance_m / velocity_mps)
    return min(maximum_timeout, max(2.0, expected_duration + 5.0))


def choose_segment_velocity(
    distance_m: float,
    configured_velocity_mps: float,
) -> float:
    if distance_m <= SHORT_SEGMENT_DISTANCE_METERS:
        return min(configured_velocity_mps, SHORT_SEGMENT_VELOCITY_CAP_MPS)
    return configured_velocity_mps


def rebase_grid_path(
    grid_points: list[tuple[float, float, float]],
    actual_start: tuple[float, float, float],
    samples: int | None = None,
) -> list[tuple[float, float, float]]:
    if not grid_points:
        return [actual_start]
    return minimum_jerk_trajectory(
        actual_start,
        grid_points[-1],
        max(2, samples or len(grid_points)),
    )


def telemetry_has_settled(
    telemetry: dict[str, object],
    target_ned: tuple[float, float, float],
    horizontal_tolerance: float = SETTLED_HORIZONTAL_TOLERANCE_METERS,
    vertical_tolerance: float = SETTLED_VERTICAL_TOLERANCE_METERS,
    velocity_tolerance: float = SETTLED_VELOCITY_TOLERANCE_MPS,
) -> bool:
    position = telemetry["position_ned"]
    velocity = telemetry["linear_velocity"]
    position_tuple = (
        float(position["x"]),
        float(position["y"]),
        float(position["z"]),
    )
    horizontal_error = sqrt(
        (position_tuple[0] - target_ned[0]) ** 2
        + (position_tuple[1] - target_ned[1]) ** 2
    )
    vertical_error = abs(position_tuple[2] - target_ned[2])
    return (
        horizontal_error <= horizontal_tolerance
        and vertical_error <= vertical_tolerance
        and euclidean_3d(
        velocity,
        (0.0, 0.0, 0.0),
        ) <= velocity_tolerance
    )


def deduplicate_points(points: list[dict[str, float]]) -> list[dict[str, float]]:
    output = []
    for point in points:
        if not output or euclidean_3d(output[-1], point) > 0.01:
            output.append(point)
    return output
