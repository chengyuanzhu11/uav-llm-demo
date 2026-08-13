from __future__ import annotations

from typing import Mapping, Sequence

from .environment import BASE, TRAJECTORY_DURATION_SECONDS, kinematics_for_motion
from .runtime import RuntimeExecution


class HtmlRuntime:
    backend_name = "html"

    def __init__(self) -> None:
        self._base = BASE

    def reset(self, base: tuple[int, int, int] = BASE) -> dict[str, object]:
        self._base = base
        return self.status()

    def execute(
        self,
        trajectory: Sequence[Mapping[str, float]],
    ) -> RuntimeExecution:
        points = tuple(
            {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "z": float(point["z"]),
            }
            for point in trajectory
        )
        if not points:
            raise ValueError("Runtime trajectory must contain at least one point.")

        start = self._point_tuple(points[0])
        end = self._point_tuple(points[-1])
        duration = 0.0 if start == end else TRAJECTORY_DURATION_SECONDS
        kinematics = kinematics_for_motion(start, end, duration)
        kinematics["source"] = self.backend_name
        return RuntimeExecution(points, duration, kinematics)

    def stop(self) -> None:
        return None

    def status(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "connected": True,
            "supports_wind": False,
            "source": self.backend_name,
        }

    def close(self) -> None:
        return None

    @staticmethod
    def _point_tuple(point: Mapping[str, float]) -> tuple[float, float, float]:
        return (float(point["x"]), float(point["y"]), float(point["z"]))
