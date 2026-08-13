from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable


class RuntimeCapabilityError(RuntimeError):
    """Raised when the active runtime does not implement a requested feature."""


class RuntimeUnavailableError(RuntimeError):
    """Raised when an external runtime cannot complete an operation."""


@dataclass(frozen=True)
class RuntimeExecution:
    trajectory: tuple[dict[str, float], ...]
    trajectory_duration: float
    kinematics: dict[str, object]


@runtime_checkable
class SimulationRuntime(Protocol):
    backend_name: str

    def reset(self, base: tuple[int, int, int]) -> dict[str, object]: ...

    def execute(
        self,
        trajectory: Sequence[Mapping[str, float]],
    ) -> RuntimeExecution: ...

    def stop(self) -> None: ...

    def status(self) -> dict[str, object]: ...

    def close(self) -> None: ...
