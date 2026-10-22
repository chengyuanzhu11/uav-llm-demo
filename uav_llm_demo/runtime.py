from typing import Protocol
from .motion import VehicleState


class RuntimeCapabilityError(RuntimeError):
    pass


class RuntimeUnavailableError(RuntimeError):
    pass


class SimulationRuntime(Protocol):
    backend_name: str

    def reset(self, base: tuple[float, float, float]) -> dict: ...
    def read_state(self) -> VehicleState: ...
    def drive(self, velocity: tuple[float, float, float], dt: float, *, yaw: float | None = None) -> VehicleState: ...
    def stop(self) -> None: ...
    def status(self) -> dict: ...
    def close(self) -> None: ...
