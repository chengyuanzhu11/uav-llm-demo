from typing import Protocol
from .motion import VehicleState


CONTROL_LEASE_SECONDS = 0.3
MAX_FEEDBACK_GAP_SECONDS = 1.0


class RuntimeUnavailableError(RuntimeError):
    pass


class StaleFeedbackError(RuntimeUnavailableError):
    """Motion must brake until a newly sampled position is available."""


class RuntimeStateError(RuntimeError):
    """An operation conflicts with vehicle/world state; the message is public."""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code


class SimulationRuntime(Protocol):
    backend_name: str

    def reset(self, base: tuple[float, float, float]) -> dict: ...
    def read_state(self) -> VehicleState: ...
    def drive(self, velocity: tuple[float, float, float], dt: float, *, yaw: float | None = None) -> VehicleState: ...
    def stop(self) -> None: ...
    def status(self) -> dict: ...
    def close(self) -> None: ...
