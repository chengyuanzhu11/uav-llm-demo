from math import atan2, isfinite, pi
from .environment import BASE
from .motion import VehicleState, SCALE, norm


class HtmlRuntime:
    backend_name = "html"

    def __init__(self):
        self.vehicle = VehicleState(tuple(float(v) for v in BASE))

    def reset(self, base=BASE):
        self.vehicle = VehicleState(tuple(float(v) for v in base))
        return self.status()

    def read_state(self):
        return self.vehicle

    def drive(self, velocity, dt, *, yaw=None):
        if yaw is not None and (isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not isfinite(yaw)):
            raise ValueError("Yaw must be a finite angle in radians.")
        previous = self.vehicle
        position = tuple(previous.position[i] + (previous.velocity[i] + velocity[i]) * 0.5 * dt / SCALE[i] for i in range(3))
        acceleration = tuple((velocity[i] - previous.velocity[i]) / dt for i in range(3))
        if yaw is None:
            yaw = atan2(velocity[0], -velocity[1]) if norm(velocity[:2]) > 0.01 else previous.yaw
        else:
            # Independent heading uses north=0 and clockwise-positive radians.
            # Limit explicit turns to 90 degrees/s and take the shortest arc.
            error = (yaw - previous.yaw + pi) % (2 * pi) - pi
            change = max(-pi / 2 * dt, min(pi / 2 * dt, error))
            yaw = (previous.yaw + change + pi) % (2 * pi) - pi
        self.vehicle = VehicleState(position, tuple(velocity), acceleration, yaw)
        return self.vehicle

    def stop(self):
        # Normal pauses decelerate through drive(); this is the emergency stop boundary.
        self.vehicle = VehicleState(self.vehicle.position, yaw=self.vehicle.yaw)

    def close(self):
        self.stop()

    def status(self):
        return dict(backend="html", connected=True, supports_wind=False, source="html")
