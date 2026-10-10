from math import atan2
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

    def drive(self, velocity, dt):
        previous = self.vehicle
        position = tuple(previous.position[i] + (previous.velocity[i] + velocity[i]) * 0.5 * dt / SCALE[i] for i in range(3))
        acceleration = tuple((velocity[i] - previous.velocity[i]) / dt for i in range(3))
        yaw = atan2(velocity[0], -velocity[1]) if norm(velocity[:2]) > 0.01 else previous.yaw
        self.vehicle = VehicleState(position, tuple(velocity), acceleration, yaw)
        return self.vehicle

    def stop(self):
        # Normal pauses decelerate through drive(); this is the emergency stop boundary.
        self.vehicle = VehicleState(self.vehicle.position, yaw=self.vehicle.yaw)

    def close(self):
        self.stop()

    def status(self):
        return dict(backend="html", connected=True, supports_wind=False, source="html")
