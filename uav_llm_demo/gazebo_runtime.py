"""PX4 SITL execution. Gazebo owns physics; this adapter never integrates position."""
from dataclasses import dataclass
import math
from pathlib import Path
import re
import threading
import time

from .motion import VehicleState, SCALE
from .runtime import (CONTROL_LEASE_SECONDS, MAX_FEEDBACK_GAP_SECONDS, RuntimeStateError,
                      RuntimeUnavailableError, StaleFeedbackError)


@dataclass(frozen=True)
class GazeboRuntimeConfig:
    port: int = 14540
    vehicle_name: str = "x500_rgbd_0"
    world: str = "visual_garden"
    partition: str = "uav_llm"

    def __post_init__(self):
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ValueError("Invalid local SITL port")
        for value in (self.vehicle_name, self.world, self.partition):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", value):
                raise ValueError("Invalid Gazebo world/vehicle name")


class GazeboRuntime:
    backend_name = "gazebo"
    supports_control_reference = True
    control_response_seconds = .5
    contact_floor_grid = 0.0

    def __init__(self, config=None):
        self.config = config or GazeboRuntimeConfig()
        self.connected = self.armed = False
        self._scene = None
        self._mav = self.camera = None
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._position = self._attitude = None
        self._last_message = self._last_position = 0.
        self._landed = None
        self._command = (0., 0., 0.)
        self._acceleration = (0., 0., 0.)
        self._yaw = 0.
        self._lease = 0.
        self._reference_time = None
        self._sent_control = None
        self._px4_setpoint_time = None
        self._takeoff_requested = False
        self._offboard_started = False
        self._landing = False
        self._land_requested = False
        self._last_arm = 0.
        self._last_mode = 0.
        self._error = None
        self._ready = False
        self._clock_origin = 0.
        self._ground_down = 0.
        self._ground_pose_z = 0.
        self._control_lock = None

    def configure_scene(self, scenario):
        if scenario.gazebo_world != self.config.world:
            raise RuntimeStateError("runtime_world_mismatch",
                "当前 Gazebo 世界与所选地图不一致。请先切换到 HTML 再更换地图，或启动对应的 Gazebo 世界。")
        self._scene = scenario
        if self._mav is None:
            self.connect()

    def connect(self):
        try:
            import fcntl
            directory=Path(__file__).resolve().parents[1]/'.sim'
            directory.mkdir(exist_ok=True)
            self._control_lock=(directory/f'px4-control-{self.config.port}.lock').open('a')
            try:
                fcntl.flock(self._control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeUnavailableError('This PX4 port already has an active task controller.') from None
            from pymavlink import mavutil
            from .gazebo_camera import GazeboCamera
            self.camera = GazeboCamera(self.config.world, self.config.vehicle_name, self.config.partition)
            self._mav = mavutil.mavlink_connection(f"udpin:127.0.0.1:{self.config.port}",
                                                   source_system=245, source_component=190)
            self._thread = threading.Thread(target=self._run, name="px4-setpoints", daemon=True)
            self._thread.start()
            deadline = time.monotonic() + 20
            while True:
                if self._position is not None and self.connected and self._landed is not None:
                    try:
                        position,_,_,_=self.camera.vehicle_state()
                        break
                    except RuntimeUnavailableError:
                        pass  # Subscriptions may initially deliver buffered samples.
                if time.monotonic() >= deadline:
                    raise RuntimeUnavailableError("No synchronized Gazebo/PX4 telemetry; start the SITL launcher.")
                time.sleep(.05)
            if not self._landed or self.armed:
                raise RuntimeStateError("runtime_requires_landed",
                    "无人机尚未降落并解除武装，无法重置或切换连接。请继续任务或下达返航降落指令，落地后重试。")
            if math.hypot(position[0]-self._scene.base[0]*3, position[1]+self._scene.base[1]*3) > .75:
                raise RuntimeStateError("runtime_requires_home",
                    "无人机尚未回到基地，无法重置或连接。请先返航降落。")
            self._ground_down = self._position.z
            self._ground_pose_z = float(position[2])-self._scene.base[2]*5
            self._clock_origin = self.camera.time
            self._ready = True
        except Exception:
            self.close()
            raise

    def _command_long(self, command, *args):
        params=list(args)+[0.]*(7-len(args))
        self._mav.mav.command_long_send(self._mav.target_system, 1, command, 0, *params)

    def _run(self):
        from pymavlink import mavutil
        m = mavutil.mavlink
        last_heartbeat = 0.
        try:
            while not self._stop_event.wait(.02):
                now=time.monotonic()
                for _ in range(100):
                    msg=self._mav.recv_match(blocking=False)
                    if msg is None: break
                    kind=msg.get_type()
                    if kind=='HEARTBEAT' and msg.autopilot==m.MAV_AUTOPILOT_PX4:
                        self._mav.target_system=msg.get_srcSystem()
                        self.connected=True;self._last_message=now
                        self.armed=bool(msg.base_mode & m.MAV_MODE_FLAG_SAFETY_ARMED)
                        self._offboard=(msg.custom_mode >> 16 & 255)==6
                        if self.armed and self._offboard_started and not self._offboard and not self._landing:
                            self._error="PX4 left Offboard mode; autonomous execution stopped"
                        if not self.armed:
                            self._offboard_started=False
                        elif self._offboard:
                            self._offboard_started=True
                    elif kind=='LOCAL_POSITION_NED':
                        self._position=msg;self._last_position=now
                    elif kind=='POSITION_TARGET_LOCAL_NED':
                        self._px4_setpoint_time=msg.time_boot_ms*.001
                    elif kind=='ATTITUDE': self._attitude=msg
                    elif kind=='EXTENDED_SYS_STATE' and msg.landed_state != m.MAV_LANDED_STATE_UNDEFINED:
                        self._landed=msg.landed_state==m.MAV_LANDED_STATE_ON_GROUND
                    elif kind=='COMMAND_ACK' and msg.result in (m.MAV_RESULT_DENIED,m.MAV_RESULT_UNSUPPORTED,m.MAV_RESULT_FAILED):
                        self._error=f"PX4 rejected command {msg.command} (result {msg.result})"
                if not self.connected: continue
                if now-last_heartbeat>1:
                    # The local task application is the SITL ground station;
                    # retain PX4's GCS-loss checks instead of disabling them.
                    self._mav.mav.heartbeat_send(m.MAV_TYPE_GCS,m.MAV_AUTOPILOT_INVALID,0,0,0)
                    last_heartbeat=now
                if not self._ready:
                    continue
                with self._lock:
                    sampled=self._reference_time
                    valid=now-self._lease<CONTROL_LEASE_SECONDS and self.camera.reference_fresh(sampled)
                    velocity=self._command if valid else (0.,0.,0.)
                    acceleration=self._acceleration if valid else (0.,0.,0.)
                    yaw=self._yaw
                    sent_time=self.camera.time
                # Position/yaw rate ignored; velocity, acceleration feedforward and yaw active.
                self._mav.mav.set_position_target_local_ned_send(int(sent_time*1000)&0xffffffff,
                    self._mav.target_system,1,m.MAV_FRAME_LOCAL_NED,2055,
                    0,0,0,-velocity[1],velocity[0],-velocity[2],
                    -acceleration[1],acceleration[0],-acceleration[2],yaw,0)
                self._sent_control=dict(pose_sim_s=sampled, send_sim_s=sent_time,
                    pose_age_at_send_s=None if sampled is None else max(0.,sent_time-sampled),
                    velocity_mps=velocity, reference_active=valid)
                if self._takeoff_requested and valid:
                    if not self.armed and now-self._last_arm>2:
                        self._command_long(m.MAV_CMD_COMPONENT_ARM_DISARM,1)
                        self._last_arm=now
                    if self.armed and not getattr(self,'_offboard',False) and now-self._last_mode>1:
                        self._mav.mav.set_mode_send(self._mav.target_system,m.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,6<<16)
                        self._last_mode=now
                    if self.armed and getattr(self,'_offboard',False):
                        self._takeoff_requested=False
                if self.armed and not self._landing and self._land_requested and valid:
                    self._landing=True
                    self._takeoff_requested=False
                if self._landing and self.armed and getattr(self, '_offboard', False):
                    if now-self._last_mode>1:
                        self._command_long(m.MAV_CMD_NAV_LAND)
                        self._last_mode=now
        except Exception as exc:
            self._error=f"PX4 transport stopped ({type(exc).__name__})"
            self.connected=False

    def reset(self, base):
        if self._scene is None or tuple(base)!=tuple(self._scene.base):
            raise ValueError("Runtime must be configured with the matching world and launch point")
        state=self.read_state()
        if self._landed is not True or self.armed:
            raise RuntimeStateError("runtime_requires_landed",
                "无人机尚未降落并解除武装，无法重置。请继续任务或下达返航降落指令，落地后重试。")
        if math.dist([state.position[i]*SCALE[i] for i in range(3)],
                     [base[i]*SCALE[i] for i in range(3)])>.75:
            raise RuntimeStateError("runtime_requires_home",
                "无人机尚未回到基地，无法重置。请先返航降落。")
        self.stop()
        self._clock_origin=self.camera.time
        self._takeoff_requested=False
        self._landing=False
        self._land_requested=False
        self._error=None
        return self.status()

    def simulation_time(self):
        if self.camera is None or self.camera.time is None:
            raise RuntimeUnavailableError("Gazebo clock unavailable")
        return max(0., self.camera.time-self._clock_origin)

    def read_state(self):
        p=self._position
        if p is None or time.monotonic()-self._last_position>2:
            raise RuntimeUnavailableError("PX4 position telemetry is missing or stale")
        # Mission geometry and camera pixels share the simulator world frame.
        # Use measured physical pose for simulation execution/acceptance, not
        # PX4's drifting estimator origin. PX4 still runs its own estimator and
        # attitude/thrust loops. This is explicitly simulator truth, not VIO.
        position,velocity,_,sampled=self.camera.vehicle_state()
        yaw=self._attitude.yaw if self._attitude else 0.
        return VehicleState(self._grid_position(position),
                            (float(velocity[0]),-float(velocity[1]),float(velocity[2])), yaw=yaw, sample_time=sampled)

    def _grid_position(self, position):
        return (float(position[0])/SCALE[0], -float(position[1])/SCALE[1],
                max(0.,float(position[2])-self._ground_pose_z)/SCALE[2])

    def measured_path(self, start, end):
        """Recover the flown curve during a delayed tick, including its endpoint."""
        samples = [(t,self._grid_position(p)) for t,p in
                   self.camera.vehicle_path(start.sample_time, end.sample_time)]
        samples.append((end.sample_time, end.position))
        previous = start.sample_time
        for t, _ in samples:
            if t-previous > MAX_FEEDBACK_GAP_SECONDS:
                raise RuntimeUnavailableError("连续超过 1 秒没有可核查的飞行轨迹，已停止任务。")
            previous = t
        return [position for _, position in samples]

    def drive(self, velocity, dt, *, yaw=None, acceleration=(0., 0., 0.), target=None, sample_time=None):
        if len(velocity)!=3 or not all(math.isfinite(v) and abs(v)<=20 for v in velocity):
            raise ValueError("Velocity must be a finite vector within 20 m/s")
        if yaw is not None and not math.isfinite(yaw): raise ValueError("Invalid yaw")
        if len(acceleration)!=3 or not all(math.isfinite(v) and abs(v)<=20 for v in acceleration):
            raise ValueError("Acceleration must be a finite vector within 20 m/s²")
        if target is not None and (len(target)!=3 or not all(math.isfinite(v) for v in target)):
            raise ValueError("Control target must be a finite position")
        if self._error: raise RuntimeUnavailableError(self._error)
        state=self.read_state()
        sampled=state.sample_time if sample_time is None else sample_time
        if not self.camera.reference_fresh(sampled):
            raise StaleFeedbackError("Gazebo pose used to compute this command is stale")
        with self._lock:
            self._reference_time=sampled
            self._command=tuple(velocity);self._lease=time.monotonic()
            self._acceleration=tuple(acceleration)
            if yaw is not None:self._yaw=yaw
            landing_target=target is not None and target[2]*SCALE[2]<=.18
            self._land_requested=landing_target and state.position[2]*SCALE[2]<.4
            if landing_target:
                self._takeoff_requested=False
            # An upward trim near the floor is not a new takeoff. Re-arm only
            # for an explicit airborne target, including subsequent missions.
            if (target is not None and target[2]*SCALE[2]>.18 and velocity[2]>.01
                    and self._landed is True and not self.armed):
                self._landing=False
                self._takeoff_requested=True
        return self.read_state()

    def capture_camera(self, request):
        return self.camera.capture(request, self._clock_origin)

    def read_telemetry(self):
        a=self._attitude
        return dict(orientation=dict(roll=a.roll if a else 0.,pitch=a.pitch if a else 0.,yaw=a.yaw if a else 0.),
                    angular_velocity=dict(x=a.rollspeed if a else 0.,y=a.pitchspeed if a else 0.,z=a.yawspeed if a else 0.),
                    estimated_height_m=-(self._position.z-self._ground_down),
                    estimated_vertical_speed_mps=-self._position.vz,
                    landed=self._landed, armed=self.armed,
                    source="gazebo_physics_pose",simulation_time=self.simulation_time())

    def stop(self):
        with self._lock:
            self._command=(0.,0.,0.);self._lease=0.
            self._acceleration=(0.,0.,0.)
            self._takeoff_requested=False
            self._reference_time=None

    def close(self):
        self._ready=False
        self.stop();self._stop_event.set()
        if self._thread and self._thread is not threading.current_thread():self._thread.join(timeout=3)
        if self._mav:self._mav.close()
        if self.camera:self.camera.close()
        if self._control_lock:
            self._control_lock.close()
            self._control_lock=None
        self.connected=False

    def status(self):
        return dict(backend=self.backend_name,source="px4_sitl",connected=self.connected and time.monotonic()-self._last_message<3,
                    scene_id=self.config.world,vehicle_name=self.config.vehicle_name,port=self.config.port,
                    partition=self.config.partition, model_version="gazebo_px4_x500_rgbd",
                    config=dict(world=self.config.world,vehicle_name=self.config.vehicle_name,port=self.config.port),
                    position_source="gazebo_physics_pose",estimator_source="px4_ekf",
                    control_timing={**(self._sent_control or {}),"px4_controller_sim_s":self._px4_setpoint_time},
                    supports_wind=False,armed=self.armed,landed=self._landed,last_error=self._error,
                    onboard_source="gazebo_rgbd")
