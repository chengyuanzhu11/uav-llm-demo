"""Owned, fresh simulator sessions for experiments; never reset a user's flight."""
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid

from .gazebo_runtime import GazeboRuntime, GazeboRuntimeConfig
from .runtime import RuntimeUnavailableError

ROOT = Path(__file__).resolve().parents[1]


class GazeboSession:
    def __init__(self, scenario, *, seed=17, scene_file=None):
        if not scenario.gazebo_world:
            raise ValueError('Scenario has no Gazebo world')
        self.scenario, self.seed = scenario, seed
        self.scene_file = Path(scene_file).resolve() if scene_file else None
        if self.scene_file:
            from .scenarios import load_asset_scenario
            if load_asset_scenario(self.scene_file).to_dict() != scenario.to_dict():
                raise ValueError("Session scene file differs from the planner scenario")
        self.partition = 'uav_batch_' + uuid.uuid4().hex
        self.runtime = GazeboRuntime(GazeboRuntimeConfig(world=scenario.gazebo_world, partition=self.partition))
        self.process = self._log = None
        self.log_dir = None
        self.manifest = {}

    def start(self, timeout):
        directory = ROOT/'logs'/'gazebo-benchmark'
        directory.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(tempfile.mkdtemp(prefix=self.scenario.scenario_id+'-', dir=directory))
        self._log = (self.log_dir/'launcher.log').open('w')
        env = dict(os.environ, GZ_PARTITION=self.partition, UAV_GZ_HEADLESS='1',
                   UAV_GZ_LOG_DIR=str(self.log_dir), UAV_GZ_SEED=str(self.seed),
                   UAV_PX4_WORK_DIR=str(self.log_dir/'px4-rootfs'))
        if self.scene_file:
            env['UAV_GZ_SCENE_FILE'] = str(self.scene_file)
        else:
            env.pop('UAV_GZ_SCENE_FILE', None)
        self.process = subprocess.Popen(['bash', str(ROOT/'scripts/start_gazebo.sh'), self.scenario.scenario_id],
                                        cwd=ROOT, env=env, stdout=self._log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                raise RuntimeUnavailableError(f'Gazebo/PX4 launcher exited; inspect {self.log_dir}')
            log = self.log_dir/'px4.log'
            if log.exists() and 'Startup script returned successfully' in log.read_text(errors='replace'):
                self.runtime.configure_scene(self.scenario)
                # Wait for a continuous stream of fresh poses before handing control
                # to the mission loop. A single startup sample may be buffered.
                stable_since = last_sample = None
                while time.monotonic() < deadline:
                    try:
                        state = self.runtime.read_state()
                        now = time.monotonic()
                        if last_sample is not None and state.sample_time > last_sample:
                            if stable_since is None:
                                stable_since = now
                            if now - stable_since >= 1.0:
                                break
                        elif last_sample is not None and state.sample_time < last_sample:
                            stable_since = None
                        last_sample = state.sample_time
                    except RuntimeUnavailableError:
                        stable_since = last_sample = None
                    time.sleep(.05)
                else:
                    raise RuntimeUnavailableError("Fresh Gazebo/PX4 pose stream did not stabilize before startup deadline")
                self.runtime.reset(self.scenario.base)
                sources = dict(world=ROOT/'.sim/worlds'/f'{self.scenario.gazebo_world}.sdf',
                               vehicle=ROOT/'.sim/worlds/models/x500_rgbd/model.sdf',
                               px4=ROOT/'.sim/PX4-Autopilot/build/px4_sitl_default/bin/px4',
                               launcher=ROOT/'scripts/start_gazebo.sh')
                if self.scenario.campus_layout:
                    sources['scenery'] = ROOT/'.sim/worlds/models'/self.scenario.scenario_id/'scenery.glb'
                    sources['target_surfaces'] = ROOT/'.sim/worlds/models'/self.scenario.scenario_id/'target-surfaces.json'
                if self.scene_file:
                    sources['scene_asset'] = self.scene_file
                self.manifest = {name+'_sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in sources.items()}
                return
            time.sleep(.1)
        raise RuntimeUnavailableError(f'Gazebo/PX4 startup timed out; inspect {self.log_dir}')

    def close(self):
        try:
            self.runtime.close()
        finally:
            if self.process:
                # Only this session's process group, including native children.
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
                except KeyboardInterrupt:
                    # Ctrl+C during cleanup still owns these children. Do not
                    # leave an armed simulator running after its controller exits.
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
                    raise
                self.process = None
            if self._log:
                self._log.close()
                self._log = None
