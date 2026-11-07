#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
scene=${1:-semantic_campus}
px4="$root/.sim/PX4-Autopilot"
[[ -x "$px4/build/px4_sitl_default/bin/px4" ]] || { echo 'Run bash scripts/setup_gazebo.sh first.' >&2; exit 1; }
export PATH="$root/.venv/bin:$PATH"
# All project launchers share one PX4 instance/UDP port. Acquire before
# building assets too, without disturbing an interactive or benchmark session.
exec 9>"$root/.sim/px4-sitl.lock"
flock -n 9 || { echo 'A project Gazebo/PX4 session is already running.' >&2; exit 1; }
world=$(python scripts/build_gazebo_scene.py --scene "$scene")
export GZ_SIM_RESOURCE_PATH="$root/.sim/worlds/models:$px4/Tools/simulation/gz/models:$px4/Tools/simulation/gz/worlds"
export PX4_GZ_MODELS="$root/.sim/worlds/models"
export PX4_GZ_WORLDS="$root/.sim/worlds"
export GZ_PARTITION=${GZ_PARTITION:-uav_llm}
# WSLg otherwise selects llvmpipe on some hybrid-GPU laptops. This affects
# only this simulator process, not the user's global graphics configuration.
if [[ -e /dev/dxg && -e /usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so ]]; then
  export GALLIUM_DRIVER=${GALLIUM_DRIVER:-d3d12}
fi
export PX4_GZ_WORLD="$scene" PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500_rgbd
# X500 + RGB-D payload: hover thrust measured in native SITL. Keep PX4's
# adaptive estimator enabled; the stock 0.60 initial value delays takeoff and
# loses height on the first lateral acceleration of this instrumented model.
export PX4_PARAM_MPC_THR_HOVER=0.73
export PX4_GZ_MODEL_POSE
PX4_GZ_MODEL_POSE=$(python -c 'from uav_llm_demo.scenarios import load_scenario; import sys; x,y,z=load_scenario(sys.argv[1]).base; print(f"{x*3},{-y*3},{z*5},0,0,0")' "$scene")
log_dir=${UAV_GZ_LOG_DIR:-"$root/logs"}
mkdir -p "$log_dir"
gz_args=(-r "$world")
if [[ ${UAV_GZ_HEADLESS:-0} == 1 ]]; then
  gz_args=(-s -r "$world")
  [[ -n ${DISPLAY:-} ]] || gz_args+=(--headless-rendering)
fi
[[ -z ${UAV_GZ_SEED:-} ]] || gz_args+=(--seed "$UAV_GZ_SEED")
gz sim "${gz_args[@]}" >"$log_dir/gazebo.log" 2>&1 &
gz_pid=$!
px4_pid=''
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$px4_pid" ]] || kill -TERM "$px4_pid" 2>/dev/null || true
  kill -TERM "$gz_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sleep 3
# Batch sessions supply a fresh directory so automatic calibration and saved
# estimator parameters cannot leak between cases. Interactive runs retain theirs.
px4_work_dir=${UAV_PX4_WORK_DIR:-"$px4/build/px4_sitl_default/rootfs"}
mkdir -p "$px4_work_dir"
"$px4/build/px4_sitl_default/bin/px4" -d -w "$px4_work_dir" "$px4/build/px4_sitl_default/etc" >"$log_dir/px4.log" 2>&1 &
px4_pid=$!
echo "Gazebo + PX4 running: $scene. Select the matching map and Gazebo in the web page. Ctrl+C stops this session."
wait -n "$gz_pid" "$px4_pid"
