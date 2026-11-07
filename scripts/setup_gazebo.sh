#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
if ! command -v gz >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y curl ca-certificates gnupg
  curl -fsSL https://packages.osrfoundation.org/gazebo.gpg | sudo tee /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg >/dev/null
  . /etc/os-release
  printf 'deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable %s main\n' "$VERSION_CODENAME" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
  sudo apt-get update
fi
sudo apt-get install -y --no-install-recommends gz-harmonic python3-gz-transport13 python3-gz-msgs10 \
  cmake ninja-build build-essential pkg-config libeigen3-dev libopencv-dev python3-dev python3-venv
mkdir -p .sim
if [[ ! -d .sim/PX4-Autopilot/.git ]]; then
  git clone --branch v1.17.0 --depth 1 https://github.com/PX4/PX4-Autopilot.git .sim/PX4-Autopilot
fi
git -C .sim/PX4-Autopilot submodule update --init --recursive --depth 1 --jobs 4 \
  Tools/simulation/gz src/modules/mavlink/mavlink src/drivers/gps/devices \
  src/lib/events/libevents src/lib/heatshrink/heatshrink \
  src/lib/crypto/monocypher src/modules/uxrce_dds_client/Micro-XRCE-DDS-Client
bash scripts/setup_wsl.sh --gazebo
if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  sudo apt-get install -y nodejs npm
fi
npm ci --ignore-scripts --no-audit --no-fund
npx playwright install chromium --only-shell
.venv/bin/python -m pip install -r .sim/PX4-Autopilot/Tools/setup/requirements.txt
export PATH="$root/.venv/bin:$PATH"
export GZ_DISTRO=harmonic
cmake -S .sim/PX4-Autopilot -B .sim/PX4-Autopilot/build/px4_sitl_default \
  -G Ninja -DCONFIG=px4_sitl_default
cmake --build .sim/PX4-Autopilot/build/px4_sitl_default --parallel 2
.venv/bin/python scripts/build_gazebo_scene.py
