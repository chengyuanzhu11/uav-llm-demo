#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv="$project_root/.venv"

if [[ $(uname -s) != Linux ]]; then
    echo "Run this script in WSL Ubuntu." >&2
    exit 1
fi
if [[ $project_root == /mnt/* ]]; then
    echo "Keep this project in the WSL Linux filesystem, e.g. ~/projects/UAV_LLM." >&2
    exit 1
fi
requirements="$project_root/requirements.txt"
case ${1:-} in
    --airsim) requirements="$project_root/requirements-airsim.txt" ;;
    "") ;;
    *) echo "Usage: bash scripts/setup_wsl.sh [--airsim]" >&2; exit 1 ;;
esac
if [[ -d $venv && ! -x $venv/bin/python ]]; then
    echo "Existing .venv is not a Linux environment; it will not be overwritten." >&2
    exit 1
fi

python3 -m venv "$venv"
"$venv/bin/python" -m pip install -r "$requirements"
"$venv/bin/python" -m pip check
echo "WSL Python is ready: $venv/bin/python"
