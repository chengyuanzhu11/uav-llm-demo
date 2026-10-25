#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$project_root"
python=${UAV_DEMO_PYTHON:-"$project_root/.venv/bin/python"}
host=$("$python" -c 'from dotenv import load_dotenv; load_dotenv(); from uav_llm_demo.airsim_connection import default_airsim_host; print(default_airsim_host())')
exec /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -NonInteractive \
    -ExecutionPolicy Bypass -File "$(wslpath -w "$project_root/scripts/start_airsim.ps1")" -HostAddress "$host" "$@"
