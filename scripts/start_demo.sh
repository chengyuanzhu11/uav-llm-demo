#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${UAV_DEMO_PYTHON:-"$HOME/.venvs/uav-llm-demo/bin/python"}
port=${UAV_DEMO_PORT:-5000}

if [[ ! $port =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1024 || 10#$port > 65535 )); then
    echo "UAV_DEMO_PORT must be an integer between 1024 and 65535." >&2
    exit 1
fi
port=$((10#$port))

if [[ ! -x $python ]] || ! "$python" -c 'import os, flask, requests; assert os.name == "posix"' 2>/dev/null; then
    echo "A Linux Python with Flask and requests is required. Run: bash scripts/setup_wsl.sh" >&2
    exit 1
fi

# Windows localhost and WSL localhost are separate in NAT mode. Detect an
# existing Windows listener before creating a second, misleading localhost URL.
if [[ -n ${WSL_INTEROP:-} ]]; then
    powershell=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
    if [[ ! -x $powershell ]]; then
        echo "Windows interop is unavailable; cannot check the Windows port." >&2
        exit 1
    fi
    if timeout 10s "$powershell" -NoProfile -NonInteractive -Command \
        "\$ErrorActionPreference='Stop'; try { \$p=Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object { \$_.LocalPort -eq $port } | Select-Object -First 1 } catch { Write-Output 'Cannot inspect Windows listeners.'; exit 2 }; if (\$p) { Write-Output \"Windows port $port is in use by PID \$(\$p.OwningProcess). Stop that service explicitly or choose UAV_DEMO_PORT.\"; exit 1 }; exit 0"; then
        :
    else
        echo "Windows port check failed; no process was stopped." >&2
        exit 1
    fi
fi

cd -- "$project_root"
echo "Open http://127.0.0.1:$port; press Ctrl+C here to stop."
exec "$python" -m flask --app app run --host 127.0.0.1 --port "$port" --no-debugger --no-reload
