#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv="$HOME/.venvs/uav-llm-demo"

if [[ $(uname -s) != Linux ]]; then
    echo "Run this script in WSL Ubuntu." >&2
    exit 1
fi

python3 -m venv "$venv"
"$venv/bin/python" -m pip install -r "$project_root/requirements.txt"
"$venv/bin/python" -m pip check
echo "WSL Python is ready: $venv/bin/python"
