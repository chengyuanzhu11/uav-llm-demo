#!/usr/bin/env bash
set -euo pipefail

qwen_root=${QWEN_RUNTIME_ROOT:-"/mnt/data14t/${USER:?USER must be set}/uav_llm"}
qwen_gpu=${QWEN_GPU:?Set QWEN_GPU to an available GPU index before starting the service}
qwen_port=${QWEN_PORT:-8001}
qwen_python="$qwen_root/envs/qwen-serve/bin/python"
qwen_model=${QWEN_MODEL_PATH:-"$qwen_root/models/Qwen3-VL-4B-Instruct"}
qwen_name=${QWEN_MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}
qwen_memory=${QWEN_GPU_MEMORY_UTILIZATION:-0.35}

[[ $qwen_gpu =~ ^[0-9]+$ ]] || { echo 'QWEN_GPU must be one GPU index.' >&2; exit 1; }
if [[ ! $qwen_port =~ ^[0-9]{1,5}$ ]] || (( 10#$qwen_port < 1024 || 10#$qwen_port > 65535 )); then
    echo 'QWEN_PORT must be an integer between 1024 and 65535.' >&2
    exit 1
fi
qwen_port=$((10#$qwen_port))
[[ $qwen_root == /* ]] || { echo 'QWEN_RUNTIME_ROOT must be an absolute Linux path.' >&2; exit 1; }
[[ -x $qwen_python ]] || { echo "Missing serving Python: $qwen_python" >&2; exit 1; }
[[ -f $qwen_model/config.json ]] || { echo "Missing local model: $qwen_model" >&2; exit 1; }
"$qwen_python" - "$qwen_memory" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
    assert math.isfinite(value) and 0 < value < 1
except (ValueError, AssertionError):
    raise SystemExit('QWEN_GPU_MEMORY_UTILIZATION must be between 0 and 1.')
PY

# Keep the port lock across exec so repeated starts cannot allocate another model.
mkdir -p "$qwen_root/run"
exec 9>"$qwen_root/run/qwen-$qwen_port.lock"
if ! flock -n 9; then
    echo "Qwen startup is already running for port $qwen_port; leaving the existing service unchanged." >&2
    exit 1
fi
"$qwen_python" - "$qwen_port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        print(f"Cannot bind 127.0.0.1:{port}; port unavailable. Existing services are not stopped or taken over.",
              file=sys.stderr)
        sys.exit(1)
PY

# Keep downloads, compilation caches and temporary files on the data volume.
export HF_HOME="$qwen_root/cache/huggingface"
export VLLM_CACHE_ROOT="$qwen_root/cache/vllm"
export TORCH_HOME="$qwen_root/cache/torch"
export CUDA_CACHE_PATH="$qwen_root/cache/cuda"
export TRITON_CACHE_DIR="$qwen_root/cache/triton"
export XDG_CACHE_HOME="$qwen_root/cache"
export TMPDIR="$qwen_root/tmp"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TMPDIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$qwen_gpu"

echo "Starting $qwen_name on GPU $qwen_gpu at http://127.0.0.1:$qwen_port/v1; Ctrl+C stops it."
exec "$qwen_python" -m vllm.entrypoints.openai.api_server \
    --model "$qwen_model" \
    --served-model-name "$qwen_name" \
    --host 127.0.0.1 --port "$qwen_port" \
    --dtype bfloat16 --max-model-len 65536 --max-num-seqs 2 \
    --gpu-memory-utilization "$qwen_memory" --limit-mm-per-prompt '{"image":3}' \
    --enable-auto-tool-choice --tool-call-parser hermes
