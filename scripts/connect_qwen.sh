#!/usr/bin/env bash
set -euo pipefail

host=${QWEN_SSH_HOST:?Set QWEN_SSH_HOST}
user=${QWEN_SSH_USER:?Set QWEN_SSH_USER}
key=${QWEN_SSH_KEY:?Set QWEN_SSH_KEY to the existing Windows private key path in WSL}
known_hosts=${QWEN_SSH_KNOWN_HOSTS:?Set QWEN_SSH_KNOWN_HOSTS to the existing Windows known_hosts path in WSL}
local_port=${QWEN_LOCAL_PORT:-8001}
remote_port=${QWEN_REMOTE_PORT:-8001}
for port in "$local_port" "$remote_port"; do
    if [[ ! $port =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1024 || 10#$port > 65535 )); then
        echo 'Ports must be integers between 1024 and 65535.' >&2
        exit 1
    fi
done
local_port=$((10#$local_port))
remote_port=$((10#$remote_port))
[[ -r $key && -f $key ]] || { echo 'The configured SSH private key is not readable.' >&2; exit 1; }
[[ -r $known_hosts && -f $known_hosts ]] || { echo 'The configured known_hosts file is not readable.' >&2; exit 1; }
[[ $(stat -f -c %T /dev/shm) == tmpfs ]] || { echo '/dev/shm must be a writable tmpfs.' >&2; exit 1; }

umask 077
tmp_key=$(mktemp /dev/shm/uav-qwen-key.XXXXXX)
ssh_pid=
cleanup() {
    trap - EXIT HUP INT TERM
    if [[ -n $ssh_pid ]]; then
        kill -TERM "$ssh_pid" 2>/dev/null || true
        wait "$ssh_pid" 2>/dev/null || true
    fi
    rm -f -- "$tmp_key"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
cat -- "$key" > "$tmp_key"
chmod 600 "$tmp_key"

printf 'Opening localhost:%s -> %s localhost:%s; Ctrl+C closes this tunnel.\n' "$local_port" "$host" "$remote_port"
ssh -F /dev/null -NT -i "$tmp_key" \
    -o IdentitiesOnly=yes -o BatchMode=yes \
    -o "UserKnownHostsFile=$known_hosts" -o StrictHostKeyChecking=yes \
    -o ExitOnForwardFailure=yes -o ConnectTimeout=12 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L "127.0.0.1:$local_port:127.0.0.1:$remote_port" \
    -l "$user" -- "$host" &
ssh_pid=$!
if wait "$ssh_pid"; then status=0; else status=$?; fi
ssh_pid=
exit "$status"
