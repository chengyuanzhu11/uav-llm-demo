"""Stop this checkout's local services, without PID files or port-based kills."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import sys
import time


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    started: str
    args: tuple[str, ...]
    cwd: Path
    exe: Path


def process_stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] not in ("Z", "X"):
            return int(fields[1]), fields[19]  # ppid, starttime (clock ticks)
    except (OSError, IndexError, ValueError):
        pass
    return None


def snapshot():
    processes = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            pid = int(directory.name)
            state = process_stat(pid)
            args = tuple(os.fsdecode(part) for part in
                         (directory / "cmdline").read_bytes().split(b"\0") if part)
            if state and args:
                processes[pid] = Process(pid, *state, args,
                                         (directory / "cwd").resolve(strict=True),
                                         (directory / "exe").resolve(strict=True))
        except OSError:
            continue  # Processes may exit while /proc is being read.
    return processes


def service_kind(process, root):
    args, cwd = process.args, process.cwd
    executable = process.exe.name
    is_python = re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable)
    script = (cwd / args[1]).resolve() if len(args) > 1 else None
    if executable == "bash":
        if script == root / "scripts/start_demo.sh":
            return "web"
        if script == root / "scripts/start_gazebo.sh":
            return "gazebo"
    if is_python:
        if script == root / "app.py":
            return "web"
        flask_args = ()
        if args[1:3] == ("-m", "flask"):
            flask_args = args[3:]
        elif script == root / ".venv/bin/flask":
            flask_args = args[2:]
        if cwd == root and "run" in flask_args:
            for index, arg in enumerate(flask_args):
                app = arg.removeprefix("--app=") if arg.startswith("--app=") else None
                if arg == "--app" and index + 1 < len(flask_args):
                    app = flask_args[index + 1]
                if app in ("app", "app:app", "app.py"):
                    return "web"
    if process.exe == root / ".sim/PX4-Autopilot/build/px4_sitl_default/bin/px4":
        return "gazebo"
    # Gazebo rewrites child argv to "gz sim server" / "gz sim gui". The
    # launcher-exported world directory identifies even orphaned native children.
    is_gz = args[0].split()[:2] == ["gz", "sim"] or any(
        Path(arg).name == "gz" and args[index + 1:index + 2] == ("sim",)
        for index, arg in enumerate(args[:2]))
    if is_gz:
        worlds = root / ".sim/worlds"
        if any(arg.endswith(".sdf") and (cwd / arg).resolve().parent == worlds for arg in args[1:]):
            return "gazebo"
        try:
            environment = Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0")
            if os.fsencode(f"PX4_GZ_WORLDS={worlds}") in environment:
                return "gazebo"
        except OSError:
            pass
    return None


def discover(root, scope, known):
    processes = snapshot()
    selected = {pid: process for pid, process in processes.items()
                if service_kind(process, root) in scope or
                (pid in known and known[pid].started == process.started)}
    # Include launch-time builders and children before a parent can exit/reparent.
    while True:
        children = {pid: process for pid, process in processes.items()
                    if process.parent in selected and pid not in selected}
        if not children:
            return selected
        selected.update(children)


def send_signal(process, sig):
    """A pidfd pins the identity; a reused PID must never receive our signal."""
    try:
        fd = os.pidfd_open(process.pid)
    except ProcessLookupError:
        return
    try:
        state = process_stat(process.pid)
        if state and state[1] == process.started:
            signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


def stop(root, scope, *, dry_run=False, timeout=10.0):
    selected = discover(root, scope, {})
    if not selected:
        print("没有正在运行的本项目服务。")
        return 0
    # Never print raw command lines or environments: they may contain secrets.
    for process in selected.values():
        kind = service_kind(process, root) or "child"
        print(f"{'将停止' if dry_run else '停止'} {kind}: PID {process.pid}", flush=True)
    if dry_run:
        return 0
    started = time.monotonic()
    sent = set()
    while selected:
        elapsed = time.monotonic() - started
        force = elapsed >= timeout
        sig = signal.SIGKILL if force else signal.SIGTERM
        if elapsed >= timeout + 3:
            print("部分进程未退出：" + ", ".join(str(pid) for pid in selected), file=sys.stderr)
            return 1
        for process in selected.values():
            identity = (process.pid, process.started, sig)
            # Let launchers clean up normally before signalling children directly.
            if identity not in sent and (process.parent not in selected or elapsed >= min(2, timeout)):
                if force:
                    print(f"正常退出超时，强制结束 PID {process.pid}。", flush=True)
                send_signal(process, sig)
                sent.add(identity)
        time.sleep(.1)
        selected = discover(root, scope, selected)
    print("所选项目服务已停止。")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="stop_demo.sh", description="停止本项目的网页服务和 Gazebo＋PX4（默认全部）。")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--web-only", action="store_true", help="只停止网页服务，包括 HTML 仿真")
    group.add_argument("--gazebo-only", action="store_true", help="只停止 Gazebo＋PX4")
    parser.add_argument("--dry-run", action="store_true", help="仅列出将停止的进程")
    args = parser.parse_args()
    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        parser.error("请在 WSL/Linux 中使用 Python 3.9+ 运行。")
    scope = {"web"} if args.web_only else {"gazebo"} if args.gazebo_only else {"web", "gazebo"}
    try:
        return stop(Path(__file__).resolve().parents[1], scope, dry_run=args.dry_run)
    except OSError as exc:
        print(f"停止失败（{type(exc).__name__}），请使用启动服务的同一个 WSL 用户运行。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
