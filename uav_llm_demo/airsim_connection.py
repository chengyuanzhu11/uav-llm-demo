"""Local Windows Blocks discovery from WSL; no simulator control calls."""
import ipaddress
import os
from pathlib import Path
import struct
import subprocess


def is_wsl():
    return os.name == "posix" and "microsoft" in os.uname().release.lower()


def windows_host():
    if not is_wsl():
        return "127.0.0.1"
    mode = subprocess.run(["wslinfo", "--networking-mode"], capture_output=True,
                          text=True, timeout=5, check=True).stdout.strip()
    if mode == "mirrored":
        return "127.0.0.1"
    for line in Path("/proc/net/route").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == "00000000" and int(fields[3], 16) & 2:
            return str(ipaddress.IPv4Address(struct.pack("<I", int(fields[2], 16))))
    raise ValueError("Cannot find the Windows WSL gateway; check WSL networking.")


def validate_airsim_host(host):
    host = str(host).strip()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return host
    if is_wsl() and host == windows_host():
        return host
    raise ValueError("AirSim must use localhost or this WSL instance's Windows gateway.")


def default_airsim_host():
    return validate_airsim_host(os.getenv("AIRSIM_HOST") or windows_host())


def windows_path(path):
    path = Path(path).resolve()
    if not is_wsl():
        return str(path)
    result = subprocess.run(["wslpath", "-w", str(path)], check=True,
                            capture_output=True, text=True, timeout=5).stdout.strip()
    if len(result) < 3 or result[1:3] != ":\\":
        raise ValueError("AIRSIM_TEXTURE_DIR must be on a Windows drive mounted in WSL, e.g. /mnt/c/...")
    return result


def texture_directory():
    configured = os.getenv("AIRSIM_TEXTURE_DIR")
    if configured:
        folder = Path(configured).expanduser()
    elif is_wsl():
        shell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        result = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command",
             "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
             "[Environment]::GetFolderPath('LocalApplicationData')"],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=10)
        local_appdata = subprocess.run(["wslpath", "-u", result.stdout.strip()],
                                      check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        folder = Path(local_appdata) / "UAV_LLM" / "AirSim" / "textures"
    else:
        import tempfile
        folder = Path(os.getenv("LOCALAPPDATA") or tempfile.gettempdir()) / "UAV_LLM" / "AirSim" / "textures"
    windows_path(folder)  # Reject Linux-only paths before creating or replacing any scene.
    folder.mkdir(parents=True, exist_ok=True)
    return folder
