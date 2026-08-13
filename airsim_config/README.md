# Cosys-AirSim Configuration

当前安装使用：

```text
Cosys-AirSim 5.5-v3.3 packaged Blocks
cosysairsim Python client 3.3.0
SimpleFlight
RPC 127.0.0.1:41451
```

`settings.simpleflight.json` 已复制到：

```text
C:\Users\kyrie\Documents\AirSim\settings.json
```

它启用 Multirotor、FastPhysicsEngine、RPC、碰撞和 NED 风场。

`settings.px4.json` 仅用于后续 PX4 SITL。PX4 在 Windows 上通常需要
WSL2 或 Linux 环境；当前安装没有启用 PX4。

启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

独立联机验证：

```powershell
& '.\.venv-airsim\Scripts\python.exe' .\scripts\verify_airsim.py
```

风场采用 NED 坐标：

- `north`: 北向速度，m/s。
- `east`: 东向速度，m/s。
- `down`: 向下速度，m/s。

AirSim 仍需要根据真实无人机的质量、惯量、动力系统、气动数据和传感器
噪声进行标定，不能直接等同于现实飞行试验。
