# AirSim 配置

项目使用 Windows Cosys-AirSim 5.5-v3.3 Blocks、Linux `cosysairsim==3.3.0` 客户端和 SimpleFlight 仿真。Blocks 不随源码迁移。

`settings.simpleflight.json` 是原始配置示例。启动器读取 Windows `Documents/AirSim/settings.json`，生成独立的 Windows 缓存配置，将监听地址设为当前 WSL 虚拟网卡；镜像网络使用 localhost。原始配置保留。

在 WSL 项目目录运行：

```bash
bash scripts/setup_wsl.sh --airsim
bash scripts/start_airsim.sh
.venv/bin/python scripts/verify_airsim.py
```

验证脚本只读 RPC。加载地图、重置以及飞行分别通过页面控件执行；共享贴图位于 Windows 本地缓存，Linux 与 Windows 各使用对应路径。防火墙如需调整，应单独授权并仅开放 WSL 接口的 TCP 41451。

`settings.px4.json` 仅保留为后续 PX4 SITL 参考，当前启动器不会使用它。完整说明见 [README](../README.md#可选wsl-连接-windows-blocks)。
