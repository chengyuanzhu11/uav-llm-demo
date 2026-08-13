# UAV LLM 双 Runtime 决策 Demo

这个项目展示 LLM 在 UAV 系统中的具体位置：LLM Agent 读取任务观测，输出结构化高层动作；任务层校验动作并规划平滑轨迹；选中的 Runtime 负责执行轨迹并返回状态。LLM 不直接控制电机、姿态环或飞控频率。

```text
            Mock / DeepSeek LLM
                    |
                Agent Layer
                    |
              MissionService
                    |
          -----------------------
          |                     |
     HTML Runtime          AirSim Runtime
   默认可视化执行器       可选最终验证环境
```

## Runtime

`HTML Simulation` 是默认模式。它不启动 Unreal、不连接 `41451`，使用与任务规划相同的 minimum-jerk 连续轨迹驱动 Three.js 场景，适合开发 Agent、观察决策日志和验证任务逻辑。

`AirSim Validation` 是可选模式。启动 Blocks 后，在同一页面切换即可。切换 Runtime 会停止自动运行并从基地重置任务，不会把 HTML 中途状态同步给 AirSim。连接失败时仍保留当前 HTML Runtime 和任务状态。

AirSim 会执行 NED 路径并返回位置、姿态、线速度、角速度、线加速度和角加速度。短航段速度上限、实际位置重定基准和稳定到达判定仍由 `airsim_runtime.py` 负责。

## 安装

默认 HTML Demo：

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m pip install -r requirements.txt
```

需要 AirSim 验证时安装可选客户端：

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m pip install -r requirements-airsim.txt
```

## 启动与停止

只启动 Flask 和默认 HTML Runtime：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

打开 `http://127.0.0.1:5000`。

需要 AirSim 时，在另一个终端显式启动 Unreal Blocks：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_airsim.ps1
```

Blocks 的 RPC 就绪后，在页面选择 `AirSim Validation`。停止 Flask 和本项目启动的 Blocks：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_demo.ps1
```

## Agent

默认 Agent 是离线 `mock`。DeepSeek 与 Runtime 相互独立：DeepSeek 可以驱动 HTML，也可以驱动 AirSim。

不要把 API key 写入代码或提交到 Git。推荐在启动前设置环境变量：

```powershell
$env:UAV_LLM_BACKEND='deepseek'
$env:DEEPSEEK_API_KEY='<your_deepseek_api_key>'
$env:DEEPSEEK_MODEL='deepseek-v4-flash'
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

也可以通过本地接口把 key 注入当前 Flask 进程内存；响应不会返回 key：

```powershell
Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:5000/api/config' `
  -ContentType 'application/json' `
  -Body '{"backend":"deepseek","api_key":"<your_deepseek_api_key>","model":"deepseek-v4-flash"}'
```

## REST API

```text
GET  /api/state          获取统一任务和 Runtime 状态
POST /api/step           执行一次观察、决策、校验、轨迹执行闭环
POST /api/reset          重置当前 Runtime 和任务
POST /api/runtime        切换 html 或 airsim，并从基地重置
POST /api/runtime/stop   停止当前 Runtime；HTML 为无操作
POST /api/runtime/wind   设置 AirSim NED 风速；HTML 返回 HTTP 409
POST /api/config         切换 mock 或 DeepSeek Agent
```

AirSim 切换请求示例：

```json
{
  "backend": "airsim",
  "config": {
    "host": "127.0.0.1",
    "port": 41451,
    "vehicle_name": "Drone1",
    "wind_ned": {"north": 0, "east": 0, "down": 0}
  }
}
```

## 测试

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m unittest discover -s tests -v
```

独立 AirSim 验证：

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' .\scripts\verify_airsim.py
```

## 项目结构

```text
uav_llm_demo/mission_service.py  Agent、任务和 Runtime 生命周期
uav_llm_demo/runtime.py          两种执行器的统一契约
uav_llm_demo/html_runtime.py     默认平滑轨迹执行器
uav_llm_demo/airsim_runtime.py   AirSim RPC、NED 映射和 6-DOF 遥测
uav_llm_demo/environment.py      三维任务、建筑、动态飞行物和路径规划
uav_llm_demo/agents.py           Mock 与 DeepSeek Agent
uav_llm_demo/static/             Three.js 场景和 Runtime 切换 UI
scripts/                         Flask、AirSim 启停和联机验证脚本
tests/                           Runtime、服务、API、飞控和界面契约测试
```

## 模型边界

AirSim 是高保真仿真，不是“完美现实”。要用于研究结论或真实部署，仍需按实际机体标定质量、惯量、电机、桨叶、传感器噪声和风场，并通过 PX4 SITL/HITL 与真实飞行测试逐级验证。
