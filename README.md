# UAV LLM

用自然语言让单架无人机执行任务，并通过机载图像识别、定位目标。模型负责理解与决策，本地规划、安全检查和 PX4 负责飞行执行。

**当前使用 Qwen3-VL-8B-Instruct：高层规划和视觉决策都调用服务器上的同一个 8B 模型。** 原始 4B 与 4B-SFT 文件仍保留；切换服务不会训练或覆盖模型。

## 先分清三个部分

| 部分 | 在哪里运行 | 用途 |
|---|---|---|
| Qwen 8B | A100 服务器 | 模型推理，接口端口 **8004** |
| SSH 连接 | 自己电脑的 WSL | 把本地 8004 转发到服务器 8004 |
| 任务网页与仿真 | 自己电脑的 WSL／浏览器 | 网页端口 **5000**；HTML 或 Gazebo 执行任务 |

**浏览器打开 5000，不是 8004。** 只启动 Qwen 不会自动打开无人机网页。

以下命令针对当前已部署环境：服务器仓库在 `/home/chengyuan/uav-llm-demo`，本地仓库在 `/home/kyrie/projects/UAV_LLM`。终端中的“服务器”和“本地 WSL”不能混用。

## 日常启动：按这三步做

### 1. 在服务器终端启动 8B

打开 VS Code 已连接 `10.62.192.84` 的远程终端，执行：

```bash
bash /mnt/data14t/chengyuan/uav_llm/run/serve-qwen8b-20261111.sh
```

这是当前服务器已经配置好的快捷入口，默认使用 GPU 1、8004 端口和原始 8B 权重。启动前确认 GPU 1 可用；如模型已在运行，跳过这一步。保持启动终端运行，等待模型加载完成。

在服务器另一个终端检查：

```bash
curl -fsS http://127.0.0.1:8004/v1/models
```

返回内容中应包含 `Qwen/Qwen3-VL-8B-Instruct`。直接访问 `http://127.0.0.1:8004/` 显示 `Not Found` 不代表服务失败。

### 2. 在本地 WSL 终端连接服务器

打开自己电脑的 Ubuntu / WSL 终端，**整段复制执行**：

```bash
cd /home/kyrie/projects/UAV_LLM
QWEN_SSH_HOST=10.62.192.84 \
QWEN_SSH_USER=chengyuan \
QWEN_SSH_KEY=/mnt/c/Users/kyrie/.ssh/id_ed25519_codex_10_62_192_84 \
QWEN_SSH_KNOWN_HOSTS=/mnt/c/Users/kyrie/.ssh/known_hosts \
QWEN_LOCAL_PORT=8004 QWEN_REMOTE_PORT=8004 \
bash scripts/connect_qwen.sh
```

看到 `Opening localhost:8004 ...` 后保持这个终端运行。若已有同一条连接，不必重复启动。

在本地另一个 WSL 终端检查连接：

```bash
curl -fsS http://127.0.0.1:8004/v1/models
```

同样应看到 8B 模型名。服务器能访问、本地不能访问时，先检查这一步连接；不要改成 DeepSeek。

### 3. 在本地另一个 WSL 终端启动网页

```bash
cd /home/kyrie/projects/UAV_LLM
bash scripts/start_demo.sh
```

在 Windows 浏览器打开 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**。

选择“北航杭州国际校园”和 **HTML**，确认顶部模型显示 `Qwen/Qwen3-VL-8B-Instruct`，再输入任务并按回车提交。视觉任务需要保持网页打开，由页面生成机载图像。Urban Crossing 仍支持 Mock／LLM 导航；校园自然语言任务使用语言模型。

## 本地模型配置：已经切到 8B

项目根目录 `.env` 中的配置如下。日常启动不需要重新填写；只有更换模型或端口时才修改。

```dotenv
LLM_API_KEY=
LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
LLM_BASE_URL=http://127.0.0.1:8004/v1
LLM_TASK_RESPONSE_FORMAT=json_schema
LLM_REQUEST_TIMEOUT_SECONDS=90
VISION_API_KEY=
VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
VISION_BASE_URL=http://127.0.0.1:8004/v1
VISION_STRATEGY=direct
```

- `LLM_*`：指令要求提取、审查、计划生成、执行复核。
- `VISION_*`：读取机载图像与工具反馈，选择观察、定位或短程移动。
- `LLM_REQUEST_TIMEOUT_SECONDS=90`：高层 HTTP 请求连接／读取等待配置为 90 秒；完整任务输入比短对话长，不能用短请求的速度判断。连续 3 次请求失败后停止自动重试，检查服务后可重新提交。这个设置不改变飞控心跳或工具规划总预算。
- 本地 SSH 转发接口不需要 API Key；外部服务使用各自凭据，不要把密钥写入 Git。

`.env` 修改后需要重启网页。网页“API 设置”只修改运行中的高层模型配置，不会替你改 `.env`；高层与视觉都换模型时，统一修改上面两组字段再重启。

## 重启和停止本地服务

以下命令在**本地 WSL 项目目录**执行。

只重启网页，重新读取 `.env`（会结束当前网页任务）：

```bash
cd /home/kyrie/projects/UAV_LLM
bash scripts/stop_demo.sh --web-only
bash scripts/start_demo.sh
```

停止本项目的网页和 Gazebo＋PX4：

```bash
bash scripts/stop_demo.sh
```

仅停止网页用 `--web-only`，仅停止 Gazebo＋PX4 用 `--gazebo-only`。这些命令不关闭服务器上的 Qwen，也不关闭独立的 SSH 连接；连接终端按 `Ctrl+C` 可断开转发。

## 停止服务器上的 Qwen（释放显存）

**当前飞行任务完成并降落、轨迹采集或评测结束后，就可以关闭 vLLM。** 暂时不用模型时无需保持服务运行；空闲服务仍会占用显存。关闭只结束推理进程，磁盘上的模型、LoRA 和训练结果会保留，下次启动直接加载。

### 启动终端还在：按 Ctrl+C

切到**服务器上启动 8B 的那个终端**，按一次 **Ctrl+C**，等待服务退出。关闭网页、断开 SSH 或退出 VS Code 后，后台启动的服务可能继续运行。

### 找不到启动终端：复制这条命令

在 VS Code 的**服务器远程终端**中执行，任何目录都可以：

```bash
pkill -TERM -u "$(id -u)" -f \
  '^/mnt/data14t/chengyuan/uav_llm/envs/qwen-serve/bin/python -m vllm[.]entrypoints[.]openai[.]api_server .*--served-model-name Qwen/Qwen3-VL-8B-Instruct .*--port 8004( |$)'
```

这条命令只匹配当前账户、本项目推理环境、8B 模型和 8004 端口，向服务发送正常退出请求。PID 每次启动会变化，这里自动查找。没有文字输出是正常现象；若服务已经关闭，命令也不会打印内容。

稍等片刻，在**服务器终端**检查：

```bash
curl -fsS --max-time 3 http://127.0.0.1:8004/v1/models
nvidia-smi
```

关闭成功后，第一条会提示无法连接 8004；第二条中对应的 vLLM 进程退出，GPU 1 显存占用下降。若该卡还有其他任务，仍会显示它们的显存占用。

### 下次使用：重新启动即可

在**服务器终端**运行：

```bash
bash /mnt/data14t/chengyuan/uav_llm/run/serve-qwen8b-20261111.sh
```

等待 `/v1/models` 返回 8B 模型名，再按前面的步骤连接本地 SSH 隧道、打开任务网页。服务器 Qwen、SSH 隧道和本地仿真分别起停；本地的 `stop_demo.sh` 负责网页和 Gazebo＋PX4。

## 需要真实飞行动力学时：打开 Gazebo＋PX4

模型与 SSH 连接仍按前面两步保持运行。**在本地新开一个 WSL 终端**启动校园世界：

```bash
cd /home/kyrie/projects/UAV_LLM
UAV_GZ_HEADLESS=0 bash scripts/start_gazebo.sh semantic_campus
```

等 Gazebo 与 PX4 就绪后，在网页选择“北航杭州国际校园”及 **Gazebo**。

小型视觉花园用 `visual_garden` 替换命令中的 `semantic_campus`，网页也选择“视觉验证花园”。切换网页地图不会替换外部 Gazebo 世界；更换世界前先结束任务、落地并停止旧仿真。

`UAV_GZ_HEADLESS=1` 表示不显示原生窗口，相机仍要渲染。HTML 使用本机运动学仿真；Gazebo 在本机运行物理与 RGB-D 仿真，PX4 执行飞控。**只有模型推理消耗服务器 GPU，仿真仍消耗启动仿真的电脑资源。**

## 常见问题

- **`scripts/...: No such file or directory`**：先执行该步骤中的 `cd`，并确认使用的是指定机器的终端。
- **8004 显示 `Not Found`**：模型没有首页；用 `/v1/models` 检查，任务网页是 5000。
- **网页仍显示 4B**：检查本地 `.env` 两组模型名与端口，重启网页并刷新浏览器；不要继续使用旧的网页内存配置。
- **请求连接失败**：先在服务器检查 8004，再在本地 WSL 检查 8004，最后检查网页 5000。这样可区分模型未启动、SSH 断开和网页未启动。
- **提示端口已占用**：通常已有实例运行；先检查对应接口，不要重复启动或按端口结束其他程序。

## 换一台机器时才需要安装

开发与运行使用 Linux 或 WSL 2，仓库应放在 Linux 文件系统。首次安装应用：

```bash
git clone https://github.com/chengyuanzhu11/uav-llm-demo.git UAV_LLM
cd UAV_LLM
bash scripts/setup_wsl.sh
cp .env.example .env
```

需要 Gazebo 时再执行 `bash scripts/setup_gazebo.sh`。模型服务需另行准备服务环境、完整权重和可用 GPU；上述服务器快捷脚本只存在于当前已部署服务器。通用入口为 [serve_qwen.sh](scripts/serve_qwen.sh)，依赖见 [requirements-qwen.txt](requirements-qwen.txt)；通过 `QWEN_MODEL_PATH`、`QWEN_MODEL_NAME`、`QWEN_PORT` 和 `QWEN_GPU_MEMORY_UTILIZATION` 选择模型。该通用脚本未指定模型时仍使用原始 4B，当前 8B 启动请用第 1 步的快捷入口。

## 数据、训练与评测入口

这些是研究开发步骤，**启动网页不需要执行训练**。当前 4B 训练结果保留，8B 尚未训练。

- **真实模型轨迹**：[collect_visual_trajectories.py](scripts/collect_visual_trajectories.py) 在独立 Gazebo＋PX4 会话中采集图像、动作和反馈。先用 `--check-cases` 检查任务清单；默认清单见 [visual_bootstrap.json](benchmarks/visual_bootstrap.json)。采集前停止已有项目 SITL 会话，使用新的输出目录。
- **合格示范导出**：[collect_demonstrations.py](scripts/collect_demonstrations.py)、[collect_visual_demonstration.py](scripts/collect_visual_demonstration.py) 提供标注专家种子；[export_demonstrations.py](scripts/export_demonstrations.py) 按 `inspect → review → audit → export` 审核并导出逐决策样本。专家示范与模型自主成功轨迹必须分开标注，完整记录不等于合格样本。
- **LoRA SFT**：[train_qwen_sft.py](scripts/train_qwen_sft.py) 的 `smoke` 验证图像处理、loss mask、梯度和保存恢复，再用 `train` 正式训练。使用独立训练环境及 [requirements-train.txt](requirements-train.txt)、[sft_pilot.json](configs/sft_pilot.json)。`status.json`、`progress.json`、`adapter/` 保存状态与结果；[merge_qwen_lora.py](scripts/merge_qwen_lora.py) 合并部署权重，不自动替换网页模型。
- **模型对比**：[compare_qwen_sft.py](scripts/compare_qwen_sft.py) 使用同一任务、工具、安全机制和预算比较多组模型，记录完整成功率、错误完成、修正重试和耗时；额外模型通过 `--profile 名称 地址 模型` 加入。先启动所需服务，参数用 `--help` 查看。任务集见 [qwen_sft_regression.json](benchmarks/qwen_sft_regression.json)。
- **功能与原生验证**：`.venv/bin/python -m unittest discover -s tests -v`；离线语义回归用 `.venv/bin/python scripts/run_semantic_benchmark.py --split all`。原生验证入口为 [verify_gazebo.py](scripts/verify_gazebo.py)，批量语义基准支持 `--runtime gazebo --mode realtime`；真实模型用 `--backend deepseek`，此处是历史接口名，实际模型仍由 `LLM_*` 决定。

训练与测试按完整轨迹、任务及场景划分，不能把同一轨迹的不同决策随机拆开。任务是否成功应由原始要求、执行遥测和独立评分决定，不能只看模型声称完成。

## 项目边界与说明

当前是单无人机 LLM／VLM 分层原型，并非直接从图像输出飞控指令的端到端 VLA。模型仍可能误解任务、无法修复计划或错误定位；升级到 8B 不代表全部任务已通过。

模型端保留 JSON 结构、数字范围和标识符约束，将代价较高的字符串长度限制交给本地完整校验；任务契约、几何和安全检查继续执行。视觉决策只使用当前图像和当前目标，先给目标框及框内实体点，再由同一 Qwen 单独读图复核，并检查深度前景及两个选点的位置是否一致，背景或孔洞选点被退回为候选反馈；试算反馈包含接近距离变化，相同视角的无效操作会被退回，多轮无进展会停止。目标框仍是模型判断；前景深度筛选在遮挡和深长物体上可能保守拒绝，需要换视角，不能替代独立的定位正确性验收。历史示范按录制时的工具协议审核。

HTML 适合任务逻辑调试，Gazebo 用于动力学与原生视觉验证。Gazebo 当前使用仿真真值位姿，不代表视觉里程计或实机验证；动态鸟群和风场尚未迁入。热检查、模拟采样和普通任务拍照是模拟观测，`visual` 才使用机载渲染图像。

机制与实验细节：[任务与工具协议](docs/task-program.md) · [校园建模](docs/beihang-campus-model.md) · [语义基准](docs/semantic-benchmark.md) · [评测及复盘](docs/evaluation-and-recovery.md) · [动态障碍物](docs/dynamic-traffic.md)。
