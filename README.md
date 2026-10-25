# UAV LLM 决策 Demo

项目包含两种单无人机演示：

- **Urban Crossing**：保留 Mock / DeepSeek 两种航点规划方式，飞到随机远端终点，再返航。
- **北航杭州国际校园**：参考校园航拍与导览图重建，输入自然语言，由 DeepSeek 生成任务与空间运动程序；本地规划、连续轨迹跟踪和避障负责执行，不提供 Mock 任务模式。

校园任务可自由描述目标、飞行形状、方向、朝向、条件和循环，也可在飞行中修改任务。未指定的参数采用默认值：按地点选择作业高度，巡检 2 秒、热检查 3 秒、模拟采样 1 秒、通用停留 2 秒；明确指定的受支持参数优先。结束位置及降落或悬停状态由任务决定，不再自动添加返航。地点不明、缺少能力或要求冲突时才提示补充或调整。

![DeepSeek API 城市飞行演示](docs/images/deepseek-urban-demo.png)

DeepSeek API 模式下的 Urban Crossing 实际飞行画面（HTML Simulation）。

```text
Urban Crossing: Mock / DeepSeek → 航点
校园任务: 自然语言 → DeepSeek 任务与运动程序 → 本地空间编译 / 转场规划
                                         ↓
                   安全检查 → 连续控制与避障 → Runtime
                                         ↓
                    轨迹覆盖 / 目标状态 / 观测记录 → 下一任务或 DeepSeek 评估
```

## Runtime

`HTML Simulation` 是默认模式。它不启动 Unreal、不连接 `41451`，由 `TrajectoryController` 以 50 ms 控制周期连续更新位置、速度和加速度，适合开发 Planner、观察决策日志和验证任务逻辑。

无人机从基地地面（z=0）启动，先垂直爬升，再执行三维航路；返航先到基地上空，再垂直下降，低速到达地面才完成任务。巡航高度为 10–50 米，低于 10 米的运动限于半径 0.75 米的已验证垂直通道：Urban Crossing 使用基地通道，校园可从共享地表与障碍几何解析其他起降或低空停留位置。两张地图均允许按建筑约束爬升、下降或绕行；园区作业点使用默认高度，指定高度上限时可降低楼外作业高度，并重新检查路线安全性。控制器限制爬升 2 米/秒、下降 1 米/秒；这是仿真参数，不代表真实机型性能。

Urban Crossing 在启动、Reset 和切换场景时随机选择安全的三维终点（高度 10–30 米），与基地水平距离至少 75 米，并排除起飞所在的近端区域；Reset 不重复上一个终点。暂停、继续、重规划和 Agent 切换保留终点；Runtime 切换也保留终点。候选点仅用本地规划检查静态可达性，执行航路仍由所选 Mock / DeepSeek 生成。园区使用固定语义地点，执行目标随任务步骤变化，Reset 清除任务。

城市 HTML 场景包含 8 只鸟，沿四条道路在 10–30 米的固定三维走廊往返，最高速度 3.6 米/秒。鸟群只依赖地图几何、交通种子和世界时间，与无人机起终点、当前位置及规划路线无关；每次 Reset 更换种子。校园的 3 只鸟沿固定道路走廊运动。

避障器只接收有范围、建筑遮挡和延迟的观测，采用历史驱动的短时预测；不读取鸟群未来曲线。交通协议、预测校准和风险指标见[动态障碍物说明](docs/dynamic-traffic.md)。

Urban Crossing 终点的实际高度处显示金色星星与光环。后端确认抵达后播放上升、散射粒子和 `★ +1` 提示，返航仍继续；每轮只收集一次，刷新不重播，Reset 恢复。

固定走廊可以经过目标点和基地附近；鸟类不因任务位置改变路线。随机目标的可达性检查仅针对静态环境，不承诺动态障碍物持续占据目标时一定能到达。

鸟类显示头部、尖喙、尾羽、振翅与滑翔；鼠标指向鸟类显示编号、高度和安全范围，接近无人机时显示琥珀色或红色提示。`LocalAvoidance` 在每个控制周期推演未来 3 秒的相对运动，沿用速度/加速度限制，检查建筑、边界和附近鸟类，并增加随预测时间增长的经验安全余量（并非概率安全保证）。水平、升降与斜向绕行同时评分，低空飞行作为偏好；安全的已有绕行每 0.5 秒重新评分，明显更优时才切换，遇险立即重选。距最终航点 6 米内检查 6 秒的接近、减速及至少 0.75 秒稳定停留，目标受阻时保留安全等待/绕行选择。页面以加粗金色实线显示原规划、蓝色虚线显示实际执行航段；避障时，灰黄虚线与橙色实线分别显示控制器的原指令预测和避让预测，并高亮相关鸟群。状态条显示实测最小安全净空，恢复航行后保留 3 秒；这些展示数据不送入规划器。

执行期间每 0.25 秒检查安全的前向航点连接，跳过已无必要的普通中间航点，不要求绕行后折返。每条连接同时通过静态全航段、动态短时运动及三维跟踪偏差检查；返航必须先低速到达基地上空的下降接近点，再进入降落段，最终目标必须低速到达。`flight.skipped_waypoints` 记录当前计划跳过的点数，Planner 原始航点仍保留。`metrics.path_length` 按水平每格 3 米、高度每层 5 米换算，`path_length_unit` 为 `m`。

HTML 飞行保护启用后，`metrics.collision_or_separation_events` 根据连续控制采样间的相对运动检查 UAV 与鸟类安全包络是否重叠，同一次持续重叠只计一次，不再搜索决策文本。`min_dynamic_clearance_m` 为最小包络净距，负值表示间距违规，并不等于已确认的网格模型物理碰撞；未测量时为 `null`。`evaluation.dynamic_incidents` 保留最近 100 次事件的鸟类编号、任务时间和净距，累计计数不受截断影响。页面事件列表在 Reset 后清除；通过 `app.py` 启动时，运行记录异步写入 `logs/mission_runs.jsonl`，可按 `run_id` 复盘，详见下方评测说明。

点击 Start 后启用本地保护：暂停任务、等待 API 或 Planner 连续失败时仍可主动躲避来袭鸟类，但不擅自继续已暂停/失败的任务。Reset、任务完成及飞控/连接故障会解除保护。短暂相遇只进行本地轨迹重规划，不消耗 API；局部避障期间剩余路线长度连续 8 秒无明显改善才请求高层重规划，动态事件调用间隔至少 10 秒，网络故障按 2、4、8、16、30 秒退避并持续重试；重复不安全路线保留拒绝航段与次数，要求更换走廊，重试间隔最多 10 秒。均不设尝试次数上限，直到成功或手动暂停、Reset。找不到无碰撞轨迹时显示红色 best-effort 告警并选择风险较低且不穿建筑的运动，不能保证任意突发情况绝对无碰撞。这是有限时域仿真避障，不是真机安全认证。AirSim 尚未接入鸟类遥测，不使用 HTML 鸟类冒充其障碍物。

`AirSim Validation` 是 Urban Crossing 的可选模式；园区的观测技能目前仅支持 HTML 仿真。启动 Blocks 后，在同一页面切换即可。切换 Runtime 会停止自动运行并从基地重置任务，不会把 HTML 中途状态同步给 AirSim。连接失败时仍保留当前 HTML Runtime 和任务状态。

AirSim 模式下，浏览器 Three.js 只保留静态任务地图、目标和基地，不显示 UAV、动态飞行物，也不回放 AirSim 返回的轨迹。实际飞行只发生在 Blocks 中；浏览器继续显示 AirSim 遥测、Planner 决策和执行结果。HTML 与 AirSim 是同一个控制层的两个 Runtime，不是先执行 HTML 再同步到 AirSim。

AirSim 会执行 NED 路径并返回位置、姿态、线速度、角速度、线加速度和角加速度。短航段速度上限、实际位置重定基准和稳定到达判定由 `airsim_runtime.py` 负责。重置后先等待机体静止；基地半径 0.75 米范围内不超过 10 厘米的 Unreal 地面接触误差在控制坐标中归零，原始 NED 遥测仍保留，其他位置及超出容差的负高度仍由安全检查拒绝。

`Urban Crossing` 使用 40×40 格（117×117 米）的任务地图，保留近端街区，并向远端扩展为 32 个地块、4 个十字路口；道路、街树、建筑和安全检查使用相同场景数据。起终点以同色的透明方形光柱标记，可透过建筑辨认位置。HTML 使用高低错落的现代楼群（最高 65 米），包括矩形、圆柱塔楼及局部斜撑立面，配合细密窗格、屋顶设备、道路纹理和暖色侧光；街树低于最低飞行高度。建筑造型及屋顶细节均限制在共享障碍包围体内，圆柱的碰撞检查采用保守包围体；任务边界外的楼群只作视觉背景。AirSim 仍按 `scene_layout.py` 将同一占地单元显示为简化方盒，实际联调方法见下方 AirSim 说明。水平每格 3 米，高度每层 5 米。

切入 AirSim 或切换场景时，程序用 Windows Blocks 的 Cube 资源生成地图并加载纯色贴图。共享地图位于 NED `(500, 500, -100)` 的独立区域，避免与原装 Blocks 建筑重叠；飞行指令和遥测统一加减该偏移。Reset 将无人机放回该地图基地。原装 Blocks 地图保留，仅更新 `UAVDemo_` 前缀的对象。Flask 在 WSL 运行，贴图写入 Windows 的 `%LOCALAPPDATA%\UAV_LLM\AirSim\textures`，经路径转换交给 Blocks 读取；资源准备成功后才替换场景。动态飞行物的真实 Unreal 运动/碰撞仍属于独立功能，不由静态地图统一提供。

HTML 中 UAV 和鸟类共用仿真时钟，页面只读取快照；暂停任务或等待 API 时，控制循环仍更新交通和本地保护。慢帧最多推进 0.1 仿真秒，因此两者一起减慢；超过 0.3 秒记录延迟而不中断 HTML 任务，AirSim 保留超时保护。页面分别显示去程到达、返航完成和重试状态；只有实际位置及速度满足条件才确认任务阶段。

Follow UAV 使用带时间戳的实际控制采样，画面缓冲约 0.35 秒后连续播放，无人机、鸟类和已飞轨迹同步；这个延迟仅影响显示，控制与避障仍使用实时状态。断流时不外推未知运动，重连缺失整段轨迹时重新同步。轨迹按实际折线增量绘制，镜头利用已收到的后续采样提前避让建筑，并检查视线与移动路径，避免楼角突然拉近。

## 安装（WSL）

在 VS Code 中打开 WSL / Ubuntu（Bash）终端；如果当前是 PowerShell，先输入 `wsl`。运行以下命令无需额外插件。

需要已安装 WSL / Ubuntu。首次使用时准备 Git 和 Python，并将仓库放在 WSL Linux 文件系统：

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/chengyuanzhu11/uav-llm-demo.git UAV_LLM
cd UAV_LLM
bash scripts/setup_wsl.sh
```

已有仓库时，直接进入仓库目录运行 `bash scripts/setup_wsl.sh`。Linux 环境位于项目内 `.venv`，无需手动激活。VS Code 使用“WSL: Open Folder in WSL”打开该目录，解释器选择 `.venv/bin/python`。

## 启动（WSL）

在 WSL 终端进入克隆的仓库目录（若克隆在其他位置，请使用实际目录）：

```bash
cd ~/projects/UAV_LLM
bash scripts/start_demo.sh
```

打开 [http://127.0.0.1:5000](http://127.0.0.1:5000)，保持启动终端打开。若提示 5000 已被占用，先停止原服务再启动。

### 页面操作

默认进入 Urban Crossing，使用 `HTML Runtime` 和 `Mock Agent`。可在工具栏切换 `Mock / DeepSeek API`；`Plan` 生成路线，`Start` 开始或暂停，`Reset` 重置并随机更换终点。

Urban Crossing 默认往返：到达目标并减速稳定后自动进入 `Returning`，根据当前位置和环境重新规划回基地的路线；返回基地上空、下降到地面并低速稳定后才标记 `Completed`。保留完整往返轨迹，返程同样支持暂停、继续、三维绕行与动态避障。DeepSeek 正常往返至少规划两次，安全重规划可能增加调用次数。

### 校园自然语言任务

选择 **北航杭州国际校园**，配置 DeepSeek key，在右侧输入指令并提交；推荐任务可点选填入。解释通过校验后自动执行，页面显示条件、循环、轨迹覆盖进度及实际执行结果；Start 可暂停/继续。执行中可直接提交“修改任务”，无需重置或回基地。

校园包含中央大屋顶及 C1–C9、R1–R3 科研组团、L1–L6 实验楼、D1–D14 宿舍、A1–A5、运动场和环校水系。全部建筑组团及主要场地共登记 56 处任务地点；地图显示编号，靠近、指向或执行任务时显示完整名称，可按楼号或名称安排任务。从南门起降点出发，例如：

- “检查图书馆温度，超过 25 摄氏度就模拟采样，否则巡检会议中心，最后返航。”
- “沿田径场顺时针飞行两圈后返航。”
- “去北区足球场中央降落停留，不用返航。”

模型按照片比例缩尺，楼高与细节为近似，显示距离为仿真距离；建模依据、碰撞边界与验证方式见 [校园模型说明](docs/beihang-campus-model.md)。

DeepSeek 可组合共享边界、参数曲线和相对折线路径，设置速度、升降及机头朝向，无需为每种飞行形状新增业务技能。田径场显示与规划使用同一份跑道中心线；任务轨迹按顺序连续跟踪，不能抄近道或仅靠回到终点确认圈数。支持数值条件、分支和循环；空间表达式、全部路径与避让约束均需通过本地校验。详见[校园任务框架](docs/task-program.md)。

修改任务时先制动并保持本地避障，DeepSeek 结合当前位置、原任务和已完成记录解释新的剩余任务；校验通过后切换，保留位置、轨迹、电量、交通时间及历史记录。新要求不可执行时安全等待，旧任务不会自动继续；迟到的旧回复不能覆盖新指令。普通航段重规划、轨迹跟踪及循环执行在本地完成；任务需要依据新观测作进一步判断时，执行评估节点会调用 DeepSeek 更新剩余程序，并保留完整的未完成要求、顺序、次数、条件和安全约束。

完成记录绑定任务、步骤、无人机、地点和仿真时间：观测动作须实际到达并满足驻留要求，运动动作须验证规定轨迹的覆盖进度，整项任务须通过独立任务契约验收：所需动作、条件分支、循环、轨迹及结束状态全部有执行证据，不能仅靠到达终点判成功。自主复核遗漏要求会自动纠错，用户主动修改任务则建立新契约。热读数是执行到现场后生成的仿真数据，模拟拍照保存结构化观测记录，**不生成照片，也不执行真实视觉识别或热成像**。初始提示不包含隐藏观测结果；已完成的历史证据会保留，暂停/偏离观测点会重新计时当前驻留。当前只有 `uav-1`，尚未实现多机分配。

## Agent

Urban Crossing 默认使用离线 `mock`，DeepSeek 可驱动 HTML 或 AirSim；园区固定使用 DeepSeek 解释任务和 HTML 执行。

点击工具栏 **API 设置**，选择 DeepSeek 官方或阿里云百炼，填写对应 Key，读取模型列表后选择模型并保存。接口地址自动填写，业务空间地址等特殊配置在“高级设置”中修改。配置失败后可直接修改重试，无需重启；更换配置后重新提交任务即可。

两家服务的 Key、模型和地址分别保存在当前 Flask 进程内存，同一服务商同一地址可留空复用 Key；切换服务商不会混用密钥。重启后需重新配置（启动环境预置除外）。Key 不写入浏览器存储、日志或 API 响应，也不要写入代码或 Git。模型列表校验不发起推理；目录可见不等于拥有该模型的推理权限，仍需任务实测。百炼地域、权限与学生券说明见[百炼接入](docs/bailian-api.md)。

DeepSeek 请求失败时，页面只显示不含响应正文的 HTTP 状态码或网络故障类别。认证失败、余额不足和参数错误会立即停止，不再重复请求；网络故障（含 TLS）、超时、限流及服务端错误按 2、4、8、16、30 秒退避并持续重试，成功后重置间隔。原始响应正文和异常文本不会写入任务轨迹。官方默认模型为 `deepseek-flash`，页面以服务商实际返回的模型列表为准。

DeepSeek 路线未通过安全检查时，使用最新观测和拒绝原因继续请求，不自动切换 Mock。等待期间保持本地避障；持续重试可能增加 API 用量，可在页面暂停或 Reset 取消。

WSL 下 DeepSeek 请求由 Linux `requests` 发出，网络与代理设置应在 WSL 中生效。

也可以在启动前通过环境变量预置 key；这不是必需步骤，Agent 模式仍在浏览器中切换：

```bash
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
export DEEPSEEK_MODEL='deepseek-flash'
bash scripts/start_demo.sh
```

输入 key 时不会回显，也不会将 key 写入命令历史。使用环境变量预置的 key 会保留在当前终端中，需要清除时执行 `unset DEEPSEEK_API_KEY`。启动脚本也会加载项目根目录的 `.env`；该文件已被 Git 忽略，界面中填写的 Key 仍只保存在进程内存。

## DeepSeek Prompt Cache

请求按固定 System 规则、Scene 地图、Mission 目标、State 当前状态、Feedback 重试反馈排序。地图 JSON 使用稳定的字段和障碍物顺序；完整地图仍随请求发送，由 DeepSeek 自动复用前缀。

通过 `app.py` 启动后，每次 DeepSeek 请求的数值统计写入 `logs/deepseek_usage.log`（JSON Lines，2 MB 轮转，保留 3 个备份）。记录 `prompt_tokens`、`prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`、输出 token 数、耗时和 HTTP 状态；不记录 key、Authorization、原始 Prompt 或模型回复。没有返回 usage 的请求保留空值，不当成零命中。`common_prefix_content_chars` 是本地字符比较，不是缓存命中 token 数。

统计连续请求时使用加权命中率 `sum(hit) / sum(prompt_tokens)`：

```bash
.venv/bin/python scripts/audit_prompt_cache.py --usage-log logs/deepseek_usage.log
```

脚本按模型及 Prompt 版本汇总，并列出最后 12 条请求；只纳入 token 数完整且 `hit + miss == prompt_tokens` 的样本。缓存落盘需要时间，新前缀应观察至少 3 次连续请求，不能把第一次冷启动或离线字符串相似度当成实测命中率。详细排查见 [Prompt Cache 报告](docs/prompt-cache-audit.md)。

## REST API

```text
GET  /api/state          获取统一任务和 Runtime 状态
POST /api/mission/task   园区提交自然语言任务：{"instruction":"..."}
POST /api/mission/plan   Urban Crossing 请求一次 waypoint 规划
POST /api/mission/start  开始或继续连续飞行控制循环
POST /api/mission/goal   Urban Crossing 更新目标并触发重规划
GET  /api/scenarios      列出两张地图、园区地点及推荐任务
POST /api/scenario       切换地图并重置任务
POST /api/reset          重置当前 Runtime 和任务
POST /api/runtime        切换 html 或 airsim，并从基地重置
POST /api/runtime/stop   暂停任务并减速；已启用的本地碰撞保护继续工作
POST /api/runtime/wind   设置 AirSim NED 风速；HTML 返回 HTTP 409
POST /api/config         页面 Agent 切换控件使用的内部接口；日常操作无需手动调用
```

AirSim 切换请求示例：

```json
{
  "backend": "airsim",
  "config": {
    "port": 41451,
    "vehicle_name": "Drone1",
    "wind_ned": {"north": 0, "east": 0, "down": 0}
  }
}
```

## 测试（WSL）

在项目目录使用上述 Linux 环境：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Follow UAV 浏览器回归需 Node.js、Playwright 及其 Chromium；使用离线 Mock 数据，不操作正在运行的服务，也不调用 DeepSeek：

```bash
.venv/bin/python scripts/capture_follow_fixture.py
node scripts/verify_follow_camera.cjs
```

报告和截图保存在 `outputs/follow-camera/`。脚本检查位置与轨迹同步、建筑遮挡及切换后的资源释放；CPU 回放耗时不代表实际显示帧率。

## 评测与复盘

校园语义任务基准使用人工标注的要求，独立检查巡检、顺序、次数、条件、实际环绕轨迹和结束状态；离线运行不会调用 API，也不操作当前服务：

```bash
.venv/bin/python scripts/run_semantic_benchmark.py
```

真实 API 评测需显式选择，运行方式、预算及结果边界见[校园语义任务基准](docs/semantic-benchmark.md)。离线回复只验证执行与验收链路，不能当作 DeepSeek 的理解准确率。

以下基准和航路回放针对 Urban Crossing；园区通过任务 API 运行，任务与证据写入同一运行日志，尚不支持该航路回放器。固定种子的离线回归不调用 API：

```bash
.venv/bin/python scripts/run_benchmark.py --mode virtual --scenario urban_crossing --seed 7 --seed 11 --seed 23 --output results.json
```

真实 DeepSeek 评测使用 `--mode realtime --backend deepseek`，需要终端环境变量中的 key；页面内存中的 key 不会传给独立脚本。实时模式计入 API 等待，`--max-seconds` 限制墙钟时间；虚拟模式限制仿真时间，只允许本地或录制的规划响应。超时明确标为 `timeout`。

每轮记录完成证据、延迟、调用量、场景种子及源码指纹。配对比较、离线响应回放和可选费用估算见 [评测与运行日志](docs/evaluation-and-recovery.md)。

## 可选：WSL 连接 Windows Blocks

Blocks 保留在 Windows；Flask、Python 和测试统一在 WSL。先安装可选依赖：

```bash
bash scripts/setup_wsl.sh --airsim
bash scripts/start_airsim.sh
.venv/bin/python scripts/verify_airsim.py
```

启动器读取 Windows `Documents/AirSim/settings.json`，在 `%LOCALAPPDATA%\UAV_LLM\AirSim\settings-wsl.json` 生成专用配置，保留原文件。NAT 模式仅监听当前 WSL 虚拟网卡，镜像网络使用 localhost；不绑定所有网卡。Blocks 路径不同时，可向启动器传入 `-BlocksExe 'C:\path\to\Blocks.exe'`。启动器仅支持 SimpleFlight 仿真。

如 Windows 防火墙拦截，仅针对实际 Blocks 程序、TCP 41451、`vEthernet (WSL)` 和 `LocalSubnet` 添加入站许可；管理员操作独立于正常启动，脚本不会自动修改防火墙。不要向 Wi-Fi 或其他局域网接口开放该端口。

后端和连接面板使用同一默认地址；可用 `AIRSIM_HOST`、`AIRSIM_PORT`、`AIRSIM_VEHICLE_NAME` 覆盖。Host 仅接受本机或当前 WSL 的 Windows 网关。共享贴图默认由 WSL 写入 `/mnt/c/.../AppData/Local/UAV_LLM/AirSim/textures`，Blocks 收到对应的 `C:\...` 路径；也可设置 `AIRSIM_TEXTURE_DIR` 为 Windows 挂载目录，不能使用 Linux 私有临时目录。

`verify_airsim.py` **只读检查** RPC、车辆、遥测和 Cube 资源，不解锁、不起飞、不移动。通过后在页面切换 `AirSim Validation` 才会加载共享地图并重置仿真无人机；点击 Start 才执行飞行任务。地图和贴图加载可单独验证：

```bash
.venv/bin/python scripts/verify_shared_scene.py
```

此流程仅面向 Windows Blocks 仿真，不连接真实无人机。校园任务仍使用 HTML 仿真。

## 项目结构

动态障碍物的观测、预测算法及固定交通评测规则见[动态障碍物说明](docs/dynamic-traffic.md)。

```text
uav_llm_demo/mission_service.py  任务生命周期、控制循环和重规划调度
uav_llm_demo/agents.py           几何规划与 DeepSeek 航路/语义接口
uav_llm_demo/semantic_tasks.py   任务契约、条件步骤与仿真执行证据
uav_llm_demo/spatial_motion.py   通用空间运动表达、共享几何与轨迹编译
uav_llm_demo/semantic_navigation.py 任务约束、所有分支的可达性预检
uav_llm_demo/safety.py           waypoint 和连续路径安全检查
uav_llm_demo/motion.py           waypoint 管理、速度/加速度限制和重规划判定
uav_llm_demo/avoidance.py        本地预测避障、短程绕行和危险缓解
uav_llm_demo/runtime.py          HTML/AirSim 的统一连续运动契约
uav_llm_demo/html_runtime.py     默认连续运动执行器
uav_llm_demo/airsim_runtime.py   AirSim RPC、NED 映射和遥测适配
uav_llm_demo/environment.py      三维任务、建筑和动态飞行物
uav_llm_demo/static/             Three.js 场景和 Runtime 切换 UI
scripts/                         Flask、AirSim 启停和联机验证脚本
tests/                           Runtime、服务、API、飞控和界面契约测试
```
