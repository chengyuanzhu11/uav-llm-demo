# 校园语义任务基准

基准按独立编写的任务要求检查实际执行结果，不能仅凭服务显示“完成”判定成功。包含 14 个公开案例，覆盖顺序、循环、条件分支、绕圈、降落停留、中途修改任务、漏项，以及未知、冲突、过期观测后的恢复。案例与标注是开发过程中包含代理辅助编写的工程夹具，尚未经过独立人工盲审。

## 运行

在项目根目录、已激活 Python 环境的 WSL 终端运行：

```bash
python scripts/run_semantic_benchmark.py --split all
python scripts/run_semantic_benchmark.py --case dev_repeat_observations
```

默认使用预设模型回复、校园几何和 HTML 执行器，不调用 API、不影响网页服务。结果直接显示在终端；需要留存时添加 `--output outputs/campus-result.json`，文件路径须未被使用。动态障碍物默认关闭，可用 `--traffic fixed` 开启固定鸟群。

校园也支持 Gazebo＋PX4。先按 README 完成安装，关闭正在运行的项目 SITL，并将网页切回 HTML：

```bash
python scripts/run_semantic_benchmark.py --runtime gazebo --mode realtime --split all --max-calls 8 --simulation-budget 900 --wall-budget 900 --output outputs/campus-px4.json
```

每例自动启动独立世界、连接 PX4、执行任务、按原有标准验收，再关闭自己创建的进程；PX4 数据目录也独立创建，避免继承前例的自动校准和保存参数。异地降落或失败不会污染下一例。已有交互式仿真占用时拒绝启动，不会重置它。启动失败、执行异常和超时全部计入分母，禁止退回 HTML。原生仿真必须使用 `realtime` 和 `--traffic none`，分别记录 Gazebo 时钟与包含启动时间的墙钟预算；真实飞行动力学不能由虚拟加速循环代替。`--backend` 独立决定预设回复或真实语言模型。

指定 `--output` 后，每例结束会原子更新同一个结果文件。Ctrl+C 会清理本次仿真并保留先前完成的结果；`selected_runs` 与 `complete` 明确区分完整批次和中断批次，不能把后者的已完成案例当作整批成功率。

`--policy all` 在相同模型、场景种子、本地规划器和安全机制下比较四种策略：`single_pass` 单次计划，`current_repair` 格式/几何修复及预置复核，`event_review` 加执行事件复核，`contract` 加独立要求提取/审查与工具迭代规划（默认生产路径，属于组合策略，不能将差异单独归因于任务契约）。前三档的内部验收依据首个计划，明确标注为 `plan_derived_baseline`；独立评分不变，可揭示服务错误完成。配对比较拒绝协议、源码、案例或种子不一致的结果，失败、超时、澄清均保留在分母。

```bash
python scripts/run_semantic_benchmark.py --case holdout_initial_omission --policy all
python scripts/run_semantic_benchmark.py --split holdout --world seeded --seed 23
```

`virtual` 使用固定 0.05 秒步长，等待本地计算时冻结世界；`--planner-latency` 指定仿真交付延迟。`realtime` 使用生产控制线程，`--control-delay-ms 120` 可注入执行器负载，单独测慢帧压力。两者分别记录仿真时间、墙钟时间、deadline miss 与慢帧差值，不能混用排名。`--world seeded` 独立生成隐藏温度，`fixture` 使用数据集明确保存的场景；未来真值和故障表均不交模型。

真实模型评测需先设置环境变量 `DEEPSEEK_API_KEY`，可选 `DEEPSEEK_MODEL` 和 `DEEPSEEK_BASE_URL`，然后运行：

```bash
python scripts/run_semantic_benchmark.py --mode realtime --backend deepseek --case dev_east_boundary --max-calls 12 --wall-budget 300
```

该命令会产生 API 用量，每例最多 12 次语义请求、运行 300 秒。首次任务或用户修改先提取和审查要求，再由模型逐轮查询几何、试算并提交计划；工具调用对应的模型请求全部计入预算。自主复核复用已接受要求并重新规划。默认预算为 6 次调用、600 秒仿真时间、300 秒实际时间，覆盖全部阶段及重试；超时和失败均计入结果。密钥不要写入命令参数或结果文件。

## 验收与扩展

数据位于 `benchmarks/campus_semantic_v1.json`。`instruction` 是测试指令，`oracle` 是独立编写的可观察要求；`offline_requirements`、`offline_audits`、`offline_responses` 分别是预设的提取、审查及计划回复，`updates` 是中途追加的用户指令。要求回复不从候选计划或评分标准自动生成；标准答案和预设回复不会交给真实模型。

- 动作按执行记录核对目标、次数、先后顺序和观测结果；条件任务按实际求值时刻及当时有效的读数验收，后续复测不会改写过去的分支判断。
- 绕圈按实际轨迹检查方向、圈数、边界和机头方向，纳入中断前的运动和续接记录；降落、悬停和停留按位置、速度、姿态及持续时间检查。
- 中途修改必须实际触发；自主复核不能遗漏仍有效的任务。

`oracle_success` 表示独立验收通过，`service_success` 是服务自身判断，`false_completion` 标记服务误报完成。退出码 0 表示案例结果符合预期。`holdout_initial_omission` 先注入要求漏项，由审查退回，再注入计划漏项，由契约拒绝，修正后必须完成全部任务；持续漏项拒绝执行另由同一运行器的测试验证。`interpretation_stage_calls` 分别记录三个阶段的请求数。

新增案例时先写可观察要求，再分别编写各阶段离线回复；保留 `combination` 在开发集与保留集之间的隔离。数据集修订号随 JSON 保存。`recovery` 统计复核后超出独立要求指定次数的成功动作，必要复测不视为重复工作；同时报告完整任务满足率、错误完成、安全违规和全部阶段调用成本。`--prices-json` 接受用户提供的每百万 token 费率，缺少 usage 时费用为未知，不能当作零成本。

离线回复验证执行与验收链路；真实 API 结果验证模型在这些案例上的表现。14 例可分别通过 HTML 或 Gazebo＋PX4 执行，结果按后端分开比较。原生后端的遥测来自 Gazebo 物理位姿，落地验收还要求 PX4 已落地并解除武装；结果包含世界、视觉网格、车辆模型及 PX4 二进制指纹。温度、巡检和普通拍照仍是模拟任务传感器，不能据此宣称真实视觉或热成像成功。原生 RGB-D 和视觉模型单独通过 [README](../README.md#gazebopx4-sitl) 的视觉验收入口检查。工程夹具不构成人群准确率、论文显著性或实机飞行结论。

独立 `physics` 后端及 `--wind-mps`、`--gust-mps` 参数已移除；`--runtime` 仅接受 `html` 或 `gazebo`。旧案例包含 `physics` 配置时明确拒绝；历史结果仍可读取和比较，但不同后端、配置和源码版本不会被配成同一实验条件。
