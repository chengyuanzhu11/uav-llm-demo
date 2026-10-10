# 恢复、时间和评测

## 执行与恢复

网络可恢复错误按 2、4、8、16、30 秒退避，之后保持 30 秒，无次数上限。认证、余额和无效参数仍要求用户处理。路线拒绝保留最近 8 次失败，反馈最近 4 次，包含拒绝航段、位置、次数和下一步修复要求；重复路线要求选择不同走廊，间隔最多 10 秒。记忆是历史证据，不永久把曾受动态障碍影响的区域列为禁区。暂停/Reset 会作废旧请求，迟到结果不执行；不自动换 Mock。

无进展时刷新观测后重规划，并保留已完成去程的证据。页面 Recovery 显示策略及剩余重试时间；Mission progress 分别显示去程目标和返航基地。任务执行器与 `MissionEvidence` 各自判断阶段，成功需要目标及基地的实际位置、末端速度证据。当前任务只含往返，不声称验证了拍摄或巡检动作。

## 时间与观测

HTML 的 UAV 和鸟类使用同一仿真时间。后台循环在未起飞、暂停和等待 API 时仍推进世界。每次推进不超过 0.1 秒；慢帧会使整个仿真相对墙钟减慢，累计差值在 diagnostics 中可见。重试间隔也基于这一仿真时间。浏览器对最近快照做显示平滑，不向未知的墙钟未来外推鸟类。

GET `/api/state` 不推进时间、不生成鸟类路线、不查询 Runtime，也不重建静态地图；返回控制线程发布快照的副本。`sequence` 表示快照版本，多次读取同一快照可得到相同序号。

`diagnostics.timing` 提供锁等待、交通、规划收集、环境检查、航点重连、避障、Runtime 和快照耗时。p95 使用最近 2048 次样本，max/count 为当前 run 累计。控制间隔超过 75ms 计一次 missed deadline；HTML 超过 300ms 记延迟事件并继续，AirSim 保留停止保护。局部碰撞检查不因调度延迟而关闭。

## 运行日志

由 `app.py` 或评测脚本启用 `logs/mission_runs.jsonl`，单文件 20MB、保留 5 个备份。日志线程处理编码和写盘，有界队列满时不阻塞飞控，`journal_dropped` / `journal_errors` 显示记录缺口。轮转前应保存需要长期保留的运行文件；异常强制结束可能丢失尚未落盘的尾部事件。

每轮 run_id 关联场景、目标/交通种子、控制参数、源码指纹、数值航点、拒绝原因、重试、阶段完成、每秒遥测和控制延迟。Reset 建立新 run_id，旧日志保留。进程内最多保留 200 条规划记录，累计调用/拒绝/错误计数不随截断丢失。日志不保存 key、请求头、原始 Prompt、原始响应或自由文本模型推理。DeepSeek 数值用量日志包含同一 run_id/request_id；未返回 usage 与尚未返回的请求会明确标记，不计为零用量。

源码指纹用于比较当前 Python 实现，不能代替依赖环境归档。任务日志保存几何场景和结构化规划响应，不是操作系统调度、图像或真实传感器的完整录制。

## 两种评测协议

- `virtual`：等待本地规划计算完成，等待期间冻结仿真时间；通过 `--planner-latency` 显式注入规划交付延迟。无需通过抢占线程或人为 sleep 决定规划能否及时返回。只支持本地 Mock/响应回放，禁止直接调用远程 API。
- `realtime`：运行生产控制线程，计入真实 API 等待、控制抖动和系统负载；`--max-seconds` 是墙钟任务预算。初始化场景的时间不计入任务预算。

两种协议分别报告 `simulation_seconds` / `wall_seconds`，不可混用排名。结果为 `completed`、`failed` 或 `timeout`，另附终止原因。虚拟本地计算还有独立的 `--planner-timeout` 墙钟保护；超时结束评测不会再发新请求，已发出的网络请求可能到自身超时才退出。

```bash
~/.venvs/uav-llm-demo/bin/python scripts/run_benchmark.py --mode virtual --scenario urban_crossing --seed 7 --seed 11 --seed 23 --planner-latency 1 --output before.json
~/.venvs/uav-llm-demo/bin/python scripts/run_benchmark.py --mode realtime --backend deepseek --scenario urban_crossing --seed 7 --max-seconds 120 --output live.json
~/.venvs/uav-llm-demo/bin/python scripts/compare_benchmarks.py before.json after.json --output comparison.json
```

真实 API 脚本需要 `DEEPSEEK_API_KEY` 环境变量，不读取 Flask 的内存 key。实时评测会消耗用户配置的 API 额度；默认命令为离线 Mock。

`--seed` 可重复指定唯一种子。比较脚本要求场景、目标、交通种子与协议一致，明确报告失败变成功及成功变失败，不忽略未匹配样本。摘要的 Wilson 区间仅是独立运行层面的描述性统计；对有依赖的任务序列或重复同一环境应另做相应统计分析，不把几次本地回归当论文结论。

成功分为 task_completed 和 safe_success。后者还要求没有记录到静态路径阻止或动态安全间距违规；HTML 间距指标不是网格物理碰撞证明，未测量动态障碍的 AirSim 对该项返回 null。电量仍是简化模型，不能用于真实能耗结论。

可选 `--prices-json prices.json` 接受 `input_miss`、`input_hit`、`output`（每百万 tokens 的数值费率）与 `currency`。费率由用户提供，不内置报价。费用只覆盖已报告 tokens；usage 缺失或请求未返回时，`complete_usage=false`。

## 离线规划响应回放

```bash
~/.venvs/uav-llm-demo/bin/python scripts/replay_run.py --journal logs/mission_runs.jsonl --run-id RUN_ID --max-seconds 120 --output replay.json
```

从日志还原初始场景、交通种子、结构化方案和记录中的仿真交付延迟，交给当前控制器重新执行，不调用 API。用于固定失败案例和成功回归案例，源码指纹不同时会在输出中标明。只支持没有手动改目标、切 Agent、暂停的单轮 HTML 任务；跨轮转文件缺少记录、手动操作或响应耗尽时明确报错/失败，不偷偷补调 API。回放不保证复现原机的 CPU 调度和网络抖动。
