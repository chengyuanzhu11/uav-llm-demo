# 校园语义任务基准

基准按人工标注的任务要求，独立检查实际执行结果，不能仅凭服务显示“完成”判定成功。包含 11 个公开案例，覆盖顺序、循环、条件分支、绕圈、降落停留、中途修改任务以及漏项。

## 运行

在项目根目录、已激活 Python 环境的 WSL 终端运行：

```bash
python scripts/run_semantic_benchmark.py --split all
python scripts/run_semantic_benchmark.py --case dev_repeat_observations
```

默认使用预设模型回复、校园几何和 HTML 执行器，不调用 API、不影响网页服务。结果直接显示在终端；需要留存时添加 `--output outputs/campus-result.json`，文件路径须未被使用。动态障碍物默认关闭，可用 `--traffic fixed` 开启固定鸟群。

真实模型评测需先设置环境变量 `DEEPSEEK_API_KEY`，可选 `DEEPSEEK_MODEL` 和 `DEEPSEEK_BASE_URL`，然后运行：

```bash
python scripts/run_semantic_benchmark.py --mode realtime --backend deepseek --case dev_east_boundary --max-calls 4 --wall-budget 300
```

该命令会产生 API 用量，每例最多 4 次任务解释请求、运行 300 秒。默认预算为 6 次调用、600 秒仿真时间、300 秒实际时间；超时和失败均计入结果。密钥不要写入命令参数或结果文件。

## 验收与扩展

数据位于 `benchmarks/campus_semantic_v1.json`。`instruction` 是用户指令，`oracle` 是人工标注的可观察要求，`offline_responses` 是离线执行样例，`updates` 是中途追加的用户指令；标准答案和预设回复不会交给真实模型。

- 动作按执行记录核对目标、次数、先后顺序和观测结果；条件任务只要求实际选中的分支。
- 绕圈按实际轨迹检查方向、圈数、边界和机头方向；降落、悬停和停留按位置、速度、姿态及持续时间检查。
- 中途修改必须实际触发；自主复核不能遗漏仍有效的任务。

`oracle_success` 表示独立验收通过，`service_success` 是服务自身判断，`false_completion` 标记服务误报完成。退出码 0 表示案例结果符合预期；故意漏项的 `holdout_initial_omission` 应验收失败，不能算作任务成功。

新增案例时先写可观察要求，再提供离线回复；保留 `combination` 在开发集与保留集之间的隔离。当前修订为 `2026-10-22-r2`，已明确“两轮，每轮测温后采样”和“条件分支完成后返航”。

离线结果验证执行与验收链路；真实 API 结果验证模型在这些案例上的表现。飞行及传感器仍是 HTML 仿真。
