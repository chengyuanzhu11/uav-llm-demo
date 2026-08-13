# UAV 仿真页面场景优先调试工作台设计

## 1. 背景与目标

当前页面已经具备 Three.js 三维场景、任务遥测、Runtime 切换、AirSim 配置、Agent 决策日志和任务历史，但所有信息长期同时显示在固定右栏中。主要问题是：

- 高频单步操作、低频 Runtime/AirSim 设置和调试证据混在同一纵向区域。
- Observation 与 Action JSON 默认以长文本块呈现，关键结论需要人工扫读。
- Agent 提案、安全仲裁和最终执行结果缺少稳定的信息层级。
- 页面大量使用相近的青绿色，选择、数据和状态的语义区分不足。
- 固定右栏不能在场景观察和详细调试之间快速切换信息密度。

本次优化的首要目标是桌面端单步调试效率。三维场景保持第一视觉中心，同时让用户能在一次点击内查看完整的 `Observation -> Agent Proposal -> Safety Guard -> Execution` 证据链。

## 2. 已确认的产品决策

- 优化方向：操作效率和信息层级优先，不以装饰性视觉升级为目标。
- 主要工作流：逐步执行并检查每一步决策。
- 目标设备：桌面端；不把手机和平板作为验收目标。
- 总体布局：场景优先，主场景默认约占 72%。
- 详情模式：双宽 Inspector，不提供任意拖拽。
- Inspector 默认宽度为 28%，展开宽度为 34%，用户可一键切换。
- 控制状态必须反映真实请求生命周期：请求尚未提交时 Pause 请求运行时取消；HTML 步骤已提交时 Pause 只停止后续 Auto 迭代，界面不得声称已经回滚。
- 默认展示结构化摘要，原始 Observation 和 Action JSON 按需展开。
- 保留理由摘要和可审计证据，不展示隐藏思维链。

## 3. 范围

### 3.1 包含

- 重组顶部命令栏、场景 HUD、Inspector、历史记录和设置入口。
- 建立 Ready、Executing、Paused、Error 和 Runtime Switching 的视觉状态。
- 使用现有 API 状态生成结构化 Observation、Agent、Guard 和 Execution 摘要。
- 将完整原始文本和 JSON 保留为可展开详情。
- 将任务历史改为紧凑表格，并支持回看单个步骤。
- 桌面视口的布局、溢出、键盘焦点和可访问性验证。

### 3.2 不包含

- 修改 Agent、任务规划、安全仲裁或 Runtime 的业务语义。
- 新增或修改 Flask REST API。
- 改变 HTML 与 AirSim Runtime 的独立执行架构。
- 增加新的仿真场景、传感器或飞行动力学能力。
- 手机和平板专项设计。
- 将 AirSim 启动并入 Flask 或浏览器 Runtime 切换。

## 4. 信息架构

### 4.1 顶部命令栏

固定高度约 52px，从左到右排列：

1. 产品标识和当前任务状态。
2. Scenario 选择器。
3. HTML/AirSim Runtime 分段选择器及连接状态。
4. 设置图标按钮。
5. Reset、Auto/Pause 和 Step 命令。

`Step` 是单步调试的主命令，使用主要按钮样式。`Auto` 和 `Reset` 为次级命令。设置使用图标按钮和 tooltip，不在主界面长期占用 AirSim Host、Port、Vehicle 和 Wind 字段。

### 4.2 三维场景

场景占据除 Inspector 外的全部工作区。场景内只保留与即时态势相关的信息：

- 左上遥测条：Position、Altitude、Battery、Phase、Dynamic Risk。
- 左下视角工具：Overview、Follow、Trajectory visibility。
- 右下执行摘要：当前 Runtime、轨迹持续时间或执行进度。
- UAV、目标、基地、建筑、动态障碍物和轨迹继续由现有 Three.js 场景渲染。

HUD 使用一条连续的信息带和分隔线，不把每个数值做成独立浮动卡片。场景控件使用熟悉的图标，并提供 tooltip 与可见焦点状态。

### 4.3 双宽 Inspector

Inspector 有两档固定宽度：

- 默认 28%：保留场景优先级，显示当前步骤的关键摘要。
- 展开 34%：同时显示完整四阶段证据和短历史。

切换通过 Inspector 标题栏中的展开/收起图标完成。选择写入 `sessionStorage`，同一浏览器会话刷新后保持；新会话仍以 28% 开始。切换只改变 CSS 布局，不重建 Three.js 场景或中断任务。

Inspector 从上到下包含：

1. 当前 Step 编号、Agent backend、任务阶段和展开按钮。
2. Observation。
3. Agent Proposal。
4. Safety Guard。
5. Execution。
6. 最近三步和“查看全部”入口。

## 5. 决策证据设计

### 5.1 Observation

默认显示现有结构化状态中的关键事实：

- 目标方向和距离。
- 最近静态障碍物及距离。
- 最近动态障碍物、预测距离和威胁等级。
- 电量和任务阶段只在与本步判断相关时重复显示。

原始 `observation.text` 放入折叠详情。折叠状态不影响 API 数据，也不删除信息。

### 5.2 Agent Proposal

优先展示：

- `action`
- `waypoint`
- `risk`
- `skill`
- `reason` 理由摘要

完整 Action JSON 放入折叠详情。理由字段用于解释可审计的决策依据，不把它描述为或扩展成模型隐藏思维链。

### 5.3 Safety Guard

必须明确并列显示：

- Agent 提议动作。
- 最终执行动作。
- 是否发生干预。
- 干预原因或安全路径进度。

未干预使用绿色 `PASS`；干预使用琥珀色 `OVERRIDE`。不能只显示最终动作，否则会掩盖 Agent 与执行器的职责边界。

### 5.4 Execution

默认显示：

- 最终位置。
- Runtime 来源。
- 执行持续时间。
- 电量变化。
- 执行结果。

AirSim 模式下补充真实遥测来源标识，并允许展开查看 6-DOF 数据；HTML 模式不伪装为真实飞控遥测。

### 5.5 历史记录

Inspector 底部显示最近三步。点击“查看全部”后，Inspector 内部切换为紧凑历史表格，场景仍保持可见。表格列为：

- Step
- Proposed action
- Executed action
- Guard result
- Execution result

点击历史行后，在同一 Inspector 中加载该步骤的四阶段只读证据，并提供返回当前步骤的按钮。历史查看不得改变任务状态。

## 6. 交互状态

### 6.1 Ready

- Step、Auto 和 Reset 可用。
- Runtime 状态显示 Ready/Connected。
- 当前证据保持可读。

### 6.2 Executing

- Step、Reset、Scenario 和 Runtime 切换禁用。
- Auto 变为 Pause；Pause 是唯一可中断当前动作的命令。
- 四阶段状态依次更新，当前阶段显示进行中状态。
- `/api/step` 请求仍在进行时显示不确定进度；AirSim 在此阶段可通过 `/api/runtime/stop` 取消候选飞行并阻止逻辑步骤提交。
- HTML Runtime 返回后，逻辑步骤已经提交，浏览器才按 `trajectory_duration` 播放轨迹。此时可以显示确定的客户端动画进度，但 Pause 只能停止后续 Auto 迭代，不能伪装成回滚已提交步骤。

### 6.3 Paused

- 如果运行时在提交前确认取消，明确显示当前动作已取消；AirSim 已执行 hover，逻辑任务仍停留在上一个已提交步骤。
- 如果 HTML 步骤已经返回并提交，显示“自动运行已在当前步骤后暂停”，不得显示“当前步骤已取消”或回退 `step_count`。
- Step、Resume Auto 和 Reset 恢复可用。
- 被取消的步骤不计入已完成历史；取消事件可以保留在错误/事件记录中。

### 6.4 Error

- 错误固定显示在对应 Runtime 或执行阶段，不覆盖 Executor Feedback。
- 提供与错误对应的恢复命令，例如 Reconnect。
- 运行失败不得前移 `step_count` 或伪装成 HTML 成功执行。
- DeepSeek 错误与 Runtime 错误使用不同标签，避免将模型请求失败误判为仿真器失败。

### 6.5 Runtime Switching

- 切换前保留“会重置任务”的确认。
- 切换期间显示 Switching，并禁用任务命令和重复切换。
- 只有候选 Runtime 成功连接并重置后才更新分段选择器。
- 失败时保留原 Runtime 与任务状态，并显示恢复操作。

## 7. 设置面板

设置图标打开桌面端模态设置面板。面板包含：

- Agent backend/model 只读摘要；API key 仍不回显。
- 当前 Scenario。
- Runtime 连接信息。
- AirSim 激活时显示 Vehicle、Host、Port、Reconnect 和 Wind NED。
- HTML 模式不显示不可用的 Wind 控件。

关闭设置面板不改变任务。只有明确提交 Runtime 配置或 Wind 时才调用现有 API。

## 8. 视觉语言

- 背景以中性炭黑和冷灰为主，避免单一青色主题。
- 青色：当前选择、路径和数据强调。
- 绿色：成功、安全、Connected。
- 琥珀色：Pause、安全干预、需要注意。
- 红色：错误、离线、危险威胁。
- 颜色不是唯一状态信号；同时使用文字和图标。
- 普通 UI 使用系统无衬线字体；坐标、动作和 JSON 使用等宽字体。
- 面板圆角不超过 7px；页面区块不做浮动卡片，卡片仅用于模态或重复数据项。
- 不使用装饰性渐变、光斑或无信息价值动画。

实现时使用 Lucide 图标。为保持本地离线运行，图标资源与 Three.js 一样存放在 `static/vendor`，不依赖 CDN。

## 9. 前端实现边界

不改变后端接口。前端继续消费现有 `/api/state`、`/api/step`、`/api/reset`、`/api/runtime`、`/api/runtime/stop`、`/api/runtime/wind` 和 `/api/scenario`。

模板新增或重组以下语义区域：

- App toolbar
- Scene telemetry
- Runtime/settings dialog
- Decision inspector
- History table
- Runtime/error status

`app.js` 仅在 UI 责任范围内拆分渲染函数，例如：

- `renderToolbarState`
- `renderTelemetry`
- `renderDecisionTrace`
- `renderHistory`
- `renderRuntimeSettings`

Three.js 场景、轨迹插值、Auto Run 串行循环和 Runtime 取消逻辑保持现有行为。避免每帧重建 Inspector DOM；只在 API 状态变化或用户切换视图时更新面板。

## 10. 错误与兼容处理

- 缺失的新旧历史字段使用 `--` 或现有文本回退，不因单个字段缺失中断整个渲染。
- 超长模型名、错误消息、坐标和动作名必须换行或省略，并可通过 title/详情查看完整内容。
- Inspector 内部允许滚动；主页面不产生双重横向滚动条。
- 保留现有桌面以下的基本 CSS fallback，但不把移动端视觉质量纳入验收。

## 11. 验证标准

### 11.1 自动测试

- 保留并运行现有 Python 测试套件。
- 更新前端契约测试，覆盖新语义节点、双宽 Inspector、设置面板和四阶段结构。
- 增加前端状态回归覆盖：Executing 禁用并发命令、Pause 调用 `/api/runtime/stop`、Runtime 切换失败保持原选择。
- 验证新图标资源为本地文件且页面不访问 CDN。

### 11.2 浏览器验收

在 `1280x720`、`1440x900` 和 `1920x1080` 检查：

- Three.js Canvas 非空且场景 framing 正确。
- 顶部命令、HUD 和 Inspector 不重叠。
- Inspector 在 28% 与 34% 两档切换时场景不重建、不闪白。
- Step 完成后四阶段数据和场景位置同步更新。
- Auto/Pause 期间按钮状态与真实请求一致。
- Runtime 切换成功、失败和 AirSim 离线状态可区分。
- 长 Observation、JSON、错误消息和模型名不撑破容器。
- 键盘焦点可见，图标按钮具有可访问名称和 tooltip。
- 浏览器控制台没有错误。

### 11.3 实际 Runtime 验收

- HTML Runtime 完成一次单步和一次 Auto/Pause。
- AirSim 可用时完成切换、单步、Pause 和切回 HTML。
- AirSim 不可用时，切换失败保留 HTML Runtime 和任务状态。

## 12. 风险与成本

- 将设置移入模态面板会降低首次发现性，但能显著减少单步调试噪声；顶部 Runtime 状态和设置图标用于补偿。
- 双宽切换比任意拖拽可定制性低，但实现和测试成本更小，布局结果更稳定。
- 结构化摘要依赖现有字段完整性，因此必须保留原始文本回退。
- 本项目应用源码当前大多仍为 Git 未跟踪文件。实施时必须逐文件检查并只暂存本次 UI、测试和本地 vendor 资源，禁止顺带提交 `.superpowers`、日志、PID 或其他运行产物。
