# DeepSeek Prompt Cache 排查

日期：2026-09-28。范围：本项目 waypoint 规划请求及缓存统计。结论依据为当前代码、三组离线请求捕获及 DeepSeek 官方缓存文档。

## 结论与证据边界

主要问题是静态地图排在动态状态后面。原 Prompt 已把规则放在前面，但 `Context` 的第二个字段就是 `step`；一旦步数变化，后面的建筑地图即使完全相同，也不能作为连续相同前缀复用。暂停时的重试仍可能被鸟类位置的变化提前截断前缀。

已完成 Prompt 重排、确定性序列化和逐次 usage 日志。没有改变规划器选择、地图几何、安全距离、输出契约、重试次数或飞行控制。

用户提供的小时统计：

- 输入合计：124,643 tokens；缓存命中：12,544；未命中：112,099。
- 加权输入命中率：10.0639%。输出 30,057 tokens 不计入该分母。
- 84 次请求平均输入 1,483.85 tokens，平均命中 149.33 tokens，平均输出 357.82 tokens。

无法把这 84 次请求逐一对应到当前代码版本：修改前未保存响应 usage，本次检查时运行中服务的 history 为 0；检查到的 Flask 日志也没有缓存统计。小时总量能证明低命中率，不能单独证明每次请求的首次变化位置或所有损失都来自同一个原因。

## A. 调用入口

| 层级 | 代码位置 | 行为 |
| --- | --- | --- |
| 业务入口 | `uav_llm_demo/mission_service.py`：`_submit_plan()` | 拷贝当前上下文，将当前 Agent 交给线程池执行 `checked_plan()` |
| 规划校验 | `uav_llm_demo/agents.py`：`checked_plan()` | 调用 `planner.plan(context, feedback)`，随后检查返回航线 |
| 唯一 DeepSeek 请求构造点 | `uav_llm_demo/agents.py`：`DeepSeekAgent.plan()` | POST `{base_url}/chat/completions`，非流式响应 |
| 非 Windows 传输 | 同文件 `_post_deepseek()` | `requests.post()` |
| Windows 传输 | 同函数 → `scripts/deepseek_request.ps1` | 请求体作为字符串经 stdin 传入，`Invoke-WebRequest` 原样发送 |

手动 Plan、Start 首次规划、返航、目标变更、安全重规划、失败重试都汇入上述入口。两个平台分支是同一请求的传输实现，不是两次 API 调用。测试和离线诊断脚本通过替身截获该接口，没有额外真实调用入口。

## B. 修改前 messages 的完整形成过程

1. `environment.py::Observation.to_dict()` 按插入顺序产生观察数据：

   ```text
   scenario_id, step, position, battery, phase, status,
   goal, goal_label, goal_distance, goal_delta, goal_direction,
   nearest_static_obstacle, dynamic_objects, nearest_dynamic_obstacle,
   altitude_range, text
   ```

2. `MissionService._context()` 在后面追加 `grid_size, buildings, base, target, no_fly_zones, orientation, task, preferred_altitude, urban_parcels, local_avoidance`。
3. 原 `build_prompt()` 浅拷贝该字典。有城市地块时，删除 `urban_parcels/buildings`，在字典末尾追加压缩后的 `building_volumes`。
4. 原 `DeepSeekAgent.plan()` 每次创建两条消息：

   ```text
   system: Return a waypoint route as a JSON object.
   user: 固定规则 + Context 整体 JSON（动态字段在前，地图在后）+ Feedback
   ```

5. 同一请求还设置固定的 JSON 输出格式、非思考模式、temperature=0.1、max_tokens=2000、stream=false。模型、地址和 key 来自 Agent 配置；没有每次随机切换模型或添加 `user_id`。

原 JSON 使用 Python 字典插入顺序，并非随机打乱；问题主要是字段顺序本身不利于缓存。Windows 传输层没有添加时间戳、重新拼接 history 或把图像插入 messages。

## C. 字段分类

| 类别 | 实际内容 | 变化方式 |
| --- | --- | --- |
| 完全固定 | 输出 schema、坐标单位/方向、距离规则、航点数量、JSON 格式、规划规则 | 只有代码修改时变化 |
| 场景级固定 | 场景 ID、地图边界、高度范围、优选高度、建筑体积、禁飞区 | 切换场景或更新地图时变化 |
| 任务/航段级固定 | task、base、target、当前 goal、goal_label | 修改任务、目标或转入返航时变化 |
| 重规划时可能变化 | UAV position、battery、phase/status、goal 距离/方向、最近障碍物、orientation、local_avoidance、step | 由运动、控制周期和任务状态决定；暂停重试时部分值相同 |
| 每次重新采集通常变化 | dynamic_objects 内的鸟类位置/速度/预测位置及距离、nearest_dynamic_obstacle | 鸟类独立时钟持续运动，UAV 不动时也会变化 |
| 请求尾部反馈 | feedback | 被拒路线/失败原因变化时变化，也可能连续相同 |

没有必然每次新建的 timestamp、request UUID、frame_id、FPS 或 latency 字段进入 Prompt；`step` 是控制步数，不是请求 ID。`text` 是含 UAV 当前坐标的重复文字说明，仍属动态内容。当前上下文含鸟类 velocity 和 UAV orientation，但没有独立 UAV velocity 字段，此次未扩充传感信息。

## D. 连续请求比较

离线诊断使用 `MissionService._context()` 生成起始和移动观察，再构造一个只改变鸟类位置及重试反馈的第三观察。三次都经过真实 `DeepSeekAgent.plan()` 的 messages 构造路径，HTTP 用测试替身截获，不调用付费 API。优化前后复用完全相同的上下文。

生成物：`artifacts/prompt-cache-before.json`、`artifacts/prompt-cache-after.json`。包含离线上下文、完整 messages 和首次差异位置；不含 key。字符偏移为零起算，未使用或假冒 DeepSeek tokenizer。

| 连续请求 | 修改前首次差异 | 修改后首次差异 | 所有 message 内容的共同前缀字符数 |
| --- | --- | --- | --- |
| 初始 → 移动重规划 | message[1] 字符 1231：`step: 0 → 100` | message[1] 字符 1530：State 的 battery | 1272 → 2922 |
| 移动重规划 → 停机重试 | message[1] 字符 1717：鸟类 position.x；地图尚未出现 | message[1] 字符 1630：State 内鸟类 position.x；地图/目标已出现 | 1758 → 3022 |

新 System 内容变长，所以不能只比较第二条 message 内的字符偏移。三条请求总内容长度分别从 5302/5318/5356 降为 5082/5098/5136 字符。共同前缀字符数不等于真实 cache hit tokens，也不能据此宣称命中率已达到某个百分比。

## E. 四类原因判定

| 检查项 | 是否存在 | 代码证据 | 影响 |
| --- | --- | --- | --- |
| 1. 动态信息出现过早 | 是，针对地图；固定规则本来就在前面 | `Observation.to_dict()` 的 `step/position` 在前，原 `build_prompt()` 的 `building_volumes` 在最后 | 高，较早变化阻断后续整张地图的前缀复用 |
| 2. 多轮 history 不稳定 | 否 | `DeepSeekAgent.plan()` 每次只有 system/user 两条消息；`_submit_plan()` 只传当前 context 和 feedback | 无，本地 history 与 UI 轨迹未进入请求 |
| 3. 每次新视觉输入 | 否 | 两条 message 的 content 均为文本，没有 image_url、base64、截图或 camera frame | 无 |
| 4. 易变辅助字段/动态 JSON | 是 | `step` 为原 context 第二项；鸟类坐标与距离高精度变化；`text` 重复包含动态坐标 | step 对早期失配影响高；动态尾部长度影响中 |

另外检查：没有完整 trajectory/path history 入 Prompt，没有随机 UUID/request_id，没有按请求改变 system 文本。返回的 provider request ID、created 等字段也未回填 messages。温度不导致客户端 Prompt 前缀变化。

## F. 原因排序

1. **主要原因（已证实）**：`step/position` 等变化发生在静态地图之前；停机重试时鸟类位置仍在地图之前变化。
2. **次要因素（已证实）**：动态鸟类快照、距离、重复 text 较长，即使排到末尾，这部分也通常无法命中；本次保留全部观察信息以避免改变规划语义。
3. **可能因素（官方机制支持，当前样本未实测）**：共同前缀需要被服务端识别并落盘；前两次可能尚无可匹配前缀，构建也需要数秒。场景/目标/模型切换、跨时段过期会影响结果。2 秒重试间隔可能落在缓存构建窗口内，但不能仅靠本地代码认定其实际贡献。
4. **基本无影响/不存在**：多轮 history 裁剪、视觉帧、时间戳/UUID、轨迹历史。

DeepSeek 当前文档描述了请求边界、共同前缀识别和固定 token 间隔三种落盘方式。特别是 A+B、A+C、A+D 的例子中，第三次才可能复用已识别的 A；因此应观察多次请求，不能只看相邻两次。不要把旧资料中的“64 tokens 块”当作本项目当前模型的已核实门槛。

## G. 已实施方案

```text
message[0] system: 固定控制规则 + 坐标定义 + 输出格式
message[1] user:
  Scene:   固定地图、边界、禁飞区、优选高度
  Mission: 任务、基地、目标、当前航段目标
  State:   step、position、鸟类快照、orientation 等当前状态
  Feedback: 拒绝原因/重试反馈
```

- JSON 固定键顺序和紧凑分隔符；建筑/地块体积/禁飞区列表规范排序。不同插入顺序不改变同一地图的前缀，实际地图修改仍正确更新请求。
- 固定规则移入 System；场景切换时也能共享通用规则。未添加缓存开关、缓存占位文本、无意义 padding 或伪造历史。
- `provider_usage.py` 只记录数值型白名单 usage；在解析路线和安全验收之前记录，所以格式错误/被安全层拒绝的付费响应也能计入。无 usage 的传输/HTTP 失败记录空值。
- `app.py` 配置 `logs/deepseek_usage.log`：JSON Lines、2 MB 轮转、3 个备份，已加入忽略范围。日志只含配置模型名、HTTP 状态、时间、耗时、字符级前缀统计和数值 token 用量；key、原始 prompt、response 文本不写入。
- 日志中的 timestamp、prompt_version 和字符统计只在本地记录，绝不追加到 API messages。
- `scripts/audit_prompt_cache.py --usage-log logs/deepseek_usage.log` 按模型和 Prompt 版本汇总，使用 `sum(hit)/sum(prompt_tokens)`；缺失或三项不一致的样本单列为未测量，不冒充 0% 命中。

验证：135 个 unittest 通过，包括缓存前缀稳定性、目标/地图变化正确失效、序列化顺序、非法返回也记录 usage、数值白名单及敏感响应不落日志。已捕获三组优化前后离线 messages。

## H. 真实连续采样结果

2026-09-28 21:57-21:58（Asia/Shanghai），用户授权后对运行中的 `deepseek-v4-flash` 执行三轮手动 Plan，未启动飞行。共尝试五次请求，三次返回 HTTP 200 且 usage 完整；三轮最终均生成计划，任务保持 paused。key 仅由用户在页面输入，没有读取或导出进程内凭据。

| 请求 | 所属轮次 | HTTP | 耗时 | 输入 tokens | 命中 | 未命中 | 命中率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 未取得响应 | 10.298 s | 未知 | 未知 | 未知 | 不计入 |
| 2 | 1 | 未取得响应 | 10.265 s | 未知 | 未知 | 未知 | 不计入 |
| 3 | 1 | 200 | 6.996 s | 1808 | 0 | 1808 | 0% |
| 4 | 2 | 200 | 2.401 s | 1843 | 896 | 947 | 48.62% |
| 5 | 3 | 200 | 2.388 s | 1849 | 896 | 953 | 48.46% |

- 三次有效响应合计：输入 5500，命中 1792，未命中 3708；加权命中率 **32.58%**，包含首次零命中样本。
- 后两次有效响应加权命中率 **48.54%**（1792/3692）；输入并不完全相同，但共同前缀仍命中 896 tokens。其字符级共同前缀为 2959，不能将字符数直接换算成缓存 token 数。
- 原小时汇总命中率为 10.06%。当前小样本显示新的请求结构确实发生前缀复用，但与历史小时不是同任务、同时间的受控 A/B，不能声称整小时命中率必然提升到 48%。
- 前两次没有收到 HTTP 状态或 usage，不能判断服务端是否处理或计费，也不能归因于缓存、DNS、代理或具体网络节点。它们不作为零命中样本混入分母。
- 后续响应变快是本次观测，不等于缓存独自造成全部加速；网络和输出长度同时可能影响耗时。三次有效响应输出分别为 219、373、242 tokens。
- 本次无需追加请求即可确认稳定前缀被复用；后续正常任务继续由 `logs/deepseek_usage.log` 记录，可用已有汇总脚本观察跨任务和返程的表现。

## 官方来源

- [DeepSeek Context Caching](https://api-docs.deepseek.com/guides/kv_cache/)：共同前缀、落盘时机、best-effort、缓存构建与过期。
- [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)：usage 三项定义，`prompt_tokens == hit + miss`。
