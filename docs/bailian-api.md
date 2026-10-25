# 使用百炼 DeepSeek 与云工开物学生券

## 先核验券，再切换接口

云工开物学生抵扣金可用于百炼模型推理，但实际适用商品动态调整。请在同一阿里云账号的“费用与成本 → 优惠券”中核验余额、有效期、适用商品和订单类型；目标 DeepSeek 型号必须在适用范围内，且需支持按量付费账单。符合条件的账单出账时自动抵扣；超额或过期后的费用需另付。

API 调用成功只能证明接入可用。抵扣是否成功以阿里云账单明细中的优惠券抵扣记录为准，不能用 HTTP 200 或模型返回文本代替核验。

## 在项目网页中配置

1. 点击地图上方的 **API 设置**，选择 **阿里云百炼（北京）** 或 **阿里云百炼（新加坡）**，地域须与 Key 一致；共享接口地址自动填写。
2. 填写百炼 Key，点击 **读取模型**，从下拉列表选择模型并保存；模型名称来自百炼目录，无需手输。列表保留 DeepSeek 作者、支持结构化输出且由阿里云提供推理的模型。
3. 使用业务空间专属地址时展开高级设置，填入控制台提供的 `https://{WorkspaceId}.{Region}.maas.aliyuncs.com/compatible-mode/v1`；接口地址改变后需填写对应 Key，再读取模型。
4. 配置失败可在原窗口修改重试，保存成功后重新提交任务。Key 在当前服务进程内按服务商和接口保存，切换回已配置的服务可留空复用；不需要重启，暂停的任务不会因保存而自动恢复。

模型发现使用官方 `GET /api/v1/models` 接口并处理分页，官方 DeepSeek 使用 `GET /models`；只查询元数据，不生成文本。模型目录可见、支持结构化输出，不保证该 Key 已获推理权限，也不保证学生券适用。真实推理、任务结果和费用抵扣仍需分别验证。百炼旧北京共享域名模型列表路径由官方 Base URL 规则推导；业务空间和新加坡列表地址在官方文档中直接给出。

Base URL 不需要 `/chat/completions`，项目会自行追加；误填该后缀会被规范化。保存前会再次检查所选模型是否仍在列表中，校验失败不覆盖当前配置、Key 或任务状态。密钥只在内存保留，不向页面返回，也不写浏览器持久存储。

## 接入行为

请求层根据解析后的主机名识别百炼北京/新加坡旧域名和 `*.maas.aliyuncs.com` 业务空间域名，向百炼发送顶层 `enable_thinking: false`。其他地址继续使用原 DeepSeek 的 `thinking: {type: disabled}`。

路由规划和语义任务共用此请求层，保留 JSON Object、输出长度限制、解析与安全校验。真实密钥、模型与地址不会因本次代码修改自动切换。第三方代理域名不会自动按百炼识别。

## 官方依据

- [云工开物百炼活动](https://help.aliyun.com/zh/model-studio/introduction-to-yungongkaiwu)
- [学生权益与适用范围](https://university.aliyun.com/buycenter)
- [优惠券使用与账单抵扣](https://help.aliyun.com/zh/user-center/how-to-use-coupons)
- [OpenAI 兼容接口与同地域密钥](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)
- [请求参数与 enable_thinking](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)
- [结构化输出支持清单](https://help.aliyun.com/zh/model-studio/qwen-structured-output)
- [查询百炼模型列表](https://help.aliyun.com/zh/model-studio/list-models)
- [各地域 Base URL](https://help.aliyun.com/zh/model-studio/base-url)
- [DeepSeek 模型列表](https://api-docs.deepseek.com/api/list-models/)
