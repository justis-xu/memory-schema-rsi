# 来源账补齐返回正文快照：为输出事实回源保留当次文本

> 2026-09-29。代码位于 `src/schema_rsi/evaluation/{pipeline,source_trace}.py`；验证脚本 `scripts/m2_verify_returned_record_trace.py`，机器结果 `results/analysis/m2_returned_record_trace_20260929.json`。零新增模型调用，不写真实记忆库或图。

## 从上一段提取输出发现的接口缺口

四次中文提取 POC 的 22 条事实均带合法来源 ID，仍有 3 条支持不足，并漏掉网球早场频率。它是新 JSON 契约，不是生产 Mem0 返回格式。核对当前代码：实验 RSI `verify_answer` 验的是拟议**答案**，`graph_admission` 验的是**图候选的当前源身份**；二者都不是输出事实验收器。现有 Mem0 路径的提取结果是记忆字符串，`EvaluationPipeline.ingest_case` 的可选来源账此前在 `batch_outcome` 只留返回 ID，无法在后续版本变化时证明当次实际返回了哪句话。

本阶段在**可选来源事件**里增加 `returned_records`：每条保存返回记录 ID、当次正文、正文 SHA256，以及有限的 `session_id`、`session_date`、`event`。保留旧 `returned_record_ids` 与 `fact_support_status=not_provided`；内容快照不等于事实子句，不自动证明语义支持。默认没有 sink 的评测写入流程不变，现有运行脚本也未自动启用 sink。

## 离线验证与边界

取已归档的中文工程师题两处真实源场话轮作输入；Recorder 将上一段模型 POC 的“早场 4 人”和“后场 5 人”输出**注入**为两条 MemoryRecord。开关来源账的 backend 请求逐批相同；两个输入/结果事件按 batch ID 配对，返回正文、正文指纹与场次元数据均能在事件中复核，任意额外 metadata 不被复制。注入文本不是历史 Mem0 实际提取结果，本试验只验来源账接口。

`tests/test_ingest_source_trace.py` 共 **11 passed**，覆盖默认兼容、返回文本与指纹、sink 修改不污染返回记录、无效运行身份在 backend 前拒绝，以及已有 sink 失败边界。脚本和测试不读写生产 Chroma/HugeGraph，没调用答题或裁判。

**结论：** 可选来源账现在能把“这批输入”与“本次 backend 实际返回的记忆正文”留在同一事件链上，避免后续只按 ID 回查时混入新版本文本。它尚未把一个记忆字符串拆成事实，也未自动将事实判为 full/partial/absent。若 backend 已写入却在返回前报错，`write_completion=unknown` 的旧边界仍在；不能从缺少返回事件推断零写入。

**置信度：** 对事件结构、正文指纹、两场 Recorder 对账、默认请求不变和测试结果为高；真实 Mem0 提取在长批次/更新事件中的完整返回行为、源支持质量与中文答题效果未知。

**下一步优化建议：** 在隔离的少量中文真实 Mem0 提取中显式启用 sink，冻结输入文件/配置/模型版本及运行尝试 ID，保存返回正文；再人工拆分实际记忆为子句，逐项核用户来源、时间、状态、遗漏与无源增强。单次返回列表也不能代替写入后完整库快照，特别是更新/失败路径；等来源和反伤过关后才做同条件答题。
