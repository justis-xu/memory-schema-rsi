# 写入来源账接口收尾与研究暂停交接

2026-09-28。用户要求先收尾、然后暂停。本阶段完成此前已开始的来源追踪接口及验证，不启动下一轮分析或模型试验。

> 2026-09-29 后续扩展：可选 `batch_outcome` 已增加返回记忆的当次正文与指纹，并支持调用者提供运行/重试身份；见 `docs/m2_returned_record_trace_zh.md` 与 `docs/m2_trace_run_identity_zh.md`。下文保留 2026-09-28 实现时的状态。

## 已实现与适用范围

`EvaluationPipeline.ingest_case` 新增可选 `source_trace_sink`，调用者可持久化两类事件：

- `batch_input`：benchmark、owner、场次与日期、实际发出的role/content及日期锚、原始消息位置与批内source_ref；LoCoMo还保留dia_id、speaker、图片URL/query/caption。
- `batch_outcome`：对应batch_id、backend返回记录数量和ID；异常仅记错误类型及写入完成状态未知，不在来源事件里复制异常文字。

`source_trace.py` 以历史字段构建独立副本和内容指纹。Gold问题、答案、evidence、原始QA metadata及has_answer不参与构造；输入修改或owner不同会改变指纹，单独改变gold不改变指纹。实际历史正文当然可能包含答案事实，排除gold字段不意味着屏蔽真实源话。

batch_id是内容身份，不是独立运行或重试次数ID。调用者仍需保存运行标识、调用顺序和事件，处理并发与落盘。现有运行脚本没有自动启用sink；默认不留新的来源事件，发送给backend的请求不变。

**批次来源不等于事实支持。** 返回的记忆ID只是这个调用的返回记录，没有将每条记忆的每个子句与source_ref连起来；事件明确标记 `fact_support_status=not_provided`。没有更改提取提示、自动语义判断、图构建、记忆归并或RSI接受门控。

## 失败语义

输入记录失败时向调用者传播异常，该批backend还未调用；结果记录失败也传播，此时backend可能已经写入，调用者不可据此假定未写入并盲目重试。backend自身报错沿用场次跳过行为，并将写入完成状态标为unknown，不宣称已回滚或零写入。记录成功也不证明backend输出事实语义正确。

## 验证证据

脚本 `scripts/m2_verify_ingest_source_trace.py`；机器结果 `results/analysis/m2_ingest_source_trace_20260928.json`。

真实中文历史回放使用独立Recorder，避免初始化Mem0/Chroma或调用模型：

| 输入 | 批次 | 原始消息 |
| --- | ---: | ---: |
| 固定五道LongMemEval完整历史 | 232 | 2,368 |
| LoCoMo conv-42完整对话 | 29 | 629 |

两遍回放分别关闭/开启追踪，逐批实际请求（owner、messages、metadata）完全相同，输入/结果事件配对。LoCoMo D9:14 的图片通道仍分别保留；Gold has_answer没有复制。

`tests/test_ingest_source_trace.py` 加两个适配器测试文件共 **11 passed**。覆盖正常非空返回ID、默认请求一致、gold隔离、源/owner指纹、截断、sink修改不污染输入、backend异常未知写入状态、输入/结果sink失败不触发自动重试。Recorder返回空列表，不能把这次验证叫做真实提取或来源语义对齐成功。

## 结论、置信度和暂停点

高置信：接口默认行为、261批真实历史请求一致、消息与图片字段保留、上述失败边界。尚未证明：自动事实支持映射、纠错/更新关系识别、归并净收益、当前纯中文Graph/Jev/RSI总体效果。

近几阶段可用于决策的判断：

1. 图和Jev不能统一补救来源不足、关系丢失、匿名对象混接或判分歧义；先找有源、入库且原上下文缺失的事实，再验证净收益。
2. 归并须同时保留子句、限定语与历史值。原题分数可能看不见事实损失，原题能力与来源保全应分别验收。
3. 中文LongMemEval知识更新需要区分同场纠错、跨场更新、历史时间窗和累计区间；不能按日期统一覆盖旧值。资格/计划/已发生的gold边界也需先注明。

**按用户要求暂停持续研究。** 本次收尾后不再继续分析或运行模型。恢复时从当前批次来源账推进输出事实级对齐，优先验证同场工程师纠错、棒球历史窗、陶艺暂停活动直证及情绪限定；先说明中文样本、对照、调用上限与反伤，保持旧结果。暂停不是整体复盘完成。
