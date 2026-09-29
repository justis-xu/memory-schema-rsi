# 中文 LoCoMo 与 LongMemEval 的 `assistant` 不是同一种来源

> 2026-09-29。脚本 `scripts/m2_audit_role_semantics.py`；机器账 `results/analysis/m2_cross_benchmark_role_semantics_20260929.json`。只读本机当前 Chroma、中文 LoCoMo 原文件与前一段冻结的中文 LongMemEval 来源 trace；零模型、embedding、答题或图调用，无库写入。当前文件 SHA 已记录，计数不冒充历史冻结版本。

## 为什么要分开

[真实隔离写入](m2_isolated_mem0_ingest_zh.md)发现 LongMemEval 工程师题中，助理说出的萨曼莎职位、套餐报价和活动建议被写进用户记忆；[提取契约审计](m2_mem0_extraction_contract_zh.md)又发现底层 `attributed_to` 标签在项目包装层未透出。一个看似直接的门控是过滤 `assistant`，但它跨基准会误删真人事实。

`LocomoDataset` 在 `src/schema_rsi/benchmarks/locomo.py` 把命名的 `speaker_a` 映射成 `role=user`，把命名的 `speaker_b` 映射成 `role=assistant`。这两位都是 LoCoMo 语料中的对话人物；原 `speaker` 和 `dia_id` 虽保留在 `BenchmarkCase` turn 中，传给 Mem0 的实际 `messages` 只含 `role/content`。LongMemEval 的输入则直接沿用 `user/assistant` 消息角色，其中 `assistant` 是助理回复。相同字符串在两个基准里指向不同来源类型。

## 当前快照与逐条见证

本机 Chroma 共 1,995 条记录，其中十个 `zhfull:locomo:*` owner 为 **1,800 条**。这 1,800 条的原始 `attributed_to` 元数据为：`user` **960**、`assistant` **689**、缺失 **151**。这些只是模型给记忆的标签分布，不能当作 689 条都正确来自真人第二说话者的准确率；更不能拿它推算 LongMemEval 污染率。

一个可核反例足以否定跨基准整类过滤：`conv-43` 的 `speaker_a=蒂姆`、`speaker_b=约翰`。当前记忆 `7d5bc1cd-026d-4baf-b6cf-007667a72128` 标为 `attributed_to=assistant`，正文写约翰与明尼苏达狼队签约、任得分后卫。中文源 `D1:3/5/7` 均由**约翰本人**说出相关事实；`conv-43_qa71/qa72` 分别问球队和位置，指定证据为 `D1:5` 与 `D1:7`。这里的 `assistant` 是 `speaker_b` 映射，不是机器编造的建议。该记忆是有用的事实载体候选；本次未跑检索和答题，不能称它是两题的唯一载体或删掉必错。

与之对照，隔离 LongMemEval 工程师场次中的 `assistant` turn 是助理给出的场地报价、联系人职位和活动安排；用户后来只复述“联系萨曼莎询问套餐”，没有确认全部商业细节。其实际提取 JSON 把五条都标为 `user`，其中三条逐子句仅获 partial 支持。因而**即便在 LongMemEval 内，按模型自报 `attributed_to=user` 放行也挡不住角色混写**；对助理 turn 整条禁用又可能损失用户后来明确接受的内容，应看支持原话与采纳关系。

## 结论、置信度、下一步优化建议

**结论：** `assistant` 在中文 LoCoMo 是第二位真人的话轮标签，在中文 LongMemEval 是助理回复标签。跨基准用一个 `assistant` 过滤器会删去至少上述一条真人自述记忆；直接信 `attributed_to=user` 又放过已观测的助理事实升格。来源类型必须先由基准/会话结构确定，再把 `speaker`、原话 turn、内容渠道与模型声明分列。

**置信度：** 适配器映射、当前 1,800 条标签计数、指定记忆 ID/原话/QA evidence 和 LongMemEval 冻结 trace 均为直接可复核，置信度高；这一个见证不估全库标签准确率、筛除损害率或最终答题变化。

**下一步优化建议：** 在来源账中引入与 `role` 分离的 `source_kind`（真人对话人物、AI 助理回复、图片描述等）和 `speaker_name`，并记录 `dia_id` 或批次内 turn；LoCoMo 按命名人物核事实，LongMemEval 对用户确认与助理建议分别标支持强度。把任何统一按 `assistant` 或统一按 `user` 放行的规则先拿两个基准的固定见证做反伤验收。图候选和 RSI 经验只能消费已核来源事实；标签透传本身不等于来源可信或答案收益。
