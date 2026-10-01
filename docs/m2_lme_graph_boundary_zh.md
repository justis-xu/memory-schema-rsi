# LongMemEval 中文侧图适用边界（收缩记录）

> 2026-10-01。零模型调用。按[阶段计划](m2_independent_graph_eval_zh.md)的收缩条件执行："若阶段 2 图结论为负且病例不足，收缩为 LME 图适用边界记录"。结论输入：[独立图配对评测](m2_independent_graph_eval_zh.md)（图净收益病例不足、0 换入必要事实）+ 9/30 索引"中文 LongMemEval-S 仅 470/500 且共同题缺背景场"。

## 现状

- LME 中文侧历史上**零图写入**（`m2_lme_*` 七份报告仅提及该边界）；本轮也未建 LME 图。
- 图基础设施已就绪且可复用：`LocalGraphStore`（检索契约与 HugeGraphStore 一致）+ 带来源准入的建图管线（`m2_build_independent_graph_zh.py`）+ 冻结配对评测框架（`m2_eval_independent_graph_pairs.py`）。把 LME 会话接入该管线是工程问题，不是能力缺口。
- **不存在 LME 侧的 A 组候选**：没有"有源、独有、当前缺失必要事实"的 LME 图收益病例清单（LoCoMo 侧 57 正例筛后也只剩 1 例且被人工核为噪声）。

## 边界结论

在 LoCoMo 受控负结果与 LME 零候选病例的双重条件下，跑 LME 图配对是"为解法找问题"。**LME 图实验的启动条件**预注册如下，满足才值得投入：

1. 先以逐事实源证筛出 ≥3 道"当前无图检索缺必要事实"的 LME 定向题（复用 `m2_lme_clause_support` 的子句 oracle 方法）；
2. 图构建须通过来源门 v3 准入（待核比例可控）；
3. 沿用本轮全部冻结纪律（争议登记表排除、ABBA×4、must-not-evict 反伤题、逐答人工核）。

在此之前，LME 侧维持零图写入边界；知识更新类病例的正确起点仍是[输出事实对齐](m2_lme_ingest_order_zh.md)指出的缺口，不是图。
