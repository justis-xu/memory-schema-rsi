# 争议题登记表与不可恢复项台账

> 2026-10-01。零模型调用。脚本 `scripts/m2_build_dispute_registry.py`，机器账 `results/analysis/m2_dispute_registry_20261001.json`。把散在 9/26–9/29 各专项文档的争议题与不可复原料整合为机器可读资产，供后续所有配对实验统一消费。

## 内容

- **争议题 21 题**，四类：gold evidence 不可达 9（结构审计机器账为源）、gold 语义错接 2、翻译语义偏移 6、翻分归因歧义 4。每条带来源文档指针与预注册处理规则。
- **缺题界限 15 组**：从 `m2_missing_case_score_bounds_20260929.json` 透传（组名、缺题数、符号稳健性），不重算。
- **不可恢复项 7 条固定台账**：旧运行缺题失败日志、Jev 首判完整上下文、历史提取完整日志、mem0 返回层字段缺口、旧缓存指纹缺失、跨运行缓存当时请求、Jev 服务失效期。每条含"为什么不可恢复 + 固定 action"。

## 使用规则（预注册）

1. 后续配对选题默认排除登记表内全部争议题；确需使用必须按 rule 预注册处理并在结果中标注。
2. 台账项不得反复尝试复原；新增运行必须自带失败记账与留痕（source_trace / jev_trace 契约）。

## 结论与置信度

清单整合自既有审计（逐题可溯源），高置信；不产生新统计。这是[完成度账](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_retrospective_completion_zh.md)第 1 项"可复原性补齐"中"不可恢复项列明"部分的落地。
