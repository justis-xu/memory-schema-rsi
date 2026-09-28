# Schema演化平均收益门控与逐题证据损失

## 问题与范围

LifecycleRSI要求零反伤，但独立EvolutionEngine用平均evidence-ID召回utility接受schema。本阶段问：历史接受步骤有没有逐题退化？平均增益能否遮住个别原有证据丢失？

范围为两份已归档tiny-fixture运行的接受步骤，加一个定向合成控制。零模型、答题、裁判、真实图写入。不是新增中文收益评测，也不是重算此前召回总体结论。

脚本 `scripts/m2_audit_evolution_case_regression.py`；首轮结果 `results/analysis/m2_evolution_case_regression_20260928.json`，新增日志验证另存 `results/analysis/m2_evolution_case_impact_logging_20260928.json`，不覆盖历史或首轮结果。

## 历史接受步骤的逐题复核

两份运行 `run-20260923T000107786191Z` 与 `run-20260923T000649010998Z`，数据SHA都与当前tiny.json相等。根据trials中的完整schema和impact的previous_schema_id，恢复每次接受修改前后train/validation selected/gold。

复算与归档的train逐题selected、train/validation平均recall与delta一致；validation逐题属于**用当前确定性代码恢复**，不是旧文件原本留有全量逐题validation日志。

每份运行两次接受，均为train一题改善/零退化、validation一题改善/零退化，也未见“召回总数不降但换掉另一个gold ID”。两份是同一个六病例合成数据，不能当两份独立真实实验，更不能证明中文不会反伤。

## 定向控制：2改善、1丢证据仍可接受

构造train/validation/test各独立owner，每组3case，固定6个Memory、相同baseline排名；候选图从第1个seed连第5个Memory，4槽内换掉原baseline第4个Memory。

- 两case需要第5个Memory，召回0→1；
- 一case需要原第4个Memory，召回1→0；
- 平均delta为+1/3，成本小于门控限额。

单候选走真实 `_search_round`、compile_graph、retrieve、evaluate_split，接受成功。validation日志中的原对照证据 `v4` 被 `v5` 换掉，明确记录1题gold-ID损失。

这是人工构造的目标权衡，不估计真实退化率；两改善case共享排名/目标，只为验证均值逻辑，不视为独立语义样本。gold-ID损失也不自动等于答题错误，可能还有同事实其他记忆，仍需事实级复核。

## 最小修改：让门控暴露代价

没有把EvolutionEngine改成零反伤策略：其目标原本是平均收益减成本，允许局部退化是该目标的真实含义，不是凭这一个控制就能决定禁止的行为。

本次仅在实验 `evolve.py` 的每个validation候选事件追加 `case_impact`：train/validation相对于**当时active schema**的改善case ID、召回退化case ID、丢失gold ID。无需额外模型或评估调用，使用既有缓存行；拒绝与接受候选都留痕。即便净召回不降但某个gold被另一个gold替代，lost_gold字段仍可见。

之前impact仍保留；没有给旧报告原地增补字段。新日志控制确认accepted事件仍接受，同时准确写入validation_harm和v4。新反伤日志测试加已有test_lab共6 passed；代码编译和diff检查通过。

## 结论、置信度、限制与下一步建议

- **高置信度，历史窄范围：** 这两份合成归档的接受转换未发现逐题退化；不证明更广场景安全。
- **高置信度，控制流与真实离线执行：** 平均utility门控可以接受原有证据丢失，单看delta/accepted不能得出零反伤。
- **高置信度，日志范围：** 新接受事件可定位具体损失，但日志仍只基于gold-ID proxy，不验证事实主体、时间、来源或答案正确性。
- **未知：** 中文Graph/RSI的局部损失频率、合理损失阈值、严格答案净收益与长期经验迁移。新字段未进入生产Evaluator，不改变候选选择目标或旧结果。

下一步：

1. 中文试验同时报告平均收益、逐题改善/退化与直接证据损失，保留回滚材料；决定零反伤、净收益或分题型风险阈值前，先核真实来源病例。
2. 不把代理训练失败定位自动当作源事实缺失：gold-ID可能有无效指针、错主体、同事实替代。后续把训练诊断中的candidate_pool_gap/channel_missing/budget_or_rank与已核源关系对应，避免学会追逐错误标签。
3. 验证后续经验收益仍需无当前金标的经验产生及独立任务；本轮只改可审计性，不能把日志完备当作RSI已有效。
