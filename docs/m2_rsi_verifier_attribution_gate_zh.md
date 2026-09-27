# RSI 验证器接受门槛：没有可定位支持 ID 时不提交修复

> 2026-09-27。实验代码 `experiments/schema_rsi/schema_rsi_lab/verify.py` 的最小修复；回归测试 `experiments/schema_rsi/tests/test_verify_support_gate.py`；只读归档审计 `scripts/m2_audit_rsi_verifier_attribution.py` 与 `results/analysis/m2_rsi_verifier_attribution_20260927.json`。零模型调用，未改生产 Mem0/Graph/Jev/答题管线。

## 问题与范围

此前中文 RSI 来源候选的 20 题探针表明：相邻 turn 能补到梅拉妮回答“画画”，也会把黛博拉妈妈朋友、卡罗琳项链的事实带进**询问另一个人**的题。旧 `audited_turn_ids` 只说明某 turn 入槽，不能作为经验接受凭据。本阶段进一步检查实验 RSI 的修复提交门槛：验证器声称答案“有证据”时，代码是否强制留下**可定位到上下文的支持 ID**？

当前 `verify_answer` 提示要求返回 `support_indices`，代码会把有效编号映射成记忆或来源 turn ID。但修复前，若验证器 JSON 写 `status=supported`、编号为空/越界/不可解析，解析后的 `support_ids` 为空，`repair_case` 仍仅凭状态将候选答案设为 `accepted=True`。用一个确定性模拟复现了“首判 contradicted、重答 supported 但编号 999”的路径：修复前接受且支持 ID 为 `[]`。这是**代码控制流漏洞**，并非当前模型输出这种坏格式的发生率。

## 归档观察与修改

英语 `conv-26` 的旧定向修复归档 `conv26_verify_repair_final.jsonl` 有 15 题、2 个 accepted（`qa133/qa140`）；优化版两题也均 accepted。两份旧记录的 `second_verdict` 只有状态、理由、检索词，没存 `support_indices` 或 `support_ids`。它们形成于支持编号契约之前，**不能因缺字段否定答案正确**；但历史归档无法证明这两个提交关联到哪条可核源话，也无法按 ID 审支持归属。

本次仅改实验验证器：

1. `supported` 若没有至少一个有效上下文编号，转为 `uncertain`，不伪装有据。
2. 最终接受分支再要求 `second.support_ids` 非空，作为防御性约束；有效编号照旧映射和接受。

确定性探针中，空列表、越界 `999`、非数字编号均变 `uncertain`；有效 `1` 仍返回 `supported` 和记忆 ID。`repair_case` 的同上下文重答模拟也确认：无有效编号保持原答案，编号有效则提交新答案。`.venv` 中 9 项相关测试通过。代码只触及 `experiments/schema_rsi`，没有据此重算任何历史分数。

## 结论、置信度与下一步优化建议

**结论：** 实验 RSI 曾允许“验证器自称 supported 但没有可定位支持 ID”的修复答案通过内部接受门槛；现已堵住这个确定性的无归属路径。它提高**可审计性约束**，还不是事实级来源校验：有效 ID 可能只是话题相近、属于错误人物、状态不符，甚至是未获原始 turn 支持的压缩记忆。旧英语正例与中文 20 题都不能证明此改动带来答题净收益。

**置信度：** 旧代码控制流、确定性复现、新路径测试和归档字段为高；旧 `qa133/qa140` 真实支持关系、当前验证器缺编号的发生率及拒绝后的答题影响未知。拒绝坏格式可能也误拒本来正确但漏填编号的修复；需真实配对验证。

**下一步：** 在中文候选的下一层验收里，对支持 ID 指向的源 turn 核人物、谓词、时间与已发生/计划/获得状态；用 `conv-26_qa111` 的真正回答与 `conv-48_qa233`、`conv-26_qa157` 的错人诱饵作为起始回归，记录误拒与误受。独立结果反馈、验证器 `supported`、可定位支持 ID 和有源关系应是不同字段，不可由一个布尔 `correct` 代替。只有门槛通过后再做固定 15 槽中文答案对照及反伤检查。
