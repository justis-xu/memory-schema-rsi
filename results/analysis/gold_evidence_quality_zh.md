# LoCoMo 金标 evidence ID：结构质量与 RSI 代理指标影响

2026-09-26。复算脚本 `scripts/audit_gold_evidence_ids.py`，机器结果 `gold_evidence_id_audit_20260926.json`。全量对齐英文原版与中文翻译的 1,986 道 QA、两版源 turn ID；另外检查 `experiments/schema_rsi/runs/conv-*_full_frozen.json` 十个冻结快照。这里仅验证**ID 结构和可达性**，不能自动证明语义支持。

## 全量结构结果

| 指标 | 数量 |
|---|---:|
| QA / evidence 列表项 / 从列表项解析出的引用 | 1,986 / 2,815 / 2,823 |
| 英文与中文 evidence 列表不同 | 0 |
| evidence 为空的 QA | 4 |
| 一个列表项塞多个 dia_id | 4 |
| 原样 ID 写法不规范但可归一化 | 2 |
| 解析不出 dia_id 的列表项 | 1 |
| 归一化后仍找不到源 turn 的引用 | 2 |

明确缺失的是 `conv-42_qa58` 的 `D10:19`（session_10 只有 16 个 turn）和 `conv-47_qa38` 的 `D4:36`（session_4 只有 25 个 turn）。`conv-42_qa88` 的单独 `D` 无法解析。`D:11:26` 可以规范化成 `D11:26`，`D30:05` 可以规范化成 `D30:5`。四个复合列表项例如 `D8:6; D9:17` 都能拆出有效 ID，但原始字符串本身不是 turn ID。

## 对已落盘 RSI 源 turn 覆盖分数的影响

十个冻结快照的 **1,536** 道非对抗且有 evidence 的题中，**9 道**至少包含一个与任何源 turn ID 都不相等的原始 evidence 项：conv-26_qa37、conv-42_qa58/88、conv-43_qa18、conv-47_qa38、conv-49_qa31/38/46、conv-50_qa69。`snapshot_conv26.py` 将 `case.evidence` 原样复制，`analyze_turn_graph.py` 等脚本再拼成 `T:user|<原样字符串>` 当目标节点。因此 9 道在现有指标定义下**不可能达到“全部金标 turn 覆盖”**；其 `all_locomo_turn_graph.json` 的 keyword/graph 全覆盖也均为 false。

这不推翻 623→557、得 10 失 76 的主要方向；9/1,536 的分母影响很小，且两臂都受同一不可能目标约束。但它说明绝对覆盖率不应当作真实证据可达率，特别是复合 ID 还把两个或四个 gold turn 错计为一个 `gold_count`。如果重做 RSI 的纯离线代理评测，应先把复合项拆分、规范化能定位的 ID；两个真正不存在的 ID 与单独 `D` 要人工核源后排除或修正，**不能猜出不存在 turn 的替代 ID**。原归档保持不动，修复前后都要留指标版本。

结构有效也不代表语义有效。时间题 `conv-43_qa56` 的 `D21:13` 的确存在，但上文 D21:12 问的是**钢琴**学了多久，金标却把“四个月”用于**小提琴**；`conv-43_qa60` 的金标 `D23:2` 是文学课对话，真正说“上周五创助攻纪录”的是 D23:3。若用这些 gold evidence 训练“成功经验”或构造 oracle 超边，错误关系会被固化。语义核查应在经验准入前发生，不能以 ID 存在代替支持事实核验。
