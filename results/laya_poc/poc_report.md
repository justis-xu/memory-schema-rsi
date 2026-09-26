# 提取质量门控 POC 判读报告：Laya 决策服务 × conv-42（29 session，双服务对比）

> 数据：`results/laya_poc/conv42_poc.json`（打点机器输出）；对话原文：`experiments/schema_rsi/runs/conv-42_full_frozen.json` + `data/locomo/locomo10.json`；
> 记忆库：`data/vector_store/chroma/chroma.sqlite3`（user 前缀 `full:` / `nf53:`）；附验台账：`docs/search_ledger.md`。
> 本报告只判读，不改脚本、不重跑打点。所有"机器算的"与"我判断的"分开标注。

---

## 0. 结论先行

1. **检测端：decider-2b 弱可用，laya 完整版不可用。** decider missing AUC = 0.698（bootstrap 95% CI [0.488, 0.874]，勉强擦着 0.5），正例均分 0.209 vs 负例 0.128（1.6×）；laya AUC = 0.569（CI [0.348, 0.782]，正负例均分差仅 0.008），分数挤在 0.34–0.53 窄带，与内容基本无关。门控选 decider。
2. **真值标签有系统性问题（本次抽查发现，非背景须知已知的"七成精度"能涵盖）。** 14 个正例 session 里 6 个（session_1/2/3/9/16/27）的"漏点"关键短语在被归因 session 自己的对话里**不存在**："Project Hail Mary" 全对话 0 次出现（只存在于 QA gold answer，且其 evidence 引用的 D10:19 这条 turn 根本不存在）；"A Court of Thorns and Roses" 只出现在 session_19，却被归因到 5 个 session，且 full flash 库里**已有**该书名记忆。按对话共现修正标签（正例 14→8）后：decider AUC 升到 0.795 [0.583, 0.955]，laya 仍 0.583 无效。
3. **阈值建议：decider missing P(B) ≥ 0.3。** 升级 3/29（10.3%：session_7 / session_19 / session_15），按机器标签 3 个全真漏；修正标签下这 3 个仍是正例。第一个假阳性 session_26（0.2227）要到 0.2227 以下才进入。代价是漏检 11/14（79%）——这是高精度低召回的兜底门，不是主力质检。
4. **核心判定：『风险 → glm-5.3 二审』这条链路现在不值得接入管线。** 收益端上限太低：POC 机器口径 nf53 覆盖 5/15（33%）；附验台账独立口径 conv-42/43 共 26 条丢失仅补回 3 条（12%），且已定位根因是 mem0 官方 prompt 收录范围而非模型档位（`docs/search_ledger.md:460-468`）。检测再准，二审也只能救回少数派漏点。正确顺序：先改提取 prompt 收录范围，decider@0.3 留作 prompt 修复后的低成本复验门控。
5. unsupported / conflict 无人工真值：本报告只描述分布（decider 三问分布有序、laya 三问挤在一起更像无信息摆动），**不下检测力结论**（JSON 里的 `auc_by_question` 是拿 missing 标签算的，不是这两个维度的检测力）。

---

## 1. 机器数字抽查（核验记录）

要求抽查 ≥3 个数字，实际做了 4 类、覆盖 20+ 个数字，**全部一致**：

| # | 抽查项 | JSON 原文 | 核验方式 | 结果 |
|---|---|---|---|---|
| 1 | session_1 laya missing | 0.4887（`conv42_poc.json:24`） | 与打点底稿 `[1/29] session_1 miss=1 laya P(B) missing=0.4887` 对照 | 一致 |
| 2 | session_7 三问双服务 | laya 0.3833/0.4872/0.6339；decider 0.5240/0.5120/0.2487（`conv42_poc.json:222-237`） | 与打点底稿 `[7/29]` 两行对照 | 一致 |
| 3 | AUC | laya 0.569048（`:986`）、decider 0.697619（`:1025`） | 用 Mann-Whitney（平局记 0.5）从 29 个 session 逐条分数独立重算 | 逐位一致 |
| 4 | 阈值表 10 行（双服务 × 0.3–0.7） | `:994-1059` | 按"分数 > 阈值"独立重放升级数/其中有真漏数/漏检数 | 10 行全一致 |
| 5 | 正负例均分 ×4、n=29、正例=14 | `:987-988, :1026-1027` | 独立重算 | 一致 |

底稿摘要在 `[20/29]` 处截断，session_21–29 的底稿行未逐行对照；但 JSON 内部数字已通过 AUC + 阈值表的整体重算覆盖校验。

**未能独立复现的数字（如实说明）**：payoff 块的 `nf53_覆盖数=5 / 漏提取条数=15`（`conv42_poc.json:1062-1065`）是机器记录值；per-session/per-item 的覆盖明细不在 JSON 里，我无法按同口径复算。旁证见 §5。

---

## 2. 双服务对比表

| 指标 | laya 完整版 | decider-2b | 判读 |
|---|---|---|---|
| missing AUC（机器=重算，一致） | 0.569 | 0.698 | decider 胜 |
| bootstrap 95% CI（我算，2 万次重采样） | [0.348, 0.782] | [0.488, 0.874] | 两者 CI 都还碰 0.5，小样本 |
| 正例 / 负例均分 | 0.442 / 0.434（差 **0.008**） | 0.209 / 0.128（差 0.081，1.63×） | laya 正负例是同一分布 |
| 分数范围 | 0.341–0.534（窄带） | 0.066–0.524 | laya 无信息先验的证据 |
| missing 选项分布 | 29 个里 28 个选 A（唯一 B 是假警报 session_13） | 28 个选 A（唯一 B 是真漏 session_7） | decider 的唯一"B"打对了 |
| top3 分数是否全为真漏 | 否（第 1 名 session_13 两边标签+审计都判负） | 是（session_7/19/15，修正标签下仍全正） | decider 胜 |
| 修正标签后 AUC（我算，正例 14→8） | 0.583 [0.344, 0.810] | **0.795 [0.583, 0.955]** | decider 唯一 CI 不含 0.5 的组合 |
| 结论 | **不可用作门控** | **弱可用，选它** | |

附注：`auc_by_question`（laya unsupported 0.624 / conflict 0.543；decider 0.640 / 0.626）是**对着 missing 标签**算的相关性，不是 unsupported/conflict 的检测力，见 §6。

---

## 3. 阈值表（机器原表 + 判读）

**laya**（机器值，已重放核验）：

| 阈值 | 升级数 | 其中有真漏 | 漏检 |
|---|---|---|---|
| 0.3 | 29 | 14 | 0 |
| 0.4 | 24 | 12 | 2 |
| 0.5 | 1 | 0 | 14 |
| 0.6 / 0.7 | 0 | 0 | 14 |

判读：无可用工作点。0.4 档升级 24/29（83%），门控等于不存在；0.5 档唯一被升级的 session_13 恰是假警报（§7 样例 2）。

**decider**（机器值，已重放核验）：

| 阈值 | 升级数 | 其中有真漏 | 漏检 |
|---|---|---|---|
| **0.3** | **3** | **3** | **11** |
| 0.4 | 2 | 2 | 12 |
| 0.5 | 1 | 1 | 13 |
| 0.6 / 0.7 | 0 | 0 | 14 |

判读（我判断的部分）：
- **建议 0.3**。0.30–0.35 之间结果相同（第 4 名 session_24 = 0.2398）；再往下到 0.23–0.2398 会带上 session_24（真漏，4/4）；第一个假阳性 session_26 = 0.2227。格点之间别过拟合，取 0.3。
- 升级的 3 个（session_7 / session_19 / session_15）在机器标签和修正标签下都是真漏，"3/3 全中"这句两头都成立（我核过）。
- 漏检 11/14 = 79% 必须直说：这个门是"拦得住最响的警报"的兜底，大多数漏提取 session 会被放行。

---

## 4. 真值标签审计（抽查发现，机器数字之外我的核查）

15 条不同漏点被归因成 25 个 session 标记，两条最大扇出项占 10/25：

| 漏点 | 归因 session 数 | 对话共现核查结果 |
|---|---|---|
| "Project Hail Mary" book | 5（s2/9/10/19/27） | **全对话 0 次出现**。原始数据集 `locomo10.json` 里该串只在 1 条 QA gold answer 里；其 evidence 引用的 `D10:19` 这条 turn 不存在。任何提取器都不可能从对话提取它 → 5 个正例全部站不住 |
| 'A Court of Thorns and Roses'（只给了 'a fantasy book series'） | 5（s1/3/15/19/23） | 书名只在 session_19（D19:15 图片说明）出现；s1/3/15/23 自己的 turn 里没有。且 **full flash 库已有** "Nate plans to read 'A Court of Thorns and Roses'…"（chroma `full:locomo:conv-42`，192 条记忆中查得）→ 4 个正例站不住，s19 保留 |
| hanging memories on a corkboard | 2（s15/27） | s15 ✓（D15:9 "a pic of my cork board"）；s27 的 D27:34 是笔记本不是 cork board → s27 站不住 |
| unverified extra entities（'a major company…'）/ unsupported 'romcoms' | 3（s2/16/9） | 这是 unsupported 类投诉，不是漏提取；且 'major company/production company' 记忆在 full 库里存在、'dramas and romcoms' 在 D1:14 对话原文里就有 → 不构成 missing 正例依据 |

按"漏点关键短语须出现在被归因 session 自己的 turn 里"修正：**正例 14 → 8**（翻负：session_1/2/3/9/16/27）。修正后 decider AUC 0.698 → 0.795，laya 0.569 → 0.583。

含义：
- 机器 AUC 是对着含噪标签算的**下界**，decider 的真实区分力可能比 0.698 好；
- 背景须知称真值"精度约七成"，本次 session 级抽查实际约 8/14 ≈ 57%，且误差方向系统性地把"对话里根本没提该事实的 session"误标为正——这类标签任何诚实的打分器都不可能命中，decider 给 session_1 打 0.199 被记成假阴性，**错的是标签不是分**（§7 样例 3）。

---

## 5. nf53 收益与链路判定（核心问题，必须回答）

**收益端数字（三源）：**
1. POC 机器记录（`conv42_poc.json:1062-1065`）：15 条漏点，nf53 覆盖 5 条 = **33% 上限**。无 per-session/per-item 明细，无法核对升级的 3 个 session 里二审能救几条（如实说明）。
2. 附验台账（`docs/search_ledger.md:460-468`，我读过原文）：conv-42/43 全部 26 条丢失条目，glm-5.3 非 flash 重灌**仅补回 3 条（12%）**；两档模型行为一致，补回的恰是 3 条硬实体类；结论"提取丢失的根因是 mem0 官方 prompt 的收录范围（7 类个人硬信息 + 具名 few-shot），与模型档位无关"。
3. 我在 chroma 库直接核了 3 个点：PHM 在 full、nf53 两库均不存在（52MB 库全文 0 命中——它不在对话里，无从提取）；ACOTAR 与 cork board 在两库**都已有**。两个收益口径（5/15 vs 3/26）不一致但方向一致：**强模型二审只能救回 12%–33% 的漏点**。

**链路判定（我判断）：不值得现在接入。**
- 检测端只有 decider 勉强可用，且召回 3/14 = 21%；
- 收益端上限 ≤1/3，而根因（prompt 收录范围）不修，二审就是在"提取范围天生不含引语/评价类"的约束下空转——台账已证明换档位只影响边角；
- 接入成本（每 session 一次决策调用 + ~10% session 一次 glm-5.3 二审）买到的期望补回量太小。
- **正确顺序**：先做提取 prompt 收录范围/两轮提取的修复（台账方向 1），修复后把 decider@0.3 作为低成本兜底门控复验——届时正例标签务必换成"漏点关键短语在被归因 session 对话内共现"口径，否则 AUC 和升级精度都会被标签噪声压低/虚高。

---

## 6. unsupported / conflict：只描述分布，不下结论

- 无人工真值；`auc_by_question` 对的是 missing 标签，只能当"与漏提取标签的相关性"旁证。
- **decider**：unsupported 0.079–0.512（均值 0.158），conflict 0.018–0.249（均值 0.081）——conflict 系统性低于 unsupported，符合"冲突应最少见"的先验；unsupported 高分段（0.38–0.51）集中在 session_7/15/19，恰是有人工标注抱怨的 session，方向上说得通（旁证，非结论）。
- **laya**：unsupported 0.333–0.487（均值 0.408）、conflict 0.238–0.634（均值 0.360）——三问分数彼此接近、整体偏高，与 missing 的 0.34–0.53 窄带一起，更像对一切输入输出"中等偏疑"的无信息先验。session_7 conflict = 0.634 是全场唯一大 outlier（该 session 恰有真漏），但单点不成结论。
- 两服务分数量化明显：decider missing 0.1172×3、0.1444×3、0.0905×3，session_14 三问同分 0.0905；laya 也有 0.4168×2 等重复。分辨率有限，阈值不要卡在重复值密集区。

---

## 7. 真实样例（3 个）

### 样例 1｜session_7：decider 全场最高分，真漏坐实，laya 打错维度
- 三问：laya missing 0.3833 / unsupported 0.4872 / **conflict 0.6339**（选项 A/A/B）；decider **missing 0.5240** / unsupported 0.5120 / conflict 0.2487（选项 B/B/A）。
- 漏点：the 'like him' comparison。对话原文（D7:5，Nate）：
  > "Thanks Jo! I picked this color because it's bright and bold — **like me**! I wanted to stand out from the regular options."
- full 库紫发记忆（chroma 实查）："Nate dyed his hair purple in early April 2022, choosing the bright, bold color because he wanted to stand out from the regular options…"——"like me" 的自比确实没收录；nf53 版本 "…to stand out and match his personality" 是改写，也没逐字收录。
- 判读：decider 把 missing 打到全场最高且唯一选 B，对了；laya 却把 conflict 打到 0.63（全场唯一 conflict 选 B）、missing 只有 0.38（低于其均值），头号警报打错了维度。

### 样例 2｜session_13：laya 全场最高分，两边都认定没漏（假警报）
- 三问：laya **missing 0.5338** / unsupported 0.3848 / conflict 0.3325（missing 选 B，laya 全场唯一）；decider 0.1505 / 0.1567 / 0.0945（全 A）。
- 标签：has_known_miss = false；§4 的对话共现审计也判负。
- session_13 内容（D13:1–23）：Nate 遛狗认识养狗夫妇、约 doggy playdates，送 Joanna 毛绒玩具，Joanna 回忆写剧本的苦与甜——闲聊型 session，无具体实体漏点。
- 判读：laya 的头号 missing 警报指向一个机器标签、修正标签、decider 三方都认为没漏的 session；这是"laya 分数与内容基本无关"的直接证据。

### 样例 3｜session_1：被标为正例，但漏点根本不在它的对话里（标签反例）
- 标签：has_known_miss = true，miss_item = "specific book title 'A Court of Thorns and Roses'"；decider missing = 0.1987（被记成假阴性）。
- session_1 的 22 个 turn（D1:1–D1:22：Nate 拿下 CS:GO 首个电竞冠军、Joanna 推荐《暖暖内含光》Eternal Sunshine）**从头到尾没有这个书名**；书名只在 session_19 的图片说明（D19:15）出现，且 full 库已提取 "Nate plans to read 'A Court of Thorns and Roses' by Sarah J. Maas…"。
- 判读：关键词归因把一个对话级事实扇出到 5 个 session，制造了至少 4 个假正例；decider 给低分是**正确行为**，被含噪标签惩罚。

---

## 8. 告诫清单

1. **小样本**：29 session / 14 正例（修正后仅 8）。AUC bootstrap CI 很宽（decider 原标签 [0.488, 0.874] 仍含 0.5），所有阈值与精度数字都不具备统计稳定性，只配当方向参考。
2. **真值标签噪声**：14 正例中 6 个站不住（PHM 全对话不存在、ACOTAR 错误扇出、库内已有）；session 级精度实测约 57%，低于背景须知宣称的"约七成"。凡与标签挂钩的数字（AUC、其中有真漏数、正负例均分差）都继承此噪声。
3. **收益口径未定**：nf53 覆盖 5/15（POC）vs 3/26（台账）两口径不一致；升级的 3 个 session 里二审能救回几条，本次 POC 数据无法回答。
4. **单一对话、单一语言、单一题材**：LoCoMo 英文双人闲聊（Nate/Joanna），无跨对话、跨语言（中文）、长 session 分布外的证据。
5. **服务条件不可控**：两服务均走公网端点（`endpoints_used.mode = public`），采样温度未知；分数量化/重复明显，阈值不应卡在重复值上。
6. **unsupported / conflict 无真值**：本报告未对这两个维度下检测力结论；`auc_by_question` 不是它们的检测力。
7. **底稿截断**：打点摘要停在 [20/29]，session_21–29 底稿行未逐行对照（JSON 内部一致性已由 AUC/阈值表整体重算覆盖）。
8. **AUC 平局约定**：我的重算按 Mann-Whitney 平局记 0.5，与机器值逐位一致，说明机器同约定；若改约定，量化严重的服务（decider）AUC 会变动。
