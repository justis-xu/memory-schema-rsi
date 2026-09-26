# 中文轨道台账（locomo-zh）

中文版数据集：`/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json`
（与英文版逐条对齐：同 sample_id / QA 数 1986 / category / evidence dia_id，日期保留英文
格式，正文全部简体中文；翻译模型 glm-5.3-flash + 284 条术语表，结构校验 0 错）。

## 轨道隔离（2026-09-25 建立）

同仓库、独立轨道，英文轨道全部资产原地不动：

| 项 | 英文轨道 | 中文轨道 |
|---|---|---|
| 配置 | config/default.yaml | **config/locomo_zh.yaml** |
| 数据 | data/locomo/locomo10.json | locomo-zh/locomo10_zh.json（绝对路径引用） |
| 向量库 | data/vector_store/chroma（schema_rsi_memories） | data/vector_store_zh/（**schema_rsi_memories_zh**） |
| user 前缀 | full:locomo:{conv} | **zhfull**:locomo:{conv} |
| 结果目录 | results/ | **results_zh/** |
| 图缓存 | data/graph/ | data/graph_zh/ |

跑法（等英文轨道终测全部结束后再启动，避免抢 API 配额）：

    .venv/bin/python scripts/eval_full.py --config config/locomo_zh.yaml --prefix zhfull
    # 断点续跑：加 --resume results_zh/zhfull_locomo_<stamp>.jsonl
    # 精准 regrade：scripts/regrade_precision.py 对 results_zh/ 的产物（判分 glm-5.3 非 flash）

## 从英文轨道继承的结论（50 轮 + 终审，详见 search_ledger.md）

1. 方法学：跨夜数字不可比，必须同夜配对 + 复现闸门 + 保留 test 对话；夜间漂移 ±15-20 题
2. rerank 必须保留（关闭 -22，两次验证）；其余检索/answer 配置轴在英文上全噪声——
   中文上是否复现待验，不要直接照抄"有效"，只照抄"方法"
3. 缺失事实四级归因框架（verify_extraction_tier.py split）+ 人工校准可直接复用，
   但**中文分词要重做**（现按空格切词，中文整句成一个 token，全部失效）
4. 提取层：换强模型无效（附验一）；丢失集中在引语/感受/清单成员——官方 prompt 收录
   范围问题，中文同样适用；原子化提取（Wave 7，+26 复现 +10）是已验证的修法
5. 判分：宽松口径虚高 ~25pp，一律以 glm-5.3 非 flash 严格 regrade 为准；adversarial
   单列看编造率

## 已知中文差异（冒烟发现，2026-09-25）

- token-F1 / contains 对中文失效（空格分词）：汇总只看 LLM judge
- 提取的日期锚正常（"2023年5月7日"入记忆）；冒烟样例：zhsmoke 3 题 1 完全对，
  2 题语义对但与 gold 表述不同（judge 是否判 exact 待正式跑验证）
- judge/answer prompt 仍为官方英文预设（中文内容直接填入）：基线先保持一致口径，
  后续如需中文化 prompt 需另开对照轮，不可与英文预设混比

## 基线记录

zhfull 全量基线（2026-09-25：mem0 提取 + 向量检索 + rerank，无图，官方 prompt 预设；
宽松 judge flash 档，精准 regrade glm-5.3 非 flash；1986 题落盘 1982，缺 4 题为跑批丢弃）：

| 口径 | 中文 zhfull | 英文 A 臂 | 判读（噪声带 ±15-20 题 ≈ ±1.0-1.3pp） |
|---|---|---|---|
| 宽松 J（类 1-4） | 91.87%（1413/1538） | 92.72%（1427/1539） | -0.85pp，带内 |
| 精准 exact J | 64.69%（995/1538） | 66.41%（1022/1539） | -1.72pp（-27 题），略超带上沿 |
| single-hop exact | 75.69% | 76.69% | 带内 |
| multi-hop exact | 54.96% | 58.16% | -9 题，带内 |
| temporal exact | 53.58% | 54.06% | 带内；日期规则未拖累（见下） |
| open-domain exact | 34.38% | 41.67% | -7 题，基数小带内 |
| adversarial 干净拒答 / 编造 | 45/444（10.1%）/ 373（84.0%） | 70/446（15.7%）/ 348（78.0%） | **编造 +6pp（约 +25 题）超带，唯一明确差异，待同夜配对消融定因** |

方法论备注：summary 与逐题重算全数一致；确定性层两个缺陷（数字与"年"间空格致日粒度解析
失败 ×1、season 自匹配恒 False ×3 且中英同病）零分损、只压低 det_vs_judge（0.7129 vs 英
0.7518），跨格式日期（May 7, 2023 ↔ 2023年5月7日）结构化匹配实测有效；token/hedge 规则中文
失效（det 决定性覆盖 11.4% vs 英 64.6%），汇总只看 LLM judge。详见 results_zh/baseline_report_zh.md。

## 提取门控 POC 中文复验（2026-09-25，conv-42 双臂闭环，工作流跑批）

沿用英文 POC 的三问 choice 框架（missing/unsupported/conflict 取 P(B)），中文全链路闭环：
zhfull（flash）提取 + nf53（glm-5.3 非 flash）对照提取 + glm-5.3 逐 session 审计造真值
（内置短语共现修正规则，抽查 7/7 成立，英文的标签扇出灾难未复发）+ 双服务打点。
产物：`scripts/zh_session_audit.py`（审计器可复用）、`results/laya_poc_zh/`（含
conv42_zh_poc.json 与 poc_report_zh.md）。

- **decider-2b@0.3 可作低成本兜底门**：升级 3/29（session_24/5/29），第 3 名 0.349 与
  第 4 名 0.277 之间断档、0.3 落中间，与英文工作点同阈值同升级量；唯一干净 session
  被 decider 按到全场最低（0.063）
- **laya 完整版中文反向**：AUC 0.286（英文 0.569），把唯一干净 session 抬到正例中位数
  之上（0.5302）；冒烟发现中文输入自动切 multilingual checkpoint，仍不可用
- **最大告诫：28 正/1 负**，AUC 0.982 不可外推（由 session_16 一个点的排位决定）；
  基础率 96.6% 下"3/3 全中"证明力弱（随机全中 89.7%）——可信的是排序事实与断档位置
- **payoff 中文闭环**：103 条审计漏点 nf53 覆盖 16（15.5%，英文区间 12-33% 低端）；
  升级的 3 个 session 共 11 条漏点仅 2 条可救回——门控全对期望也只补 2 条记忆
- **根因跨语言成立**：救回的是硬实体（蜘蛛侠胸针/毛绒小狗），救不回的是评价/情感类
  （"意义重大""很乐意帮忙"）——先修提取 prompt 收录范围，门控不替代修复
- 待查信号：中文 flash 提取 28/29 session 审计均有漏（英文修复标签后 8/29）——或审计
  更敏感、或中文提取更漏，待 zh 基线全量后用同口径复核
- 观察：审计运行时 `zhfull:locomo:conv-42` 已存在 193 条记忆（本工作流未灌，疑另一
  窗口已开始中文 ingest——双窗口操作前先对齐状态）

## 图臂配对结果（2026-09-25 夜，同夜背靠背双臂全量）

设计：A=无图基线（zhfull）vs B=s2+v8cluster 融合图臂（节点中心聚合召回进统一 rerank，
Laya/Jev 全关），同夜背靠背各跑全量，对照英文终审检验"图无增益"是否复现。
数据：`results/precision_regrade_base/graph.jsonl` + `precision_summary_20260925_175715.json`。

- **复现英文结论：图/schema 层无增益。** 精准 exact 982→999（63.77%→64.87%），
  **Δ+17（+1.10pp），McNemar p=0.2183 不显著**；169 题（11.0%）exact 状态双向互翻
  （基线独得 76 / 图臂独得 93），散布≫净值，与英文"~15% 互翻、净值≈0（Δ+7，p=0.64）"同形态
- 分对话 Δ 从 −5（conv-30）到 +11（conv-26），5 正 4 负 1 零（符号检验 p=1.0）——
  conv-26 一对话的 +11 就占净值三分之二，无跨夜可预期方向
- **temporal 优势不迁移**：英文唯一方向性信号 +3.2pp 在中文为 +0.31pp（p=1.0）；
  中文自己的方向性正信号换成 open-domain +4.17pp（p=0.481）与 multi-hop +2.13pp（p=0.405），
  照英文先例只标**待同夜复验**（英文 50 轮正效应伪影率 100%）
- **adversarial 同向略差**：干净拒答 56→43（−3.0pp，p=0.160），与英文 70→61 同向，
  图臂多编造少拒答，是唯一两轨一致的负方向
- 硬证据抽查：93 个图臂独得 exact 里，金标事实"只在图候选、不在基线召回池"的仅 1 例
  （conv-43_qa25，`entity:signed basketball` 桥接出约翰的签名篮球）——翻正主力是
  rerank/生成扰动而非图召回；反例 conv-41_qa86 是挤出型伤害（2 条 concept 边图候选
  挤掉"车抛锚"事实致 wrong）。融合臂 69.2%（1371/1980）的题有图候选实际进上下文
- 汇总文件 adversarial transitions 缺 2 个非零格（exact→partial 1、partial→exact 2，
  7 格和 430≠n=433）；逐题重算补齐后守恒，统计量本身无误。judge=null 判 wrong：
  基线 357 / 图臂 374 条，两臂对称，翻转样本中 18/169 涉及
- 详见 `results_zh/graph_arm_report_zh.md`（配对表/分对话表/英文对比表/样例/告诫）

## Jev 停止准则双服务结果（2026-09-25 深夜，三臂同夜背靠背全量）

设计：control=纯融合臂（s2+v8cluster，无决策服务）vs jevdec=融合+Jev 停止
（decider-2b 驱动）vs jevlaya=融合+Jev 停止（laya-rl-agent 驱动），机制为
p(证据充足)<0.4 → 聚合宽度加倍一次性扩拉+二次评分（pipeline.py:237-254）。
数据：三份 precision_regrade_*.jsonl + summary_20260925_212001.json（其 paired 节
只有一对且实为 jevdec↔jevlaya，fused 两对由逐题重算补齐，守恒抽查全过）。

- **Jev 停止准则在中文无增益，两臂都是**：对 control 精准 exact（J 类 n=1539）
  jevdec **Δ−2（995→993，p=0.9396）**、jevlaya **Δ−5（995→990，p=0.7723）**，
  均噪声带内微负；宽松口径同样 −9/−7。这是"无差异"，不是"被噪声覆盖的方向性"
  （唯一例外 multi-hop 两臂一致 +9/+12，p=0.211/0.134，待同夜复验；temporal 一致负）
- **背靠背 jevdec↔jevlaya 分不出高下**：Δ−3（994→991，n=1540，p=0.8785），
  adversarial +5（p=0.6305）——净值统计无差异；但成本差 4.4 倍：触发率
  **decider 55.1%（1091/1980，调用约 3071 次）vs laya 12.4%（246/1979，约 2223 次）**，
  laya 以 1/4 的扩拉量拿到同噪声带结果，单选运营选 laya
- **机制层负证据**：decider 首判均值 0.40、扩拉后 **96%（1052/1091）仍判不足**——
  分数锚定问题难度而非上下文证据量；扩拉题上 jevdec 救回 67/反伤 81（净−14），
  且 50% 扩拉（544/1087）重排后 top-15 上下文一字不变（纯开销）
- **样例一救一不救**：conv-49_qa72 扩拉换入"Sam 以前很喜欢徒步但很久没去"直接命中
  题眼（partial→exact）；conv-26_qa26 的书《没有什么是不可能的》根本不在索引里，
  加倍宽度拉不出缺失证据，反从诚实拒答退化为编造（both wrong）——病根在提取侧，
  门控不替代修复，与提取门控 POC 结论一致
- **与英文 Wave 7 +6 不可比**：那是 dev 616 单轮、未终审、叠加原子池（r601/r605 均
  +atomic），且同配置复现轮自波动 ±16；中文三臂 atomic_pool=False——中文的"无增益"
  只对无原子池的融合管线成立，"原子池+Jev"组合在中文未测
- 告诫：单夜单配对，子信号一律待复验；配对集 1968 共同题（J 配对 1539-1540）；
  judge=null 判分 342/364/343 条两臂近似对称；decider 55% 触发率混杂"阈值校准过严"
  与"机制无效"两个解释，复验时应先扫阈值（0.3/0.4/0.5）再下机制结论；
  laya 臂 2 题 jev_stop 记录缺失（均答对，调用量按可见 2223 次+2 未知计）
- 详见 `results_zh/jev_stop_report_zh.md`（三臂表/三对转移矩阵/机制拆解/样例/告诫）

## 语言污染事故与修复（2026-09-26 发现并当日修复）

**事故**：中文库 1993 条记忆里 616 条是英文（conv-41 155 英/210、conv-47 205/214、
conv-49 102/159 最重，conv-42/48/50 轻微，nf53:conv-42 122/207）。根因：mem0 官方提取
prompt 只说"跟随输入语言"，flash 在部分对话上开头输出英文后整段跟随（更新阶段看到英文
旧记忆会继续英文）。全部写入发生在 09-25 上午 10-11 点的单次进程；基线工作流 11:58 启动
时全部对话已有记忆而走"复用跳过"——**09-25 全部中文数字（基线 64.69%、图臂、Jev 臂）
均基于混合语言库**。分对话 exact 率显示污染对话未显著偏低（conv-47★ 甚至最高 71.3%，
跨语言向量检索部分起效），配对类结论（双臂共享同一库）仍成立。

**修复**（scripts/repair_zh_store.py，工作流 dwfrun-ab043509）：
- config/locomo_zh.yaml 的 mem0.fact_extraction_prompt：官方原文为底，语言行替换为
  "一律简体中文记录"硬指令（仅中文轨道，英文轨道未动）
- **重要技术坑**：vendored mem0 2.1.0 V3 管线要求输出 JSON 键为 `memory`（main.py:978
  只解析 memory 键），照官方原文写 `facts` 会导致提取结果被静默丢弃（+0 条且无报错）——
  自定义提取 prompt 必须用 memory 键格式
- 7 个污染 user 删除重灌（~460 次调用）；全库 11 个应存在 user 逐条复核 100% 中文
  （专有名词如 巫师3/Python/VR 按原样保留属正常）；冒烟作答正常（conv-47_qa1 temporal
  全中：保龄球）
- 教训：校验脚本遍历了 nf53×10 个本不存在的 user，把"空"误报为 FAIL——校验前先声明
  哪些 user 应该存在

**遗留**：绝对值数字待有额度时重测（重跑基线 eval-only+regrade 即可，~1h）；上午那个
写入进程的源头未定位，若复发优先查 history db 时间线。
