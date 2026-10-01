# Graph、Jev、RSI 中文记忆复盘：跨机结论与证据索引

> 2026-09-30。把 `memory-schema-rsi` 与 `memory-prompt` 两仓已经落盘的阶段结论接成一个阅读入口。本文件只整理既有报告，不重跑统计、模型、检索或答题；旧文档、机器结果和脚本均保留。两仓各报告注明的样本、数据版本和日期仍是结论边界。**决策级复盘在此收尾，研究目标已暂停；当前纯中文 Graph、Jev、RSI 的受控总体净收益仍未知。**

## 怎么看

1. **先看总判断：**[本仓综合复盘](m2_integrated_retrospective_20260929_zh.md)。它分开 Graph、Jev、RSI、来源、角色和评测，不把局部机制写成总体收益。
2. **再看跨仓完成度：**[`memory-prompt` 完成度与停止条件](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_retrospective_completion_zh.md)。它说明历史文件盘点已做什么、哪些日志不可恢复、哪些总体证据仍缺。这里的“严格研究完成条件未满足”与“本轮复盘可收尾”并不矛盾：前者指尚不能证明模块净收益，后者指已把现有证据整理成可决策结论。
3. **想核数字和逐例：**按下表进入专项；各报告链接脚本与 `results/analysis/m2_*` 原始机器账。早期全量复算可从[首轮深度复盘](deep_retrospective_zh.md)、[历史运行覆盖](m2_historical_run_coverage_zh.md)、[非 JSONL 版本清单](m2_artifact_version_coverage_zh.md)查起。
4. **想找任意阶段：**[两仓 `m2_*_zh.md` 报告完整目录](m2_cross_machine_report_catalog_zh.md)按仓库和文件名列出全部阶段文档；本页下表负责筛出值得先读的关键结论。

## 跨机合并后的判断

| 主题 | 已确认的事实与限制 | 证据入口 |
| --- | --- | --- |
| **Graph：有局部机制，未有当前净收益** | 旧中文配对中图候选入槽 2,042 次、挤出旧候选 2,084 次，入槽图记忆缺来源日期；同一最终上下文的 482 题仍有 59 题 exact 翻分。历史三道图正例迁到当前中文库后，曼城和番茄两题已由无图答全；签名篮球人工补位可救局部缺口，但自动多人拆问尚未显示稳定收益。**2026-10-01 受控实验收口**：独立可核源中文图（1,698 Memory 全带来源准入字段）+ 11 题冻结配对下，图臂 39/44 vs base 36/44（噪声量级），换入必要事实 0/11、反伤侵占 0——"当前图收益病例不足"被证实，最大翻转（conv-47_qa6）人工核为推断+模糊金标上的作答波动而非图机制。见 [独立图配对评测](m2_independent_graph_eval_zh.md)。 | [综合复盘](m2_integrated_retrospective_20260929_zh.md)；[固定单槽正反对照](m2_frozen_graph_swaps_poc_zh.md)；[跨仓旧正例迁移](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_graph_transfer_decision_zh.md)；[缓存就绪](m2_current_cache_graph_readiness_zh.md)。 |
| **Jev：分数不能直接当答案或证据充分性** | 旧首判分数对最终 exact 的 AUC 约 0.53，评分器仅看每条记忆前 60 字；扩拉中不少题最终没有选入首判池外 ID。另一台机器的中文短命题探针显示 `noul` 有真假排序信息，却与原固定阈值不匹配；它和记忆答题是不同任务，不能互相推算效果。旧首判完整有序上下文缺失，严格同次反事实不可重建。 | [Jev 分数审计](jev_score_audit_zh.md)；[扩拉追踪](m2_jev_expansion_zh.md)；[跨仓三种标签边界](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_jev_decision_gate_zh.md)；[短命题阈值](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_jev_noul_calibration_zh.md)。 |
| **RSI：来源线索与成功经验要分开** | 英文离线源 ID 覆盖增益提示有可用线索，但九道增益题回源后，五条先前“成功经验”的 mined turn 不足以支持其完整答案或人物。把五条经验整体拒收，在固定回放中又失去五个目标题的源 ID 增益，其中四题失去的 turn 对目标问题确有支持。两边都不是中文最终答案收益。 | [主仓逐题退化](m2_evolution_case_regression_zh.md)；[跨仓九题回源](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_rsi_mined_gain_source_zh.md)；[整条拒收代价](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_rsi_accept_gate_tradeoff_zh.md)。 |
| **事实来源是最明确的断点** | 固定 12 对跨场重复有真实旧源独有细节，整条去重会丢事实；公路同行者身份无源。一次真实隔离 Mem0 写入的 5 条中，3 条整句仅部分受源支持。跨仓八次中文提取输出 36 条事实，人工核为 28 full、7 partial、1 absent；合法指针或严格 JSON 都不能自动排除语义错误。最近固定两例四次提取仍把“4 人含本人”写成事实。 | [12 对事实合并](m2_carryover_fact_merge_zh.md)；[真实写入](m2_isolated_mem0_ingest_zh.md)；[跨仓门槛回放](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_fact_acceptance_gate_replay_zh.md)；[两类角色提取](m2_role_aware_extraction_zh.md)。 |
| **跨运行缓存与时间错挂不能笼统归为幻觉** | 跨仓隔离回放发现 `delete_all` 清向量记录但不清同 owner 历史消息；后场原话可在重跑首场进入提取提示。四条未来事实错挂有“旧内容→删除→新首场 ADD→后场重入”的时间线，但缺当时真实请求，不能断言单一成因。人工修正文日期在固定题上带来局部答题改善；自动修订尚未稳定还原来源日期。 | [跨运行缓存可见性](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_cross_run_cache_visibility_zh.md)；[未来事实线索](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_future_origin_cache_trace_zh.md)；[错时记忆来源](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_memory_origin_triage_zh.md)；[人工修订答题](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_trip_answer_repair_zh.md)；[自动修订边界](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_trip_source_repair_zh.md)。 |
| **评测与版本边界决定分数能否解释** | 中文 LoCoMo 1,986 QA；中文 LongMemEval-S 仅 470/500，且共同题有背景场缺失。中文八臂实际检索到的 1,780 条记忆可见正文一致，但完整库/图版本未冻结。历史清单中当前机器可读 133/135 份 JSONL；八臂缺题的多数小分差可跨零。金标、相对时间、翻译、裁判和答案附加句均有逐例问题。跨仓 LongMemEval 中文源文件变化后，旧批次 ID 有漂移，不能跨版本沿用。 | [中文 HF 覆盖](m2_hf_dataset_coverage_zh.md)；[产物版本](m2_artifact_version_coverage_zh.md)；[缺题界限](m2_missing_case_score_bounds_zh.md)；[跨仓文件可读性](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_historical_file_availability_zh.md)；[源文件漂移](https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_lme_source_drift_zh.md)。 |

## 最终决策与未解决项

**现在能做的决策：** 不用旧 exact 小幅净增为 Graph/Jev/RSI 单独背书；不把模型写出的合法 `source_ref`、`attributed_to` 或成功写入当作事实可靠；不按 `assistant` 载体角色跨基准整类过滤，LoCoMo 第二位说话者也是真人。[角色语义](m2_cross_benchmark_role_semantics_zh.md)与[十条回源](m2_role_label_source_sample_zh.md)给出病例。不要按整条删除重复记忆或整条拒收 RSI 经验连带删除有用原始来源。

**仍不能回答的研究问题：** 当前纯中文同版本、同源、同槽、同答题器/裁判下，Jev、RSI 各自对必要事实、答案、反伤和成本的净效果（Jev 因 GPU 服务失效停车；RSI 接受器两门不一致见[阶段 3 重放](m2_rsi_acceptor_replay_zh.md)）；旧运行缺题、首判和部分提取日志的具体原因。**已回答项（2026-10-01）：** ① 来源门——语义层未对齐人工口径但确定性守卫使部署形态过停止门（[POC](m2_fact_source_gate_zh.md)、[v3 重放](m2_fact_source_gate_v3_zh.md)）；② **Graph 受控净效果——独立可核源图 + 冻结配对下"当前图收益病例不足"成立，0 换入必要事实、0 反伤侵占**（[独立图配对评测](m2_independent_graph_eval_zh.md)）。不可恢复项已固定为[台账](m2_dispute_registry_zh.md)，不再反复尝试。当前没有充分证据给 Jev/RSI 净效果一个百分比。

**保存与同步边界：** 两仓 Git 文档、脚本、已跟踪机器账可跨机拉取；本地忽略的数据集版本、Chroma、图服务、模型、凭据及未跟踪旧日志不能因 `git push` 自动视为同步。各报告内的文件指纹、时间和模型调用账优先于“本机当前”字样。历史交接文档保留原时间语境，不删除或重写为最新结论。本轮只新增导航索引，研究目标继续暂停。
