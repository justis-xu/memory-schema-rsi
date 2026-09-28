# RSI训练诊断：活动路径优先与两跳边界

## 问题与范围

承接平均收益门控审计，本阶段检查训练失败定位是否能可靠决定修改方向。EvolutionEngine读取train的gold-ID遗漏，输出displaced_baseline_evidence、channel_missing、hop_limit、budget_or_rank或candidate_pool_gap；这些标签没有语义来源校验。

范围是实际演化代码与5个合成离线结构控制。每个控制包括train/validation/test合法case和固定Memory/hub/边；只执行真实compile/evaluate/train diagnose，零模型、答案、真实图调用。不估中文错误率，不复算已知历史总体。

脚本 `scripts/m2_audit_evolution_diagnosis_paths.py`；before/after分别在 `results/analysis/m2_evolution_diagnosis_{before,after}_20260928.json`，保留输入、schema、编译图、实际selected和代码SHA。

## 直接复现

旧诊断从原始候选池寻找首条两跳内最短路径，未优先核活动编译图，邻接顺序继承输入边顺序。

两个控制只交换候选边顺序：seed→gold同时有未启用r0和已启用r1直接边；活动schema只启用r1。编译图与真实train selected完全相同，访问预算为1导致gold尚未出堆即停止。但r0先出现时诊断channel_missing，r1先出现时诊断budget_or_rank。输入顺序改变了学习症状标签，虽然活动图和失败相同。

另一个控制有未启用的一跳r0，与已启用/编译可用的两跳r1。旧诊断只看更短r0，报缺通道，忽略当前max_hops=1与可用r1两跳路径之间的限制。扩大跳数或增加通道都可能改变结果，不能把缺通道称为唯一必要原因；控制还设访问预算1，说明多个限制可同时成立。

|控制|旧标签|新标签/范围|
|---|---|---|
|未启用直边先出现|channel_missing|budget_or_rank|
|已启用直边先出现|budget_or_rank|budget_or_rank|
|活动两跳替代路径，当前只许一跳|channel_missing|hop_limit|
|启用边低于min_score，未编入图|budget_or_rank|compile_filter_or_degree|
|原始池存在三跳路径|candidate_pool_gap|保留该标签，显式path_horizon=2|

三跳控制证明旧README“在候选池中根本不可达”的解释过强；代码实际只搜索两跳，而schema也只支持1/2跳。不存在支持范围内路径不等于全池无路径。

## 最小修复

仅改实验诊断，流程为：

1. 查活动编译图的正分有向路径，两跳内可达时根据当前hop上限报hop_limit或budget_or_rank；
2. 活动图无路径而启用通道原始正分池可达时，报compile_filter_or_degree；
3. 再查全部原始正分池，存在路径才报缺通道并累计其通道，否则报有界candidate_pool_gap；
4. 邻接排序固定，输出path_horizon、diagnostic_seed_k和target_semantics=unvalidated_evidence_ids。

编译图按节点top-degree后可能有方向性，所以此层按实际有向弧走，不把原始双向边当活动可用路径。诊断使用引擎的固定候选seed预算，输出该值以便解释；它仍是启发式定位，不是每种参数修改后的反事实保证。

同步收窄README的candidate_pool_gap解释。未改gold、候选池、检索算法、评分目标、接受门槛或历史patterns归档。诊断优先级/通道累计变化可能改变新run的deep排序，因此不能承诺所有未来演化轨迹不变。

修复后5控制符合预期；新增诊断测试加既有test_lab共7 passed；编译/diff检查通过。没有扩大到模型/中文答案测试。

## 结论、置信度、限制与下一步

- **高置信度，确定性复现：** 旧训练标签可由无关输入边顺序改变，不能视作稳定失败归因；活动路径优先消除了本次顺序缺口。
- **高置信度，代码边界：** candidate_pool_gap仅有两跳范围意义；候选pool有路径不等于源事实支持。
- **未知：** 历史真实运行是否因此选错策略、中文改善幅度、最有效动作。现有两份历史tiny运行的旧patterns未作无依据修改。
- 新标签仍不能区分访问预算、graph_slots、打分排序的唯一根因；compile_filter_or_degree亦未拆阈值与degree。受控反事实才可证明具体动作有效，不能靠细分标签名字代替执行证据。

下一步将训练目标准入与结构诊断分离：已核的错主体、弱关系和无效gold指针不能直接驱动新增通道。先用已核中文来源病例建立“目标有源/待核/错配”侧车，再在有源目标上做局部修改；保留结构可达性、源关系支持和答案结果三个字段。这样RSI学到的是可验证的失败修复，而非更努力地追逐一个不稳标签。
