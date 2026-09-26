# 图记忆的完整链路：conv-26 隔离试验

这部分在 `schema_rsi_lab/lifecycle.py`、`provenance.py`、`verify.py` 和 `lifecycle_rsi.py` 中实现。它与本目录早先的固定边池优化器互相独立。所有实测数据都来自 `runs/` 内的冻结快照，**没有改动正在运行的 Mem0、Chroma 或 HugeGraph**。

## 上游方法与本地实现的对应

本地核对了 [RSIAgent 的 2026-09-14 代码快照](https://github.com/AetherLabsAI/RSIAgent/tree/e8e8c22018da9455c6adc438954b571d875b8a46)：`core/self_evolving_loop.py` 明确分开 Actor、Verifier、失败后的学习和新的目标尝试；`explore/phase1_wave.py` 将分支经验蒸馏、审计后串行提交到 canonical memory。该仓库没有可直接复用的“记忆图检索器”，所以本实验复现的是**反馈与提交机制**，不是复制图实现。当前 `main` 的对应代码文件与上述快照没有差异。

[OaK](https://arxiv.org/html/2608.22974) 的启发是把结构与读写算子当成同一个 kernel：这里的 `LifecyclePolicy` 同时约束建边、抽象枢纽、提炼前读图和回答前遍历；边标签 `r0/r1/r2` 是机器通道，不表示人类本体。尚未实现 OaK 的 LLM 驱动 schema 修复或可执行函数合成。[WikiSkill](https://arxiv.org/html/2608.27454) 的启发是不可变原始经验、可修订的图/策略、验证通过才发布；这里保留原始对话轮次及图片描述，答案修复经独立证据检查后才接受。尚未实现它的长期 wiki 影响记录。

## 实际链路

1. **提炼前读图**：用新会话原文在同一用户已存在的记忆中找种子，受预算限制遍历图，给提炼器提供少量旧事实作为身份与矛盾消解背景。`add_session_with_graph` 会明确指示 Mem0 只从新消息提取事实。当前 conv-26 的离线回放只能记录这一步本来会取到什么；原有 138 条事实已经提取完成，不能声称读图改善了提炼质量。
2. **写记忆与图**：Mem0 完成写入并返回稳定 ID 后，才读取它存下的向量并写图。`r0` 连接局部向量近邻，`r1` 接入固定随机投影桶的潜在枢纽，`r2` 连接同次提炼的近邻事实。查询和建边均受每用户隔离、扫描上限、跳数与边访问预算约束。可用 `upsert/delete` 处理事实更新或删除。
3. **保留原始来源**：`SourceGraph` 存储带日期的原始 turn 和图片描述，以用户、会话、turn ID 定位；与同会话事实建立稀疏联系，并对原文建倒排索引。这样图可从压缩事实回到原始证据，即使某个细节在 Mem0 提炼时丢失也仍有恢复路径。
4. **回答前检索**：先使用已有向量加重排的 15 条结果；图候选只在同样的 15 条上下文预算内替换部分记忆。原始来源 turn 也占预算，不额外给答案模型更多文本槽位。
5. **验证和修复**：先检查基线答案与现有证据。答案有足够支持就保持；答案与现有证据冲突，则在原上下文重答；缺证据才查图、来源轮次和已核实的成功经验路径。候选重答再做一次证据检查，通过才接受，否则回退原答案。验证器在同一次调用中返回支持答案的上下文编号，映射为记忆或原始 turn ID；只有再得到独立结果反馈才允许提交经验节点。验证器不接触 gold answer；gold 只用于离线评测。这个新增的支持编号链路尚未做真实模型配对验证。
6. **RSI 门槛**：`LifecycleRSI` 可探索“仅验证、记忆图、记忆加来源图、稀疏来源图、已核实的成功经验路径”五类完整链路策略，按 conversation 隔离训练、验证与冻结测试，记录 `raw/trials.jsonl`、`wiki/impact.jsonl` 和生效策略；验证集没有修复或出现回归时回退。它要求至少五个独立 conversation，并通过注入的真实回答/judge 评估器判分。conv-26 只有一个 conversation，因此本轮没有把定向诊断当作策略进化的验证集；跨 conversation 的真实评估器和候选搜索仍待接入。

## conv-26 当前证据

冻结了 19 个会话、419 个原始 turn、138 条 Mem0 事实、199 道已完成基线题。正式计分的四类共 152 题，原有基线 145 题正确。初版无条件扩图让 196/199 题的上下文改变，图中 138 个事实节点、30 个抽象枢纽、2248 条有向弧；这说明**必须在回答前按失败类型做门控**，不能把高覆盖率当成效果。

本轮门控诊断覆盖 7 道基线错题，以及按题号选取的 8 道基线正确题；这是刻意偏向错题的诊断集，不是随机测试集，也不能推算为整体 J-score。在 7 道错题里，原始 turn 的图片描述让 `qa133`（咖啡馆告示）和 `qa140`（海报文字）能够找到压缩事实里缺失的答案。`qa26` 在候选答案中得到正确的 2022 年，但验证器因原文没有明确书名而拒绝提交；这是验证门槛的误拒案例。`qa66`、`qa75`、`qa85` 的关键线索原本就在前 15 条记忆中，主要是回答或验证的选择问题。`qa59` 没有得到足够来源证据。

最终记录为 15/15 题正常执行：错题修复 2/7，已答对的对照题 8/8 仍正确。随后将来源候选限制为前三个锚记忆会话与最多三个稀有查询词，`qa133/qa140` 在优化版再次判对。对冻结的 138 条事实、419 个 turn 的本地索引做单进程微测：建事实图约 22 ms，建立来源索引约 22 ms；199 次“图候选 + 来源候选”本地查询平均约 0.5 ms、95 分位约 0.8 ms。**这些数值不含向量检索、重排、模型调用、Chroma/HugeGraph 网络或持久化开销**，不能外推到生产延迟。建边当前对每个用户做受上限保护的本地向量扫描，规模大于 2000 条时必须换 ANN。

以上描述以 `runs/conv26_verify_repair_final.jsonl` 和优化版两题复核 `runs/conv26_source_optimized.jsonl` 为准。原始 turn 与事实快照含数据集文本和答案，放在 `runs/`（Git 忽略）中，不应作为代码材料提交。

随后对全部十段 LoCoMo 对话及已完成的 LongMemEval-S 前 20 题做了离线检查，见 [SUCCESS_GRAPH.md](SUCCESS_GRAPH.md)。固定向量邻边与潜在哈希枢纽在十段对话上只带来有限的原始证据覆盖增益；训练自其他成功题的抽象“问题 → 证据组”路径更有信号，但目前用官方证据标注代替真实验证器产生的支持来源，尚不能发布为在线有效的 schema。

为检查“提炼前读图”是否真的进入模型调用，又在独立的 `runs/conv26_graph_ingest_01/` 中运行 `ingest_conv26_isolated.py`：19/19 会话成功，第二个会话起都拿到旧图上下文且发生邻域扩展，最终 Mem0 持久化 140 条事实，图中也是 140 个事实节点，并保留 419 个来源 turn。提炼阶段累计约 252 秒，包括模型与向量调用；没有同环境同时间的无图配对耗时。原基线有 138 条事实，两次提炼输出不完全相同，不能把条数差当成效果。新事实仍未包含咖啡馆告示与 “Trans Lives Matter” 的完整文字；这支持保留原始来源层。该独立集合的问答准确率尚未判完，不能声称它优于原基线。

主项目此前的 HugeGraph S0 A/B 结果另有两个已确认缺陷：不同用户的同名实体被合并，图候选中出现跨用户记忆；图组获得 20 条回答上下文而基线只有 15 条。旧 S0 结果和据此写出的提升数字已撤回，详见根目录 README。修正后的主项目图检索和建图代码仍需在重建图后重新配对评估；本目录的冻结快照局部试验不依赖旧 S0 图。

## 复现

```bash
PYTHONPATH=src:third_party/mem0-src .venv/bin/python experiments/schema_rsi/snapshot_conv26.py \
  --baseline results/full_locomo_20260923_073809.jsonl \
  --output experiments/schema_rsi/runs/conv26_frozen.json

.venv/bin/python experiments/schema_rsi/replay_conv26.py \
  --snapshot experiments/schema_rsi/runs/conv26_frozen.json \
  --output experiments/schema_rsi/runs/conv26_replay_v1.json

PYTHONPATH=src:third_party/mem0-src .venv/bin/python experiments/schema_rsi/try_conv26.py \
  --snapshot experiments/schema_rsi/runs/conv26_frozen.json \
  --output experiments/schema_rsi/runs/conv26_verify_repair_final.jsonl
```

每次输出必须使用新文件名。最后一步会调用已配置的回答模型、验证模型和官方口径 judge。它只读取冻结快照。生产接入前还需将图及来源索引持久化、在独立 Mem0 collection 重跑提炼，并且对更大规模用户语料改用 ANN 写边。
