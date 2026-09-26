# Schema RSI：独立离线原型

**完整记忆链路实验**见 [LIFECYCLE.md](LIFECYCLE.md)：提炼前读图、事实与来源写图、回答检索、验证后修复，并使用已完成的 conv-26 做隔离诊断。下文仍记录较早的“固定候选边池优化器”，两者的结果不能混同。

**从成功题学习图路由**见 [SUCCESS_GRAPH.md](SUCCESS_GRAPH.md)：对 145 道已答对题做无向量检索替代与成功轨迹枢纽诊断，清楚区分问答判分、来源覆盖和检索 ID 代理指标。

这个目录试验一件事：**在记忆记录不变、上下文长度不变的条件下，自动调整图的稀疏连接和遍历算子，看看图邻域能否补回原检索漏掉的证据记忆。** 它只读取本目录提供的冻结快照，不访问正在运行的 Mem0、Chroma、HugeGraph 或模型 API。

## 三篇论文在这里各起什么作用

- [OaK](https://arxiv.org/html/2608.22974) 的核心是把 schema 与可执行函数作为一个 kernel，一起构建、评估和修复。这里的 `Schema` 同时决定关系通道、每点度数、阈值、跳数、访问预算及上下文槽位；图编译和遍历是受它约束的算子。这里采用可检验的机器结构，并未复制 OaK 的人工语义本体或 OWL/HermiT 验证。
- [RSIAgent v1（2026-09-14）](https://arxiv.org/html/2609.15364v1) 及其[代码](https://github.com/AetherLabsAI/RSIAgent)中的广度探索、针对失败的深度探索、独立验证与冻结使用，在这里对应：先遍历不同通道和预算，再按漏证据原因修改 schema，以验证集决定接受，最后冻结后仅评一次测试集。这里是确定性搜索，不是论文中的多智能体软件操作系统。
- [WikiSkill](https://arxiv.org/html/2608.27454) 区分原始经验、持续累积的知识和当前生效技能。这里分别保存 `raw/trials.jsonl`、`wiki/patterns.json` 与 `wiki/impact.jsonl`、`active_schema.json`；拒绝的候选不会覆盖当前 schema，失败记录仍保留。

## 图表示

`M` 是记忆节点，`H` 是可选的抽象枢纽节点；`r0/r1/...` 只是机器通道 ID，没有规定“人物”“偏好”等人类可读含义。候选边池可以来自 ANN 邻居、时间邻近、共同检索或无监督聚类。原型只消费**已生成的稀疏候选边**；不会在每轮做全体记忆的两两相似度计算。

每轮 schema 只从候选边池挑选边：为每个通道设置权重、最低边分数和每点最大度数。检索先取基础排序的前 `seed_k` 条，再按权重做至多两跳的优先遍历；只将 `M` 放进答案上下文，并限制 `graph_slots`、`context_k` 和 `max_visits`。节点与边都限定同一 `user_id`。评测时，图组和基础组都只拿 `context_k` 条记忆；图加入的条目必须在基础组前 `context_k` 之外。

这个 V0 的进化范围是**固定候选边池上的图结构和遍历参数**。它可以启用、组合或压制抽象通道，也能回退无效修改；暂未让模型发明新的候选边生成程序。若证据节点在候选池中根本不可达，诊断会明确记录 `candidate_pool_gap`，不会假装调权重能修复。

**当前评分是证据记忆 ID 的 Recall@context_k，辅以遍历边数、图边数和本地遍历耗时。它不是问答正确率。** 验证集上的增益需要超过资源惩罚及设定阈值，否则回退。训练集诊断只读取训练题的 gold；验证题只给接受/拒绝的汇总分，测试题只在冻结后评一次。

## 运行

使用项目现有 Python 3.11 环境即可，无额外依赖，也没有模型调用：

```bash
.venv/bin/python experiments/schema_rsi/run.py
```

默认读取 `fixtures/tiny.json`，输出到本目录 `runs/` 下的新目录。这个微型数据是**人为构造的链路示例**：其中 `r0` 是直接记忆边，`r1` 经抽象枢纽两跳到达另一条记忆，`r2` 是无帮助的候选。它只验证搜索、遍历、接受/回退和隔离约束，不提供真实效果证据。

使用自己的冻结快照：

```bash
.venv/bin/python experiments/schema_rsi/run.py \
  --dataset /absolute/path/snapshot.json \
  --output /absolute/path/schema-rsi-run-001 \
  --context-k 15 --deep-rounds 3
```

如果已有**冻结的记忆向量**、按用户隔离的基础检索排序和独立确定的证据记忆 ID，可以先生成候选边池与抽象枢纽：

```bash
.venv/bin/python experiments/schema_rsi/prepare.py \
  --input /absolute/path/vector-input.json \
  --output /absolute/path/snapshot.json
```

`vector-input.json` 含 `memories: [{"id", "user_id", "embedding": [浮点数]}]`、下述 `cases`，可选 `ann_edges: [{"source", "target", "score": 0.0..1.0}]`。本机现有虚拟环境已包含 `numpy`，只有这一步需要它。默认用分块余弦乘法从每位用户最多 2000 条记忆中产生 `r0` 候选邻边；超限必须提供来自现有向量索引的 `ann_edges`。`r1` 枢纽由固定种子的随机投影分桶产生，分桶大小有上限。生成边时不读取题目的 `gold`；只原样传递到输出以供后续验证。

输入是一个 JSON 对象，包含：

- `nodes`: `{"id": "...", "user_id": "...", "kind": "M" 或 "H"}`；
- `edges`: `{"source": "...", "target": "...", "channel": "r0", "score": 0.0..1.0}`；边池须在题目 gold 之外独立生成；
- `cases`: `{"id", "user_id", "split": "train" / "validation" / "test", "base_ranked": [记忆 ID], "gold": [证据记忆 ID]}`。每一类 split 都必须存在，ID 必须在对应用户内有效。

运行目录保存输入哈希、每个候选的训练轨迹、持续累积的诊断、每次验证的接受记录、生效 schema 与冻结测试报告。输出目录必须是新目录，以免覆盖上一轮实验。

## 接入当前项目的边界

主项目的 [GraphBuilder](../../src/schema_rsi/graph/builder.py) 可构建按用户隔离的 S0/S1，也保留同会话相邻的 V0；它还不能直接执行本目录的 `r0/r1/r2` 多通道拓扑。[EvaluationPipeline](../../src/schema_rsi/evaluation/pipeline.py) 已允许图候选在同一上下文预算内替换尾部向量记忆，并记录实际上下文 ID，但这仍是主项目的 S0/S1 检索，不等于本目录的完整生命周期策略。本目录的[完整链路试验](LIFECYCLE.md)已在独立 Mem0 集合中完成一次带图提炼，并在冻结快照上检查图检索和修复；跨 conversation 的正式策略晋级尚未接入主评测链路。

基线 JSONL 只保存检索到的部分记忆，不能直接当完整图。本目录已从原 Chroma 只读冻结十段 LoCoMo 对话的全部记忆和向量，并保留原始 `dia_id`；[成功题图路由诊断](SUCCESS_GRAPH.md)区分了原始 turn 覆盖、原检索记忆 ID 代理指标和真实问答判分。LongMemEval-S cleaned 版只有答案会话 ID，其覆盖指标更粗。用基准 gold 构造的成功证据组仅用于上限诊断；实际 RSI 必须从无 gold 的验证结果获得支持来源。
