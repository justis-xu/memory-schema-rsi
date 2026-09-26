# memory-schema-rsi

长期记忆 Graph Schema（图模式）自动演化实验项目 —— **Phase 0：实验基础设施**。

参考方向：OaK（Ontology-as-a-Kernel）、RSIAgent（递归式自我改进智能体）。
主评测管线仍是 Phase 0 基线；[独立实验目录](experiments/schema_rsi/LIFECYCLE.md) 已开始试验图结构、读写算子和验证修复链路，尚未接入主评测。

```
LoCoMo / LongMemEval-S
        ↓
       Mem0 (OSS, Chroma 本地向量库)          ← Memory = Source of Truth（事实源）
        ↓
    Memory Records
        ↓
   GraphBuilder（schema 驱动，可随时重建）      ← Graph = Derived Artifact（派生产物）
        ↓
HugeGraph Server (单机 + RocksDB)
        ↓
 Retrieval (向量 + 可选 rerank + 可选 graph) / Evaluation
        ↓
Benchmark Result (results/*.jsonl)
```

关键解耦：**不使用 Mem0 自带的 Graph Memory**。图实验层完全由本项目控制，
Mem0 只作为记忆来源；图可以随时删除并根据 Memory + Schema 重建
（`GraphBuilder.build(memories, schema)` 即为此预留，也是未来 SchemaRSI 的接入点）。

## 快速开始

环境要求：macOS Apple Silicon、[Homebrew](https://brew.sh)、[uv](https://docs.astral.sh/uv/)。
不需要 Docker / Redis / PostgreSQL / Neo4j。

```bash
# 1. Python 3.11 虚拟环境 + 依赖（uv 会自动下载独立 CPython 3.11，不动系统 Python；
#    mem0 为 vendored 源码 third_party/mem0-src，editable 安装，改动直接改源码）
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
uv pip install -e third_party/mem0-src --no-deps

# 2. 配置模型端点（OpenAI-compatible，不写死厂商）
cp .env.example .env
#    编辑 .env 填入 LLM_API_KEY / EMBEDDING_API_KEY（/ RERANK_API_KEY）

# 3. 下载数据集（LoCoMo 官方英文版 2.8MB；LongMemEval-S 默认引用本机已有文件）
./scripts/download_data.sh

# 3b. （换机分析用）拉取历次评测产物 JSONL（~690MB，存 HF 私有仓库，可 INCLUDE 只拉部分）
# ./scripts/pull_results.sh

# 4. 安装并启动 HugeGraph（首次会 brew 安装 openjdk@11 并下载 ~900MB 发行包，解压校验后自动
#    修剪 pd/store 子包，最终约 410MB）
./scripts/setup_hugegraph.sh
./scripts/start_hugegraph.sh      # 停止: ./scripts/stop_hugegraph.sh

# 5. 环境体检（含 LLM/Embedding/Rerank 各一次小调用实测）
.venv/bin/python scripts/check_env.py

# 6. 冒烟测试
.venv/bin/python scripts/smoke_mem0.py
.venv/bin/python scripts/smoke_hugegraph.py
.venv/bin/python scripts/smoke_locomo.py
.venv/bin/python scripts/smoke_longmemeval.py
.venv/bin/python scripts/smoke_e2e.py        # Mem0 → Graph 端到端 + 小规模评测对照

# 单元测试（不依赖外部服务）
.venv/bin/python -m pytest tests/ -q
```

## 目录结构

```
├── config/
│   ├── default.yaml                 # 全部行为参数（top_k/graph_enabled/截断/路径…）
│   └── hugegraph.properties.example # 对 HugeGraph 后端配置的记录
├── data/                            # 数据集放置说明（大文件不入 git）
├── src/schema_rsi/
│   ├── config.py                    # .env + YAML → 类型化 Settings
│   ├── llm/                         # chat.py（OpenAI-compatible）/ rerank.py（百炼）
│   ├── memory/                      # MemoryBackend 协议 + Mem0Backend（Chroma 封装在内）
│   ├── graph/                       # GraphStore 协议 + HugeGraphStore + GraphSchema + GraphBuilder
│   ├── benchmarks/                  # BenchmarkCase + LocomoDataset + LongMemEvalDataset
│   └── evaluation/                  # Pipeline / Retriever / Answerer / Evaluator / Result
├── scripts/                         # check_env / download_data / setup|start|stop_hugegraph / 5 个 smoke
├── tests/                           # pytest（mini fixture + FakeGraphStore，不依赖服务）
└── results/                         # 评测 JSONL 输出
```

## 模型与配置

模型端点全部走 `.env`（OpenAI-compatible，厂商可换）：

| 角色 | 环境变量 | 当前实验配置 |
|---|---|---|
| LLM（Mem0 提取 + Answerer） | `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL` | 智谱 BigModel coding 端点 / glm-5.3-flash |
| Embedding（Mem0 向量化） | `EMBEDDING_*` | 阿里百炼 compatible-mode / qwen3.7-text-embedding-flash（dim=1024） |
| Rerank（可选，检索重排） | `RERANK_*` | 百炼原生 text-rerank API / qwen3.7-text-rerank |
| Judge（全量评测启用） | `JUDGE_*`，留空则复用 `LLM_*` | 官方口径 LoCoMo / LongMemEval 判分；调用失败的题必须重试 |

行为参数在 `config/default.yaml`：`mem0.{top_k,vector_store_path,…}`、
`evaluation.graph_enabled`（**Mem0 Only vs Mem0 + Graph 的对照开关**）、
`evaluation.graph_seed_k`、`evaluation.graph_slots`、`rerank.enabled`、
`user_id_strategy`（per_case / global）、数据集路径、e2e 截断参数（控制 LLM 调用量）、
`hugegraph.*`。环境变量 `HUGEGRAPH_URL / LOCOMO_DATASET_PATH / LONGMEMEVAL_DATASET_PATH` 可覆盖。

## 核心接口（下一阶段的入口）

```python
# 记忆后端（Vector Store 封装在内，可替换）
class MemoryBackend(Protocol):
    add_memory(user_id, messages, metadata=None) -> list[MemoryRecord]
    search_memory(query, user_id=None, top_k=None) -> list[MemoryRecord]
    get_all_memories(user_id=None) -> list[MemoryRecord]
    reset_memories(user_id=None) -> None

# 图存储（动态 schema：vertex/edge/property/index 均可动态创建）
class GraphStore(Protocol):
    create_schema(schema) / schema_exists / drop_schema
    upsert_vertex(label, key, properties) / upsert_edge(...)
    get_vertex(label, key) / get_neighbors(label, key, ...) / count_vertices(label)
    reset_graph(drop_schema=True) / run_gremlin(query)

# Schema 演化实验的落点：Schema S0/S1/S2... 都用 GraphSchema 表达
GraphBuilder.build(memories, schema, *, reset=True) -> GraphReport
```

`make_test_schema()`（Memory 顶点 + RELATED_TO 边 + 两个二级索引）只是基础设施
冒烟测试；主项目已有 S0/S1 实体、事件与偏好图。独立目录
[experiments/schema_rsi](experiments/schema_rsi/README.md) 另实现机器通道与
抽象枢纽的 RSI 原型；目前尚未接入主评测管线。

## 评测骨架

```
BenchmarkCase → ingest(Mem0) → retrieve(top_k, 可选 rerank)
              → [graph_enabled 时: 用户隔离图遍历，替换尾部检索记忆]
              → Answerer(LLM) → Evaluator(token-F1 / contains) → results/*.jsonl
```

- `graph_enabled=false/true` 是对照开关；要比较效果，需固定记忆、题目与回答上下文预算，并配对判分。
- GraphRetriever 的 V0 为一跳链路检查；S0/S1 支持经实体、事件、偏好节点扩展，返回记忆时校验用户归属。
- 基础指标有 token-F1、contains、exact；全量评测另调用官方口径 judge，异常不计作答错。

## 无图基线验证（Phase 1 第一步）

不需要启动 HugeGraph（`graph_store=None`，`graph_enabled=false`），单独验证 Mem0-only 的效果：

```bash
.venv/bin/python scripts/eval_baseline.py               # 首次：选 case + ingest + 评测（~20 分钟，LLM 提取为主）
.venv/bin/python scripts/eval_baseline.py --skip-ingest # 复跑：只重新评测（~2 分钟）
```

- **子集构造**（保证 gold 在 ingest 窗口内，指标才有意义，参数在 `config/default.yaml` 的 `baseline:`）：
  LoCoMo 用 evidence dia_id 定位（排除需日期换算的 temporal / 不可答设计的 adversarial）；
  LongMemEval 用"答案子串出现在窗口文本"粗筛（可能有假阳性）。
- **记忆只灌一次**：存在 Chroma 里按 user_id 复用（`baseline:locomo:{conv}` / `baseline:lme:{case}`），
  同 conversation 的多个 QA 共享一次 ingest；后续 Schema 实验直接复用这批记忆（Memory=事实源）。
- **检索与作答**：top_k=15 + rerank（`mem0.top_k`/`rerank.top_k`），Answerer 提示词要求
  "列出全部具体细节、跨记忆综合"（多记忆综合题依赖此设置）。
- **提取三要素**（准确率的关键，缺一不可；现为 vendored 源码 + 官方默认提取 prompt）：
  1. 关闭模型深度思考——vendored mem0 源码的 `disable_thinking` 配置项
     （`third_party/mem0-src`，经 extra_body `thinking: disabled`）；
     思考 token 会挤占提取输出甚至把 content 吃空，且慢 3 倍；
  2. 提取 prompt 用 mem0 官方默认（自研 dense prompt 已回退删除）；
  3. ingest 时按 session 注入日期锚（system 头）——mem0 OSS 不支持 timestamp 参数，
     不锚定的话"yesterday/next month"全被错误解析成今天（2026-09）的相对日期。
- **当前基线**（2026-09-23，官方 benchmark 口径复现，无图）：

  | 数据集 | 规模 | 官方口径成绩 | 对照 |
  |---|---|---|---|
  | LoCoMo 全量 | 10 对话 / 1986 题 | **观测 J-score 91.8%**（类别 1-4，1540 题；adversarial 46.9% 不计分；部分 judge 请求遇到 429，尚非干净复核值） | mem0 官方公布 ~66-70% |
  | LongMemEval-S | 前 20 case | **judge 95.0%**（注：这 20 题均为 single-session-user，类型代表性有限） | mem0 官方公布 ~66% |

  六类题型的 40 题分层任务曾启动，但连续 429 使第 7、8 题出现部分会话缺失，任务停止且没有形成可用分层成绩；续跑前需处理这两题的部分记忆。评测脚本现会在新提炼报错或发现明显的历史会话缺口时停止评分。

  分类别（LoCoMo）：temporal 93.8% / multi-hop 92.9% / single-hop 92.4% / open-domain 76.0%。
  本地模型、日期锚和评测配置均与官方发表值不同，不能从这张表单独归因差距。
  **观测到的 J-score 内 127 个错题归因**（`scripts/analyze_badcases.py --exclude-adversarial`）：
  作答综合 50% / 判分争议 24% / 检索召回不足 20% / 提取丢失仅 7%
  ——这组归因也受 judge 限流影响，需逐题复核后才能用于确定图层优先级。
- **复现方式**：
  ```bash
  .venv/bin/python -u scripts/eval_full.py                    # LoCoMo 全量（断点可续 + 渐进式并发 8→20，限流降档）
  .venv/bin/python -u scripts/eval_full_lme.py --num 20       # LME 前 N case（每 case 完整 haystack，约 8.5 分钟/case）
  .venv/bin/python scripts/analyze_badcases.py results/full_locomo_*.jsonl --exclude-adversarial
  ```

## LongMemEval-S 分层基线（2026-09-24，官方口径，无图）

按 question_type 轮转取 40 case（6 种类型各 6-7 题）：

| question_type | 题数 | judge 准确率 |
|---|---|---|
| knowledge-update | 7 | **100%** |
| temporal-reasoning | 6 | **100%** |
| single-session-preference | 7 | **100%** |
| multi-session | 7 | 85.7% |
| single-session-user | 6 | 83.3% |
| single-session-assistant | 6 | **50%** ← 最弱 |
| **总体** | **39** | **87.2%** |

`single-session-assistant` 低的原因：问的是"助手在某 session 说了什么"，而 mem0 提取
以用户事实为主——这是记忆系统设计偏好，不是 bug。mem0 官方同类型也偏低。

## Schema S0 vs S1 对比（2026-09-24，user-scoped + 同预算，同 127 错题）

修正提取模板 bug 后重跑（S1 图：5336 顶点 / 9517 边，graph_memories 全部非空）：

| | S0（Entity+MENTIONS） | S1（+Event+Preference+关系） |
|---|---|---|
| 翻正数 | 35（27.6%） | **36（28.3%）** |
| 仅此 Schema 翻正 | 7 | **8** ← 增量贡献 |
| 共同翻正 | 28 | 28 |
| S0∪S1 合计 | 43/127（33.9%） | — |

S1 的 8 个独有翻正命中了设计目标：
- open-domain 2 题（"religious?" / "travel hobby"）→ **Preference 节点**
- multi-hop 2 题（"how many children/games"）→ **Event 聚合结构**
- single-hop 4 题 → 实体+事件+偏好多路跳转扩大召回

结论：**S0→S1 的净增仅 +1 翻正（噪音范围内），但 Schema 多样性显著**：
S0 和 S1 各自翻正了不同的题（仅 28 重叠 / 43 总翻正），不同 Schema 结构
确实补不同类型的错。这对 SchemaRSI 的启示是：mutator 应产出**多样化的
Schema 组合**而非单一"最优"Schema。

## 图层价值验证（Schema S0，2026-09-24 user-scoped 同预算全量）

修正跨用户泄露和上下文预算不公后的干净对照（1986 题，user-scoped entity key +
graph 候选在同一上下文预算内替换向量尾部，非额外增加）：

| 指标 | 无图基线 | graph on（S0 scoped） | 提升 |
|---|---|---|---|
| **J-score（类别 1-4）** | **91.8%**（1413/1540） | **92.0%**（1417/1540） | **+0.2pp** |
| multi-hop | 92.9% | **94.0%** | +1.1pp |
| single-hop | 92.4% | 92.4% | 0.0pp |
| temporal | 93.8% | 94.1% | +0.3pp |
| open-domain | 76.0% | 76.0% | 0.0pp |

诚实结论：**S0（Entity + MENTIONS）在正确评测下仅提供边际提升（+0.2pp）**。
价值集中在 multi-hop 跨记忆综合（+1.1pp），对 single-hop 直接查证和
open-domain 推理无增益。这说明：
1. 简单的"实体+提及"结构不足以显著改善记忆检索
2. 此前报告的更高提升来自跨用户泄露和上下文预算不公
3. SchemaRSI 需要更丰富的结构（S1+）和更智能的检索策略才能产生实质增量

```bash
.venv/bin/python -u scripts/eval_graph_ab.py results/full_locomo_*.jsonl --control 30 --workers 3 --start-conc 2
```

- **已知限制**：无日期事件的会话日期锚定（"just launched"→会话当天）由提取 prompt
  实现，flash 级模型对指令的遵循是概率性的（同一对话重灌，多数事实带日期、个别不带）。
  结构化出路：把日期作为图属性（Event 节点挂 session date）——即 Schema S0 的动机之一。
- **已知噪声源**（读结果时要意识到）：
  1. LLM 供应商内容过滤有时会拦截个别 session（按 session 容错跳过并在输出里列出）；
     历史上对 LoCoMo conv-26 整体受损的判断已被完整重跑推翻：提取了 138 条记忆，
     其正式计分四类题的单次基线为 145/152；
  2. 答案子串粗筛可能匹配到无关 session（假阳性）；
  3. 中断的 ingest 会留下不完整的记忆（复用前看 ingest 行数是否合理，必要时手动 reset）。
- 诊断方法（判断丢在哪一环）：先查 `get_all_memories(user_id)` 和检索上下文，
  再回到对应原始 turn 与图片描述。答案在上下文里仍答错，是回答或验证问题；
  只在原始 turn 里存在，是提炼遗漏；原始材料也没有时需单独处理数据或标注边界。

## 精准度重判（2026-09-24，glm-5.3 严格判分）

官方宽松 judge 的三个膨胀源（部分给分 / 日期 ±14 天容差 / 同指代即算对）会高估
准确率。新增第二判分口径 `scripts/regrade_precision.py`（对已落盘结果重判，零答题成本）：

- **确定性严格检查**（`evaluation/precision.py`，免费）：temporal 日期/时长按 gold
  粒度精确匹配；其余题型 gold 实词+数字必须齐全。16 个单测（`tests/test_precision.py`）。
- **LLM 严格判分**（`evaluation/precision_judge.py`，glm-5.3 非 flash 档；结果按内容
  哈希缓存 `data/precision_judge_cache.json`，断点续跑零成本）：对宽松判 CORRECT 的题
  分级 exact（全对且精准）/ partial（基本对但缺元素、粒度不足、含糊表述）/ wrong。
  分题型规则内置于 prompt，盲判（只看题/gold/pred，与组别无关）。

两个全量 run（1986 题）的重判结果（J 类别 1-4，n=1540）：

| 口径 | 无图基线 | graph on（S0 scoped） |
|---|---|---|
| 官方宽松 J-score | 91.8% | 92.0% |
| **精准 exact-only** | **66.5%**（1024） | **67.3%**（1037；McNemar p=0.32，不显著） |
| exact+partial | 87.0% | 88.8% |
| 宽松判对中"模糊"（partial）占比 | 22.4%（316 题） | 23.3% |

分题型 exact 率（基线）：single-hop **77.8%** ≫ multi-hop **55.3%** ≈ temporal **54.2%**
> open-domain **41.7%**。

关键发现：

1. **宽松口径虚高约 25pp**：约 1/5 宽松判对的答案缺元素或含糊（316/1540）——
   列表缺项（multi-hop）与日期含糊（temporal）正是"精准"要发力的空间。
2. **temporal（类别 2）是可单独结构化判分的题型**：gold 多为日期/时长，可确定性
   精确匹配；其 113 题 partial 大量是 "Around <正确日期>" 式含糊（answer prompt
   输出风格问题，可针对性优化）。
3. **adversarial 真实编造率 74%**（330/446）：宽松口径只显示约 47%。干净拒答仅
   83 题、拒答但附带猜测 33 题——宽松 judge 对空 gold 的"同指代"规则放过了编造。
4. **S0 图在精准口径下无显著增益**（+0.9pp，p=0.32）；此前宽松口径下 adversarial
   的反编造优势在严格判分下消失（干净拒答 83→70，p=0.20 不显著）。图的价值
   需要换接入方式（融入检索/提炼，而非旁路增强）。
5. S1 在 127 道基线错题上恢复 15 exact + 13 partial（宽松口径 36 翻正中约 8 题属
   "宽松对但严格不过"）。

判分质量：确定性检查与 LLM 严格判分一致率约 74%（确定性规则机械偏严，LLM 为主
口径）；人工抽检 15 例（编造 5 / 日期含糊 5 / 列表缺项 5）分级全部成立。

## 图融入检索主流程（Schema S2 fused，2026-09-24）

按"图不是旁路增强，要融入检索流程本身"的思路重构接入（`scripts/eval_full_fused.py`）：

- **Schema S2 = S1 + 抽象 Concept 节点**（`schema.make_schema_s2` + `graph/extractor_topics.py`
  + `scripts/build_concepts.py`）：34 个受控词表主题（career/emotions/outdoors/...）
  批量标注 1803/1829 条记忆（缓存 `data/graph/topic_cache.json`），生成 274 个
  用户隔离 Concept 顶点 + 3787 条 HAS_TOPIC 边，增量建在 S1 图上（不 reset）。
- **节点中心聚合召回**（`GraphRetriever.retrieve_fused`）：种子记忆 → 关联
  Entity/Event/Preference/Concept 节点 → 一次带出该节点下全部记忆（单节点上限
  15；support = 拉回该记忆的不同节点数），替代 bypass 的"尾部 3 槽替换"。
- **统一重排同预算**（`pipeline.evaluate_case` 的 `graph_mode=fused`）：向量 top15 +
  图聚合候选（≤30）同池进 rerank，取同样 15 条预算；图与向量公平竞争，无固定席位。

全量 1986 题（有效 1985；1 题判分重试耗尽）三向对照，J 类别 n=1540：

| 指标 | 无图基线 | bypass（S0 尾部替换） | **fused（S2 融合）** |
|---|---|---|---|
| 宽松 J-score | 91.8% | 92.0% | **92.9%** |
| 精准 exact-only | 66.5%（1024） | 67.3%（1037） | **67.6%（1041）** |
| 精准口径 wrong | 200 | 173 | 179 |
| multi-hop exact | 55.3% | 56.0% | **59.2%** |
| temporal exact | 54.2% | 56.4% | **56.7%** |
| single-hop exact | 77.8% | 78.2% | 77.6% |
| open-domain exact | 41.7% | 41.7% | 40.6% |
| adversarial 干净拒答 | 83/446 | 70/446 | 75/446 |

配对 McNemar（exact 翻转）：fused vs base 63:80（p=0.18）；bypass vs base 65:78
（p=0.32）。两口径上 fused 均一致优于 bypass，但单次全量仍未达统计显著。

**机制归因**（提升确实来自图召回入选上下文，而非采样噪声）：

- 图候选入选上下文的 885 题：exact 67.3% → 69.8%（净 +22）
- 图候选未入选的 655 题（rerank 全保留向量，等价基线臂）：65.3% → 64.6%（净 -5，噪声内）
- 其中 multi-hop 且图入选（159 题）：exact 52.2% → 59.7%（**+7.5pp**）——列表
  完整性正是聚合召回的设计目标
- 图入选均值 1.2 条/题：聚合候选大多被统一 rerank 过滤，入选者以高相关记忆为主

诚实标注：fused 臂同时改了两件事——接入方式（旁路→融合）与 Schema（S0→S1+Concept），
对照不是单变量；bypass 臂跑在 S0 图上。要拆分归因需补一组 S0-fused 或 S2-bypass。
已知混合因素：DashScope rerank 偶发 SSL 失败（10/1986 题池重排降级为纯向量）。

结论：**融合接入在宽松与精准两个口径上都优于旁路接入，增益集中在精准口径最弱的
multi-hop/temporal**；总样本下单次配对检验未达显著（p=0.18）。剩余空间：
Concept 词表分层扩容、support 加权进 rerank、temporal 的 113 题 partial 多为
"around" 式含糊（answer 层可修复，与图无关）。

## 100 轮多图组合探索（最终：50 轮完成 + 同夜双臂终审，2026-09-25）

> 图层的完整中文说明（每种顶点/边/属性、真实样例、聚类原理、检索全流程走例、
> 实测效果）见 **[docs/graph_schema_说明.md](docs/graph_schema_说明.md)**。

50 轮探索（27+12+11）+ 同夜背靠背双臂全量终审（glm-5.3 @2 并发严格判分）的
**最终结论**：

| 终审臂 | 宽松 J | 精准 exact | 配对 |
|---|---|---|---|
| A：无图基线（official） | 92.7% | **66.4%** | — |
| B：全栈（s2+v8 融合 pn30 + precise） | 92.0% | **66.8%** | p=0.64 **无统计差异** |

1. **图/多图/Laya/聚合参数/采样数——所有配置轴在控变量后全部落入噪声带**。
   此前报告的 precise +17 / pn30 +20 / r111 +23 复现轮全部失败，均为跨夜漂移
   （±15-20 题）与 dev 选择偏差的伪影；test 对话上 dev 增益消失（+6 → +1）
2. **唯一可靠效应：rerank 关闭 -22 题（两次验证）——rerank 必须保留**
3. **精准口径 66.4-66.8% 是当前栈的硬区间**：剩余缺口在提取层（single-hop 76.7%
   封顶=mem0 没存到）、列表完整性（提取端）与判分长尾，检索侧已无空间
4. 方法学资产（这是本次最有价值的产出）：同夜括号校准 + 复现闸门 + test 对话
   保留 + 两级判分——正是这套流程抓住了全部伪影，避免错误结论入库

基础设施（全部保留可用）：7 图变体注册表（`graph/variants.py`）、跨图聚合召回
（`retrieve_fused(hops=...)`）、Laya 决策服务接入（`llm/laya.py` + pipeline
`laya_route`/`laya_filter` 轴）、探索跑批器（`scripts/search_rounds.py`，支持
rerank/laya/answer_samples 轮次轴）、`analyze_search.py --append-ledger` 自动台账。
全部 50 轮逐题结果在 `results/search_rounds.jsonl` + `data/search/rounds/`。

下一步唯一有希望的方向：**提取层重做**（详见 `docs/search_ledger.md` 终审章节）。

评测方法学（详见 `docs/search_ledger.md`，这套比结果本身更重要，它抓住了上面
所有伪影）：

- **dev/test 按 conversation 切分**（7/3），探索只看 dev，test 留给终审
- **同夜括号校准**：参考臂必须与轮次同条件测量；夜间漂移 ±15-20 题 > 任何配置
  效应，跨夜比较一律不可信
- **复现闸门**：任何"效应"必须复现轮通过才算数（r111/pn30/precise 三个"大效应"
  全部死在复现上）
- **两级判分**：内环 glm-5.3-flash 严格（配对差值抵消偏差）；终审 glm-5.3 @≤2 并发
- 选择信号 = 执行层严格判分（RSI 综述要求），bootstrap 稳定性选择（ESPO）

## 提取层重做 + Jev 融入（wave 7，2026-09-25）——突破

50 轮终审归因（提取层是天花板）被控变量实验证实，并叠加 Jev-Mem 机制：

- **原子事实提取**（`extraction/atomic.py` + `scripts/extract_atomic.py`）：
  4888 条"一事一条"原子事实（mem0 合并式仅 1829 条），字面量/日期/列表项全保留，
  独立向量池 `schema_rsi_atomic`，检索时与主池 RRF 合流（原子 ×1.25）
- **Laya/decider 双端点**（`llm/laya.py`）：Jev 四维并行候选评分（一次请求四命题）、
  证据充足判定（Jev-Mem 停止准则：不足→聚合宽度加倍重拉一次）

同夜配对结果（dev 616，对照 347）：**原子池 +26**（复现 +10，效应真实）；
原子+Jev 停止 = 379；**全叠加（原子+停止+四维过滤+precise）= 380/616，全场最高**
（61.7% vs 对照 56.3%；换算全量约 71-72%，未做全量终审）。四维过滤单独略负
（阈值 0.35 偏激进，待调）。图在原子池之上仍无独立贡献。

下一步空间：原子提取密度再提（当前 5-25 条/会话）、过滤阈值校准、全量终审。

## Troubleshooting（本阶段实测踩坑记录）


| 问题 | 原因 | 解决 |
|---|---|---|
| `setup_hugegraph.sh` 卡住不动 | macOS 的 `/usr/bin/java` 是 stub，无 JVM 时执行 `-version` 会挂起 | 脚本已把 stub 识别为"无 Java"，走 brew `openjdk@11`（keg-only 免 sudo；卸载 `brew uninstall openjdk@11`） |
| **HugeGraph 必须用 JDK 11** | 1.7.0 捆绑的 gremlin-groovy/Groovy 3 不支持 JDK 17 字节码（`Unsupported class file major version 61`，gremlin 引擎初始化失败） | 三个脚本均优先 `/opt/homebrew/opt/openjdk@11`；JDK 17 会出现 REST 正常但 gremlin 报错的诡异状态 |
| REST 404（`/apis/versions`） | HugeGraph 1.7.0 起 REST **无 `/apis` 前缀**（老版本在 `/apis` 下） | 客户端自动探测前缀；脚本健康检查两种都试 |
| REST /gremlin 报 `No such property: g` | 官方包两处缺陷：`gremlin-server.yaml` 的 `graphs: {}` 为空 + `scripts/empty-sample.groovy` 未定义 `g` | `setup_hugegraph.sh` 幂等修复：补 graphs 绑定 + 追加 `globals << [g : graph.traversal()]`（改前备份 `.bak`） |
| `clear` 接口 400 | 确认串是 `I'm sure to delete all data`（不是 "drop"），且走 **query 参数** 非 body；`/clear` 会连 schema 一起清 | `reset_graph` 已内置；`reset_graph(drop_schema=False)` 用最近 schema 自动重建 |
| 顶点更新 PUT 400/415 | 需 `?action=append`（小写），大写 MERGE/APPEND 都不行 | 已内置 |
| 边创建 400 `Unrecognized field outV_label` | 1.7.0 字段为驼峰 `outVLabel`/`inVLabel` | 已内置 |
| propertykey 400 | cardinality 要大写 `SINGLE/SET/LIST` | 已内置 |
| indexlabel 400 | 字段是 `base_type: VERTEX_LABEL` + `base_value` + `fields`（数组） | 已内置 |
| downloads.apache.org 404 | 主镜像未收录该文件名 | 脚本自动回退 `archive.apache.org` |
| glm-5.3-flash 返回空字符串 | 推理模型思考 token 计入 completion，max_tokens 太小被思考吃光 | `ChatClient` 已设 max_tokens 下限 256，Answerer 用 1024 |
| mem0 add 时警告 ignoring metadata['user_id'] | mem0 2.x 身份字段必须走参数不能进 metadata | `Mem0Backend.add_memory` 已自动剥离身份字段 |
| mem0 search 报 entity parameters not supported | mem0 2.x 改用 `filters={"user_id":...}` + `top_k` | 封装层已适配 |
| mem0 get_all 报 filters must contain... | mem0 2.x 强制按 user_id 过滤 | `get_all_memories(user_id=...)` 必须显式传 user_id |
| LoCoMo 解析出 0 sessions | 官方 locomo10.json 的 sessions 在嵌套 `conversation` dict 里，category 是 int 1-5 | 适配器按真实结构解析（1-5 → single_hop/multi_hop/temporal/open_domain/adversarial） |
| LongMemEval 无 history/evidence 字段 | cleaned 版用 `haystack_session_ids/dates/sessions` 三平行列表 | 适配器双格式兼容（原版 history+evidence 也支持） |
| rerank 的 OpenAI 风格 /rerank 不可用 | 百炼 rerank 只有原生 text-rerank API | `RerankClient` 自动回退原生端点；失败时检索降级为向量序 |
| 首次运行 mem0 很慢 | tiktoken 联网下载 encoding（一次） | 正常现象 |
| LongMemEval 加载峰值内存 ~2.6GB | 265MB JSON → Python 对象膨胀 | 独立进程跑完即释放；16GB 机器无压力 |
| chroma 提示不支持关键词检索 | Chroma 无 BM25 | 无影响，mem0 自动退化为纯语义检索 |

## 资源占用（实测，2026-09-22）

- 磁盘：HugeGraph 发行包 410MB（server 子包 403MB 必需；官方三合一包中的 `pd`/`store`
  子包共 561MB 单机模式用不到，已删除）；`.venv` 360MB；openjdk@11 296MB（brew keg-only，
  `brew uninstall openjdk@11` 可移除）；uv 管理的 CPython 3.11 ~100MB。合计 ~1.2GB
- 内存：HugeGraph JVM `-Xmx1g`（start 脚本 `-j` 显式设置，`HUGEGRAPH_XMX` 可调，实测 RSS ~0.9GB）；
  LongMemEval 加载峰值 ~2.6GB（独立进程跑完即释放）
- LLM/Embedding/Rerank 全部远端 API，本地零模型开销

## 系统边界

**当前完成**：Mem0 + HugeGraph(RocksDB) + LoCoMo/LongMemEval-S 基础环境；
Memory → Graph 最小端到端；统一 BenchmarkCase；评测管线骨架（graph on/off 对照）。

**仍未完成**：主评测链路中的完整 Schema RSI，以及跨 conversation 的真实问答策略进化验证。
独立的 [图记忆链路实验](experiments/schema_rsi/LIFECYCLE.md) 已覆盖提炼前读图、稀疏建图、
原始来源回溯、问答前检索和验证修复。另见[十段 LoCoMo 对话与 LongMemEval 的图路由诊断](experiments/schema_rsi/SUCCESS_GRAPH.md)：
固定图扩展对完整来源覆盖的收益很小；受验证约束的成功证据组在独立 conversation 上有窄幅离线增益，
仍待无 gold 的支持来源定位和配对问答验证。已完成的 conv-26 成功题 24 题配对实验中，
唯一的原始 judge 分歧是两组答案都未给出金标准国家的判分不一致，不能记作图的真实修复。
