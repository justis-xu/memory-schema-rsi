#!/usr/bin/env python
"""中文轨道建图：对 zhfull:locomo:conv-* 构建 s2 + v8cluster 图（与英文终版 B 臂同构）。

流程（每 conversation，串行）：
  1. 从中文 mem0 库（data/vector_store_zh/chroma）取 zhfull:locomo:{conv} 记忆
     （无图基线已 ingest，本脚本不重复 ingest）
  2. StructuredExtractor 实体/事件/偏好/关系抽取（LLM=主 LLM 档，
     缓存 data/graph_zh/structured_cache.json，批间断点续跑）
  3. GraphBuilder.build_s1(reset=False) —— ⚠️ reset=False 必须：
     HugeGraph 共用 hugegraph 实例（user 域隔离，见下），reset 会清掉英文轨道全部图数据
  4. TopicExtractor 受控词表主题（缓存 data/graph_zh/topic_cache.json）
     + GraphBuilder.build_concepts → Concept/HAS_TOPIC（S2）
  5. VARIANTS["v8cluster"].build：嵌入缓存 data/graph_zh/memory_embeddings.json，
     k-means 按 user_id（=每 conv）独立分组，单 conv 调用与全量调用结果等价

图数据隔离（为什么共用 hugegraph 实例是安全的）：
  - 所有派生顶点主键带 owner 前缀（len:owner:name，GraphBuilder._scoped_key /
    variants._build_v8_cluster 的 cluster_key），zhfull: 与英文 full:/nf53: 零碰撞
  - Memory 顶点主键 = 中文 mem0 库的 memory id（独立 chroma 库，与英文库不同源）
  - 检索侧 retrieve_fused._same_owner 只放行 user_id == 锚点记忆 owner 的 Memory
  - 本脚本绝不调用 reset_graph / drop_schema

幂等/断点：
  - LLM 抽取缓存按 memory_id 落盘，中断重跑只补缺
  - 顶点/边 upsert 幂等；v8cluster 依赖嵌入缓存（memory_id 键）
  - 图内该 uid 的 Memory 顶点已齐且 V8Cluster 已存在 → 整段跳过（--force 强制重建）

用法：
    .venv/bin/python -u scripts/build_graph_zh.py                        # 全部 10 conv
    .venv/bin/python -u scripts/build_graph_zh.py --conv conv-42         # 冒烟/断点单对话
    .venv/bin/python -u scripts/build_graph_zh.py --force                # 重建（仍不 reset）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.graph import GraphBuilder, HugeGraphStore, make_schema_s1  # noqa: E402
from schema_rsi.graph.extractor_s1 import StructuredExtractor  # noqa: E402
from schema_rsi.graph.extractor_topics import TopicExtractor  # noqa: E402
from schema_rsi.graph.variants import VARIANTS  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402


def zh_counts(store: HugeGraphStore, uid: str) -> dict[str, int]:
    """gremlin 精确数该 uid 的子图顶点（跨轨道共图下的准确口径）。

    Entity/Event/Preference 顶点无 user_id 属性（主键是 scoped key），
    故用「Memory(user_id) --出边--> 邻居」遍历计数：构建时出边只从该用户
    记忆发出，天然限定在该用户子图内。
    """
    edge_of = {
        "Entity": "MENTIONS",
        "Event": "DESCRIBES_EVENT",
        "Preference": "EXPRESSES_PREFERENCE",
        "Concept": "HAS_TOPIC",
        "V8Cluster": "V8_IN_CLUSTER",
    }
    out: dict[str, int] = {}
    queries = {"Memory": f"g.V().hasLabel('Memory').has('user_id','{uid}').count()"}
    for label, edge in edge_of.items():
        queries[label] = (
            f"g.V().hasLabel('Memory').has('user_id','{uid}').out('{edge}').dedup().count()"
        )
    for label, q in queries.items():
        try:
            data = store.run_gremlin(q) or {}
            results = data.get("result", {}).get("data", [])
            out[label] = int(results[0]) if results else 0
        except Exception as e:  # noqa: BLE001
            print(f"    ⚠ gremlin 计数失败 {label}: {str(e)[:100]}")
            out[label] = -1
    return out


def isolation_check(store: HugeGraphStore, uid: str) -> int:
    """隔离核对：从该 uid 记忆经 Memory→节点→Memory 全部跳转边能带出的
    异 user_id Memory 数（retrieve_fused 的实际路径），应为 0。"""
    edges = "MENTIONS','DESCRIBES_EVENT','EXPRESSES_PREFERENCE','HAS_TOPIC','V8_IN_CLUSTER"
    q = (
        f"g.V().hasLabel('Memory').has('user_id','{uid}')"
        f".out('{edges}').in('{edges}')"
        f".hasLabel('Memory').has('user_id', neq('{uid}')).dedup().count()"
    )
    try:
        data = store.run_gremlin(q) or {}
        results = data.get("result", {}).get("data", [])
        return int(results[0]) if results else -1
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠ 隔离核对 gremlin 失败: {str(e)[:120]}")
        return -1


def main() -> int:
    conv_filter = {
        sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--conv" and i + 1 < len(sys.argv)
    }
    config_path = (
        sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv
        else "config/locomo_zh.yaml"
    )
    prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "zhfull"
    force = "--force" in sys.argv
    settings = get_settings(config_path)

    print("=" * 72)
    print(f"中文建图 s2+v8cluster（config={config_path}, prefix={prefix}, force={force}）")
    print(f"  图实例: {settings.hugegraph.graph} @ {settings.hugegraph.url}（user 域隔离，不 reset）")
    print("=" * 72)

    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    convs = sorted({str(c.metadata.get("conversation_id")) for c in ds.cases})
    if conv_filter:
        unknown = conv_filter - set(convs)
        if unknown:
            print(f"!! 未知 conversation: {unknown}（数据集里有 {convs}）")
            return 1
        convs = sorted(conv_filter)
    print(f"目标 conversations: {convs}")

    backend = Mem0Backend(settings)
    store = HugeGraphStore(settings)

    # schema 幂等预建（S1 全量标签 + Concept；与英文同 schema，label 主键相同）
    store.create_schema(make_schema_s1())
    from schema_rsi.graph.schema import make_schema_s2
    store.create_schema(make_schema_s2())

    builder = GraphBuilder(store)
    extractor = StructuredExtractor(settings)
    topic_extractor = TopicExtractor(settings)

    total_report: dict[str, dict] = {}
    for conv in convs:
        uid = f"{prefix}:locomo:{conv}"
        memories = backend.get_all_memories(user_id=uid)
        if not memories:
            print(f"\n[{conv}] !! mem0 无 {uid} 记忆（先跑 scripts/eval_full.py ingest）")
            return 1
        before = zh_counts(store, uid)
        if (not force and before.get("Memory") == len(memories) and before.get("V8Cluster", 0) > 0):
            print(f"\n[{conv}] 已建（Memory {before['Memory']}/{len(memories)}, "
                  f"V8Cluster {before['V8Cluster']}）→ 跳过（--force 重建）")
            total_report[conv] = before
            continue

        print(f"\n[{conv}] {uid}: {len(memories)} 条记忆；现状 {before}")

        # ---- S1：结构化抽取 + 建图（reset=False！）----
        t0 = time.time()
        extraction = extractor.extract(memories)
        n_e = sum(len(v.get("entities", [])) for v in extraction.values())
        n_v = sum(len(v.get("events", [])) for v in extraction.values())
        n_p = sum(len(v.get("preferences", [])) for v in extraction.values())
        n_r = sum(len(v.get("relationships", [])) for v in extraction.values())
        print(f"  S1 提取: {n_e} 实体 / {n_v} 事件 / {n_p} 偏好 / {n_r} 关系 ({time.time()-t0:.0f}s)")
        report = builder.build_s1(memories, extraction, make_schema_s1(), reset=False)
        print(f"  S1 写入: {report.label_counts} | edges={report.edges_written} "
              f"| {report.duration_s}s")

        # ---- S2：主题 Concept（增量，不 reset）----
        t0 = time.time()
        topics = topic_extractor.extract(memories)
        tagged = sum(1 for v in topics.values() if v)
        rep2 = builder.build_concepts(memories, topics)
        print(f"  S2 主题: {tagged}/{len(topics)} 条有主题 → {rep2.label_counts} "
              f"| HAS_TOPIC 边={rep2.edges_written} ({time.time()-t0:.0f}s)")

        # ---- v8cluster：嵌入缓存（data/graph_zh/）+ 每用户独立 k-means ----
        t0 = time.time()
        counts = VARIANTS["v8cluster"].build(store, memories, settings)
        print(f"  v8cluster: {counts} ({time.time()-t0:.0f}s)")

        after = zh_counts(store, uid)
        print(f"  ✓ [{conv}] 完成：{after}")
        total_report[conv] = after

    # ---- 汇总（中文子图精确规模 + 隔离核对）----
    print("\n" + "=" * 72)
    print("建图汇总（按 user_id 前缀精确计数）：")
    sums: dict[str, int] = {}
    for conv, c in total_report.items():
        print(f"  {conv:10} {c}")
        for k, v in c.items():
            sums[k] = sums.get(k, 0) + max(v, 0)
    print(f"  {'合计':10} {sums}")
    # 隔离核对：中文记忆经图跳转（retrieve_fused 路径）带不出异 user_id 的记忆
    leaks = {conv: isolation_check(store, f"{prefix}:locomo:{conv}") for conv in convs}
    print(f"  隔离核对（经图跳转可达的异 user_id 记忆数，应全 0）: {leaks}")
    print("✓ 中文建图完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
