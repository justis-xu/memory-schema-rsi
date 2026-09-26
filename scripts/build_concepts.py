#!/usr/bin/env python
"""在现有 S1 图上增量构建 Concept（抽象主题）节点 —— Schema S2。

流程：
1. 从 mem0 加载全部 full:locomo:{conv} 记忆
2. TopicExtractor 批量分配受控词表主题（缓存 data/graph/topic_cache.json）
3. GraphBuilder.build_concepts 增量写入 Concept 顶点 + HAS_TOPIC 边（不 reset）

用法：.venv/bin/python scripts/build_concepts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.graph import GraphBuilder, HugeGraphStore  # noqa: E402
from schema_rsi.graph.extractor_topics import TOPIC_VOCAB, TopicExtractor  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402


def main() -> int:
    settings = get_settings()
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    backend = Mem0Backend(settings)
    conv_user = {str(c.metadata.get("conversation_id")):
                 f"full:locomo:{c.metadata.get('conversation_id')}" for c in ds.cases}

    all_memories = []
    for uid in conv_user.values():
        all_memories.extend(backend.get_all_memories(user_id=uid))
    print(f"记忆: {len(all_memories)} 条（{len(conv_user)} conversations）")

    store = HugeGraphStore(settings)
    counts = {lab: store.count_vertices(lab) for lab in ("Memory", "Entity", "Event", "Preference")}
    print(f"现有图（应为 S1）: {counts}")
    if not counts.get("Memory"):
        print("!! 图为空：请先跑 S1 建图（scripts/eval_s1_ab.py 的 build 步骤）")
        return 1

    extractor = TopicExtractor(settings)
    t0 = time.time()
    topics = extractor.extract(all_memories)
    tagged = sum(1 for v in topics.values() if v)
    from collections import Counter
    dist = Counter(t for v in topics.values() for t in v)
    print(f"主题提取: {tagged}/{len(topics)} 条有主题，词表 {len(TOPIC_VOCAB)}；"
          f"top5={dist.most_common(5)} ({time.time()-t0:.0f}s)")

    report = GraphBuilder(store).build_concepts(all_memories, topics)
    print(f"Concept 写入: {report.label_counts} | HAS_TOPIC 边={report.edges_written} "
          f"| {report.duration_s}s")
    print(f"Concept 顶点总数: {store.count_vertices('Concept')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
