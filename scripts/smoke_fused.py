#!/usr/bin/env python
"""融合检索冒烟：少量题目端到端验证 fused 链路（聚合召回→统一重排→作答→判分）。

用法：.venv/bin/python scripts/smoke_fused.py [n_cases]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation import EvaluationPipeline  # noqa: E402
from schema_rsi.evaluation.judge import make_judge_client  # noqa: E402
from schema_rsi.graph import HugeGraphStore  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    settings = get_settings()
    settings.raw.setdefault("evaluation", {})["graph_mode"] = "fused"

    store = HugeGraphStore(settings)
    counts = {lab: store.count_vertices(lab)
              for lab in ("Memory", "Entity", "Event", "Preference", "Concept")}
    print(f"图状态: {counts}")
    assert counts.get("Concept"), "无 Concept 顶点，先跑 build_concepts.py"

    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    # 挑不同类别各 1 题，覆盖 4 条跳转路径
    picked: dict[str, object] = {}
    for case in ds.cases:
        cat = case.category or "?"
        conv = str(case.metadata.get("conversation_id"))
        if cat not in picked and (case.metadata.get("smoke_seen") or conv):
            picked[cat] = case
        if len(picked) >= 4:
            break
    cases = list(picked.values())[:n]

    backend = Mem0Backend(settings)
    pipeline = EvaluationPipeline(
        backend, settings, graph_store=store, judge_client=make_judge_client(settings)
    )
    ok = 0
    for case in cases:
        uid = f"full:locomo:{case.metadata.get('conversation_id')}"
        r = pipeline.evaluate_case(case, graph_enabled=True, user_id=uid)
        md = r.metadata
        via_examples = [g.get("via")[:2] for g in (r.graph_memories or [])[:3]]
        print(f"\n[{case.category}] {case.question[:70]}")
        print(f"  gold: {case.answer[:70]}")
        print(f"  pred: {(r.predicted_answer or '')[:90].replace(chr(10), ' ')}")
        print(f"  judge={r.metrics.get('judge_correct')} | 图候选={len(r.graph_memories or [])} "
              f"入上下文={md.get('fused_graph_in_context')} 池={md.get('fused_pool_size')} "
              f"预算={md.get('answer_context_budget')}")
        print(f"  图候选 via 示例: {via_examples}")
        ok += 1
    print(f"\n✓ 冒烟通过 {ok} 题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
