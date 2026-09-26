#!/usr/bin/env python3
"""Compare equal-budget local routes against LME answer-session IDs.

The cleaned benchmark supplies answer session IDs, not annotated answer turns.
Session coverage is a coarse screening proxy, not answer correctness.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from schema_rsi_lab.graph_only import GraphOnlyRetriever
from schema_rsi_lab.lifecycle import LifecyclePolicy
from schema_rsi_lab.provenance import SourceGraph
from try_conv26 import build_graph


def represented_sessions(ids: list[str], graph, sources) -> set[str]:
    sessions = set()
    for item in ids:
        if item.startswith("T:"):
            turn = sources.turns.get(item[2:])
            if turn:
                sessions.add(turn["session_id"])
        elif item in graph.memories:
            sessions.add(str(graph.memories[item].get("metadata", {}).get("session_id", "")))
    return sessions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    files = sorted(args.snapshots.glob("*.json"))
    if not files:
        raise ValueError("no snapshots")
    rows = []
    for path in files:
        snap = json.loads(path.read_text(encoding="utf-8"))
        case = snap["cases"][0]
        graph = build_graph(snap, LifecyclePolicy(max_visits=40, max_hub_members=12))
        sources = SourceGraph(graph, snap["sessions"])
        retriever = GraphOnlyRetriever(graph, sources)
        keyword = retriever.retrieve(snap["user_id"], case["question"], graph_enabled=False)
        route = retriever.retrieve(snap["user_id"], case["question"], graph_enabled=True)
        answers = set(case["answer_session_ids"])
        baseline = represented_sessions(case["base_ranked"][:15], graph, sources)
        key_sessions = represented_sessions(keyword["selected"], graph, sources)
        graph_sessions = represented_sessions(route["selected"], graph, sources)
        rows.append({
            "id": case["id"], "category": case["category"],
            "baseline_correct": case["baseline_correct"],
            "memories": len(graph.memories), "source_turns": len(sources.turns),
            "answer_session_count": len(answers),
            "baseline_answer_sessions": len(answers & baseline),
            "keyword_answer_sessions": len(answers & key_sessions),
            "graph_answer_sessions": len(answers & graph_sessions),
            "baseline_all_sessions": bool(answers) and answers <= baseline,
            "keyword_all_sessions": bool(answers) and answers <= key_sessions,
            "graph_all_sessions": bool(answers) and answers <= graph_sessions,
            "graph_added": len(route["graph_added"]),
            "keyword_local_ms": keyword["local_ms"], "graph_local_ms": route["local_ms"],
        })
        print(f"{case['id']}: {len(answers & baseline)}/{len(answers)} baseline; "
              f"{len(answers & key_sessions)} keyword; {len(answers & graph_sessions)} graph", flush=True)
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["category"]].append(row)
    summary = {cat: {"cases": len(group),
                     "baseline_correct": sum(r["baseline_correct"] is True for r in group),
                     "baseline_all_sessions": sum(r["baseline_all_sessions"] for r in group),
                     "keyword_all_sessions": sum(r["keyword_all_sessions"] for r in group),
                     "graph_all_sessions": sum(r["graph_all_sessions"] for r in group)}
               for cat, group in by_type.items()}
    output = {"method": "15-context local retrieval against answer-session IDs",
              "limits": "Answer-session coverage is not answer correctness; the baseline and new routes use the same memory store but different retrieval.",
              "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
