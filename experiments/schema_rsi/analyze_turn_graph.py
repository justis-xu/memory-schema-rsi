#!/usr/bin/env python3
"""Offline source-evidence comparison for cross-session turn graph paths."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from schema_rsi_lab.graph_only import GraphOnlyRetriever
from schema_rsi_lab.lifecycle import LifecyclePolicy
from schema_rsi_lab.provenance import SourceGraph
from schema_rsi_lab.turn_graph import TurnGraph
from try_conv26 import build_graph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.2)
    ap.add_argument("--allow-source-replacement", action="store_true")
    ap.add_argument("--recurrence-only", action="store_true")
    args = ap.parse_args()
    rows = []
    builds = []
    for path in sorted(args.snapshots.glob("conv-*_full_frozen.json")):
        snap = json.loads(path.read_text(encoding="utf-8"))
        graph = build_graph(snap, LifecyclePolicy(max_visits=40, max_hub_members=12))
        sources = SourceGraph(graph, snap["sessions"])
        retriever = GraphOnlyRetriever(graph, sources)
        started = time.perf_counter()
        turns = TurnGraph(sources)
        builds.append({"conversation_id": snap["conv"], "build_ms": round((time.perf_counter()-started)*1000, 3),
                       **turns.stats()})
        for case in snap["cases"]:
            if case["category"] == "adversarial" or not case.get("evidence"):
                continue
            target = {f"T:{snap['user_id']}|{eid}" for eid in case["evidence"]}
            result = turns.retrieve(retriever, snap["user_id"], case["question"],
                                    slots=args.slots, min_score=args.min_score,
                                    preserve_sources=not args.allow_source_replacement,
                                    recurrence_only=args.recurrence_only)
            rows.append({"id": case["id"], "conversation_id": snap["conv"],
                         "category": case["category"], "baseline_correct": case["baseline_correct"],
                         "gold_count": len(target),
                         "keyword_hits": len(target & set(result["keyword"])),
                         "turn_graph_hits": len(target & set(result["selected"])),
                         "keyword_all": target <= set(result["keyword"]),
                         "turn_graph_all": target <= set(result["selected"]),
                         "added": len(result["added"]), "local_ms": result["local_ms"]})
        print(f"{snap['conv']}: {turns.stats()['directed_arcs']} directed turn arcs", flush=True)

    def summary(group: list[dict]) -> dict:
        return {"cases": len(group), "keyword_all": sum(r["keyword_all"] for r in group),
                "turn_graph_all": sum(r["turn_graph_all"] for r in group),
                "gained": sum(r["turn_graph_all"] and not r["keyword_all"] for r in group),
                "lost": sum(r["keyword_all"] and not r["turn_graph_all"] for r in group),
                "hit_gain": sum(r["turn_graph_hits"]-r["keyword_hits"] for r in group),
                "mean_added": round(sum(r["added"] for r in group)/len(group),3) if group else 0}
    by_cat = defaultdict(list)
    by_conv = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
        by_conv[row["conversation_id"]].append(row)
    latency = sorted(r["local_ms"] for r in rows)
    output = {"method": "bounded cross-session rare-term source-turn edges; 15 contexts, at most five source turns",
              "policy": {"slots": args.slots, "min_score": args.min_score,
                         "preserve_sources": not args.allow_source_replacement,
                         "recurrence_only": args.recurrence_only},
              "limits": "Exact LoCoMo gold turn ID coverage is not answer accuracy; graph topology uses the same source text as keyword retrieval.",
              "summary": summary(rows), "by_category": {k: summary(v) for k,v in by_cat.items()},
              "by_conversation": {k: summary(v) for k,v in by_conv.items()},
              "local_query_ms": {"median": statistics.median(latency), "p95": latency[int(.95*len(latency))]},
              "builds": builds, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:output[k] for k in ("summary","by_category","local_query_ms")},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
