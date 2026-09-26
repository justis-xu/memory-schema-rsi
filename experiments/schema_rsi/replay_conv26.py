#!/usr/bin/env python3
"""Replay the full graph lifecycle on frozen Mem0 output without touching live stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from schema_rsi_lab.lifecycle import LifecycleGraph, LifecyclePolicy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--write-knn", type=int, default=5)
    ap.add_argument("--hub-bits", type=int, default=5)
    ap.add_argument("--graph-slots", type=int, default=2)
    ap.add_argument("--min-query-overlap", type=float, default=0.12)
    args = ap.parse_args()
    raw_bytes = args.snapshot.read_bytes()
    raw = json.loads(raw_bytes)
    policy = LifecyclePolicy(write_knn=args.write_knn, hub_bits=args.hub_bits,
                             query_graph_slots=args.graph_slots,
                             min_query_overlap=args.min_query_overlap)
    graph = LifecycleGraph(policy)
    by_session: dict[str, list[dict]] = defaultdict(list)
    for record in raw["memories"]:
        by_session[str(record["metadata"].get("session_id"))].append(record)
    ingest_trace = []
    for session in raw["sessions"]:
        sid = str(session["session_id"])
        pre = graph.before_extract(raw["user_id"], session["turns"])
        writes = sorted(by_session.pop(sid, []), key=lambda item: item["id"])
        # The extracted facts are frozen Mem0 output. This checks graph reads
        # and writes but cannot measure how graph context would alter extraction.
        for record in writes:
            graph.upsert(record)
        ingest_trace.append({"session": sid, "prior_memory_count": len(graph.memories) - len(writes),
                             "pre_read_ids": pre["ids"], "pre_read_graph_added": pre["graph_added"],
                             "pre_read_edge_visits": pre["edge_visits"], "memory_writes": len(writes)})
    if by_session:
        raise ValueError(f"memories outside conversation sessions: {sorted(by_session)}")

    qa = []
    for case in raw["cases"]:
        result = graph.for_question(raw["user_id"], case["question"], case["base_ranked"], context_k=15)
        qa.append({"id": case["id"], "category": case["category"],
                   "baseline_correct": case["baseline_correct"], **result})
    report = {
        "snapshot_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "conv": raw["conv"], "policy": asdict(policy), "graph": graph.stats(),
        "ingest": {"sessions": len(ingest_trace),
                   "sessions_with_graph_context": sum(bool(t["pre_read_graph_added"]) for t in ingest_trace),
                   "pre_read_edge_visits": sum(t["pre_read_edge_visits"] for t in ingest_trace),
                   "memory_writes": sum(t["memory_writes"] for t in ingest_trace),
                   "trace": ingest_trace},
        "retrieval": {"cases": len(qa), "changed": sum(bool(q["graph_added"]) for q in qa),
                      "graph_added": sum(len(q["graph_added"]) for q in qa),
                      "edge_visits": sum(q["edge_visits"] for q in qa),
                      "changed_baseline_wrong": sum(bool(q["graph_added"]) and q["baseline_correct"] is False for q in qa),
                      "changed_baseline_right": sum(bool(q["graph_added"]) and q["baseline_correct"] is True for q in qa),
                      "rows": qa},
        "limitations": [
            "The graph pre-read is replayed against frozen Mem0 facts; it did not influence their extraction.",
            "Selection differences and graph statistics are not answer-accuracy gains.",
            "The result file holds only baseline top-15; graph candidates can be compared at the same 15-memory budget.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "graph": report["graph"],
                      "ingest": {k: v for k, v in report["ingest"].items() if k != "trace"},
                      "retrieval": {k: v for k, v in report["retrieval"].items() if k != "rows"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
