#!/usr/bin/env python3
"""Offline evidence and successful-route diagnostic on every LoCoMo conversation.

No answer model is called. Evidence-turn coverage and baseline top-five memory
recall are distinct proxies; neither is answer accuracy. Each question's own
baseline retrieval is excluded from its learned route.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from analyze_success_traces import similarity
from schema_rsi_lab.graph_only import GraphOnlyRetriever, _terms
from schema_rsi_lab.lifecycle import LifecyclePolicy
from schema_rsi_lab.provenance import SourceGraph
from try_conv26 import build_graph


def analyze(snapshot: dict) -> tuple[list[dict], dict]:
    graph = build_graph(snapshot, LifecyclePolicy(max_visits=40, max_hub_members=12))
    sources = SourceGraph(graph, snapshot["sessions"])
    retriever = GraphOnlyRetriever(graph, sources)
    eligible = [c for c in snapshot["cases"] if c["category"] != "adversarial"]
    successes = [c for c in eligible if c["baseline_correct"] is True]
    terms = {c["id"]: _terms(c["question"]) for c in successes}
    df = Counter(t for tokens in terms.values() for t in tokens)
    idf = {t: math.log1p((len(successes) + 1) / (n + 1)) for t, n in df.items()}
    rows = []
    for case in eligible:
        keyword = retriever.retrieve(snapshot["user_id"], case["question"], graph_enabled=False)
        route = retriever.retrieve(snapshot["user_id"], case["question"], graph_enabled=True)
        evidence = {f"T:{snapshot['user_id']}|{eid}" for eid in case.get("evidence") or []}
        row = {
            "id": case["id"], "conversation_id": snapshot["conv"],
            "category": case["category"], "baseline_correct": case["baseline_correct"],
            "evidence_count": len(evidence),
            "keyword_evidence": len(evidence & set(keyword["selected"])),
            "graph_evidence": len(evidence & set(route["selected"])),
            "keyword_all_evidence": bool(evidence) and evidence <= set(keyword["selected"]),
            "graph_all_evidence": bool(evidence) and evidence <= set(route["selected"]),
            "graph_added": len(route["graph_added"]),
            "keyword_local_ms": keyword["local_ms"], "graph_local_ms": route["local_ms"],
        }
        if case["baseline_correct"] is True:
            peers = []
            for prior in successes:
                if prior["id"] == case["id"]:
                    continue
                score = similarity(terms[case["id"]], terms[prior["id"]], idf)
                if score >= 0.15:
                    peers.append((score, prior))
            peers.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
            paths: dict[str, float] = defaultdict(float)
            for score, prior in peers[:3]:
                for rank, mid in enumerate(prior["base_ranked"][:5]):
                    paths[mid] += score / (rank + 1)
            trace = [mid for mid, _ in sorted(paths.items(), key=lambda p: (-p[1], p[0]))][:15]
            # Match the keyword arm's memory/source mix. Otherwise a route made
            # entirely of memory IDs gets more slots for the memory-ID proxy.
            keyword_turns = [mid for mid in keyword["selected"] if mid.startswith("T:")]
            memory_slots = 15 - len(keyword_turns)
            combined = trace[:memory_slots]
            for mid in keyword["selected"]:
                if not mid.startswith("T:") and mid not in combined and len(combined) < memory_slots:
                    combined.append(mid)
            combined += keyword_turns
            target = set(case["base_ranked"][:5])
            row.update({
                "peer_similarity": peers[0][0] if peers else 0,
                "trace_top5_recall": len(target & set(trace)) / len(target) if target else 0,
                "keyword_top5_recall": len(target & set(keyword["selected"])) / len(target) if target else 0,
                "combined_top5_recall": len(target & set(combined)) / len(target) if target else 0,
            })
        rows.append(row)
    return rows, {"memory_nodes": len(graph.memories), "source_turns": len(sources.turns),
                  "latent_hubs": graph.stats()["latent_hubs"], "directed_arcs": graph.stats()["directed_arcs"]}


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    result = {}
    for cat, group in sorted(groups.items()):
        successes = [r for r in group if r["baseline_correct"] is True]
        with_evidence = [r for r in group if r["evidence_count"]]
        result[cat] = {
            "cases": len(group), "baseline_successes": len(successes),
            "keyword_all_evidence": sum(r["keyword_all_evidence"] for r in with_evidence),
            "graph_all_evidence": sum(r["graph_all_evidence"] for r in with_evidence),
            "with_evidence": len(with_evidence),
            "keyword_top5_recall": sum(r["keyword_top5_recall"] for r in successes) / len(successes) if successes else None,
            "trace_top5_recall": sum(r["trace_top5_recall"] for r in successes) / len(successes) if successes else None,
            "combined_top5_recall": sum(r["combined_top5_recall"] for r in successes) / len(successes) if successes else None,
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.snapshots.glob("conv-*_full_frozen.json"))
    if len(paths) != 10:
        raise ValueError(f"expected ten frozen conversations, got {len(paths)}")
    all_rows = []
    per_conv = {}
    for path in paths:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        rows, graph_stats = analyze(snapshot)
        all_rows.extend(rows)
        per_conv[snapshot["conv"]] = {"graph": graph_stats, "summary": summarize(rows)}
        print(f"{snapshot['conv']}: {len(rows)} scored cases", flush=True)
    output = {"method": "keyword vs bounded graph; leave-one-question-out successful-route proxy",
              "limits": "Exact source turn ID coverage and baseline top-five ID recall are not answer correctness. Success routes are learned from other questions in the same conversation.",
              "per_conversation": per_conv, "summary": summarize(all_rows), "rows": all_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "summary": output["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
