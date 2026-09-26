#!/usr/bin/env python3
"""Leave-one-question-out probe: can successful retrieval traces route future queries?"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from schema_rsi_lab.graph_only import GraphOnlyRetriever, _terms
from schema_rsi_lab.lifecycle import LifecyclePolicy
from schema_rsi_lab.provenance import SourceGraph
from try_conv26 import build_graph


def similarity(left: set[str], right: set[str], idf: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    intersection = sum(idf.get(term, 0.0) for term in left & right)
    union = sum(idf.get(term, 0.0) for term in left | right)
    return intersection / union if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    snap = json.loads(args.snapshot.read_text(encoding="utf-8"))
    successes = [c for c in snap["cases"] if c["baseline_correct"] is True and c["category"] != "adversarial"]
    graph = build_graph(snap, LifecyclePolicy(max_visits=40, max_hub_members=12))
    sources = SourceGraph(graph, snap["sessions"])
    keyword = GraphOnlyRetriever(graph, sources)
    tokens = {c["id"]: _terms(c["question"]) for c in successes}
    df = Counter(term for ts in tokens.values() for term in ts)
    idf = {term: math.log1p((len(successes) + 1) / (count + 1)) for term, count in df.items()}

    rows = []
    for case in successes:
        peers = []
        for prior in successes:
            if prior["id"] == case["id"]:
                continue  # no direct answer-route leakage
            score = similarity(tokens[case["id"]], tokens[prior["id"]], idf)
            if score >= 0.15:
                peers.append((score, prior))
        peers.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        selected_peers = peers[:3]
        paths: dict[str, float] = defaultdict(float)
        for score, prior in selected_peers:
            for rank, mid in enumerate(prior["base_ranked"][:5]):
                paths[mid] += score / (rank + 1)
        trace_memories = [mid for mid, _ in sorted(paths.items(), key=lambda pair: (-pair[1], pair[0]))][:15]
        # Fill unused slots with a vector-free lexical route, which is also
        # evaluated separately; the trace contribution remains explicit.
        lexical = keyword.retrieve(snap["user_id"], case["question"], graph_enabled=False)
        selected = trace_memories.copy()
        for mid in lexical["selected"]:
            if mid not in selected and len(selected) < 15:
                selected.append(mid)
        target = set(case["base_ranked"][:5])
        rows.append({
            "id": case["id"], "category": case["category"],
            "peer_ids": [peer["id"] for _, peer in selected_peers],
            "peer_similarity": [round(score, 4) for score, _ in selected_peers],
            "trace_memories": trace_memories, "selected": selected,
            "target_top5": list(target),
            "trace_top5_recall": len(target & set(trace_memories)) / 5,
            "combined_top5_recall": len(target & set(selected)) / 5,
            "keyword_top5_recall": len(target & set(lexical["selected"])) / 5,
        })
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    summary = {cat: {"cases": len(group),
                     "trace_top5_recall": sum(x["trace_top5_recall"] for x in group) / len(group),
                     "combined_top5_recall": sum(x["combined_top5_recall"] for x in group) / len(group),
                     "keyword_top5_recall": sum(x["keyword_top5_recall"] for x in group) / len(group)}
               for cat, group in by_cat.items()}
    output = {"method": "leave-one-question-out successful-trace hyperedges",
              "note": "Target is the baseline top-five retrieval IDs, not verified answer evidence; same conversation may share topics.",
              "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
