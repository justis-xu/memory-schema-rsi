#!/usr/bin/env python3
"""Check no-gold support mining and provisional experience paths on LoCoMo."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from schema_rsi_lab.graph_only import GraphOnlyRetriever, _terms
from schema_rsi_lab.lifecycle import LifecyclePolicy
from schema_rsi_lab.provenance import SourceGraph
from schema_rsi_lab.success_paths import _similarity, budgeted_route
from schema_rsi_lab.support_miner import mine_supports
from try_conv26 import build_graph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--miner", choices=("strict", "weighted"), default="strict")
    args = ap.parse_args()
    rows = []
    for path in sorted(args.snapshots.glob("conv-*_full_frozen.json")):
        snap = json.loads(path.read_text(encoding="utf-8"))
        graph = build_graph(snap, LifecyclePolicy(max_visits=40, max_hub_members=12))
        sources = SourceGraph(graph, snap["sessions"])
        retriever = GraphOnlyRetriever(graph, sources)
        cases = [c for c in snap["cases"] if c["category"] != "adversarial" and c.get("evidence")]
        successes = [c for c in cases if c["baseline_correct"] is True]
        mined = {c["id"]: mine_supports(retriever, snap["user_id"], c["question"],
                                        c["baseline_answer"], mode=args.miner) for c in successes}
        peers = [c for c in successes if mined[c["id"]]]
        df = Counter(term for peer in peers for term in _terms(peer["question"]))
        def similarity(q: str, prior: str) -> float:
            qt, pt = _terms(q), _terms(prior)
            idf = {term: math.log1p((len(peers) + 1) / (df.get(term, 0) + 1))
                   for term in qt | pt}
            return _similarity(qt, pt, idf)
        for case in cases:
            uid = snap["user_id"]
            target = {f"T:{uid}|{eid}" for eid in case["evidence"]}
            selected = retriever.retrieve(uid, case["question"], graph_enabled=False)["selected"]
            direct = mined.get(case["id"], [])
            direct_ids = {item["id"] for item in direct}
            matches = []
            for peer in peers:
                if peer["id"] == case["id"]:
                    continue
                score = similarity(case["question"], peer["question"])
                if score >= 0.5:
                    matches.append((score, peer))
            matches.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
            support_scores: dict[str, float] = defaultdict(float)
            for score, peer in matches[:3]:
                for item in mined[peer["id"]]:
                    support_scores[item["id"]] += score
            route = [mid for mid, _ in sorted(support_scores.items(), key=lambda pair: (-pair[1], pair[0]))]
            augmented = budgeted_route(selected, route, slots=2)
            rows.append({
                "id": case["id"], "conversation_id": snap["conv"],
                "category": case["category"], "baseline_correct": case["baseline_correct"],
                "gold_evidence_count": len(target),
                "mined_supports": direct if case["baseline_correct"] is True else [],
                "mined_gold_hits": len(direct_ids & target) if case["baseline_correct"] is True else None,
                "keyword_all_evidence": target <= set(selected),
                "provisional_path_all_evidence": target <= set(augmented),
                "provisional_path_used": bool(set(route) & set(augmented)),
                "peer_count": len(matches),
            })
        print(f"{snap['conv']}: {len(cases)} cases, {len(peers)} provisional episodes", flush=True)

    successes = [r for r in rows if r["baseline_correct"] is True]
    supported = [r for r in successes if r["mined_supports"]]
    precision_denominator = sum(len(r["mined_supports"]) for r in supported)
    all_by_conv = defaultdict(list)
    for row in rows:
        all_by_conv[row["conversation_id"]].append(row)
    def route_summary(group: list[dict]) -> dict:
        return {"cases": len(group),
                "keyword_covered": sum(r["keyword_all_evidence"] for r in group),
                "provisional_covered": sum(r["provisional_path_all_evidence"] for r in group),
                "gained": sum(r["provisional_path_all_evidence"] and not r["keyword_all_evidence"] for r in group),
                "lost": sum(r["keyword_all_evidence"] and not r["provisional_path_all_evidence"] for r in group),
                "path_used": sum(r["provisional_path_used"] for r in group)}
    output = {
        "method": f"{args.miner} answer and question support locator; leave-one-question-out provisional path",
        "limits": "Support turns are heuristic and may not entail the answer. Baseline correctness comes from benchmark judge; gold turn IDs only score the locator and retrieval. This is not an active verified path graph.",
        "support_mining": {
            "successful_cases": len(successes), "cases_with_candidates": len(supported),
            "candidate_turns": precision_denominator,
            "candidate_gold_hits": sum(r["mined_gold_hits"] for r in supported),
            "cases_any_gold_hit": sum(r["mined_gold_hits"] > 0 for r in supported),
            "cases_all_candidates_gold": sum(r["mined_gold_hits"] == len(r["mined_supports"]) for r in supported),
        },
        "retrieval": route_summary(rows),
        "by_conversation": {conv: route_summary(group) for conv, group in all_by_conv.items()},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: output[k] for k in ("support_mining", "retrieval")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
