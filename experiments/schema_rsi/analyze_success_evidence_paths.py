#!/usr/bin/env python3
"""Upper-bound probe for learned query-to-evidence hyperedges on LoCoMo.

Prior successful QAs supply gold evidence IDs here, solely as a diagnostic
surrogate for verifier-attested support. A deployed system must record its own
support IDs; benchmark gold is never available to the runtime.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from schema_rsi_lab.graph_only import GraphOnlyRetriever
from schema_rsi_lab.lifecycle import LifecyclePolicy
from schema_rsi_lab.provenance import SourceGraph
from schema_rsi_lab.success_paths import SuccessPathGraph, budgeted_route
from try_conv26 import build_graph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    configs = [(t, op, slots) for t in (0.15, 0.3, 0.5) for op in (False, True) for slots in (1, 2, 3)]
    observations = []
    for path in sorted(args.snapshots.glob("conv-*_full_frozen.json")):
        snap = json.loads(path.read_text(encoding="utf-8"))
        graph = build_graph(snap, LifecyclePolicy(max_visits=40, max_hub_members=12))
        sources = SourceGraph(graph, snap["sessions"])
        retriever = GraphOnlyRetriever(graph, sources)
        cases = [c for c in snap["cases"] if c["category"] != "adversarial" and c.get("evidence")]
        peers = [c for c in cases if c["baseline_correct"] is True]
        path_graph = SuccessPathGraph(sources)
        for peer in peers:
            support = [f"T:{snap['user_id']}|{eid}" for eid in peer["evidence"]
                       if f"{snap['user_id']}|{eid}" in sources.turns]
            if support:
                path_graph.record(user=snap["user_id"], episode_id=peer["id"],
                                  question=peer["question"], support_ids=support,
                                  attested=True,
                                  attestation_source="benchmark_gold")  # diagnostic surrogate
        for case in cases:
            keyword = retriever.retrieve(snap["user_id"], case["question"], graph_enabled=False)["selected"]
            target = {f"T:{snap['user_id']}|{eid}" for eid in case["evidence"]}
            base = target <= set(keyword)
            entry = {"id": case["id"], "conversation_id": snap["conv"], "category": case["category"],
                     "baseline_correct": case["baseline_correct"], "keyword_all_evidence": base,
                     "policies": {}}
            for threshold, same_operation, slots in configs:
                started = time.perf_counter()
                match = path_graph.query(user=snap["user_id"], question=case["question"],
                                         threshold=threshold, same_operation=same_operation,
                                         exclude_episode=case["id"])
                local_ms = (time.perf_counter() - started) * 1000
                route = match["support_ids"]
                selected = budgeted_route(keyword, route, slots=slots)
                key = f"{threshold}:{int(same_operation)}:{slots}"
                entry["policies"][key] = {"all_evidence": target <= set(selected),
                                          "route_used": bool(set(selected) & set(route)),
                                          "peer_similarity": round(match["max_similarity"], 4),
                                          "path_query_ms": round(local_ms, 4)}
            observations.append(entry)
        print(f"{snap['conv']}: {len(cases)} eligible", flush=True)

    conversations = sorted({r["conversation_id"] for r in observations})
    if len(conversations) != 10:
        raise ValueError("expected ten conversations")
    splits = {"train": conversations[:6], "validation": conversations[6:8], "test": conversations[8:]}
    def report(rows: list[dict], key: str) -> dict:
        return {"cases": len(rows),
                "base_covered": sum(r["keyword_all_evidence"] for r in rows),
                "covered": sum(r["policies"][key]["all_evidence"] for r in rows),
                "gained": sum(r["policies"][key]["all_evidence"] and not r["keyword_all_evidence"] for r in rows),
                "lost": sum(r["keyword_all_evidence"] and not r["policies"][key]["all_evidence"] for r in rows),
                "activated": sum(r["policies"][key]["route_used"] for r in rows)}
    train = [r for r in observations if r["conversation_id"] in splits["train"]]
    valid = [r for r in observations if r["conversation_id"] in splits["validation"]]
    test = [r for r in observations if r["conversation_id"] in splits["test"]]
    ranked = sorted(((-report(train, f"{t}:{int(op)}:{slots}")["gained"]
                      + report(train, f"{t}:{int(op)}:{slots}")["lost"],
                      t, op, slots) for t, op, slots in configs))
    initial_key = f"{ranked[0][1]}:{int(ranked[0][2])}:{ranked[0][3]}"
    initial_validation = report(valid, initial_key)
    attempts = [{"policy": initial_key, "train": report(train, initial_key),
                 "validation": initial_validation,
                 "accepted": initial_validation["gained"] > 0 and initial_validation["lost"] == 0}]
    selected_key = initial_key
    if not attempts[0]["accepted"]:
        # Failure-driven refinement: use the best train candidate with no
        # evidence regression, breaking ties by fewer added source slots.
        safe = []
        for threshold, same_operation, slots in configs:
            key = f"{threshold}:{int(same_operation)}:{slots}"
            train_report = report(train, key)
            if train_report["lost"] == 0 and train_report["gained"] > 0:
                safe.append((-train_report["gained"], slots, threshold, same_operation, key))
        if safe:
            selected_key = sorted(safe)[0][-1]
            revised_validation = report(valid, selected_key)
            attempts.append({"policy": selected_key, "train": report(train, selected_key),
                             "validation": revised_validation,
                             "accepted": revised_validation["gained"] > 0 and revised_validation["lost"] == 0})
    promoted = attempts[-1]["accepted"]
    latencies = sorted(r["policies"][selected_key]["path_query_ms"] for r in observations)
    output = {"method": "leave-one-question-out gold-support hyperedge upper-bound",
              "limit": "Prior successful questions use benchmark gold evidence, unavailable in deployment. Target gold is used only for scoring. No answer model called.",
              "splits": splits, "selected_train_policy": initial_key,
              "attempts": attempts, "selected_policy": selected_key,
              "train": report(train, selected_key), "validation": report(valid, selected_key),
              "promoted": promoted,
              "frozen_test": report(test, selected_key) if promoted else None,
              "local_path_query_ms": {"median": statistics.median(latencies),
                                      "p95": latencies[int(0.95 * len(latencies))]},
              "all_policy_train_validation": {f"{t}:{int(op)}:{slots}":
                                              {"train": report(train, f"{t}:{int(op)}:{slots}"),
                                               "validation": report(valid, f"{t}:{int(op)}:{slots}")}
                                              for t, op, slots in configs},
              "rows": observations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: output[k] for k in ("attempts", "promoted", "frozen_test", "local_path_query_ms")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
