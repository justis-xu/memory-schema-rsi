#!/usr/bin/env python3
"""Bounded fused rerank of frozen Chinese vector and graph candidate lists."""

import hashlib
import json
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.llm.rerank import RerankClient, RerankError


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "results/analysis/m2_rare_entity_graph_priority_20260927.json"
OLD = ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json"
RECENT = ROOT / "results/analysis/m2_current_zh_stratified_retrieval_20260927.json"
OUT = ROOT / "results/analysis/m2_rare_entity_fused_rerank_20260927.json"
CASE_IDS = ("locomo_conv-43_qa25", "locomo_conv-43_qa2", "locomo_conv-43_qa32")
JOHN = "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"
ORDERS = ("reversed", "sorted")
POLICIES = ("baseline", "rare_first")
MAX_CALLS = len(CASE_IDS) * len(ORDERS) * len(POLICIES)


def save(report):
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    settings = get_settings()
    if not settings.rerank_enabled or not settings.rerank.api_key:
        raise SystemExit("rerank not configured")
    client = RerankClient(base_url=settings.rerank.base_url, api_key=settings.rerank.api_key,
                          model=settings.rerank.model)
    graph = json.loads(GRAPH.read_text())
    lookup = {c["case_id"]: c for source in (OLD, RECENT)
              for c in json.loads(source.read_text())["cases"]}
    report = {"scope": "Frozen current Chinese 15 vector + 30 graph candidates reranked to 15; three cases, two graph orderings, two candidate policies; no answer/judge calls",
              "inputs": {"graph": str(GRAPH.relative_to(ROOT)),
                         "old_retrieval": str(OLD.relative_to(ROOT)),
                         "recent_retrieval": str(RECENT.relative_to(ROOT))},
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (GRAPH, OLD, RECENT)},
              "max_logical_rerank_calls": MAX_CALLS, "logical_rerank_calls": 0,
              "stopped_early": False, "runs": [],
              "limits": ["Current reranker may differ from the historical one; service version is not archived",
                         "Candidate order and membership may both change; this is not an order-only causal test",
                         "Final context ID exposure is not answer correctness"]}
    for cid in CASE_IDS:
        case = lookup[cid]
        vec = case["reranked"][:15]
        assert len(vec) == 15
        vector = [{"id": m["id"], "content": m["content"]} for m in vec]
        vector_ids = {m["id"] for m in vector}
        for order in ORDERS:
            graph_run = graph["runs"][order][cid]
            graph_records = {m["id"]: m for pool in ("baseline_top30", "rare_first_top30")
                             for m in graph_run[pool]}
            for policy in POLICIES:
                ids = graph_run["baseline_ids" if policy == "baseline" else "rare_first_ids"]
                assert len(ids) == 30 and len(set(ids)) == 30 and not vector_ids.intersection(ids)
                records = vector + [graph_records[mid] for mid in ids]
                docs = [r["content"] for r in records]
                event = {"case_id": cid, "question": case["question"], "order": order,
                         "policy": policy, "vector_ids": [r["id"] for r in vector],
                         "graph_ids": ids,
                         "documents_sha256": hashlib.sha256(json.dumps(docs, ensure_ascii=False).encode()).hexdigest(),
                         "input_count": len(docs)}
                try:
                    ranked = client.rerank(case["question"], docs, top_n=15)
                    report["logical_rerank_calls"] += 1
                    indices = [r["index"] for r in ranked]
                    if len(indices) != 15 or len(set(indices)) != 15 or any(not 0 <= i < len(docs) for i in indices):
                        event["error_type"] = "invalid_result_shape"
                        report["stopped_early"] = True
                    else:
                        event["endpoint_style"] = client.last_endpoint
                        event["top15"] = [{"rank": i, "id": records[r["index"]]["id"],
                                           "source": "vector" if r["index"] < 15 else "graph",
                                           "score": r["score"]}
                                          for i, r in enumerate(ranked, 1)]
                        event["john_in_top15"] = JOHN in {m["id"] for m in event["top15"]}
                        event["vector_retained"] = sum(m["source"] == "vector" for m in event["top15"])
                except RerankError as exc:
                    report["logical_rerank_calls"] += 1
                    event["error_type"] = type(exc).__name__
                    report["stopped_early"] = True
                report["runs"].append(event)
                save(report)
                if report["stopped_early"]:
                    print("stopped after", report["logical_rerank_calls"], "logical rerank call(s)")
                    return
    print("completed", report["logical_rerank_calls"], "logical rerank calls")
    for event in report["runs"]:
        print(event["case_id"], event["order"], event["policy"],
              "john", event["john_in_top15"], "vector", event["vector_retained"])


if __name__ == "__main__":
    main()
