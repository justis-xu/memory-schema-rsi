"""Pair archived Chinese base/fused answers where a graph-only ID entered context."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
GRAPH = ROOT / "results_zh/zhfull_fused_locomo_20260925_161049.jsonl"
IDS = ROOT / "results/analysis/m2_graph_only_memory_ids_20260926.json"
OUT = ROOT / "results/analysis/m2_graph_only_answer_exposure_20260926.json"


def load(path):
    return {row["case_id"]: row for row in map(json.loads, path.open())}


def context_records(row):
    by_id = {m["id"]: m["content"] for m in row.get("retrieved_memories") or []}
    by_id.update({m["id"]: m["content"] for m in row.get("graph_memories") or []})
    ids = row.get("metadata", {}).get("answer_context_ids") or []
    return ids, by_id


def main():
    graph_only = {item["id"] for item in json.loads(IDS.read_text())["ids"]}
    base, graph = load(BASE), load(GRAPH)
    rows = []
    for case_id, g in graph.items():
        b = base.get(case_id)
        if not b:
            continue
        gid, gtext = context_records(g)
        chosen = [mid for mid in gid if mid in graph_only]
        if not chosen:
            continue
        bid, btext = context_records(b)
        gained = [mid for mid in gid if mid not in bid]
        displaced = [mid for mid in bid if mid not in gid]
        assert all(mid in gtext for mid in gid)
        assert all(mid in btext for mid in bid)
        rows.append({
            "case_id": case_id, "category": g["metadata"].get("category"),
            "question": g["question"], "gold": g["expected_answer"],
            "base_answer": b["predicted_answer"], "graph_answer": g["predicted_answer"],
            "base_archived_judge_correct": (b.get("metrics") or {}).get("judge_correct"),
            "graph_archived_judge_correct": (g.get("metrics") or {}).get("judge_correct"),
            "base_context_ids": bid, "graph_context_ids": gid,
            "selected_graph_only_memories": [{"id": mid, "content": gtext[mid]} for mid in chosen],
            "gained_memories": [{"id": mid, "content": gtext[mid]} for mid in gained],
            "displaced_memories": [{"id": mid, "content": btext[mid]} for mid in displaced],
        })
    assert len(rows) == 8
    counts = {"paired_cases": len(rows), "by_category": dict(Counter(r["category"] for r in rows)),
              "selected_graph_only_occurrences": sum(len(r["selected_graph_only_memories"]) for r in rows),
              "gained_context_memories": sum(len(r["gained_memories"]) for r in rows),
              "displaced_context_memories": sum(len(r["displaced_memories"]) for r in rows),
              "archived_judge_base_only": sum(r["base_archived_judge_correct"] is True and r["graph_archived_judge_correct"] is False for r in rows),
              "archived_judge_graph_only": sum(r["base_archived_judge_correct"] is False and r["graph_archived_judge_correct"] is True for r in rows)}
    OUT.write_text(json.dumps({
        "scope": "One archived Chinese no-graph vs fused-graph pair; cases where a graph-only ID was selected",
        "base_run": str(BASE.relative_to(ROOT)), "graph_run": str(GRAPH.relative_to(ROOT)),
        "counts": counts, "cases": rows,
        "limits": ["Cross-run paired answers are observational; generation and grading noise remain",
                   "The graph-only memory is one of multiple context changes and cannot be isolated as the cause",
                   "Gold support and answer correctness require source/semantic review"],
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
