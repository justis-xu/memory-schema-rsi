#!/usr/bin/env python3
"""Offline source-turn candidate comparison: legacy ASCII vs Chinese bigrams."""

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/schema_rsi"))
from schema_rsi_lab.lifecycle import _tokens as ascii_tokens  # noqa: E402
from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402

DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
LABELS = ROOT / "results/analysis/m2_current_zh_stratified_labels_20260927.json"
OUT = ROOT / "results/analysis/m2_zh_source_tokenizer_comparison_20260927.json"
CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
TOP_K = (2, 15)


def bigram_tokens(value):
    out = set(ascii_tokens(value))
    for run in CJK_RUN.findall(value):
        out.update(run[i:i + 2] for i in range(len(run) - 1))
    return out


def rank(question_terms, turns):
    if not question_terms:
        return {"query_terms": 0, "shared_candidates": 0, "gate_candidates": 0,
                "top2": [], "top15": []}
    scored = []
    for turn_id, turn_terms in turns:
        overlap = len(question_terms & turn_terms) / len(question_terms)
        if overlap:
            scored.append((turn_id, overlap))
    scored.sort(key=lambda x: (-x[1], x[0]))
    gate = [(tid, score) for tid, score in scored if score >= 0.3]
    return {
        "query_terms": len(question_terms),
        "shared_candidates": len(scored),
        "gate_candidates": len(gate),
        "top2": [{"turn_id": tid, "overlap": round(score, 4)} for tid, score in gate[:2]],
        "top15": [{"turn_id": tid, "overlap": round(score, 4)} for tid, score in gate[:15]],
    }


def summarize(rows, method):
    vals = [row[method] for row in rows]
    return {
        "cases": len(rows),
        "nonempty_queries": sum(v["query_terms"] > 0 for v in vals),
        "any_shared_candidate": sum(v["shared_candidates"] > 0 for v in vals),
        "any_gate_candidate": sum(v["gate_candidates"] > 0 for v in vals),
        "total_shared_candidates": sum(v["shared_candidates"] for v in vals),
        "total_gate_candidates": sum(v["gate_candidates"] for v in vals),
        "cases_with_at_least_15_gate_candidates": sum(v["gate_candidates"] >= 15 for v in vals),
    }


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    cases = LocomoDataset(path=DATASET).cases
    sources = {}
    for case in cases:
        cid = case.metadata["conversation_id"]
        if cid in sources:
            continue
        turns = [turn for session in case.history for turn in session["turns"]]
        sources[cid] = {
            "ascii": [(turn["dia_id"], ascii_tokens(turn["content"])) for turn in turns],
            "cjk_bigram": [(turn["dia_id"], bigram_tokens(turn["content"])) for turn in turns],
        }
    rows = []
    for case in cases:
        cid = case.metadata["conversation_id"]
        rows.append({
            "case_id": case.case_id, "conversation_id": cid, "category": case.category,
            "ascii": rank(ascii_tokens(case.question), sources[cid]["ascii"]),
            "cjk_bigram": rank(bigram_tokens(case.question), sources[cid]["cjk_bigram"]),
        })
    labels = json.loads(LABELS.read_text())
    audited = []
    by_id = {row["case_id"]: row for row in rows}
    for item in labels["cases"]:
        row = by_id[item["case_id"]]
        source_ids = set(item["source_turns"])
        audited.append({
            "case_id": item["case_id"], "label": item["label"],
            "audited_turn_ids": sorted(source_ids),
            **{method: {
                "audited_in_top2": sorted(source_ids & {m["turn_id"] for m in row[method]["top2"]}),
                "audited_in_top15": sorted(source_ids & {m["turn_id"] for m in row[method]["top15"]}),
                "gate_candidates": row[method]["gate_candidates"],
            } for method in ("ascii", "cjk_bigram")},
        })
    result = {
        "scope": "Read-only source-turn lexical candidate comparison on the same 1,986 Chinese LoCoMo questions and 5,882 source turns; no model calls",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "methods": {
            "ascii": "Existing schema_rsi_lab.lifecycle._tokens, overlap >=0.3",
            "cjk_bigram": "Same ASCII tokens plus contiguous CJK character bigrams, overlap >=0.3; no learned embeddings or stopword tuning",
        },
        "ranking": "All source turns within the same conversation ranked by fraction of distinct question tokens present; score descending then dia_id ascending; top2/top15 after >=0.3 gate",
        "ascii": summarize(rows, "ascii"), "cjk_bigram": summarize(rows, "cjk_bigram"),
        "audited_20": audited, "by_case": rows,
        "limits": ["This is a candidate-generation ablation, not the full SourceGraph with memory links, sessions, graph hops or answer calls", "Audited turn IDs are diagnostic source references and may include misleading or incomplete benchmark evidence; top-k exposure is not answer support", "No claim of Chinese answer accuracy, Graph/Jev benefit or deployable tokenizer quality", "CJK bigrams may rank common names and question wording above the fact-bearing turn"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for method in ("ascii", "cjk_bigram"):
        print(method, result[method])
    for method in ("ascii", "cjk_bigram"):
        print(method, "audited any top2/top15",
              sum(bool(a[method]["audited_in_top2"]) for a in audited),
              sum(bool(a[method]["audited_in_top15"]) for a in audited))


if __name__ == "__main__":
    main()
