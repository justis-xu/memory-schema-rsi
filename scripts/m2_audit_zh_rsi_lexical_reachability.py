#!/usr/bin/env python3
"""Read-only Chinese reachability audit for the English-only RSI lexical gates."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/schema_rsi"))
from schema_rsi_lab.graph_only import _terms  # noqa: E402
from schema_rsi_lab.lifecycle import _tokens  # noqa: E402

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from m2_audit_zh_graph_cache_alignment import current_memories, signature  # noqa: E402


DATASET_DIR = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh")
OUT = ROOT / "results/analysis/m2_zh_rsi_lexical_reachability_20260927.json"
MANIFEST = ROOT / "results/analysis/m2_current_zh_stratified_manifest_20260927.json"


def source_index(cases):
    conversations = {}
    for case in cases:
        cid = case.metadata["conversation_id"]
        if cid not in conversations:
            turns = [turn for session in case.history for turn in session["turns"]]
            conversations[cid] = [(_tokens(turn["content"]), turn["dia_id"]) for turn in turns]
    return conversations


def coverage(question_terms, indexed_terms, threshold):
    if not question_terms:
        return {"any_shared": False, "passes_gate": False, "max_overlap": 0.0}
    mx = max((len(question_terms & terms) / len(question_terms) for terms in indexed_terms), default=0.0)
    return {"any_shared": mx > 0, "passes_gate": mx >= threshold, "max_overlap": round(mx, 4)}


def analyze_dataset(path, memories=None):
    cases = LocomoDataset(path=path).cases
    sources = source_index(cases)
    memory_by_conv = defaultdict(list)
    if memories is not None:
        for mid, m in memories.items():
            uid = m["user_id"] or ""
            if uid.startswith("zhfull:locomo:"):
                memory_by_conv[uid.removeprefix("zhfull:locomo:")].append(_tokens(m["content"] or ""))
    rows = []
    for case in cases:
        cid = case.metadata["conversation_id"]
        qt = _tokens(case.question)
        assert bool(qt) == bool(_terms(case.question)), "tokenizer reachability diverged"
        row = {
            "case_id": case.case_id, "conversation_id": cid, "category": case.category,
            "question_has_lexical_terms": bool(qt),
            "question_terms": sorted(qt),
            "only_numeric_terms": bool(qt) and all(t.isdigit() for t in qt),
            "source_turn": coverage(qt, (terms for terms, _ in sources[cid]), 0.3),
        }
        if memories is not None:
            row["current_memory"] = coverage(qt, memory_by_conv[cid], 0.12)
        rows.append(row)
    summary = {
        "questions": len(rows),
        "zero_lexical_terms": sum(not r["question_has_lexical_terms"] for r in rows),
        "only_numeric_terms": sum(r["only_numeric_terms"] for r in rows),
        "source_turn_any_shared": sum(r["source_turn"]["any_shared"] for r in rows),
        "source_turn_passes_0_3_gate_upper_bound": sum(r["source_turn"]["passes_gate"] for r in rows),
        "source_turn_count": sum(map(len, sources.values())),
        "source_turn_zero_lexical_terms": sum(not terms for turns in sources.values() for terms, _ in turns),
        "by_category": {},
    }
    for category in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == category]
        summary["by_category"][category] = {
            "questions": len(sub),
            "zero_lexical_terms": sum(not r["question_has_lexical_terms"] for r in sub),
            "source_turn_passes_0_3_gate_upper_bound": sum(r["source_turn"]["passes_gate"] for r in sub),
        }
    if memories is not None:
        summary["current_memory_any_shared"] = sum(r["current_memory"]["any_shared"] for r in rows)
        summary["current_memory_passes_0_12_gate_upper_bound"] = sum(r["current_memory"]["passes_gate"] for r in rows)
    return summary, rows


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    memories = current_memories()
    zh = [m for m in memories.values() if (m["user_id"] or "").startswith("zhfull:locomo:")]
    assert len(zh) == 1800
    english_path = DATASET_DIR / "locomo10.json"
    chinese_path = DATASET_DIR / "locomo10_zh.json"
    en_summary, en_rows = analyze_dataset(english_path)
    zh_summary, zh_rows = analyze_dataset(chinese_path, memories)
    assert len(en_rows) == len(zh_rows) == 1986
    assert [r["case_id"] for r in en_rows] == [r["case_id"] for r in zh_rows]
    selected = {r["case_id"] for r in json.loads(MANIFEST.read_text())["cases"]}
    result = {
        "scope": "Exact legacy RSI tokenizer and lexical gates on English/Chinese LoCoMo source turns and current zhfull Chroma memories; no model calls",
        "code_paths": ["experiments/schema_rsi/schema_rsi_lab/lifecycle.py:_tokens/_overlap",
                       "experiments/schema_rsi/schema_rsi_lab/graph_only.py:_terms",
                       "experiments/schema_rsi/schema_rsi_lab/provenance.py:SourceGraph.for_question"],
        "dataset_sha256": {"english": hashlib.sha256(english_path.read_bytes()).hexdigest(),
                           "chinese": hashlib.sha256(chinese_path.read_bytes()).hexdigest()},
        "current_memory_id_content_user_session_sha256": signature(memories),
        "current_zhfull_memories": len(zh),
        "current_zhfull_memories_zero_lexical_terms": sum(not _tokens(m["content"] or "") for m in zh),
        "english": en_summary, "chinese": zh_summary,
        "selected_prior_20": [r for r in zh_rows if r["case_id"] in selected],
        "chinese_by_case": zh_rows,
        "limits": ["Source turn gate is an upper bound across all turns in the same conversation; actual SourceGraph uses restricted seeds, sessions and slots", "Current memory gate is an upper bound across all same-user current memories; actual LifecycleGraph requires a neighbor candidate first", "The audit concerns the experiments/schema_rsi lexical prototype, not production Mem0 vector search or HugeGraph", "ASCII tokens in Chinese questions may be years or English names, not semantically adequate evidence", "No answer, graph or judge calls"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for key in ("english", "chinese"):
        print(key, result[key]["questions"], "zero query terms", result[key]["zero_lexical_terms"],
              "source gate upper bound", result[key]["source_turn_passes_0_3_gate_upper_bound"])
    print("current zhfull memory zero terms", result["current_zhfull_memories_zero_lexical_terms"])


if __name__ == "__main__":
    main()
