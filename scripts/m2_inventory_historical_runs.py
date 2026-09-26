"""Inventory archived run coverage and logging fields without rerunning models."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/m2_historical_run_inventory_20260926.json"
DATASET_SIZE = {"locomo": 1986, "longmemeval_s_en": 500, "longmemeval_s_zh": 470}
CONFIG_KEYS = (
    "graph_enabled", "graph_mode", "graph_seed_k", "graph_slots",
    "atomic_pool", "answerer_model", "answer_samples",
)


def inspect_jsonl(path):
    rows, case_ids = 0, set()
    fields, benchmarks, languages, categories, config_variants = Counter(), Counter(), Counter(), Counter(), Counter()
    malformed = 0
    for line in path.open():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        rows += 1
        if not isinstance(row, dict):
            fields["non_object"] += 1
            continue
        cid = row.get("case_id") or row.get("id")
        if cid is not None:
            case_ids.add(str(cid))
        if row.get("benchmark"):
            benchmarks[str(row["benchmark"])] += 1
        question = str(row.get("question") or "")
        if question:
            languages["zh" if any("\u4e00" <= ch <= "\u9fff" for ch in question) else "non_zh"] += 1
        metadata = row.get("metadata") or {}
        cat = metadata.get("category") or row.get("category")
        if cat is not None:
            categories[str(cat)] += 1
        for key in ("question", "expected_answer", "predicted_answer", "retrieved_memories", "graph_memories", "metrics", "metadata"):
            if key in row:
                fields[key] += 1
        for key in ("answer_context_ids", "graph_context_ids", "answer_candidates", "decision_model"):
            if key in metadata:
                fields[f"metadata.{key}"] += 1
        if row.get("benchmark") and "predicted_answer" in row:
            cfg = tuple((k, str(metadata.get(k))) for k in CONFIG_KEYS)
            config_variants[str(cfg)] += 1
    kind = "evaluation" if fields["predicted_answer"] else "derived_or_experiment"
    return {
        "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
        "kind": kind, "rows": rows, "unique_case_ids": len(case_ids), "malformed_lines": malformed,
        "benchmarks": dict(benchmarks), "question_language_rows": dict(languages),
        "categories": dict(categories), "field_presence": dict(fields),
        "config_variant_count": len(config_variants),
        "config_variants": [{"signature": k, "rows": v} for k, v in config_variants.most_common(3)],
    }


def main():
    run_paths = sorted((ROOT / "results").glob("*.jsonl"))
    run_paths += sorted((ROOT / "results_zh").glob("*.jsonl"))
    search_paths = sorted((ROOT / "data/search/rounds").glob("*.jsonl"))
    experiment_paths = sorted((ROOT / "experiments/schema_rsi/runs").rglob("*.jsonl"))
    files = [inspect_jsonl(p) for p in run_paths + search_paths + experiment_paths]
    evals = [f for f in files if f["kind"] == "evaluation"]
    counts = {
        "jsonl_files": len(files), "evaluation_files": len(evals),
        "search_round_files": len(search_paths), "experiment_jsonl_files": len(experiment_paths),
        "chinese_locomo_eval_files": sum(bool(f["benchmarks"].get("locomo", 0) and f["question_language_rows"].get("zh", 0)) for f in evals),
        "longmemeval_eval_files": sum(bool(f["benchmarks"].get("longmemeval_s")) for f in evals),
        "longmemeval_eval_files_with_chinese_questions": sum(bool(f["benchmarks"].get("longmemeval_s") and f["question_language_rows"].get("zh")) for f in evals),
    }
    chinese = [f for f in evals if f["benchmarks"].get("locomo") and f["question_language_rows"].get("zh")]
    selected = sorted(chinese, key=lambda f: f["path"])
    dataset_path = Path("/Users/xu/git/eval-datasets/locomo-zh/locomo10_zh.json")
    raw = json.loads(dataset_path.read_text())
    expected = {f"locomo_{conv['sample_id']}_qa{i}": str(q.get("category"))
                for conv in raw for i, q in enumerate(conv["qa"])}
    del raw
    observed = {}
    for file in selected:
        ids = set()
        for line in (ROOT / file["path"]).open():
            if line.strip():
                ids.add(json.loads(line)["case_id"])
        observed[file["path"]] = ids
    all_sets = list(observed.values())
    common_ids = set.intersection(*all_sets)
    union_ids = set.union(*all_sets)
    missing_by_run = {name: sorted(set(expected) - ids) for name, ids in observed.items()}
    missing_all = set(expected) - common_ids
    coverage = {
        "dataset_case_ids": len(expected), "run_count": len(selected),
        "case_ids_in_all_runs": len(common_ids), "case_ids_in_any_run": len(union_ids),
        "case_ids_missing_from_at_least_one_run": len(missing_all),
        "case_ids_missing_from_every_run": len(set(expected) - union_ids),
        "missing_by_run": missing_by_run,
        "missing_category_codes": dict(Counter(expected[cid] for cid in missing_all)),
        "unexpected_case_ids_by_run": {name: sorted(ids - set(expected)) for name, ids in observed.items() if ids - set(expected)},
    }
    payload = {
        "scope": "Top-level JSONL runs in results/ and results_zh/, search-round JSONL, experiment JSONL; derived JSON and binary stores listed separately by docs",
        "counts": counts,
        "chinese_locomo_runs": selected,
        "chinese_locomo_case_coverage": coverage,
        "longmemeval_runs": [f for f in evals if f["benchmarks"].get("longmemeval_s")],
        "all_jsonl_manifest": files,
        "limits": ["Question language uses a Chinese-character heuristic, not dataset file provenance",
                   "A row count is not an independent sample count or causal comparison",
                   "Historical run names alone do not prove shared memory store, prompt, random seed or model settings"],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"counts": counts, "chinese_run_rows": [(f["path"], f["rows"], f["unique_case_ids"], f["config_variant_count"]) for f in selected]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
