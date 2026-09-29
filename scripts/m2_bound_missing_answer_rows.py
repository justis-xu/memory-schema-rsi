"""Sharp full-dataset exact-score bounds for missing rows in archived runs.

Only missing outcomes vary. All recorded outcomes are held fixed; the bounds
do not address generation noise, judge errors, or incompatible interventions.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "results/analysis/m2_historical_run_inventory_20260926.json"
OUT = ROOT / "results/analysis/m2_missing_case_score_bounds_20260929.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path: Path) -> dict[str, int]:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        case_id = row["case_id"]
        assert case_id not in rows, (path, case_id)
        exact = row["metrics"]["exact"]
        assert isinstance(exact, bool), (path, case_id, exact)
        rows[case_id] = int(exact)
    return rows


def bound(a: dict[str, int], b: dict[str, int], universe: set[str]) -> dict:
    assert set(a) <= universe and set(b) <= universe
    shared = set(a) & set(b)
    only_a = set(a) - shared
    only_b = set(b) - shared
    neither = universe - (set(a) | set(b))
    paired_difference = sum(a[i] - b[i] for i in shared)
    # On only-A rows B can be 0 or 1; on only-B rows A can be 0 or 1.
    lower = paired_difference + sum(a[i] - 1 for i in only_a) - sum(b[i] for i in only_b) - len(neither)
    upper = paired_difference + sum(a[i] for i in only_a) + sum(1 - b[i] for i in only_b) + len(neither)
    assert lower <= upper and upper - lower == len(universe) - len(shared)
    return {
        "paired_questions": len(shared),
        "missing_from_a": len(only_b) + len(neither),
        "missing_from_b": len(only_a) + len(neither),
        "missing_from_either": len(universe) - len(shared),
        "paired_exact_difference_a_minus_b": paired_difference,
        "full_1986_exact_difference_lower": lower,
        "full_1986_exact_difference_upper": upper,
        "sign_robust_to_missing_only": "positive" if lower > 0 else "negative" if upper < 0 else "undetermined",
    }


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    run_specs = inventory["chinese_locomo_runs"]
    assert len(run_specs) == 8
    names = [spec["path"] for spec in run_specs]
    no_graph = [n for n in names if Path(n).name.startswith("zhfull_locomo_")]
    graph = [n for n in names if Path(n).name.startswith("zhfull_fused_locomo_")]
    jev = [n for n in names if "_jev" in Path(n).name]
    assert (len(no_graph), len(graph), len(jev)) == (3, 3, 2)
    universe = set()
    runs = {}
    hashes = {}
    for name in names:
        path = ROOT / name
        hashes[name] = digest(path)
        runs[name] = load_run(path)
        universe |= set(runs[name])
        assert len(runs[name]) == next(s["rows"] for s in run_specs if s["path"] == name)
    assert len(universe) == inventory["chinese_locomo_case_coverage"]["dataset_case_ids"] == 1986
    assert all(universe - set(runs[n]) == set(inventory["chinese_locomo_case_coverage"]["missing_by_run"][n]) for n in names)
    comparisons = []
    for group, lefts, rights in (("graph_minus_no_graph", graph, no_graph),
                                 ("jev_minus_graph", jev, graph)):
        for a in lefts:
            for b in rights:
                comparisons.append({"group": group, "a": a, "b": b, **bound(runs[a], runs[b], universe)})
    summary = {}
    for group in ("graph_minus_no_graph", "jev_minus_graph"):
        group_rows = [x for x in comparisons if x["group"] == group]
        summary[group] = {
            "comparisons": len(group_rows),
            "robust_positive": sum(x["sign_robust_to_missing_only"] == "positive" for x in group_rows),
            "robust_negative": sum(x["sign_robust_to_missing_only"] == "negative" for x in group_rows),
            "undetermined": sum(x["sign_robust_to_missing_only"] == "undetermined" for x in group_rows),
        }
    OUT.write_text(json.dumps({
        "scope": "Fixed eight Chinese LoCoMo archived runs; exact metric; full 1986-question bounds when each missing exact outcome is independently 0 or 1. No causal interpretation.",
        "inventory_sha256": digest(INVENTORY),
        "run_sha256": hashes,
        "dataset_questions": len(universe),
        "summary": summary,
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
