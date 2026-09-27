"""Verify local HF tree payloads by content, and identify later local files."""

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TREE_DIR = ROOT / ".cache/huggingface/trees"
OUT = ROOT / "results/analysis/m2_hf_payload_integrity_20260927.json"
DATASETS = {
    "locomo_en": ROOT / "data/locomo/locomo10.json",
    "locomo_zh": Path("/Users/xu/git/eval-datasets/locomo-zh/locomo10_zh.json"),
    "lme_en": Path("/Users/xu/git/eval-datasets/longmemeval-zh/longmemeval_s_cleaned.json"),
    "lme_zh": Path("/Users/xu/git/eval-datasets/longmemeval-zh/longmemeval_s_cleaned_zh.json"),
}


def digest(path: Path, method: str) -> str:
    h = hashlib.new(method)
    if method == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    trees = [(p, json.loads(p.read_text())["files"]) for p in TREE_DIR.glob("*.json")]
    if not trees:
        raise FileNotFoundError(TREE_DIR)
    tree_path, entries = max(trees, key=lambda item: len(item[1]))
    statuses = []
    for rel, meta in sorted(entries.items()):
        path = ROOT / rel
        row = {"path": rel, "size_expected": meta["size"]}
        if not path.is_file():
            row["status"] = "missing"
        else:
            row["size_actual"] = path.stat().st_size
            if "lfs_sha256" in meta:
                expected, actual = meta["lfs_sha256"], digest(path, "sha256")
                row["hash_type"] = "lfs_sha256"
            else:
                expected, actual = meta["blob_id"], digest(path, "sha1")
                row["hash_type"] = "git_blob_sha1"
            row["status"] = "match" if actual == expected else "content_changed"
            if actual != expected:
                row.update(hash_expected=expected, hash_actual=actual)
        statuses.append(row)

    roots = (ROOT / "data", ROOT / "results", ROOT / "results_zh", ROOT / "experiments/schema_rsi/runs")
    local_files = {str(p.relative_to(ROOT)) for folder in roots for p in folder.rglob("*") if p.is_file()}
    extra = sorted(local_files - set(entries))
    payload = {
        "scope": "Local cached HF tree only; no remote freshness check; benchmark files checked for presence and size only",
        "manifest": str(tree_path.relative_to(ROOT)),
        "manifest_entries": len(entries),
        "manifest_status_counts": dict(Counter(row["status"] for row in statuses)),
        "manifest_content_changes": [row for row in statuses if row["status"] != "match"],
        "manifest_matches_by_root": dict(Counter(row["path"].split("/")[0] for row in statuses if row["status"] == "match")),
        "local_files_outside_manifest_by_root": dict(Counter(path.split("/")[0] for path in extra)),
        "local_files_outside_manifest": extra,
        "benchmark_files": {name: {"path": str(path), "present": path.is_file(), "size": path.stat().st_size if path.is_file() else None}
                            for name, path in DATASETS.items()},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "local_files_outside_manifest"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
