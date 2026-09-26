#!/usr/bin/env python3
"""Bound the observed Chroma byte-hash change after a retrieval-only probe."""

import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
PRIOR = ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json"
PROBE = ROOT / "results/analysis/m2_signed_ball_paths_20260926.json"
OUT = ROOT / "results/analysis/m2_signed_ball_integrity_20260926.json"


def main():
    prior = json.loads(PRIOR.read_text())
    probe = json.loads(PROBE.read_text())
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    counts = con.execute("select string_value,count(*) from embedding_metadata where key='user_id' group by string_value order by string_value").fetchall()
    checked = 0
    missing = []
    changed = []
    for case in prior["cases"]:
        for m in case["vector"]:
            checked += 1
            row = con.execute(
                "select em.string_value from embeddings e join embedding_metadata em on em.id=e.id where e.embedding_id=? and em.key='data'",
                (m["id"],),
            ).fetchone()
            if row is None:
                missing.append(m["id"])
            elif row[0] != m["content"]:
                changed.append(m["id"])
    digest = hashlib.sha256(DB.read_bytes()).hexdigest()
    result = {
        "scope": "Post-probe read-only integrity check after Chroma SQLite byte hash changed",
        "probe_hash_before": probe["vector_db_sha256_before"],
        "probe_hash_after": probe["vector_db_sha256_after"],
        "current_file_hash": digest,
        "embedding_count": con.execute("select count(*) from embeddings").fetchone()[0],
        "user_counts": {user: n for user, n in counts},
        "previously_retrieved_instances_checked": checked,
        "prior_candidate_ids_missing": missing,
        "prior_candidate_contents_changed": changed,
        "limits": [
            "The byte-level change is real; this check cannot reconstruct its cause or verify every prior record because no full logical snapshot was saved before the probe.",
            "The probe script did not call memory write APIs, but Chroma initialization or another process may have changed SQLite bytes.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("embedding_count", "previously_retrieved_instances_checked", "prior_candidate_ids_missing", "prior_candidate_contents_changed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
