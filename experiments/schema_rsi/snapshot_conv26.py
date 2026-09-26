#!/usr/bin/env python3
"""Freeze a completed LoCoMo conversation and its Mem0 embeddings for offline replay.

This reads Chroma and an existing result JSONL. It never writes to the live store.
The output belongs under runs/ (ignored by Git): it contains benchmark conversation
and memory text, plus vectors and gold answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "mem0-src"))

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    settings = get_settings()
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    cases = [c for c in ds.cases if c.metadata.get("conversation_id") == args.conv]
    if not cases:
        raise ValueError(f"unknown conversation {args.conv}")
    uid = f"full:locomo:{args.conv}"
    result_rows = {}
    malformed = 0
    with args.baseline.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if row.get("metadata", {}).get("user_id") == uid:
                result_rows[row["case_id"]] = row
    missing = [c.case_id for c in cases if c.case_id not in result_rows]
    if missing:
        raise ValueError(f"baseline missing {len(missing)} cases; first={missing[0]}")

    backend = Mem0Backend(settings)
    records = {r.id: r for r in backend.get_all_memories(uid)}
    collection = backend.raw.vector_store.collection
    raw = collection.get(where={"user_id": uid}, include=["embeddings", "metadatas"])
    vectors = raw["embeddings"]
    if vectors is None or len(raw["ids"]) != len(records):
        raise ValueError("incomplete Chroma embedding snapshot")
    memories = []
    for idx, mid in enumerate(raw["ids"]):
        rec = records[mid]
        memories.append({
            "id": mid, "user_id": uid, "content": rec.content,
            "metadata": rec.metadata, "embedding": vectors[idx].tolist(),
        })
    memories.sort(key=lambda item: item["id"])
    ids = set(records)
    output_cases = []
    for case in cases:
        row = result_rows[case.case_id]
        ranked = [r["id"] for r in row["retrieved_memories"]]
        if any(mid not in ids for mid in ranked):
            raise ValueError(f"baseline IDs missing from snapshot: {case.case_id}")
        output_cases.append({
            "id": case.case_id, "question": case.question, "answer": case.answer,
            "category": case.category, "evidence": case.evidence,
            "base_ranked": ranked, "baseline_correct": row["metrics"].get("judge_correct"),
            "baseline_answer": row["predicted_answer"],
        })
    document = {
        "conv": args.conv, "user_id": uid, "baseline_path": str(args.baseline),
        "baseline_malformed_lines": malformed, "memories": memories,
        "sessions": cases[0].history, "cases": output_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False)
    print(json.dumps({"snapshot": str(args.output), "memories": len(memories),
                      "sessions": len(document["sessions"]), "cases": len(output_cases),
                      "malformed_baseline_lines": malformed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
