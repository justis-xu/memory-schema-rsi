#!/usr/bin/env python3
"""Freeze completed LongMemEval cases for read-only graph diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "mem0-src"))

from schema_rsi.benchmarks.longmemeval import LongMemEvalDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = {}
    with args.baseline.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["case_id"] in rows:
                raise ValueError(f"duplicate case: {row['case_id']}")
            rows[row["case_id"]] = row
    ds = LongMemEvalDataset(get_settings().longmemeval_path)
    ds.load()
    cases = {c.case_id: c for c in ds.cases if c.case_id in rows}
    if set(cases) != set(rows):
        raise ValueError("result rows absent from dataset")
    backend = Mem0Backend(get_settings())
    collection = backend.raw.vector_store.collection
    args.output_dir.mkdir(parents=True)
    for qid, case in cases.items():
        uid = f"full:lme:{qid}"
        records = {r.id: r for r in backend.get_all_memories(uid)}
        raw = collection.get(where={"user_id": uid}, include=["embeddings"])
        vectors = raw["embeddings"]
        if vectors is None or len(raw["ids"]) != len(records):
            raise ValueError(f"incomplete memory snapshot for {qid}")
        memories = []
        for index, mid in enumerate(raw["ids"]):
            record = records[mid]
            memories.append({"id": mid, "user_id": uid, "content": record.content,
                             "metadata": record.metadata, "embedding": vectors[index].tolist()})
        ranked = [memory["id"] for memory in rows[qid]["retrieved_memories"]]
        if not set(ranked) <= set(records):
            raise ValueError(f"baseline memory IDs missing: {qid}")
        document = {
            "conv": qid, "user_id": uid, "memories": memories, "sessions": case.history,
            "cases": [{"id": qid, "question": case.question, "answer": case.answer,
                       "category": case.category, "answer_session_ids": case.metadata["answer_session_ids"],
                       "question_date": case.metadata["question_date"],
                       "base_ranked": ranked,
                       "baseline_correct": rows[qid]["metrics"].get("judge_correct"),
                       "baseline_answer": rows[qid]["predicted_answer"]}],
        }
        (args.output_dir / f"{qid}.json").write_text(json.dumps(document, ensure_ascii=False))
        print(f"{qid}: {len(memories)} memories, {len(case.history)} sessions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
