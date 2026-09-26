#!/usr/bin/env python3
"""Evaluate a separately ingested graph-memory collection on conv-26 cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from schema_rsi.benchmarks.base import parse_session_date  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation.answerer import Answerer  # noqa: E402
from schema_rsi.evaluation.judge import judge_answer, make_judge_client  # noqa: E402
from schema_rsi.evaluation.retriever import Mem0Retriever  # noqa: E402
from schema_rsi.llm.chat import make_chat_client  # noqa: E402
from schema_rsi.memory import Mem0Backend, build_mem0_config  # noqa: E402

from schema_rsi_lab.lifecycle import LifecycleGraph, LifecyclePolicy  # noqa: E402
from schema_rsi_lab.provenance import SourceGraph  # noqa: E402
from schema_rsi_lab.verify import repair_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--ingest-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--correct-controls", type=int, default=8)
    args = ap.parse_args()
    snap = json.loads(args.snapshot.read_text(encoding="utf-8"))
    summary = json.loads((args.ingest_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("errors"):
        raise ValueError("isolated ingestion had errors")
    settings = get_settings()
    cfg = build_mem0_config(settings)
    cfg["vector_store"]["config"]["path"] = str((args.ingest_dir / "chroma").resolve())
    cfg["vector_store"]["config"]["collection_name"] = "schema_rsi_conv26_graph_ingest"
    cfg["history_db_path"] = str((args.ingest_dir / "history.db").resolve())
    backend = Mem0Backend(settings, config=cfg)
    durable = backend.get_all_memories(snap["user_id"])
    if len(durable) != summary["durable_memories"]:
        raise ValueError("durable memory count changed after ingestion")
    rows = backend.raw.vector_store.collection.get(where={"user_id": snap["user_id"]},
                                                    include=["embeddings"])
    by_id = {r.id: r for r in durable}
    embedded = {mid: vector.tolist() for mid, vector in zip(rows["ids"], rows["embeddings"])}
    graph = LifecycleGraph(LifecyclePolicy(max_visits=40, max_hub_members=12,
                                           min_query_overlap=0.3))
    write_order = []
    for line in (args.ingest_dir / "ingest.jsonl").open(encoding="utf-8"):
        write_order.extend(json.loads(line).get("new_memory_ids", []))
    if set(write_order) != set(by_id) or len(write_order) != len(by_id):
        raise ValueError("ingest journal does not match durable memory IDs")
    for mid in write_order:
        rec = by_id[mid]
        graph.upsert({"id": rec.id, "user_id": snap["user_id"],
                      "content": rec.content, "metadata": rec.metadata,
                      "embedding": embedded[rec.id]})
    sources = SourceGraph(graph, snap["sessions"])

    original_wrong = [c for c in snap["cases"] if c["baseline_correct"] is False and c["category"] != "adversarial"]
    controls = [c for c in snap["cases"] if c["baseline_correct"] is True and c["category"] != "adversarial"][:args.correct_controls]
    selected = original_wrong + controls
    retriever = Mem0Retriever(backend, settings)
    answerer = Answerer(settings)
    verifier = make_chat_client(settings)
    judge = make_judge_client(settings)
    if judge is None:
        raise RuntimeError("judge unavailable")
    reference_date = parse_session_date(snap["sessions"][-1].get("date"))
    memories = {rec.id: {"id": rec.id, "user_id": snap["user_id"],
                         "content": rec.content, "metadata": rec.metadata} for rec in durable}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.open(encoding="utf-8"):
            if line.strip():
                old = json.loads(line)
                if "error" not in old:
                    completed.add(str(old["id"]))
    with args.output.open("a", encoding="utf-8") as stream:
        for index, case in enumerate(selected, 1):
            if str(case["id"]) in completed:
                continue
            try:
                retrieved = retriever.retrieve(case["question"], user_id=snap["user_id"], top_k=15)
                initial, _ = answerer.answer(case["question"], retrieved, reference_date=reference_date,
                                             benchmark="locomo")
                initial_correct, _ = judge_answer(case["question"], case["answer"], initial, judge,
                                                  category=case["category"], preset="official", benchmark="locomo")
                adapted = {**case, "base_ranked": [r.id for r in retrieved],
                           "baseline_answer": initial, "baseline_correct": initial_correct}
                repaired = repair_case(adapted, memories, graph, sources, answerer, verifier,
                                       reference_date=reference_date)
                if repaired.get("accepted"):
                    final_correct, _ = judge_answer(case["question"], case["answer"], repaired["final_answer"], judge,
                                                    category=case["category"], preset="official", benchmark="locomo")
                else:
                    final_correct = initial_correct
                row = {"id": case["id"], "original_baseline_correct": case["baseline_correct"],
                       "new_baseline_correct": initial_correct, "final_correct": final_correct,
                       "new_memory_count": len(durable), "retrieved_ids": adapted["base_ranked"],
                       **repaired}
            except Exception as exc:
                row = {"id": case["id"], "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"progress": f"{index}/{len(selected)}", "id": case["id"],
                              "new_baseline_correct": row.get("new_baseline_correct"),
                              "final_correct": row.get("final_correct"),
                              "action": row.get("action"), "error": row.get("error")}, ensure_ascii=False), flush=True)
    latest = {}
    for line in args.output.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            latest[str(row["id"])] = row
    print(json.dumps({"completed": sum("error" not in row for row in latest.values()),
                      "pending_retry": [cid for cid, row in latest.items() if "error" in row]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
