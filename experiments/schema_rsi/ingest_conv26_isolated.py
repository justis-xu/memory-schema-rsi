#!/usr/bin/env python3
"""Run graph-conditioned Mem0 extraction in a new, isolated local collection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from schema_rsi.benchmarks.base import parse_session_date  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.memory import Mem0Backend, build_mem0_config  # noqa: E402

from schema_rsi_lab.lifecycle import LifecycleGraph, LifecyclePolicy, add_session_with_graph  # noqa: E402
from schema_rsi_lab.provenance import SourceGraph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="new directory under ignored runs/")
    args = ap.parse_args()
    snap = json.loads(args.snapshot.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    cfg = build_mem0_config(settings)
    cfg["vector_store"]["config"]["path"] = str((args.output / "chroma").resolve())
    cfg["vector_store"]["config"]["collection_name"] = "schema_rsi_conv26_graph_ingest"
    cfg["history_db_path"] = str((args.output / "history.db").resolve())
    backend = Mem0Backend(settings, config=cfg)
    policy = LifecyclePolicy(max_visits=40, max_hub_members=12, min_query_overlap=0.3)
    graph = LifecycleGraph(policy)
    sources = SourceGraph(graph, [])
    errors = []
    trace_path = args.output / "ingest.jsonl"
    with trace_path.open("x", encoding="utf-8") as trace:
        for index, session in enumerate(snap["sessions"], 1):
            messages = [{"role": t.get("role", "user"), "content": t["content"]}
                        for t in session["turns"]]
            date = parse_session_date(session.get("date"))
            if date:
                messages.insert(0, {"role": "system", "content": (
                    f"Conversation date: {date}. Use this date to resolve relative time "
                    "expressions like 'yesterday', 'next month', 'in three weeks'.")})
            t0 = time.perf_counter()
            try:
                records, pre = add_session_with_graph(
                    backend, graph, snap["user_id"], messages,
                    metadata={"benchmark": "locomo", "session_id": session["session_id"],
                              "session_date": session.get("date")},
                    source_graph=sources, source_session=session,
                )
                row = {"session": session["session_id"], "index": index,
                       "pre_read_ids": pre["ids"], "pre_read_graph_added": pre["graph_added"],
                       "new_memory_ids": [r.id for r in records], "new_memory_count": len(records),
                       "elapsed_s": round(time.perf_counter() - t0, 3)}
            except Exception as exc:
                row = {"session": session["session_id"], "index": index,
                       "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                       "elapsed_s": round(time.perf_counter() - t0, 3)}
                errors.append(row)
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace.flush()
            print(json.dumps({"session": index, "of": len(snap["sessions"]),
                              "memories": row.get("new_memory_count"),
                              "prior_graph_context": len(row.get("pre_read_ids", [])),
                              "error": row.get("error")}, ensure_ascii=False), flush=True)
    durable = backend.get_all_memories(snap["user_id"])
    result = {"output": str(args.output), "durable_memories": len(durable),
              "graph": graph.stats(), "source_nodes": len(sources.turns),
              "source_links": sum(map(len, sources.memory_to_turns.values())),
              "errors": errors}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if not errors and len(durable) == len(graph.memories) else 1


if __name__ == "__main__":
    raise SystemExit(main())
