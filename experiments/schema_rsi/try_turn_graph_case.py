#!/usr/bin/env python3
"""One paired answer trial for keyword vs event-conditioned turn graph."""

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
from schema_rsi.memory.base import MemoryRecord  # noqa: E402

from schema_rsi_lab.graph_only import GraphOnlyRetriever  # noqa: E402
from schema_rsi_lab.lifecycle import LifecyclePolicy  # noqa: E402
from schema_rsi_lab.provenance import SourceGraph  # noqa: E402
from schema_rsi_lab.turn_graph import TurnGraph  # noqa: E402
from try_conv26 import build_graph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    snap = json.loads(args.snapshot.read_text(encoding="utf-8"))
    case = next(c for c in snap["cases"] if c["id"] == args.case_id)
    graph = build_graph(snap, LifecyclePolicy(max_visits=40, max_hub_members=12))
    sources = SourceGraph(graph, snap["sessions"])
    retriever = GraphOnlyRetriever(graph, sources)
    turns = TurnGraph(sources)
    route = turns.retrieve(retriever, snap["user_id"], case["question"],
                           slots=2, min_score=0.2, preserve_sources=False,
                           recurrence_only=True)
    settings = get_settings()
    answerer = Answerer(settings)
    judge = make_judge_client(settings)
    if judge is None:
        raise RuntimeError("judge unavailable")
    reference_date = parse_session_date(snap["sessions"][-1].get("date"))
    result = {"id": case["id"], "question": case["question"],
              "gold": case["answer"], "gold_evidence": case["evidence"],
              "baseline_answer": case["baseline_answer"],
              "arms": {}, "method": "event-conditioned turn graph, 15 contexts"}
    for arm, selected in (("keyword", route["keyword"]), ("turn_graph", route["selected"])):
        records = [MemoryRecord(id=mid, content=retriever.content(mid)) for mid in selected]
        try:
            predicted, _ = answerer.answer(case["question"], records,
                                           reference_date=reference_date, benchmark="locomo")
            correct, note = judge_answer(case["question"], case["answer"], predicted, judge,
                                         category=case["category"], preset="official", benchmark="locomo")
            result["arms"][arm] = {"selected": selected, "answer": predicted,
                                   "correct": correct if not note.startswith("judge error") else None,
                                   "judge_note": note}
        except Exception as exc:
            result["arms"][arm] = {"selected": selected,
                                   "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    result["graph_added"] = route["added"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"id": case["id"], "added": route["added"],
                      "keyword_correct": result["arms"]["keyword"].get("correct"),
                      "graph_correct": result["arms"]["turn_graph"].get("correct"),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
