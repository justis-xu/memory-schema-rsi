#!/usr/bin/env python3
"""Paired diagnostic: baseline answers vs graph-aware verify/repair on conv-26."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation.answerer import Answerer  # noqa: E402
from schema_rsi.evaluation.judge import judge_answer, make_judge_client  # noqa: E402
from schema_rsi.llm.chat import make_chat_client  # noqa: E402

from schema_rsi_lab.lifecycle import LifecycleGraph, LifecyclePolicy  # noqa: E402
from schema_rsi_lab.provenance import SourceGraph  # noqa: E402
from schema_rsi_lab.verify import repair_case  # noqa: E402


def build_graph(snapshot: dict, policy: LifecyclePolicy) -> LifecycleGraph:
    graph = LifecycleGraph(policy)
    by_session: dict[str, list[dict]] = defaultdict(list)
    for record in snapshot["memories"]:
        by_session[str(record["metadata"].get("session_id"))].append(record)
    for session in snapshot["sessions"]:
        for record in sorted(by_session.pop(str(session["session_id"]), []), key=lambda r: r["id"]):
            graph.upsert(record)
    if by_session:
        raise ValueError("snapshot has memories without a session")
    return graph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--correct-controls", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--case-id", action="append", default=[])
    args = ap.parse_args()
    snap = json.loads(args.snapshot.read_text(encoding="utf-8"))
    graph = build_graph(snap, LifecyclePolicy(query_graph_slots=2, max_visits=40,
                                             min_query_overlap=0.3, max_hub_members=12))
    sources = SourceGraph(graph, snap["sessions"])
    eligible = [c for c in snap["cases"] if c["category"] != "adversarial"]
    wrong = [c for c in eligible if c["baseline_correct"] is False]
    controls = [c for c in eligible if c["baseline_correct"] is True][:args.correct_controls]
    selected = wrong + controls
    if args.case_id:
        selected = [c for c in snap["cases"] if c["id"] in set(args.case_id)]
        if len(selected) != len(set(args.case_id)):
            raise ValueError("some --case-id values are absent from the snapshot")
    if args.max_cases:
        selected = selected[:args.max_cases]
    settings = get_settings()
    answerer = Answerer(settings)
    verifier = make_chat_client(settings)
    judge = make_judge_client(settings)
    if judge is None:
        raise RuntimeError("judge client unavailable")
    memories = {m["id"]: m for m in snap["memories"]}
    from schema_rsi.benchmarks.base import parse_session_date
    date = parse_session_date(snap["sessions"][-1].get("date"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("x", encoding="utf-8") as stream:
        for index, case in enumerate(selected, 1):
            try:
                row = repair_case(case, memories, graph, sources, answerer, verifier, reference_date=date)
                if row["accepted"]:
                    correct, note = judge_answer(case["question"], case["answer"], row["final_answer"], judge,
                                                 category=case["category"], preset="official", benchmark="locomo")
                    row["final_correct"] = correct
                    row["judge_note"] = note
                else:
                    row["final_correct"] = case["baseline_correct"]
            except Exception as exc:
                row = {"id": case["id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"progress": f"{index}/{len(selected)}", "id": case["id"],
                              "action": row.get("action"), "baseline_correct": row.get("baseline_correct"),
                              "final_correct": row.get("final_correct"), "error": row.get("error")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
