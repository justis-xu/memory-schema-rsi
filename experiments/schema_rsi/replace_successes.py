#!/usr/bin/env python3
"""Can a vector-free graph route preserve already-correct LoCoMo answers?"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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

from try_conv26 import build_graph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--all-successes", action="store_true")
    ap.add_argument("--resume", action="store_true", help="append completed retries to an existing JSONL")
    ap.add_argument("--max-new", type=int, default=0, help="limit newly processed cases in this invocation")
    args = ap.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    graph = build_graph(snapshot, LifecyclePolicy(max_visits=40, max_hub_members=12))
    sources = SourceGraph(graph, snapshot["sessions"])
    retriever = GraphOnlyRetriever(graph, sources)
    categories = ("single-hop", "multi-hop", "temporal", "open-domain")
    eligible = [c for c in snapshot["cases"] if c["baseline_correct"] is True and c["category"] in categories]
    if args.all_successes:
        selected = eligible
    else:
        by_category = defaultdict(list)
        for case in eligible:
            by_category[case["category"]].append(case)
        selected = [case for category in categories for case in by_category[category][:args.per_category]]

    settings = get_settings()
    answerer = Answerer(settings)
    judge = make_judge_client(settings)
    if judge is None:
        raise RuntimeError("judge unavailable")
    reference_date = parse_session_date(snapshot["sessions"][-1].get("date"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        with args.output.open(encoding="utf-8") as previous:
            for line in previous:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing[item["id"]] = item
    with args.output.open("a" if args.resume else "x", encoding="utf-8") as stream:
        processed = 0
        for index, case in enumerate(selected, 1):
            row = existing.get(case["id"], {"id": case["id"], "category": case["category"],
                                            "baseline_correct": True, "arms": {}})
            def completed(arm: str) -> bool:
                item = row["arms"].get(arm, {})
                return "correct" in item and not str(item.get("judge_note", "")).startswith("judge error")
            if all(completed(arm)
                   for arm in ("keyword", "graph")):
                continue
            if args.max_new and processed >= args.max_new:
                break
            processed += 1
            for arm in ("keyword", "graph"):
                if completed(arm):
                    continue
                try:
                    selection = retriever.retrieve(snapshot["user_id"], case["question"],
                                                   graph_enabled=arm == "graph", context_k=15)
                    records = [MemoryRecord(id=mid, content=retriever.content(mid))
                               for mid in selection["selected"]]
                    predicted, _ = answerer.answer(case["question"], records,
                                                   reference_date=reference_date, benchmark="locomo")
                    correct, note = judge_answer(case["question"], case["answer"], predicted, judge,
                                                 category=case["category"], preset="official", benchmark="locomo")
                    if note.startswith("judge error"):
                        raise RuntimeError(note)
                    row["arms"][arm] = {**selection, "answer": predicted,
                                        "correct": correct, "judge_note": note,
                                        "baseline_overlap": len(set(selection["selected"]) & set(case["base_ranked"]))}
                except Exception as exc:
                    row["arms"][arm] = {"error": f"{type(exc).__name__}: {str(exc)[:250]}"}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"progress": f"{index}/{len(selected)}", "id": case["id"],
                              "keyword": row["arms"]["keyword"].get("correct"),
                              "graph": row["arms"]["graph"].get("correct"),
                              "graph_added": len(row["arms"]["graph"].get("graph_added", []))},
                             ensure_ascii=False), flush=True)
    print(json.dumps({"selected": len(selected), "by_category": dict(Counter(c["category"] for c in selected)),
                      "output": str(args.output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
