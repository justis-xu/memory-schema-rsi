#!/usr/bin/env python
"""阶段 2d 前置：聚合配对运行 → 逐题两臂对照表（供人工事实核，零调用）。

按题汇总：两臂 judge 通过数/答案多样性、换入/挤出、must_not_evict 侵占、
逐条答案原文（人工核读用）。产出 results/analysis/m2_graph_pair_analysis_20261001.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN = PROJECT_ROOT / "results/analysis/m2_graph_pair_run_20261001.json"
OUT = PROJECT_ROOT / "results/analysis/m2_graph_pair_analysis_20261001.json"


def norm_answer(s: str) -> str:
    return re.sub(r"[\s，。、；：！？\"'（）,.:;!?]", "", (s or "").strip())


def main() -> int:
    run = json.loads(RUN.read_text(encoding="utf-8"))
    rows = []
    for case in run["cases"]:
        if case.get("aborted"):
            rows.append({"case_id": case["case_id"], "aborted": case["aborted"]})
            continue
        calls = [c for c in case.get("calls", []) if "answer" in c]
        arms = {"base": [c for c in calls if c["arm"] == "base"],
                "graph": [c for c in calls if c["arm"] == "graph"]}
        mne_hit = sorted({m["id"] for m in case.get("swapped_out", [])
                          if m["id"] in set(case.get("must_not_evict", []))})
        row = {
            "case_id": case["case_id"], "group": case.get("group"),
            "question": case["question"], "gold": case["gold"],
            "swap_in": [{"id": s["id"], "content_head": s["content"][:60],
                         "admission": s.get("admission_status")} for s in case.get("swapped_in", [])],
            "swap_out": [{"id": s["id"], "content_head": s["content"][:60]} for s in case.get("swapped_out", [])],
            "must_not_evict_violated": mne_hit,
            "arms": {},
        }
        for arm, cs in arms.items():
            votes = [c.get("judge_correct") for c in cs]
            distinct = {norm_answer(c["answer"]) for c in cs}
            row["arms"][arm] = {
                "n": len(cs),
                "judge_pass": sum(1 for v in votes if v),
                "distinct_answers": len(distinct),
                "answers": [{"i": c["index"], "answer": c["answer"][:400],
                             "judge": c.get("judge_correct")} for c in cs],
            }
        rows.append(row)

    agg = {
        "scope": "配对运行聚合：逐题两臂对照 + 换入挤出 + must_not_evict 侵占；人工事实核按 answers 原文读。",
        "questions": len(rows),
        "aborted": sum(1 for r in rows if r.get("aborted")),
        "arm_summary": {},
        "mne_violations": [r["case_id"] for r in rows if r.get("must_not_evict_violated")],
        "rows": rows,
    }
    for arm in ("base", "graph"):
        cs = [r["arms"][arm] for r in rows if arm in r.get("arms", {})]
        agg["arm_summary"][arm] = {
            "questions": len(cs),
            "judge_pass_total": sum(c["judge_pass"] for c in cs),
            "answers_total": sum(c["n"] for c in cs),
        }
    OUT.write_text(json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(agg["arm_summary"], ensure_ascii=False))
    print("must_not_evict 侵占:", agg["mne_violations"] or "无")
    for r in rows:
        if r.get("aborted"):
            continue
        b, g = r["arms"].get("base", {}), r["arms"].get("graph", {})
        print(f"  [{r['group']}] {r['case_id']}: base {b.get('judge_pass',0)}/{b.get('n',0)}"
              f" vs graph {g.get('judge_pass',0)}/{g.get('n',0)}"
              f" 换入{len(r['swap_in'])}换出{len(r['swap_out'])}"
              f"{' ⚠MNE' if r.get('must_not_evict_violated') else ''}")
    print(f"✓ {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
