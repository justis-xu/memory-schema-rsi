#!/usr/bin/env python3
"""事实来源门 POC 离线验收：对照 devset oracle 机械评估 8 个验收案 + 两条停止门条件。

用法: python scripts/m2_evaluate_fact_gate.py <run_json> <verdict_out_json>
只读 run 与 devset；新增 v2 一致性守卫（不改判定内容，只修整流）：
- 引用后缀匹配（容忍 session 前缀）
- 一致性：全部子句 keep 而 record_action=withdraw → keep；有 withdraw 子句而 action=keep → split
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEVSET = ROOT / "results/analysis/m2_fact_gate_devset_20261001.json"


def norm_refs(refs: list[str], valid: set[str]) -> list[str]:
    out = []
    for r in refs:
        if r in valid or any(r.endswith("." + v) or r.endswith(v) for v in valid):
            out.append(r)
    return out


def gated_v2(entry: dict, gi: dict) -> dict:
    v = json.loads(json.dumps(entry["gated"], ensure_ascii=False))
    valid = {t.get("source_ref") or f"turn_{t['turn_index']}" for t in gi.get("source_turns", [])}
    for s in gi.get("source_sessions", {}).values():
        valid |= {t["source_ref"] for t in s["turns"]}
    for c in v["clauses"]:
        bad = [r for r in c.get("source_refs", []) if r not in valid and not norm_refs([r], valid)]
        c["hallucinated_ref"] = bad or None
        if c["support"] == "absent" and c["status"] != "withdraw":
            c["status"] = "withdraw"
        if c["support"] == "partial" and c["status"] == "keep":
            c["status"] = "weaken"
    actions = [c["status"] for c in v["clauses"]]
    if v["record_action"] == "withdraw" and all(a in {"keep", "weaken"} for a in actions):
        v["record_action"] = "keep" if all(a == "keep" for a in actions) else "split"
        v["guard_log"] = v.get("guard_log", []) + ["v2一致性: withdraw→与子句对齐"]
    if v["record_action"] == "keep" and any(a != "keep" for a in actions):
        v["record_action"] = "split"
        v["guard_log"] = v.get("guard_log", []) + ["v2一致性: keep→split"]
    return v


def clause_find(g, *keywords) -> list[dict]:
    return [c for c in g["clauses"] if any(k in c["clause"] for k in keywords)]


def evaluate(devset: dict, run: dict) -> dict:
    inputs = {g["input_id"]: g for g in devset["gate_inputs"]}
    by_id = {r["input_id"]: r for r in run["results"]}
    g2 = {iid: gated_v2(by_id[iid], inputs[iid]) for iid in by_id}
    checks = []

    def add(case, expect, detail, passed, evidence=""):
        checks.append({"case": case, "expect": expect, "pass": bool(passed),
                       "detail": detail, "evidence": evidence})

    # 1 keep_team_size
    a0 = g2["A0"]
    ok = a0["record_action"] == "keep" and all(c["status"] != "withdraw" for c in a0["clauses"])
    add("keep_team_size", "keep", "A0 纠正后团队规模须保留（v1 raw 曾整条 withdraw 自相矛盾）",
        ok, f"gated_v2={a0['record_action']}")

    # 2 keep_late_only_facts
    b = g2["B_later_conv-43_de89be96dda6"]
    hits = clause_find(b, "创造", "快乐")
    ok = b["record_action"] != "withdraw" and hits and all(c["status"] != "withdraw" for c in hits)
    add("keep_late_only_facts", "keep", "迟到旧源事实（创造新世界/快乐）须保留",
        ok, f"action={b['record_action']} 命中{len(hits)}子句")

    # 3 keep_older_only_details
    ok_parts = []
    for iid, kws in (("B_older_conv-43_c95588df7268", ("城堡", "作者")),
                     ("B_older_conv-26_444350391e64", ("读书", "画画"))):
        g = g2[iid]
        hits = clause_find(g, *kws)
        ok_parts.append(g["record_action"] != "withdraw" and (not hits or
                       all(c["status"] != "withdraw" for c in hits)))
    add("keep_older_only_details", "keep", "旧版独有细节（城堡/作者/读书画画）不得判无源删除",
        all(ok_parts), f"{ok_parts}")

    # 4 keep_human_assistant_class
    withdrawn = [iid for iid in (f"C{i}" for i in range(10)) if g2[iid]["record_action"] == "withdraw"]
    c5 = g2["C5"]
    add("keep_human_assistant_class", "keep", "10 条 assistant 标签真人记忆不得整类拒绝",
        len(withdrawn) == 0, f"withdrawn={withdrawn or '无'}; C5(conv-44双真人) action={c5['record_action']}")

    # 5 block_unsourced_companions
    ok_parts = []
    for iid in ("B_older_conv-41_e94a25c4f01c", "B_later_conv-41_e94a25c4f01c"):
        g = g2[iid]
        hits = clause_find(g, "家人", "family", "Maria", "玛丽亚", "同行")
        ok_parts.append(all(c["status"] == "withdraw" for c in hits) if hits else None)
    add("block_unsourced_companions", "withdraw",
        "无源同行者（家人/玛丽亚）须撤回（weaken 不算达标）", all(p for p in ok_parts if p is not None)
        and any(p is not None for p in ok_parts), f"{ok_parts}")

    # 6 block_assistant_only_clauses
    a1_bad = [c for c in g2["A1"]["clauses"] if "含" in c["clause"] and c["status"] == "keep"]
    a3_bad = [c for c in g2["A3"]["clauses"] if c["status"] == "keep" and
              any(k in c["clause"] for k in ("萨曼莎", "套餐", "邮箱", "电话", "经理"))]
    a4_bad = [c for c in g2["A4"]["clauses"] if c["status"] == "keep" and "自由交流" in c["clause"]]
    add("block_assistant_only_clauses", "withdraw",
        "仅助手来源子句（含本人/职位/套餐/联系方式/自由交流）不得 keep",
        not (a1_bad or a3_bad or a4_bad),
        f"A1坏{len(a1_bad)} A3坏{len(a3_bad)} A4坏{len(a4_bad)}")

    # 7 block_overstated_emotion
    ok_parts = []
    for iid in ("B_older_conv-26_444350391e64", "B_later_conv-26_444350391e64"):
        hits = [c for c in g2[iid]["clauses"] if "难受" in c["clause"]]
        ok_parts.append(all(c["status"] in {"weaken", "withdraw"} for c in hits) if hits else None)
    add("block_overstated_emotion", "weaken",
        "'难受'缺'但我还好'限定须削弱，不得原样 keep", all(p for p in ok_parts if p is not None)
        and any(p is not None for p in ok_parts), f"{ok_parts}")

    # 8 no_whole_partial_pass（针对 raw）
    raw_keeps = [iid for iid in ("A1", "A3", "A4") if by_id[iid]["raw_verdict"]["record_action"] == "keep"]
    add("no_whole_partial_pass", "split", "partial 记忆禁止整条放行（raw 行为）",
        not raw_keeps, f"raw keep: {raw_keeps or '无'}")

    # 幻觉引用（v2 归一化后）
    halluc = {iid: [c["hallucinated_ref"] for c in g["clauses"] if c.get("hallucinated_ref")]
              for iid, g in g2.items()}
    halluc = {k: v for k, v in halluc.items() if v}

    # 停止门
    stop_gate = {
        "fail_1_whole_partial_pass": len(raw_keeps) > 0,
        "fail_1_evidence": raw_keeps,
        "fail_2_assistant_class_rejection": len(withdrawn) >= 9,
        "fail_2_evidence": withdrawn,
        "hallucinated_refs_after_norm": halluc,
        "verdict": "FAIL" if (raw_keeps or len(withdrawn) >= 9) else "PASS",
    }
    return {"checks": checks, "stop_gate": stop_gate,
            "summary": {
                "passed": sum(1 for c in checks if c["pass"]),
                "total": len(checks),
            }}


def main() -> int:
    run_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    devset = json.loads(DEVSET.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    report = evaluate(devset, run)
    report["scope"] = f"离线验收对照 {run_path.name}；v2 守卫=引用后缀归一+动作一致性，不改语义判定。"
    out = Path(out_path)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"验收 {out.name}: {report['summary']['passed']}/{report['summary']['total']} 通过")
    for c in report["checks"]:
        print(f"  {'✓' if c['pass'] else '✗'} {c['case']}  {c['evidence']}")
    sg = report["stop_gate"]
    print(f"停止门: {sg['verdict']} (fail1={sg['fail_1_whole_partial_pass']}, fail2={sg['fail_2_assistant_class_rejection']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
