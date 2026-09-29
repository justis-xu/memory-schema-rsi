#!/usr/bin/env python3
"""Freeze manual clause-level review of four Chinese extraction responses."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/analysis/m2_lme_update_extraction_20260929.json"
PACKET = ROOT / "results/analysis/m2_lme_update_operations_20260928.json"
OUT = ROOT / "results/analysis/m2_lme_update_extraction_review_20260929.json"

# Labels concern each exact output fact, not overall answer correctness.
LABELS = {
    ("031748ae", "base"): [
        ("full", "后场本人明确提到高级软件工程师职位。"),
        ("full", "后场本人明确说现在带领五人工程师团队。"),
        ("full", "早场本人明确纠正为带领四名；输出事实正确，但 status 未记纠正关系。"),
        ("full", "早场本人两次称雷切尔为经理。"),
        ("partial", "用户说场地听起来不错、会联系对接人询问；‘计划参加团建’比已定计划更强。"),
        ("full", "后场本人明确选团队徒步。"),
    ],
    ("031748ae", "source_policy"): [
        ("full", "早场本人明确称新职位为高级软件工程师。"),
        ("full", "早场先说五名工程师加经理，后本人明确纠正为四名工程师；来源链完整。"),
        ("full", "早场本人说打算邀请经理雷切尔。"),
        ("full", "用户称场地听起来不错，作意向表述可接受。"),
        ("partial", "用户只说会联系萨曼莎；‘City View Rooftop 活动经理’身份来自助手 turn:5，所引用户 turn:8 不能独立支持。"),
        ("full", "后场本人明确说现在带领五人工程师团队。"),
        ("full", "后场本人说需容纳六人并选团队徒步。"),
    ],
    ("f685340e", "source_policy"): [
        ("full", "早场本人说与朋友们每周打网球。"),
        ("full", "早场本人说上周日在公园看到别人打网球。"),
        ("full", "早场本人计划本周日与朋友打球；场次日期可锚到 2023-03-12。"),
        ("full", "后场本人计划当天周日与朋友在附近公园打球。"),
        ("full", "后场本人说每隔一周这样打球；输出却未保留惯常星期日限定。"),
    ],
    ("f685340e", "base"): [
        ("partial", "每周打球有源；time_scope 写‘周日活动’把一次周日观察/计划推成惯常星期。"),
        ("full", "早场本人计划本周日与朋友打球。"),
        ("full", "后场本人计划当天周日与朋友打球。"),
        ("full", "后场本人说每隔一周这样打球；输出未保留惯常星期日限定。"),
    ],
}

CORE_CHECKS = {
    ("031748ae", "base"): {
        "early_corrected_four_retained": True, "later_five_retained": True,
        "early_correction_link_explicit": False, "retracted_five_as_current": False,
        "unsupported_or_overstated_extra": True,
        "failure_reasons": ["早场四人只写普通陈述，未记五人被纠正。", "场地‘计划参加’强于源话意向。"],
    },
    ("031748ae", "source_policy"): {
        "early_corrected_four_retained": True, "later_five_retained": True,
        "early_correction_link_explicit": True, "retracted_five_as_current": False,
        "unsupported_or_overstated_extra": True,
        "failure_reasons": ["萨曼莎的活动经理身份来自助手，所引用户轮次仅说会联系她。"],
    },
    ("f685340e", "source_policy"): {
        "early_weekly_retained": False, "early_every_sunday_asserted": False,
        "later_alternate_weekly_retained": True, "later_sunday_habit_retained": False,
        "failure_reasons": ["漏早场每周与朋友打球。", "后场‘每隔一周的周日’只留下每隔一周。"],
    },
    ("f685340e", "base"): {
        "early_weekly_retained": True, "early_every_sunday_asserted": True,
        "later_alternate_weekly_retained": True, "later_sunday_habit_retained": False,
        "failure_reasons": ["原始输出带 Markdown 代码围栏，非严格 JSON。", "旧每周活动被写成周日活动，来源不支持固定周日。", "后场惯常星期日限定丢失。"],
    },
}


def parse_raw(raw):
    try:
        return json.loads(raw), True, None
    except ValueError:
        if raw.startswith("```json\n") and raw.endswith("\n```"):
            return json.loads(raw[len("```json\n"):-len("\n```")]), False, "markdown_json_fence"
        raise


def main():
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")
    run_bytes = RUN.read_bytes()
    run = json.loads(run_bytes)
    packet = json.loads(PACKET.read_text())
    by_case = {case["question_id"]: case for case in packet["cases"]}
    by_request = {(r["question_id"], r["arm"]): r for r in run["requests"]}
    reports = []
    for call in run["calls"]:
        key = (call["question_id"], call["arm"])
        request = by_request[key]
        parsed, strict, recovery = parse_raw(call["raw_response"])
        facts = parsed["facts"]
        assert len(facts) == len(LABELS[key])
        source_turns = {}
        for session in by_case[key[0]]["source"]["chinese"]["answer_sessions"]:
            for turn in session["turns"]:
                ref = f"{session['session_id']}:turn:{turn['index']}"
                if ref in request["source_refs"]:
                    source_turns[ref] = {"role": turn["role"], "text": turn["content"]}
        assert set(source_turns) == set(request["source_refs"])
        fact_rows = []
        for index, (fact, (support, reason)) in enumerate(zip(facts, LABELS[key])):
            refs = fact.get("source_turn_ids", [])
            fact_rows.append({"output_index": index, "fact": fact,
                              "source_refs_legal": all(ref in source_turns for ref in refs),
                              "cited_source_turns": [{"source_ref": ref, **source_turns[ref]}
                                                     for ref in refs if ref in source_turns],
                              "manual_support": support, "manual_reason": reason})
        reports.append({"question_id": key[0], "arm": key[1],
                        "raw_response_sha256": hashlib.sha256(call["raw_response"].encode()).hexdigest(),
                        "strict_json": strict, "review_parse_recovery": recovery,
                        "fact_count": len(facts), "facts": fact_rows,
                        "core_checks": CORE_CHECKS[key]})
    assert len(reports) == 4
    assert sum(r["fact_count"] for r in reports) == 22
    result = {
        "scope": "四次固定中文提取输出的逐事实人工源审；不评价答题收益",
        "run_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "reviewed_calls": len(reports),
        "reviewed_output_facts": sum(r["fact_count"] for r in reports),
        "all_cited_refs_legal": all(f["source_refs_legal"] for r in reports for f in r["facts"]),
        "strict_json_calls": sum(r["strict_json"] for r in reports),
        "calls": reports,
        "limits": ["人工支持标签是离线诊断 oracle，不是自动验收器。",
                   "代码围栏响应仅为离线审查剥除围栏，原始协议失败仍保留。",
                   "定向两题每臂一次，不能估模型稳定性、总体收益或答题正确率。"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("reviewed_calls", "reviewed_output_facts",
                                            "all_cited_refs_legal", "strict_json_calls")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
