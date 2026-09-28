#!/usr/bin/env python3
"""Freeze manually reviewed clause support for five Chinese update cases.

This is an offline diagnostic oracle over the archived source packet. It does
not align any Mem0 output or send gold labels into extraction.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/analysis/m2_lme_update_operations_20260928.json"
OUTPUT = ROOT / "results/analysis/m2_lme_clause_support_20260928.json"

# Each reference is (session_id, zero-based turn index, required text fragment).
SPEC = {
    "031748ae": [
        ("early_five_engineers_plus_manager", "早场曾说五名工程师加一名经理", "full", "同场后来由用户纠正，不再作为初始带领人数", [("answer_8748f791_1", 4, "5名工程师加一位经理")]),
        ("early_corrected_four_engineers", "刚任职时带领四名工程师", "full", "用户在同场明确纠正；优先于早场未纠正数字", [("answer_8748f791_1", 10, "带领一个4人的工程师团队")]),
        ("current_five_engineers", "后来带领五名工程师", "full", "跨场的新带领人数，不能覆盖历史四名", [("answer_8748f791_2", 2, "带领一个五人工程师团队")]),
        ("four_includes_self", "早场四名工程师包含用户本人", "partial", "用户只说带领四人；含本人是助手后续概括，不能作为用户明确来源", [("answer_8748f791_1", 10, "带领一个4人的工程师团队"), ("answer_8748f791_1", 11, "含你本人")]),
    ],
    "0ddfec37": [
        ("first_three_months_fifteen", "开始收藏三个月时共有十五颗签名棒球", "full", "第十五颗是该时间窗的收藏数量", [("answer_a22b654d_1", 0, "从三个月前开始收藏以来，这已经是我第15颗签名棒球")]),
        ("later_added_twenty", "后来几个月新增二十颗签名棒球", "full", "新增量属于较晚时间窗，不替换早期十五颗", [("answer_a22b654d_2", 0, "过去几个月我新增了20个签名棒球")]),
        ("current_total_thirty_five", "现在总共有三十五颗签名棒球", "partial", "15 加 20 需假设两窗口衔接且期间无出售或其他变动；原话未直接给当前总量", [("answer_a22b654d_1", 0, "第15颗签名棒球"), ("answer_a22b654d_2", 0, "新增了20个签名棒球")]),
    ],
    "f685340e": [
        ("earlier_weekly_tennis", "过去每周与朋友打网球", "full", "原话有每周频率，但没有说固定在周日", [("answer_25df025b_1", 0, "每周的网球活动")]),
        ("earlier_sunday_plan", "当时计划这周日与朋友在本地公园打网球", "full", "一次计划，不自动等于每周日惯例", [("answer_25df025b_1", 6, "这周日和朋友们在本地公园打网球")]),
        ("earlier_every_sunday", "过去每周日都与朋友在本地公园打网球", "partial", "每周频率加一次周日计划不足以确定固定星期；中英文原话均如此，gold 的周日精度更强", [("answer_25df025b_1", 0, "每周的网球活动"), ("answer_25df025b_1", 6, "这周日和朋友们在本地公园打网球")]),
        ("current_alternate_sundays", "现在每隔一周的周日在附近公园与朋友打网球", "full", "新场直接说明惯常频率", [("answer_25df025b_2", 0, "我们每隔一周都会这样")]),
    ],
    "50635ada": [
        ("silver_eligible", "早期已有资格获得 Premier Silver", "full", "资格模态是用户原话", [("answer_dcd74827_1", 4, "有资格获得 Premier Silver 等级")]),
        ("silver_attained", "早期已经取得 Premier Silver 会员等级", "partial", "原话只明确有资格；gold 以前一等级指代 Silver，取得关系未直接闭合", [("answer_dcd74827_1", 4, "有资格获得 Premier Silver 等级")]),
        ("gold_attained", "后来已经达到 Premier Gold 会员等级", "full", "用户在后场明确说达到", [("answer_dcd74827_2", 2, "刚达到 Premier Gold 会员等级")]),
    ],
    "71315a70": [
        ("earlier_ocean_five_to_six", "同一抽象海洋雕塑早期已投入约五至六小时", "full", "已投入的累计区间", [("answer_c44b9df4_1", 4, "抽象海洋雕塑，到目前为止已经花了大约5-6个小时")]),
        ("later_ocean_ten_to_twelve", "该抽象海洋雕塑后来累计投入十至十二小时", "full", "后场两次确认，仍是累计区间", [("answer_c44b9df4_2", 0, "抽象海洋雕塑上花了不少时间——已经投入了10-12个小时"), ("answer_c44b9df4_2", 10, "我的抽象海洋雕塑已经投入了10-12个小时")]),
        ("sum_ocean_fifteen_to_eighteen", "该雕塑累计投入十五至十八小时", "absent", "两次说的都是同一项目已投入的累计区间，不能相加", [("answer_c44b9df4_1", 4, "到目前为止已经花了大约5-6个小时"), ("answer_c44b9df4_2", 0, "已经投入了10-12个小时")]),
        ("sunset_separate_project", "日落雕塑是另一个计划对象", "full", "后场先问日落作品技巧，随即另提已经在做的抽象海洋雕塑", [("answer_c44b9df4_2", 0, "创作一件以日落为灵感的雕塑")]),
    ],
}


def main() -> None:
    packet_bytes = SOURCE.read_bytes()
    packet = json.loads(packet_bytes)
    cases = {case["question_id"]: case for case in packet["cases"]}
    assert set(cases) == set(SPEC)
    rows = []
    for question_id, claims in SPEC.items():
        case = cases[question_id]
        source = case["source"]["chinese"]
        sessions = {session["session_id"]: session for session in source["answer_sessions"]}
        claim_rows = []
        for claim_id, claim, support, rationale, refs in claims:
            witnesses = []
            for session_id, index, expected in refs:
                session = sessions[session_id]
                turn = session["turns"][index]
                assert turn["index"] == index and expected in turn["content"], (question_id, claim_id, session_id, index)
                witnesses.append({
                    "session_id": session_id,
                    "session_date": session["date"],
                    "source_ref": f"turn:{index}",
                    "role": turn["role"],
                    "text": turn["content"],
                })
            claim_rows.append({"claim_id": claim_id, "claim": claim, "source_support": support,
                               "manual_reason": rationale, "witnesses": witnesses})
        rows.append({"question_id": question_id, "question": source["question"],
                     "gold": source["answer"], "claims": claim_rows})
    counts = Counter(claim["source_support"] for row in rows for claim in row["claims"])
    result = {
        "scope": "五道固定中文 LongMemEval 知识更新题的人工子句来源支持；只用已归档答案场源话",
        "source_packet": str(SOURCE.relative_to(ROOT)),
        "source_packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "support_definition": {
            "full": "所引用户原话在其限定语和时间窗内直接支持该子句",
            "partial": "原话支持部分内容，但某个限定、身份、总量或取得关系未直接闭合",
            "absent": "所引原话不支持该子句，或其时间/累计关系与子句冲突",
        },
        "counts": dict(counts),
        "cases": rows,
        "limits": [
            "这是人工诊断 oracle，不是提取器实际输出事实的自动对齐，也不进入运行提取或检索输入。",
            "仅对历史 packet 中两场指定答案来源作语义审计，未穷尽各题所有背景场次。",
            "source_ref 是已归档答案场内的零起始 turn 位置；当前本机 HF 原文件缺失，无法重算完整 batch_id 与真实适配器请求。",
            "支持强度是按明确用户原话作保守判断；gold 可比原话更具体，不能直接当训练接受条件。",
            "没有模型调用、真实记忆库或图写入，也没有中文答题表现测量。",
        ],
    }
    if OUTPUT.exists():
        raise SystemExit(f"refusing overwrite: {OUTPUT}")
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "claims": sum(len(row["claims"]) for row in rows),
                      "support_counts": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
