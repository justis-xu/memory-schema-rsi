#!/usr/bin/env python3
"""零调用校验：源相对日算术何时可用于中文时间题的证据复核。"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from m2_audit_relative_day_funnel import DATASET, GRADES, RELATIVE_DAYS, RELATIVE_PATTERN, ROOT, SESSION_PATTERN, load_jsonl

OUTPUT = ROOT / "results/analysis/m2_relative_day_gate_poc_20260926.json"
DATE_QUESTION = re.compile(r"什么时候|何时|哪天|哪一天|几号|什么日期|哪日")
GOLD_DAY = re.compile(r"(?<!\d)((?:19|20)\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日")


def source_date(value: str) -> date:
    return datetime.strptime(value.split(" on ", 1)[1], "%d %B, %Y").date()


def main() -> None:
    dataset = json.loads(DATASET.read_text())
    grades = load_jsonl(GRADES)
    counts = Counter()
    cases = []
    for entry in dataset:
        conv_id = entry["sample_id"]
        conversation = entry["conversation"]
        by_id = {
            turn["dia_id"]: (session_id, turn)
            for session_id, turns in conversation.items()
            if SESSION_PATTERN.fullmatch(session_id) and isinstance(turns, list)
            for turn in turns
        }
        for index, qa in enumerate(entry["qa"]):
            case_id = f"locomo_{conv_id}_qa{index}"
            grade = grades.get(case_id)
            if not grade or grade["category"] != "temporal":
                continue
            counts["temporal_J_cases"] += 1
            hits = []
            for dia_id in qa.get("evidence") or []:
                if dia_id not in by_id:
                    continue
                session_id, turn = by_id[dia_id]
                anchor = source_date(conversation[f"{session_id}_date_time"])
                for word in sorted(set(RELATIVE_PATTERN.findall(turn.get("text", "")))):
                    hits.append({
                        "dia_id": dia_id,
                        "session_id": session_id,
                        "session_date": anchor.isoformat(),
                        "word": word,
                        "calculated_date": (anchor + timedelta(days=RELATIVE_DAYS[word])).isoformat(),
                        "turn": turn,
                    })
            if not hits:
                continue
            counts["cases_with_relative_day_in_gold_evidence"] += 1
            calculated = sorted({hit["calculated_date"] for hit in hits})
            gold_days = sorted({
                date(int(year), int(month), int(day)).isoformat()
                for year, month, day in GOLD_DAY.findall(qa["answer"])
            })
            question_gate = bool(DATE_QUESTION.search(qa["question"]))
            selected = question_gate and len(calculated) == 1
            if question_gate:
                counts["question_requests_date"] += 1
            if selected:
                counts["gate_selected"] += 1
                counts["selected_gold_day_present"] += bool(gold_days)
                counts["selected_calculation_matches_gold_day"] += bool(set(calculated) & set(gold_days))
            else:
                counts["gate_excluded"] += 1
            cases.append({
                "case_id": case_id,
                "grade": grade["final"],
                "question": qa["question"],
                "gold": qa["answer"],
                "question_requests_date": question_gate,
                "calculated_dates": calculated,
                "gold_calendar_dates": gold_days,
                "gate_selected": selected,
                "selected_date_matches_gold": bool(set(calculated) & set(gold_days)) if selected else None,
                "source_hits": hits,
            })
    report = {
        "scope": "旧中文无图归档对应的 temporal J 题；源 gold evidence 含昨天/前天/明天/后天",
        "gate": "仅当题面问什么时候/何时/哪天等明确日期，且 gold evidence 中的相对日算术只生成一个不同日期时，给出待核日期。",
        "counts": dict(counts),
        "cases": cases,
        "limits": [
            "这是利用 benchmark gold evidence 的离线证据校验 POC，不是运行时记忆提取或检索策略；gold evidence 本身可能错误。",
            "日期匹配 gold 只校验算术和这个数据集的金标一致性，不能证明事件、人物或答案可靠。",
            "词面问句门只覆盖四个相对日词和列出的问法；其他时间表达不在范围内。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
