#!/usr/bin/env python3
"""Conservative explicit year/month comparison of English and Chinese LoCoMo golds.

Only flags disjoint explicit year or month sets. Relative dates and number words
need manual source-turn review; a clean result is not translation validation.
"""

from __future__ import annotations

import calendar
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "data/locomo/locomo10.json"
ZH = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUTPUT = ROOT / "results/analysis/zh_gold_date_audit_20260926.json"
EN_MONTHS = {
    spelling.lower(): number
    for number in range(1, 13)
    for spelling in (calendar.month_name[number], calendar.month_abbr[number])
}
ZH_MONTHS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
}


def en_dates(value: str) -> tuple[set[int], set[int]]:
    years = {int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", value)}
    months = {
        number for word, number in EN_MONTHS.items()
        if re.search(rf"\b{re.escape(word)}\b", value, re.I)
    }
    return years, months


def zh_dates(value: str) -> tuple[set[int], set[int]]:
    years = {int(x) for x in re.findall(r"(?:19|20)\d{2}(?=年|\b)", value)}
    months = {int(x) for x in re.findall(r"(?<!\d)(1[0-2]|[1-9])月", value)}
    months |= {ZH_MONTHS[x] for x in re.findall(r"(十一|十二|十|[一二三四五六七八九])月", value)}
    return years, months


def main() -> None:
    en = {row["sample_id"]: row for row in json.loads(EN.read_text())}
    zh = {row["sample_id"]: row for row in json.loads(ZH.read_text())}
    assert en.keys() == zh.keys()
    paired = explicit_year_pairs = explicit_month_pairs = 0
    mismatches = []
    for conv in sorted(en):
        assert len(en[conv]["qa"]) == len(zh[conv]["qa"])
        for index, (source, translated) in enumerate(zip(en[conv]["qa"], zh[conv]["qa"])):
            paired += 1
            en_y, en_m = en_dates(str(source.get("answer") or ""))
            zh_y, zh_m = zh_dates(str(translated.get("answer") or ""))
            explicit_year_pairs += bool(en_y and zh_y)
            explicit_month_pairs += bool(en_m and zh_m)
            if (en_y and zh_y and en_y.isdisjoint(zh_y)) or (en_m and zh_m and en_m.isdisjoint(zh_m)):
                mismatches.append({
                    "case_id": f"locomo_{conv}_qa{index}", "category": source["category"],
                    "en_question": source["question"], "zh_question": translated["question"],
                    "en_gold": source["answer"], "zh_gold": translated["answer"],
                    "en_years": sorted(en_y), "zh_years": sorted(zh_y),
                    "en_months": sorted(en_m), "zh_months": sorted(zh_m),
                })
    report = {
        "scope": "Explicit year/month values in aligned English and Chinese LoCoMo gold answers only.",
        "warning": "A value detected on only one side is not flagged. The audit does not validate relative dates, source consistency, numbers expressed as words, or other translation content.",
        "paired_qa": paired, "paired_explicit_years": explicit_year_pairs,
        "paired_explicit_months": explicit_month_pairs, "mismatches": mismatches,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
