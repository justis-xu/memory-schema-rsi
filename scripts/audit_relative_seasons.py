#!/usr/bin/env python3
"""Align LoCoMo 'last summer' turns with Chinese turns and gold evidence.

This lists contexts and associated golds; it does not choose a universal
interpretation of the ambiguous English phrase.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "data/locomo/locomo10.json"
ZH = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUTPUT = ROOT / "results/analysis/last_summer_alignment_20260926.json"


def main() -> None:
    en = {entry["sample_id"]: entry for entry in json.loads(EN.read_text())}
    zh = {entry["sample_id"]: entry for entry in json.loads(ZH.read_text())}
    aligned = []
    for conv, entry in en.items():
        assert len(entry["qa"]) == len(zh[conv]["qa"])
        for session_key, turns in entry["conversation"].items():
            if not re.fullmatch(r"session_\d+", session_key):
                continue
            zh_turns = {turn["dia_id"]: turn for turn in zh[conv]["conversation"][session_key]}
            for turn in turns:
                if not re.search(r"\blast summer\b", turn.get("text") or "", re.I):
                    continue
                dia_id = turn["dia_id"]
                linked = []
                for index, question in enumerate(entry["qa"]):
                    refs = {
                        m.group().replace("D:", "D")
                        for token in question.get("evidence") or []
                        for m in re.finditer(r"D:?(\d+):\d+", str(token))
                    }
                    if dia_id in refs:
                        linked.append({
                            "case_id": f"locomo_{conv}_qa{index}",
                            "category": question["category"],
                            "en_question": question["question"],
                            "en_gold": question.get("answer"),
                            "zh_gold": zh[conv]["qa"][index].get("answer"),
                        })
                aligned.append({
                    "conversation": conv, "dia_id": dia_id, "session": session_key,
                    "session_date": entry["conversation"][f"{session_key}_date_time"],
                    "en_text": turn["text"], "zh_text": zh_turns[dia_id]["text"],
                    "zh_says_previous_year_summer": "去年夏天" in zh_turns[dia_id]["text"],
                    "linked_qa": linked,
                })
    report = {
        "method": "All English source turns containing exact phrase 'last summer', aligned by dia_id to Chinese turns and QA evidence references.",
        "warning": "Linked QA golds may be unrelated to the date or have no answer. The English phrase is context-dependent; this listing is not a translation-error count.",
        "matched_turns": len(aligned),
        "translated_as_previous_year_summer": sum(row["zh_says_previous_year_summer"] for row in aligned),
        "rows": aligned,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"matched_turns": report["matched_turns"], "translated_as_previous_year_summer": report["translated_as_previous_year_summer"], "linked": [(r["conversation"], r["dia_id"], [(q["case_id"], q["en_gold"]) for q in r["linked_qa"] if q["en_gold"]]) for r in aligned]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
