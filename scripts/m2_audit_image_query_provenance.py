#!/usr/bin/env python3
"""审计中文 LoCoMo 图片 query/caption 的输入来源与归档记忆；零模型调用。"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from schema_rsi.benchmarks.locomo import _turn_text

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
ARCHIVES = ROOT / "results_zh"
OUTPUT = ROOT / "results/analysis/m2_image_query_provenance_20260926.json"
TARGET_CASE = "locomo_conv-50_qa65"
TARGET_MEMORY = "a98ed3ea-f911-4666-92ae-15f4904cc492"
SAMPLE_SEED = 20260926
SAMPLE_SIZE = 30
CLEAR_SAMPLE_CONFLICTS = {
    ("conv-26", "D7:11"): "query 与对话均为《Becoming Nicole》，caption 却是船上的狗",
    ("conv-26", "D10:16"): "query 与对话均为流星雨，caption 却是飞机尾迹",
    ("conv-48", "D10:13"): "query 与对话均为艾森豪威尔矩阵，caption 却是纸上的剪刀",
}


def load_jsonl_case(path: Path, case_id: str) -> dict:
    matches = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("case_id") == case_id:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {case_id}, got {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--archives", type=Path, default=ARCHIVES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    counts = Counter()
    query_caption_turns = []
    sunrise_sunset = []
    conv50 = None
    for entry in dataset:
        conv_id = entry["sample_id"]
        conv = entry["conversation"]
        if conv_id == "conv-50":
            conv50 = entry
        for session_id, turns in conv.items():
            if not re.fullmatch(r"session_\d+", session_id) or not isinstance(turns, list):
                continue
            counts["sessions"] += 1
            for turn in turns:
                counts["turns"] += 1
                caption = turn.get("blip_caption") or ""
                query = turn.get("query") or ""
                img_url = turn.get("img_url")
                counts["turns_with_caption"] += bool(caption)
                counts["turns_with_query"] += bool(query)
                counts["turns_with_img_url"] += bool(img_url)
                counts["turns_with_query_and_caption"] += bool(query and caption)
                if not query or not caption:
                    continue
                if query not in _turn_text(turn) or caption not in _turn_text(turn):
                    raise AssertionError(f"adapter omitted image metadata: {conv_id} {turn.get('dia_id')}")
                item = {
                    "conversation_id": conv_id,
                    "session_id": session_id,
                    "dia_id": turn.get("dia_id"),
                    "text": turn.get("text"),
                    "query": query,
                    "blip_caption": caption,
                }
                query_caption_turns.append(item)
                q_words = set(re.findall(r"[a-z]+", query.lower()))
                c_words = set(re.findall(r"[a-z]+", caption.lower()))
                if ("sunrise" in q_words and "sunset" in c_words) or (
                    "sunset" in q_words and "sunrise" in c_words
                ):
                    sunrise_sunset.append(item)

    assert conv50 is not None
    sample = random.Random(SAMPLE_SEED).sample(query_caption_turns, SAMPLE_SIZE)
    for item in sample:
        key = (item["conversation_id"], item["dia_id"])
        if key in CLEAR_SAMPLE_CONFLICTS:
            item["manual_note"] = CLEAR_SAMPLE_CONFLICTS[key]
    if len([x for x in sample if "manual_note" in x]) != len(CLEAR_SAMPLE_CONFLICTS):
        raise AssertionError("fixed sample changed; review manual notes")

    session = conv50["conversation"]["session_27"]
    target = next(turn for turn in session if turn["dia_id"] == "D27:6")
    qa = conv50["qa"][65]
    if qa["evidence"] != ["D27:6"]:
        raise AssertionError("target QA evidence changed")
    other_waterfall_mentions = [
        turn["dia_id"] for turn in session
        if "waterfall" in ((turn.get("text") or "") + " " + (turn.get("blip_caption") or "")).lower()
        or "瀑布" in ((turn.get("text") or "") + " " + (turn.get("blip_caption") or ""))
    ]
    if other_waterfall_mentions:
        raise AssertionError(f"waterfall appears in text/caption: {other_waterfall_mentions}")
    white_mountains_mentions = []
    for session_id, turns in conv50["conversation"].items():
        if not re.fullmatch(r"session_\d+", session_id) or not isinstance(turns, list):
            continue
        for turn in turns:
            for field in ("text", "query", "blip_caption"):
                value = turn.get(field) or ""
                if any(token in value.lower() for token in ("white mountains", "new hampshire", "白山", "新罕布什尔")):
                    white_mountains_mentions.append({
                        "session_id": session_id, "dia_id": turn.get("dia_id"),
                        "field": field, "value": value,
                    })
    if white_mountains_mentions != [{
        "session_id": "session_27", "dia_id": "D27:6", "field": "query",
        "value": "waterfall white mountains new hampshire",
    }]:
        raise AssertionError("white mountains provenance changed")

    archived = []
    for path in sorted(args.archives.glob("zhfull*locomo*.jsonl")):
        row = load_jsonl_case(path, TARGET_CASE)
        by_id = {m["id"]: m for m in row["retrieved_memories"] + row.get("graph_memories", [])}
        if TARGET_MEMORY not in by_id:
            raise AssertionError(f"target memory missing: {path}")
        context_ids = row["metadata"].get("answer_context_ids") or []
        archived.append({
            "archive": path.name,
            "predicted_answer": row["predicted_answer"],
            "target_memory_selected": TARGET_MEMORY in context_ids,
            "target_memory": by_id[TARGET_MEMORY],
            "clock_tower_memories_selected": [
                {"id": mid, "content": by_id[mid]["content"]}
                for mid in context_ids if mid in by_id and "钟楼" in by_id[mid]["content"]
            ],
        })

    report = {
        "scope": "中文 LoCoMo 原始 turn、实际适配器输出与 2026-09-25 八份评测归档；零模型调用",
        "dataset": str(args.dataset),
        "counts": dict(counts),
        "query_caption_exact_sunrise_sunset_conflicts": sunrise_sunset,
        "fixed_sample_seed": SAMPLE_SEED,
        "fixed_sample_size": SAMPLE_SIZE,
        "fixed_sample": sample,
        "fixed_sample_clear_conflicts_manual_lower_bound": len(CLEAR_SAMPLE_CONFLICTS),
        "target": {
            "case_id": TARGET_CASE,
            "question": qa["question"],
            "gold": qa["answer"],
            "evidence": qa["evidence"],
            "session_date": conv50["conversation"]["session_27_date_time"],
            "source_turn": target,
            "adapter_content": _turn_text(target),
            "waterfall_mentions_in_session_text_or_caption": other_waterfall_mentions,
            "white_mountains_mentions_in_full_conversation": white_mountains_mentions,
            "archive_rows": archived,
        },
        "limits": [
            "query/caption 冲突不能直接确定哪一方正确；需要结合对话文字及原图。",
            "固定样本的三条人工核查是下界，不代表 888 条的总体冲突率。",
            "八份归档复用同一批记忆，不能当作八次独立提取。",
            "此审计确认输入路径和一条归档污染事实，但不证明删除 query 能提升总体得分。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "counts": report["counts"],
        "sunrise_sunset_conflicts": len(sunrise_sunset),
        "fixed_sample_clear_conflicts_manual_lower_bound": len(CLEAR_SAMPLE_CONFLICTS),
        "target_archives": len(archived),
        "target_archives_select_memory": sum(r["target_memory_selected"] for r in archived),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
