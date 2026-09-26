#!/usr/bin/env python3
"""从旧中文归档找跨 session 近重复记忆，附原始两场会话供人工核查。"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
RAW = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
OUTPUT = ROOT / "results/analysis/m2_cross_session_carryover_packet_20260926.json"
JACCARD_MIN = 0.37
RATIO_MIN = 0.55
PHOTO_SENTINEL = {
    "a5f1a15b-1ec4-4f1d-b5be-1e4e79fef5e0",
    "3a4760bc-5e23-43a0-8e00-499b54e0232c",
}


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def trigrams(value: str) -> set[str]:
    return {value[i:i + 3] for i in range(max(0, len(value) - 2))}


def pair_id(conv_id: str, left: dict, right: dict) -> str:
    ids = sorted((left["id"], right["id"]))
    return f"{conv_id}_{hashlib.sha256('|'.join(ids).encode()).hexdigest()[:12]}"


def main() -> None:
    dataset = {entry["sample_id"]: entry["conversation"] for entry in json.loads(DATASET.read_text())}
    by_conv: dict[str, dict[str, dict]] = defaultdict(dict)
    contexts: dict[str, set[str]] = {}
    for line in RAW.open():
        if not line.strip():
            continue
        row = json.loads(line)
        conv_id = row["case_id"].split("_")[1]
        contexts[row["case_id"]] = set(row["metadata"]["answer_context_ids"])
        for memory in row["retrieved_memories"]:
            prior = by_conv[conv_id].get(memory["id"])
            if prior and (prior["content"] != memory["content"] or prior["metadata"] != memory["metadata"]):
                raise ValueError(f"memory ID changed across archive rows: {memory['id']}")
            by_conv[conv_id][memory["id"]] = memory
    candidates = []
    counts = Counter()
    sentinel_found = False
    for conv_id, memories in sorted(by_conv.items()):
        arr = list(memories.values())
        norm = [normalized(m["content"]) for m in arr]
        grams = [trigrams(x) for x in norm]
        counts["visible_unique_memories"] += len(arr)
        for i, left in enumerate(arr):
            for j in range(i + 1, len(arr)):
                right = arr[j]
                s_left = (left.get("metadata") or {}).get("session_id")
                s_right = (right.get("metadata") or {}).get("session_id")
                if not s_left or not s_right or s_left == s_right:
                    continue
                counts["cross_session_pairs_examined"] += 1
                is_sentinel = {left["id"], right["id"]} == PHOTO_SENTINEL
                a, b = grams[i], grams[j]
                jaccard = len(a & b) / len(a | b) if a or b else 0.0
                if jaccard < JACCARD_MIN and not is_sentinel:
                    continue
                ratio = difflib.SequenceMatcher(None, norm[i], norm[j], autojunk=False).ratio()
                if ratio < RATIO_MIN and not is_sentinel:
                    continue
                counts["lexical_candidate_pairs"] += not is_sentinel
                sentinel_found |= is_sentinel
                conv = dataset[conv_id]
                source_sessions = {
                    session_id: {
                        "date": conv.get(f"{session_id}_date_time"),
                        "turns": conv.get(session_id),
                    }
                    for session_id in {s_left, s_right}
                }
                candidates.append({
                    "pair_id": pair_id(conv_id, left, right),
                    "conversation": conv_id,
                    "selection": "photo_sentinel" if is_sentinel else "lexical_threshold",
                    "char_trigram_jaccard": round(jaccard, 4),
                    "sequence_ratio": round(ratio, 4),
                    "left": left,
                    "right": right,
                    "co_selected_case_ids": [
                        case_id for case_id, selected_ids in contexts.items()
                        if left["id"] in selected_ids and right["id"] in selected_ids
                    ],
                    "source_sessions": source_sessions,
                })
    if not sentinel_found:
        raise AssertionError("known cross-session photo pair missing from archive")
    counts["manual_photo_sentinel_pairs"] = 1
    counts["packet_pairs"] = len(candidates)
    counts["pair_case_co_selections"] = sum(len(pair["co_selected_case_ids"]) for pair in candidates)
    counts["unique_cases_with_pair_co_selected"] = len({
        case_id for pair in candidates for case_id in pair["co_selected_case_ids"]
    })
    report = {
        "scope": "旧中文无图归档中，每个 conversation 的检索记忆 ID 并集；只比较不同 session 的记忆正文",
        "selection": "去空白小写后，字符 3-gram Jaccard >= 0.37 且 SequenceMatcher ratio >= 0.55；另加一对已知词面漏检的家庭照 sentinel。",
        "counts": dict(counts),
        "pairs": sorted(candidates, key=lambda x: (x["selection"] != "photo_sentinel", -x["sequence_ratio"], x["pair_id"])),
        "limits": [
            "高相似度只生成候选；跨会话再次提及同一事实是正常情况，必须读两场原始 turn。",
            "词面阈值漏掉跨语言或大幅改写的重复；检索并集也不是完整历史库。",
            "记忆 metadata 的 session_id 是写入场次，不是已验证的事实来源场次。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"counts": report["counts"],
                      "pair_ids": [pair["pair_id"] for pair in report["pairs"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
