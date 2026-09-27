#!/usr/bin/env python3
"""Read-only aligned English/Chinese packet for the frozen 20-case sample."""

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh")
MANIFEST = ROOT / "results/analysis/m2_current_zh_stratified_manifest_20260927.json"
OUT = ROOT / "results/analysis/m2_zh_translation_sample_20260927.json"
SESSION = re.compile(r"session_\d+$")


def all_turns(sample):
    for key, value in sample["conversation"].items():
        if SESSION.fullmatch(key) and isinstance(value, list):
            yield from value


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    paths = {"english": DATASET_DIR / "locomo10.json",
             "chinese": DATASET_DIR / "locomo10_zh.json"}
    datasets = {lang: {s["sample_id"]: s for s in json.loads(path.read_text())}
                for lang, path in paths.items()}
    assert datasets["english"].keys() == datasets["chinese"].keys()
    counts = {"turns": 0, "chinese_turns_with_caption": 0,
              "caption_identical_to_english": 0, "chinese_turns_with_query": 0,
              "query_identical_to_english": 0, "chinese_turns_with_image_url": 0,
              "chinese_text_with_ascii_word_len_ge_3": 0}
    for cid in datasets["english"]:
        a, b = datasets["english"][cid], datasets["chinese"][cid]
        en_turns = list(all_turns(a))
        zh_turns = list(all_turns(b))
        assert len(en_turns) == len(zh_turns)
        for en, zh in zip(en_turns, zh_turns):
            assert en["dia_id"] == zh["dia_id"]
            counts["turns"] += 1
            if zh.get("blip_caption"):
                counts["chinese_turns_with_caption"] += 1
                counts["caption_identical_to_english"] += en.get("blip_caption") == zh.get("blip_caption")
            if zh.get("query"):
                counts["chinese_turns_with_query"] += 1
                counts["query_identical_to_english"] += en.get("query") == zh.get("query")
            counts["chinese_turns_with_image_url"] += bool(zh.get("img_url"))
            counts["chinese_text_with_ascii_word_len_ge_3"] += bool(re.search(r"[A-Za-z]{3,}", zh.get("text", "")))
    manifest = json.loads(MANIFEST.read_text())
    selected = []
    for meta in manifest["cases"]:
        cid, i = meta["conversation_id"], meta["qa_index"]
        a, b = datasets["english"][cid], datasets["chinese"][cid]
        en_qa, zh_qa = a["qa"][i], b["qa"][i]
        assert en_qa["category"] == zh_qa["category"]
        assert en_qa.get("evidence") == zh_qa.get("evidence")
        evidence = []
        for turn_id in en_qa.get("evidence", []):
            day, turn = turn_id.split(":")
            key, index = "session_" + day[1:], int(turn) - 1
            en_turn, zh_turn = a["conversation"][key][index], b["conversation"][key][index]
            assert en_turn["dia_id"] == zh_turn["dia_id"] == turn_id
            evidence.append({
                "turn_id": turn_id,
                "english": {"speaker": en_turn.get("speaker"), "text": en_turn.get("text"),
                            "caption": en_turn.get("blip_caption"), "query": en_turn.get("query")},
                "chinese": {"speaker": zh_turn.get("speaker"), "text": zh_turn.get("text"),
                            "caption": zh_turn.get("blip_caption"), "query": zh_turn.get("query")},
            })
        selected.append({
            "case_id": meta["case_id"], "category": meta["category"],
            "english": {"question": en_qa["question"], "answer": en_qa.get("answer"),
                        "adversarial_answer": en_qa.get("adversarial_answer")},
            "chinese": {"question": zh_qa["question"], "answer": zh_qa.get("answer"),
                        "adversarial_answer": zh_qa.get("adversarial_answer")},
            "evidence": evidence,
        })
    result = {
        "scope": "Frozen 20 Chinese LoCoMo QA and aligned English original evidence, plus full-dataset image text field counts; no model calls",
        "dataset_sha256": {lang: hashlib.sha256(path.read_bytes()).hexdigest() for lang, path in paths.items()},
        "selection_manifest": str(MANIFEST.relative_to(ROOT)),
        "field_counts": counts, "selected_cases": selected,
        "limits": ["Aligned packet is not a semantic translation judge", "Image caption and query counts concern dataset fields, not whether a model used them", "The selected 20 cases are equal by category, not a random estimate of full translation quality"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(counts)
    print("aligned cases", len(selected))


if __name__ == "__main__":
    main()
