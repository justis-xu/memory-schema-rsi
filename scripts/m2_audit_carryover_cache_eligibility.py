"""Bound the ten-message cache explanation for the fixed 12 carryover pairs.

Historical fact inventory is preserved. Two composite labels are corrected in
this new sidecar before checking exact retained source-turn positions.
"""

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

from mem0.memory.storage import SQLiteManager

from schema_rsi.benchmarks.locomo import LocomoDataset

from m2_audit_last_messages_tie import batch_messages


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
INVENTORY = ROOT / "results/analysis/m2_carryover_fact_inventory_20260926.json"
HISTORY = ROOT / "data/vector_store_zh/mem0_history_zh.db"
OUTPUT = ROOT / "results/analysis/m2_carryover_cache_eligibility_20260928.json"


def correct_inventory(original):
    corrected = copy.deepcopy(original)
    by_id = {p["pair_id"]: p for p in corrected["pairs"]}

    pottery = by_id["conv-26_444350391e64"]
    emotion = next(f for f in pottery["facts"] if f["fact_id"].endswith(":2"))
    assert emotion["claim_zh"] == "暂停陶艺让梅拉妮难受，但她说自己还好"
    assert "这让她很难受" in pottery["later_memory"]["content"]
    assert "我还好" not in pottery["later_memory"]["content"]
    emotion["claim_zh"] = "暂停陶艺让梅拉妮难受"
    emotion["correction_zh"] = "源话还说但我还好；两版记忆都未保存这一限定语"

    writing = by_id["conv-43_de89be96dda6"]
    world = next(f for f in writing["facts"] if f["fact_id"].endswith(":3"))
    assert world["claim_zh"] == "写作给他快乐，并让他创造全新世界"
    assert "快乐" in writing["older_memory"]["content"]
    assert "全新的世界" in writing["later_memory"]["content"]
    world["claim_zh"] = "写作能创造全新世界"
    world["correction_zh"] = "旧版已有写作快乐；仅创造世界是晚版独有"
    joy = copy.deepcopy(world)
    joy["fact_id"] = "conv-43_de89be96dda6:4"
    joy["claim_zh"] = "写作带给蒂姆快乐"
    joy["in_older_memory"] = True
    joy["in_later_memory"] = True
    joy["correction_zh"] = "从原复合标签拆出两版共有子句"
    writing["facts"].append(joy)

    facts = [f for p in corrected["pairs"] for f in p["facts"]]
    corrected["counts"] = {
        "pairs": len(corrected["pairs"]), "facts": len(facts),
        "supported": sum(f["support"] != "unsupported" for f in facts),
        "unsupported": sum(f["support"] == "unsupported" for f in facts),
        "supported_only_in_later_memory_from_older_source": sum(
            f["support"] != "unsupported" and f["in_later_memory"] and not f["in_older_memory"] for f in facts),
        "supported_only_in_older_memory": sum(
            f["support"] != "unsupported" and f["in_older_memory"] and not f["in_later_memory"] for f in facts),
        "supported_in_both": sum(
            f["support"] != "unsupported" and f["in_older_memory"] and f["in_later_memory"] for f in facts),
    }
    assert corrected["counts"] == {"pairs": 12, "facts": 39, "supported": 37, "unsupported": 2,
                                   "supported_only_in_later_memory_from_older_source": 6,
                                   "supported_only_in_older_memory": 5, "supported_in_both": 26}
    return corrected


def cache_after_source_sessions(pairs):
    dataset = LocomoDataset(path=SOURCE)
    cases = {case.metadata["conversation_id"]: case for case in dataset.cases if case.case_id.endswith("_qa0")}
    needed = {}
    for pair in pairs:
        conv = pair["pair_id"].split("_")[0]
        needed.setdefault(conv, set()).add(pair["older_memory"]["session_id"])
    caches = {}
    for conv, sessions in needed.items():
        cache = SQLiteManager(":memory:")
        scope = f"user_id=zhfull:locomo:{conv}"
        for session in cases[conv].history:
            messages = batch_messages(session)
            cache.save_messages(messages, scope)
            if session["session_id"] not in sessions:
                continue
            retained = [row["content"] for row in cache.get_last_messages(scope, limit=10)]
            assert retained == [row["content"] for row in messages[:10]], (conv, session["session_id"])
            current_turns = {turn["content"]: turn["dia_id"] for turn in session["turns"]}
            cached_ids = [current_turns.get(content, "date_anchor") for content in retained]
            caches[(conv, session["session_id"])] = {"batch_message_count": len(messages),
                                                       "cached_dia_ids": cached_ids,
                                                       "source_session_date": session["date"]}
        cache.connection.close()
    return caches


def history_adds(pairs):
    conn = sqlite3.connect(f"file:{HISTORY}?mode=ro", uri=True)
    try:
        out = {}
        for pair in pairs:
            records = []
            for key in ("older_memory", "later_memory"):
                memory = pair[key]
                row, = conn.execute("SELECT created_at,new_memory FROM history WHERE memory_id=? AND event='ADD'", (memory["id"],)).fetchall()
                assert row[1] == memory["content"], (pair["pair_id"], key)
                records.append({"id": memory["id"], "session_id": memory["session_id"], "created_at": row[0]})
            assert records[0]["created_at"] < records[1]["created_at"]
            out[pair["pair_id"]] = records
        return out
    finally:
        conn.close()


def main():
    original = json.loads(INVENTORY.read_text())
    corrected = correct_inventory(original)
    caches = cache_after_source_sessions(corrected["pairs"])
    adds = history_adds(corrected["pairs"])
    rows = []
    for pair in corrected["pairs"]:
        conv = pair["pair_id"].split("_")[0]
        old_n = int(pair["older_memory"]["session_id"].split("_")[-1])
        new_n = int(pair["later_memory"]["session_id"].split("_")[-1])
        assert new_n == old_n + 1
        cached = caches[(conv, pair["older_memory"]["session_id"])]
        eligible = []
        for fact in pair["facts"]:
            if not fact["in_later_memory"] or fact["support"] == "unsupported":
                continue
            refs = [turn["dia_id"] for turn in fact["source_turns"]]
            assert all(turn["session_id"] == pair["older_memory"]["session_id"] for turn in fact["source_turns"])
            eligible.append({"fact_id": fact["fact_id"], "claim_zh": fact["claim_zh"], "source_dia_ids": refs,
                             "source_in_cached_first_ten": bool(set(refs) & set(cached["cached_dia_ids"]))})
        rows.append({"pair_id": pair["pair_id"], "older_session_id": pair["older_memory"]["session_id"],
                     "later_session_id": pair["later_memory"]["session_id"], "source_cache": cached,
                     "old_and_later_add": adds[pair["pair_id"]], "later_supported_facts": eligible,
                     "unsupported_later_facts": [f["fact_id"] for f in pair["facts"] if f["in_later_memory"] and f["support"] == "unsupported"],
                     "all_later_supported_source_refs_in_cache": all(f["source_in_cached_first_ten"] for f in eligible)})
    later = [f for row in rows for f in row["later_supported_facts"]]
    counts = {"pairs": len(rows), "adjacent_and_add_order_verified": len(rows),
              "later_supported_facts": len(later),
              "later_supported_facts_with_source_in_cache": sum(f["source_in_cached_first_ten"] for f in later),
              "pairs_all_later_supported_refs_in_cache": sum(row["all_later_supported_source_refs_in_cache"] for row in rows),
              "unsupported_later_facts": sum(len(row["unsupported_later_facts"]) for row in rows)}
    assert counts == {"pairs": 12, "adjacent_and_add_order_verified": 12,
                      "later_supported_facts": 32, "later_supported_facts_with_source_in_cache": 31,
                      "pairs_all_later_supported_refs_in_cache": 11, "unsupported_later_facts": 1}, counts
    corrected_merges = []
    for pair in corrected["pairs"]:
        accepted = [fact for fact in pair["facts"] if fact["support"] != "unsupported"]
        rejected = [fact for fact in pair["facts"] if fact["support"] == "unsupported"]
        assert all(fact["source_turns"] for fact in accepted)
        corrected_merges.append({"canonical_memory_id": pair["older_memory"]["id"],
                                 "alias_memory_id_to_retire": pair["later_memory"]["id"],
                                 "accepted_fact_ids": [fact["fact_id"] for fact in accepted],
                                 "retracted_fact_ids": [fact["fact_id"] for fact in rejected],
                                 "accepted_claims_zh": [fact["claim_zh"] for fact in accepted],
                                 "source_turn_ids": sorted({turn["dia_id"] for fact in accepted for turn in fact["source_turns"]})})
    assert sum(len(row["accepted_fact_ids"]) for row in corrected_merges) == corrected["counts"]["supported"]
    pottery_cache = next(row for row in rows if row["pair_id"] == "conv-26_444350391e64")["source_cache"]
    assert "D17:9" in pottery_cache["cached_dia_ids"] and "D17:10" not in pottery_cache["cached_dia_ids"]
    pottery_case = next(case for case in LocomoDataset(path=SOURCE).cases if case.case_id == "locomo_conv-26_qa0")
    pottery_session = next(session for session in pottery_case.history if session["session_id"] == "session_17")
    assistant_guess, = [turn for turn in pottery_session["turns"] if turn["dia_id"] == "D17:9"]
    self_report, = [turn for turn in pottery_session["turns"] if turn["dia_id"] == "D17:10"]
    payload = {
        "scope": "Fixed 12 manually sourced carryover pairs; corrected composite claim labels; actual current SQLiteManager cache replay and read-only Mem0 ADD history",
        "input_sha256": {"historical_inventory": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
                         "chinese_source": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                         "history_db": hashlib.sha256(HISTORY.read_bytes()).hexdigest()},
        "historical_counts_before_correction": original["counts"], "corrected_inventory_counts": corrected["counts"],
        "corrected_fact_inventory": corrected, "corrected_merge_contract": corrected_merges,
        "label_corrections": [
            {"fact_id": "conv-26_444350391e64:2", "old_label": "暂停陶艺让梅拉妮难受，但她说自己还好",
             "new_label": "暂停陶艺让梅拉妮难受", "source_qualification_missing_from_both_memories": "但我还好"},
            {"fact_id": "conv-43_de89be96dda6:3", "old_label": "写作给他快乐，并让他创造全新世界",
             "new_label": "写作能创造全新世界", "new_shared_fact_id": "conv-43_de89be96dda6:4",
             "new_shared_fact": "写作带给蒂姆快乐"}],
        "counts": counts, "pairs": rows,
        "pottery_emotion_speaker_boundary": {"cached_other_speaker_turn": assistant_guess,
                                             "uncached_self_report_turn": self_report},
        "limits": ["缓存重放证明旧源原话可进入下一场提示，不证明历史模型实际从该通道提取",
                   "Existing Memories检索输入未冻结，无法在两个旧事实通道之间做归因",
                   "本结果修正旧人工复合标签，旧归档保留为历史版本；没有自动事实判别或答题净收益"]}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
