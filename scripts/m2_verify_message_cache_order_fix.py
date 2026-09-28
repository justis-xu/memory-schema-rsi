#!/usr/bin/env python3
"""Verify fixed Mem0 cache order and preserve three historical replays."""

import hashlib
import json
import tempfile
from pathlib import Path

from mem0.memory.storage import SQLiteManager
from schema_rsi.benchmarks.locomo import LocomoDataset

import m2_audit_carryover_cache_eligibility as carryover
import m2_audit_last_messages_tie as tie
import m2_poc_ordered_message_cache as ordered
from m2_legacy_message_cache import LegacySQLiteManager, STORAGE_SHA256_BEFORE_FIX


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
HISTORY = ROOT / "data/vector_store_zh/mem0_history_zh.db"
STORAGE = ROOT / "third_party/mem0-src/mem0/memory/storage.py"
OUTPUT = ROOT / "results/analysis/m2_message_cache_order_fix_20260928.json"
REPLAYS = (
    (tie, "m2_last_messages_tie_20260928.json"),
    (carryover, "m2_carryover_cache_eligibility_20260928.json"),
    (ordered, "m2_ordered_message_cache_poc_20260928.json"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing overwrite: {OUTPUT}")
    db_before = sha(HISTORY)
    dataset = LocomoDataset(path=SOURCE)
    cases = {case.metadata["conversation_id"]: case for case in dataset.cases if case.case_id.endswith("_qa0")}
    counts = {"conversations": len(cases), "sessions": 0, "long_batches": 0,
              "legacy_first_ten": 0, "fixed_last_ten": 0}
    examples = {}
    for conv, case in sorted(cases.items()):
        old = LegacySQLiteManager(":memory:")
        fixed = SQLiteManager(":memory:")
        try:
            scope = f"user_id=zhfull:locomo:{conv}"
            for session in case.history:
                messages = tie.batch_messages(session)
                counts["sessions"] += 1
                old.save_messages(messages, scope)
                fixed.save_messages(messages, scope)
                if len(messages) <= 10:
                    continue
                counts["long_batches"] += 1
                old_contents = [row["content"] for row in old.get_last_messages(scope, 10)]
                fixed_contents = [row["content"] for row in fixed.get_last_messages(scope, 10)]
                counts["legacy_first_ten"] += old_contents == [row["content"] for row in messages[:10]]
                counts["fixed_last_ten"] += fixed_contents == [row["content"] for row in messages[-10:]]
                if (conv, session["session_id"]) in (("conv-26", "session_18"), ("conv-41", "session_31")):
                    by_content = {turn["content"]: turn["dia_id"] for turn in session["turns"]}
                    examples[f"{conv}:{session['session_id']}"] = {
                        "legacy_dia_ids": [by_content.get(content, "date_anchor") for content in old_contents],
                        "fixed_dia_ids": [by_content.get(content, "date_anchor") for content in fixed_contents],
                    }
        finally:
            old.connection.close()
            fixed.connection.close()
    assert counts == {"conversations": 10, "sessions": 272, "long_batches": 272,
                      "legacy_first_ten": 272, "fixed_last_ten": 272}

    archived_replay_equal = {}
    with tempfile.TemporaryDirectory(prefix="m2_cache_replay_") as tmp:
        for module, name in REPLAYS:
            original_output = module.OUTPUT
            module.OUTPUT = Path(tmp) / name
            try:
                module.main()
                archived_replay_equal[name] = json.loads(module.OUTPUT.read_text()) == json.loads(
                    (ROOT / "results/analysis" / name).read_text())
            finally:
                module.OUTPUT = original_output
    assert all(archived_replay_equal.values())
    db_after = sha(HISTORY)
    assert db_after == db_before
    result = {
        "scope": "isolated in-memory replay of Chinese LoCoMo; no model or real store writes",
        "source_sha256": sha(SOURCE),
        "history_db_sha256_before": db_before,
        "history_db_sha256_after": db_after,
        "production_storage_sha256": sha(STORAGE),
        "historical_storage_sha256": STORAGE_SHA256_BEFORE_FIX,
        "counts": counts,
        "target_examples": examples,
        "archived_replay_equal": archived_replay_equal,
        "limits": [
            "Real historical extraction input and model output were not frozen.",
            "Correct cache order does not prevent old facts entering through Existing Memories.",
            "No extraction or QA benefit is measured here.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"counts": counts, "archived_replay_equal": archived_replay_equal}, ensure_ascii=False))


if __name__ == "__main__":
    main()
