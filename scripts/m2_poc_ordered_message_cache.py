"""Offline cache-order intervention over Chinese LoCoMo sessions and fixed facts.

The subclass changes only SQL tie-breaking inside an in-memory SQLiteManager.
No real history database, model, vector store, or graph is written.
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from mem0.memory.storage import SQLiteManager
from mem0.memory.utils import parse_messages

from schema_rsi.benchmarks.locomo import LocomoDataset

from m2_audit_last_messages_tie import batch_messages
from m2_legacy_message_cache import LegacySQLiteManager


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
INVENTORY = ROOT / "results/analysis/m2_carryover_cache_eligibility_20260928.json"
OUTPUT_AUDIT = ROOT / "results/analysis/m2_output_origin_sample_20260928.json"
OUTPUT = ROOT / "results/analysis/m2_ordered_message_cache_poc_20260928.json"
PROMPT_TARGETS = {"conv-26": ("session_18", "session_19"), "conv-41": ("session_31", "session_32")}


class OrderedSQLiteManager(SQLiteManager):
    """Isolated prototype: break batch timestamp ties by insertion rowid."""

    def save_messages(self, messages, session_scope):
        if not messages:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                now = datetime.now(timezone.utc).isoformat()
                for message in messages:
                    self.connection.execute(
                        "INSERT INTO messages (id,session_scope,role,content,name,created_at) VALUES (?,?,?,?,?,?)",
                        (str(uuid.uuid4()), session_scope, message.get("role"), message.get("content"), message.get("name"), now),
                    )
                self.connection.execute("""DELETE FROM messages WHERE session_scope=? AND id NOT IN (
                    SELECT id FROM (SELECT id FROM messages WHERE session_scope=?
                                    ORDER BY created_at DESC, rowid DESC LIMIT 10))""",
                                        (session_scope, session_scope))
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise

    def get_last_messages(self, session_scope, limit=10):
        with self._lock:
            rows = self.connection.execute("""SELECT role,content,name,created_at FROM (
                SELECT role,content,name,created_at,rowid AS insertion_id FROM messages
                WHERE session_scope=? ORDER BY created_at DESC, rowid DESC LIMIT ?)
                ORDER BY created_at ASC, insertion_id ASC""", (session_scope, limit)).fetchall()
        return [{"role": row[0], "content": row[1], "name": row[2], "created_at": row[3]} for row in rows]


def prompt_parts(prompt):
    marker_a, marker_b = "## Last k Messages\n", "\n\n## Recently Extracted Memories\n"
    before, rest = prompt.split(marker_a, 1)
    last_k, after = rest.split(marker_b, 1)
    return before, last_k, after


def main():
    inventory = json.loads(INVENTORY.read_text())
    output_audit = json.loads(OUTPUT_AUDIT.read_text())
    output_by_conv = {item["conversation"]: item for item in output_audit["items"] if not item["origin_matches_written_session"]}
    assert set(output_by_conv) == set(PROMPT_TARGETS)
    dataset = LocomoDataset(path=SOURCE)
    cases = {case.metadata["conversation_id"]: case for case in dataset.cases if case.case_id.endswith("_qa0")}
    source_sessions = {(pair["pair_id"].split("_")[0], pair["older_session_id"]) for pair in inventory["pairs"]}
    source_sessions |= {(conv, source) for conv, (source, _) in PROMPT_TARGETS.items()}
    cached = {}
    all_counts = {"sessions": 0, "long_batches": 0, "current_first_ten": 0,
                  "prototype_last_ten": 0, "same_new_message_batches": 0}
    for conv, case in sorted(cases.items()):
        # Reproduce the historical pre-fix arm; SQLiteManager now has the fix.
        current = LegacySQLiteManager(":memory:")
        ordered = OrderedSQLiteManager(":memory:")
        scope = f"user_id=zhfull:locomo:{conv}"
        for session in case.history:
            messages = batch_messages(session)
            all_counts["sessions"] += 1
            # The intervention touches only the history cache. Each next
            # session's New Messages are built once and passed unchanged.
            expected_new = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            current.save_messages(messages, scope)
            ordered.save_messages(messages, scope)
            assert json.dumps(messages, ensure_ascii=False, sort_keys=True) == expected_new
            all_counts["same_new_message_batches"] += 1
            old_cache = current.get_last_messages(scope, 10)
            new_cache = ordered.get_last_messages(scope, 10)
            old_text = [row["content"] for row in old_cache]
            new_text = [row["content"] for row in new_cache]
            if len(messages) > 10:
                all_counts["long_batches"] += 1
                all_counts["current_first_ten"] += old_text == [row["content"] for row in messages[:10]]
                all_counts["prototype_last_ten"] += new_text == [row["content"] for row in messages[-10:]]
            key = conv, session["session_id"]
            if key in source_sessions:
                by_content = {turn["content"]: turn["dia_id"] for turn in session["turns"]}
                cached[key] = {"current": old_cache, "ordered": new_cache,
                               "current_dia_ids": [by_content.get(x, "date_anchor") for x in old_text],
                               "ordered_dia_ids": [by_content.get(x, "date_anchor") for x in new_text],
                               "batch_message_count": len(messages)}
        current.connection.close()
        ordered.connection.close()
    assert all_counts == {"sessions": 272, "long_batches": 272, "current_first_ten": 272,
                          "prototype_last_ten": 272, "same_new_message_batches": 272}, all_counts

    pairs = []
    for pair in inventory["pairs"]:
        key = pair["pair_id"].split("_")[0], pair["older_session_id"]
        item = cached[key]
        facts = []
        for fact in pair["later_supported_facts"]:
            refs = set(fact["source_dia_ids"])
            old = bool(refs & set(item["current_dia_ids"]))
            new = bool(refs & set(item["ordered_dia_ids"]))
            assert old == fact["source_in_cached_first_ten"]
            facts.append({"fact_id": fact["fact_id"], "claim_zh": fact["claim_zh"],
                          "source_dia_ids": fact["source_dia_ids"], "current_cache_has_source": old,
                          "ordered_cache_has_source": new})
        pairs.append({"pair_id": pair["pair_id"], "batch_message_count": item["batch_message_count"],
                      "current_cached_dia_ids": item["current_dia_ids"],
                      "ordered_cached_dia_ids": item["ordered_dia_ids"], "facts": facts,
                      "unsupported_later_fact_ids": pair["unsupported_later_facts"]})
    all_facts = [fact for pair in pairs for fact in pair["facts"]]
    counts = {"current_cache_source_facts": sum(f["current_cache_has_source"] for f in all_facts),
              "ordered_cache_source_facts": sum(f["ordered_cache_has_source"] for f in all_facts),
              "supported_later_facts": len(all_facts),
              "pairs_with_any_source_in_ordered_cache": sum(any(f["ordered_cache_has_source"] for f in p["facts"]) for p in pairs)}
    assert counts == {"current_cache_source_facts": 31, "ordered_cache_source_facts": 7,
                      "supported_later_facts": 32, "pairs_with_any_source_in_ordered_cache": 3}, counts

    prompts = {}
    for conv, (source_session, written_session) in PROMPT_TARGETS.items():
        case = cases[conv]
        next_session = next(session for session in case.history if session["session_id"] == written_session)
        next_messages = batch_messages(next_session)
        new_messages = parse_messages(next_messages)
        old_carriers = output_by_conv[conv]["same_topic_source_session_records"]
        existing = [{"id": str(i), "text": record["content"]} for i, record in enumerate(old_carriers)]
        source_cache = cached[(conv, source_session)]
        kwargs = {"summary": None, "recently_extracted_memories": None,
                  "existing_memories": existing, "new_messages": new_messages,
                  "current_date": "2026-09-25", "timestamp": None, "custom_instructions": None}
        old_prompt = generate_additive_extraction_prompt(last_k_messages=source_cache["current"], **kwargs)
        new_prompt = generate_additive_extraction_prompt(last_k_messages=source_cache["ordered"], **kwargs)
        old_parts, new_parts = prompt_parts(old_prompt), prompt_parts(new_prompt)
        assert old_parts[0] == new_parts[0] and old_parts[2] == new_parts[2]
        assert old_parts[1] != new_parts[1]
        prompts[conv] = {"source_session": source_session, "written_session": written_session,
                         "controlled_existing_memories": existing,
                         "new_messages_sha256": hashlib.sha256(new_messages.encode()).hexdigest(),
                         "system_prompt_sha256": hashlib.sha256(ADDITIVE_EXTRACTION_PROMPT.encode()).hexdigest(),
                         "only_last_k_section_changed": True,
                         "current_user_prompt_sha256": hashlib.sha256(old_prompt.encode()).hexdigest(),
                         "ordered_user_prompt_sha256": hashlib.sha256(new_prompt.encode()).hexdigest(),
                         "current_last_k_dia_ids": source_cache["current_dia_ids"],
                         "ordered_last_k_dia_ids": source_cache["ordered_dia_ids"],
                         "current_user_prompt": old_prompt, "ordered_user_prompt": new_prompt}
    payload = {"scope": "Isolated source-input intervention: current SQLiteManager versus a rowid-tiebroken in-memory prototype; no model or real store writes",
               "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "inventory_sha256": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
               "all_session_checks": all_counts, "fact_eligibility_counts": counts, "pairs": pairs,
               "controlled_prompt_checks": prompts,
               "limits": ["Retained source turn is an input opportunity, not evidence that the extractor emitted the fact from it",
                          "Existing Memories are held to two known old carriers only for prompt controls; historical top-10 retrieval and custom instructions are not reconstructed",
                          "Only source positions and prompt sections are tested, not extraction quality, downstream retrieval or answer effect"]}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**all_counts, **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
