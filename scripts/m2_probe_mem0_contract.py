#!/usr/bin/env python3
"""Offline, isolated probe of the current Chinese Mem0 extraction contract."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")

from schema_rsi.config import get_settings
from schema_rsi.memory.mem0_backend import Mem0Backend, build_mem0_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/locomo_zh.yaml"
OUT = ROOT / "results/analysis/m2_mem0_extraction_contract_20260929.json"


def main():
    if OUT.exists():
        raise SystemExit("refusing to overwrite result")
    settings = get_settings(str(CONFIG))
    prompt = settings.mem0["fact_extraction_prompt"]
    result = {
        "scope": "当前中文 Mem0 提取配置与两场合成事实的离线持久化探针",
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "custom_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_checks": {
            "fixed_today_2026_09_26": "Today's date is 2026-09-26." in prompt,
            "instructs_array_of_strings": '"memory" key containing an array of strings' in prompt,
            "includes_literal_messages_placeholder": "{messages}" in prompt,
            "instructs_assistant_extraction": "user and assistant messages only" in prompt,
            "instructs_false_public_origin_reply": "publicly available sources on internet" in prompt,
        },
        "model_calls": 0,
        "embedding_http_calls": 0,
        "answer_or_judge_calls": 0,
    }
    with tempfile.TemporaryDirectory(prefix="m2-mem0-contract-") as directory:
        cfg = build_mem0_config(settings)
        cfg["vector_store"]["config"]["path"] = str(Path(directory) / "chroma")
        cfg["vector_store"]["config"]["collection_name"] = "m2_contract_probe"
        cfg["history_db_path"] = str(Path(directory) / "history.db")
        backend = Mem0Backend(settings=settings, config=cfg)
        embedder = backend.raw.embedding_model
        embedder.embed = lambda text, memory_action=None: [0.0] * 1536
        embedder.embed_batch = lambda texts, memory_action="add": [[0.0] * 1536 for _ in texts]

        backend.raw.llm.generate_response = lambda **kwargs: json.dumps({"memory": [{
            "id": "0", "text": "用户喜欢画画", "attributed_to": "user",
        }]}, ensure_ascii=False)
        first = backend.add_memory("m2:contract-probe", [{"role": "user", "content": "我喜欢画画。"}],
                                   metadata={"session_id": "s1", "session_date": "2023-01-01"})
        first_id = first[0].id
        backend.raw.llm.generate_response = lambda **kwargs: json.dumps({"memory": [{
            "id": "0", "text": "用户也喜欢写作", "attributed_to": "user",
            "linked_memory_ids": [first_id],
        }]}, ensure_ascii=False)
        second = backend.add_memory("m2:contract-probe", [{"role": "user", "content": "我也喜欢写作。"}],
                                    metadata={"session_id": "s2", "session_date": "2023-02-01"})
        second_id = second[0].id
        payload = backend.raw.vector_store.get(vector_id=second_id).payload
        inventory = backend.get_all_memories("m2:contract-probe")
        second_record = next(r for r in inventory if r.id == second_id)
        result["control"] = {
            "input_link_target_is_first_record": True,
            "first_and_second_distinct": first_id != second_id,
            "second_raw_payload_has_attributed_to_user": payload.get("attributed_to") == "user",
            "second_raw_payload_has_linked_memory_ids": "linked_memory_ids" in payload,
            "second_wrapper_has_attributed_to": "attributed_to" in second_record.metadata,
            "second_wrapper_has_linked_memory_ids": "linked_memory_ids" in second_record.metadata,
            "second_wrapper_has_session_date": second_record.metadata.get("session_date") == "2023-02-01",
            "persisted_record_count": len(inventory),
        }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["control"], ensure_ascii=False))


if __name__ == "__main__":
    main()
