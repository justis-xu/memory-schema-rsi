#!/usr/bin/env python3
"""对 D27:6 的图片 query 做两臂、至多两次提取调用；不写入 Mem0。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from mem0.llms.openai import OpenAILLM
from mem0.memory.utils import parse_messages
from openai import OpenAI

from schema_rsi.benchmarks.base import parse_session_date
from schema_rsi.benchmarks.locomo import _turn_text
from schema_rsi.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUTPUT = ROOT / "results/analysis/m2_image_query_ablation_poc_20260926.json"
TARGET_DIA_IDS = ("D27:4", "D27:5", "D27:6", "D27:7", "D27:8")
ARMS = ("original", "without_d27_6_query")


def build_request(entry: dict, arm: str, custom_instructions: str) -> tuple[list[dict], list[dict]]:
    conv = entry["conversation"]
    by_id = {turn["dia_id"]: turn for turn in conv["session_27"]}
    subset = [dict(by_id[dia_id]) for dia_id in TARGET_DIA_IDS]
    if arm == "without_d27_6_query":
        subset[2]["query"] = ""
    iso_datetime = parse_session_date(conv["session_27_date_time"])
    iso_date = iso_datetime.split("T")[0] if iso_datetime else None
    if iso_date != "2023-10-29":
        raise AssertionError(f"unexpected date: {iso_datetime}")
    messages = [{
        "role": "system",
        "content": (
            f"Conversation date: {iso_datetime}. "
            "Use this date to resolve relative time expressions like "
            "'yesterday', 'next month', 'in three weeks'."
        ),
    }]
    for turn in subset:
        messages.append({
            "role": "user" if turn["speaker"] == conv["speaker_a"] else "assistant",
            "content": _turn_text(turn),
        })
    user_prompt = generate_additive_extraction_prompt(
        existing_memories=[],
        new_messages=parse_messages(messages),
        last_k_messages=[],
        current_date="2026-09-26",
        timestamp=iso_date,
        custom_instructions=custom_instructions,
    )
    return [
        {"role": "system", "content": ADDITIVE_EXTRACTION_PROMPT},
        {"role": "user", "content": user_prompt},
    ], messages


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="执行尚未完成的调用；默认仅展示计划")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    entry = next(x for x in json.loads(args.dataset.read_text()) if x["sample_id"] == "conv-50")
    requests = {}
    source_messages = {}
    for arm in ARMS:
        requests[arm], source_messages[arm] = build_request(
            entry, arm, settings.mem0["fact_extraction_prompt"]
        )
    for a, b in zip(source_messages[ARMS[0]], source_messages[ARMS[1]]):
        if a != b and ("D27:6" not in TARGET_DIA_IDS or
                       "waterfall white mountains new hampshire" not in a["content"] or
                       "waterfall white mountains new hampshire" in b["content"]):
            raise AssertionError("ablation changed more than the targeted query")
    changed = sum(a != b for a, b in zip(source_messages[ARMS[0]], source_messages[ARMS[1]]))
    if changed != 1:
        raise AssertionError(f"expected exactly one changed message, got {changed}")
    hashes = {
        arm: hashlib.sha256(json.dumps(requests[arm], ensure_ascii=False).encode()).hexdigest()
        for arm in ARMS
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text())
        if payload["request_sha256"] != hashes:
            raise AssertionError("existing output was made from a different request")
    else:
        payload = {
            "objective": "隔离 D27:6 的错误图片 query，比较提取结果是否包含白山瀑布与上个月拍钟楼关系",
            "case_id": "locomo_conv-50_qa65",
            "dataset": str(args.dataset),
            "model": settings.llm.model,
            "temperature": settings.llm.temperature,
            "max_calls": 2,
            "target_dia_ids": TARGET_DIA_IDS,
            "arms": ARMS,
            "request_sha256": hashes,
            "only_changed_message_index": next(i for i, (a, b) in enumerate(
                zip(source_messages[ARMS[0]], source_messages[ARMS[1]])
            ) if a != b),
            "original_d27_6_input": source_messages[ARMS[0]][3]["content"],
            "ablated_d27_6_input": source_messages[ARMS[1]][3]["content"],
            "method_limits": [
                "只取 session_27 的 D27:4–8，existing memories 与 last k messages 置空；不重放原全量 ingest。",
                "将 observation date 显式设为 2023-10-29；旧运行靠 system 日期行而 Mem0 内部 observation date 默认当前日。",
                "新中文提示词与 2026-09-25 旧混合语言记忆库使用的提取配置不同。",
                "每臂只调用一次；差异可能含采样波动，不能外推总体收益。",
            ],
            "results": {},
        }
    if not args.execute:
        print(json.dumps({
            "planned_arms": ARMS,
            "completed_arms": list(payload["results"]),
            "only_changed_message_index": payload["only_changed_message_index"],
            "model": settings.llm.model,
            "request_sha256": hashes,
        }, ensure_ascii=False, indent=2))
        return

    if not settings.llm.api_key or not settings.llm.base_url or not settings.llm.model:
        raise RuntimeError("LLM endpoint configuration is incomplete")
    usage = []

    def on_response(_llm, response, _params):
        if response.usage:
            usage.append(response.usage.model_dump())

    llm = OpenAILLM({
        "model": settings.llm.model,
        "api_key": settings.llm.api_key,
        "openai_base_url": settings.llm.base_url,
        "temperature": settings.llm.temperature,
        "max_tokens": 1800,
        "disable_thinking": True,
        "response_callback": on_response,
    })
    llm.client = OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        max_retries=0,
        timeout=90,
    )
    for arm in ARMS:
        if arm in payload["results"]:
            continue
        started = time.monotonic()
        try:
            raw = llm.generate_response(requests[arm], response_format={"type": "json_object"})
            parsed = json.loads(raw)
            memories = parsed.get("memory")
            if not isinstance(memories, list):
                raise ValueError("response lacks memory array")
            payload["results"][arm] = {
                "status": "ok",
                "raw_response": raw,
                "memories": memories,
                "usage": usage[-1] if usage else None,
                "duration_seconds": round(time.monotonic() - started, 2),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(f"{arm}: ok, {len(memories)} memories", flush=True)
        except Exception as exc:
            payload["results"][arm] = {
                "status": "error", "error_type": type(exc).__name__,
                "error": str(exc)[:300],
                "duration_seconds": round(time.monotonic() - started, 2),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(f"{arm}: error ({type(exc).__name__})", flush=True)
            write_result(args.output, payload)
            break
        write_result(args.output, payload)


if __name__ == "__main__":
    main()
