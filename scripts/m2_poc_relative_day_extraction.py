#!/usr/bin/env python3
"""三臂、最多三次的中文相对日提取探针；不写 Mem0 或调用 embedding。"""

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
OUTPUT = ROOT / "results/analysis/m2_relative_day_extraction_poc_20260926.json"
TARGET_IDS = ("D11:13", "D11:14", "D11:15")
ARMS = ("default_observation_date", "source_observation_date", "source_date_no_hedge")
NO_HEDGE = (
    "对源话中明确的‘昨天/前天/明天/后天’，用会话日期算出确定的日历日。"
    "同时保留原相对词和事件主体；不得给该确定日期添加‘约/左右/前后/around’。"
    "对‘最近/刚/上周’等不确定表达，仍保留原有不确定性。"
)


def request_for(entry: dict, arm: str, custom: str) -> tuple[list[dict], list[dict], str]:
    conv = entry["conversation"]
    by_id = {turn["dia_id"]: turn for turn in conv["session_11"]}
    iso_datetime = parse_session_date(conv["session_11_date_time"])
    if not iso_datetime or iso_datetime.split("T")[0] != "2023-05-11":
        raise AssertionError(f"unexpected session date: {iso_datetime}")
    messages = [{
        "role": "system",
        "content": (
            f"Conversation date: {iso_datetime}. "
            "Use this date to resolve relative time expressions like "
            "'yesterday', 'next month', 'in three weeks'."
        ),
    }]
    for dia_id in TARGET_IDS:
        turn = by_id[dia_id]
        messages.append({
            "role": "user" if turn["speaker"] == conv["speaker_a"] else "assistant",
            "content": _turn_text(turn),
        })
    observation = None if arm == "default_observation_date" else "2023-05-11"
    instructions = custom + ("\n" + NO_HEDGE if arm == "source_date_no_hedge" else "")
    user_prompt = generate_additive_extraction_prompt(
        existing_memories=[],
        new_messages=parse_messages(messages),
        last_k_messages=[],
        current_date="2026-09-26",
        timestamp=observation,
        custom_instructions=instructions,
    )
    return [
        {"role": "system", "content": ADDITIVE_EXTRACTION_PROMPT},
        {"role": "user", "content": user_prompt},
    ], messages, observation or "2026-09-26"


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    entry = next(e for e in json.loads(DATASET.read_text()) if e["sample_id"] == "conv-30")
    requests, observations = {}, {}
    for arm in ARMS:
        requests[arm], messages, observations[arm] = request_for(
            entry, arm, settings.mem0["fact_extraction_prompt"]
        )
    a = requests[ARMS[0]][1]["content"]
    b = requests[ARMS[1]][1]["content"]
    c = requests[ARMS[2]][1]["content"]
    if a.replace("## Observation Date\n2026-09-26", "## Observation Date\n2023-05-11") != b:
        raise AssertionError("A/B differ beyond Observation Date")
    if b.replace("\n\n# Output:", "\n" + NO_HEDGE + "\n\n# Output:") != c:
        raise AssertionError("B/C differ beyond no-hedge instruction")
    hashes = {
        arm: hashlib.sha256(json.dumps(requests[arm], ensure_ascii=False).encode()).hexdigest()
        for arm in ARMS
    }
    if OUTPUT.exists():
        payload = json.loads(OUTPUT.read_text())
        if payload["request_sha256"] != hashes:
            raise AssertionError("existing output belongs to different requests")
    else:
        payload = {
            "objective": "区分当前 Mem0 Observation Date 冲突与无依据日期模糊词的提取影响",
            "case_id": "locomo_conv-30_qa19",
            "source_dia_ids": TARGET_IDS,
            "source_session_date": entry["conversation"]["session_11_date_time"],
            "source_messages": messages,
            "model": settings.llm.model,
            "temperature": settings.llm.temperature,
            "max_calls": 3,
            "arms": ARMS,
            "observation_dates": observations,
            "request_sha256": hashes,
            "limits": [
                "只取 D11:13–15，existing memories 和 last k messages 置空；不是旧全量 ingest 的重放。",
                "当前 2026-09-26 中文配置不同于 2026-09-25 旧库提取配置。",
                "每臂一次生成；差异可能含模型波动，不能据此估计总体效果。",
                "只比较提取文本，不写库、不做 embedding、检索、答题或裁判。",
            ],
            "results": {},
        }
    if not args.execute:
        print(json.dumps({"arms": ARMS, "completed": list(payload["results"]),
                          "observation_dates": observations, "request_sha256": hashes},
                         ensure_ascii=False, indent=2))
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
    llm.client = OpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url,
                        max_retries=0, timeout=90)
    for arm in ARMS:
        if arm in payload["results"]:
            continue
        started = time.monotonic()
        try:
            raw = llm.generate_response(requests[arm], response_format={"type": "json_object"})
            parsed = json.loads(raw)
            if not isinstance(parsed.get("memory"), list):
                raise ValueError("response lacks memory array")
            payload["results"][arm] = {
                "status": "ok", "raw_response": raw, "memories": parsed["memory"],
                "usage": usage[-1] if usage else None,
                "duration_seconds": round(time.monotonic() - started, 2),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            payload["results"][arm] = {
                "status": "error", "error_type": type(exc).__name__,
                "error": str(exc)[:300], "duration_seconds": round(time.monotonic() - started, 2),
            }
            save(OUTPUT, payload)
            raise
        save(OUTPUT, payload)
    print(json.dumps({arm: {"status": payload["results"][arm]["status"],
                            "usage": payload["results"][arm].get("usage")}
                      for arm in ARMS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
