#!/usr/bin/env python3
"""Six-call continuation of the frozen Chinese source-relation diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

from m2_poc_source_relations_v2 import BASE_V2, POLICY_V2


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "results/analysis/m2_source_relations_v2_poc_20260928.json"
OUTPUT = ROOT / "results/analysis/m2_source_relations_v3_20260928.json"
FIXTURE_IDS = (
    "animal_question_caption_real",
    "gina_agreement_real",
    "question_only_synthetic",
)


def save(data: dict) -> None:
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing overwrite: {OUTPUT}")
    prior_bytes = PREVIOUS.read_bytes()
    fixtures = {item["fixture_id"]: item for item in json.loads(prior_bytes)["fixtures"]}
    report = {
        "scope": "three frozen Chinese snippets, two extraction prompt arms; no Memory/Graph/answer/judge calls",
        "prior_packet_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        "fixture_ids": FIXTURE_IDS,
        "max_calls": 6,
        "max_tokens_per_call": 1200,
        "temperature": 0.0,
        "sdk_max_retries": 0,
        "stop_on_first_failure": True,
        "requests": [],
        "calls": [],
        "limits": [
            "Not historical Mem0 extraction; new common JSON contract.",
            "One response per arm cannot establish stability, prevalence or answer gain.",
            "Image caption is supplied text; no image is fetched or viewed.",
            "Manual acceptance and QA gold are not sent to the model.",
        ],
    }
    for index, fixture_id in enumerate(FIXTURE_IDS):
        fixture = fixtures[fixture_id]
        lines = []
        for turn in fixture["turns"]:
            lines.append(f"[{turn['dia_id']}] {turn['speaker']}: {turn['text']}")
            if turn.get("blip_caption"):
                lines.append(f"[{turn['dia_id']} image_caption] {turn['blip_caption']}")
        suffix = "\n会话日期：" + fixture["session_date"] + "\n对话：\n" + "\n".join(lines)
        order = ("base", "source_policy") if index % 2 == 0 else ("source_policy", "base")
        for arm in order:
            user = BASE_V2 + (POLICY_V2 if arm == "source_policy" else "") + suffix
            report["requests"].append({
                "fixture_id": fixture_id,
                "origin": fixture["origin"],
                "arm": arm,
                "system": "你是事实提取器。仅输出有效JSON。",
                "user": user,
                "user_sha256": hashlib.sha256(user.encode()).hexdigest(),
            })
    assert len(report["requests"]) == report["max_calls"]
    save(report)
    print(f"prepared {len(report['requests'])} frozen requests; no model calls")


def run() -> None:
    report = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if report["calls"]:
        raise SystemExit("refusing to rerun a started diagnostic")
    from schema_rsi.config import get_settings
    from schema_rsi.llm.chat import make_chat_client

    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    report["model"] = settings.llm.model
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0, timeout=45)
    save(report)
    for request in report["requests"]:
        call = {
            "call_index": len(report["calls"]) + 1,
            "fixture_id": request["fixture_id"],
            "arm": request["arm"],
        }
        assert call["call_index"] <= report["max_calls"]
        report["calls"].append(call)
        save(report)
        try:
            raw, usage = client.complete(
                system=request["system"], user=request["user"],
                max_tokens=report["max_tokens_per_call"], temperature=report["temperature"],
            )
            call["raw_response"] = raw
            call["usage"] = usage
            try:
                call["parsed_response"] = json.loads(raw)
                call["json_parse_ok"] = True
            except ValueError:
                call["json_parse_ok"] = False
            if not raw:
                raise RuntimeError("empty response")
            print(call["call_index"], call["fixture_id"], call["arm"], "received", flush=True)
        except Exception as exc:
            call["error_type"] = type(exc).__name__
            call["http_status"] = getattr(exc, "status_code", None)
            report["stopped_after_failure"] = True
            save(report)
            print(call["call_index"], call["error_type"], call["http_status"], flush=True)
            return
        save(report)
    report["completed_planned_calls"] = True
    save(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    prepare() if args.prepare else run()
