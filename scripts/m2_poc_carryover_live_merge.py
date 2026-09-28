#!/usr/bin/env python3
"""Bounded Chinese source-grounded merge diagnostic; no persistent memory writes."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
INVENTORY = ROOT / "results/analysis/m2_carryover_cache_eligibility_20260928.json"
WITNESSES = ROOT / "results/analysis/m2_carryover_current_witnesses_20260928.json"
OUTPUT = ROOT / "results/analysis/m2_carryover_live_merge_v1_20260928.json"
TIM_IDS = ("conv-43_de89be96dda6", "conv-43_c95588df7268")
POTTERY_ID = "conv-26_444350391e64"
ROAD_ID = "current_conv-41_509b61c0"

BASE = """请用简体中文根据“候选记忆”和“来源对话”输出逐事实核查表，只输出有效 JSON：
{"facts":[{"case_id":"病例ID","claim_zh":"单一事实","support":"full/partial/absent","source_turn_ids":["D场:轮"],"source_session_id":"session_N 或 null","source_speakers":["说话者"],"covered_by_memory_ids":["记忆ID"],"action":"保留/合并/撤回"}]}
把复合内容拆成可独立核查的最小事实；同一事实若出现在多个记忆版本，用 covered_by_memory_ids 标明。来源对话中没有证据的主张也列出并标 absent/撤回。不要用候选记忆本身充当来源。来源字段必须引用输入中的真实 turn。
"""
POLICY = """额外来源与反伤规则：
- 区分事实首次出现的来源场次与候选记忆写入场次；后一场重述旧事，不会自动改变原始来源日期。
- 旧版遗漏而新版包含的有源细节要保留；新版遗漏的旧版有源细节也要保留，不能取较短版本覆盖。
- 区分本人明确自述、他人猜测、图片附带文字。不能把他人推测写成本人自述，图片 caption 也不等于看过原图。
- 保留“但还好”、愿望、时间范围等限定语，不把部分支持升格成确定结论。
- “我们”未说明同行者身份时保留旅行事实，但撤回具体同行者身份；不要依据人物关系猜测。
- 每个被接受的事实都给足以支持其主体、对象和限定语的 turn；无法支持时用 absent/撤回，不凭常识补足。
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_lines(conversation: dict, session_ids: tuple[str, ...]) -> str:
    lines = []
    for session_id in session_ids:
        lines.append(f"## {session_id} 日期：{conversation.get(session_id + '_date_time')}")
        for turn in conversation[session_id]:
            did = turn["dia_id"]
            lines.append(f"[{did} text] {turn['speaker']}：{turn.get('text', '')}")
            if turn.get("blip_caption"):
                lines.append(f"[{did} image_caption] {turn['blip_caption']}")
            if turn.get("query"):
                lines.append(f"[{did} image_query] {turn['query']}")
    return "\n".join(lines)


def prepare() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing overwrite: {OUTPUT}")
    data = {item["sample_id"]: item for item in json.loads(DATA.read_text())}
    inventory = json.loads(INVENTORY.read_text())
    by_pair = {pair["pair_id"]: pair for pair in inventory["corrected_fact_inventory"]["pairs"]}
    witnesses = json.loads(WITNESSES.read_text())
    road = witnesses["road_trip_companion"]["current_memory"]
    assert road["id"] == "509b61c0-35da-4fe7-94a0-e6924cf50f07"

    fixtures = [
        ("tim_writing_castle", "conv-43", ("session_15", "session_16"),
         [by_pair[pair_id] for pair_id in TIM_IDS]),
        ("melanie_pottery", "conv-26", ("session_17", "session_18"),
         [by_pair[POTTERY_ID]]),
    ]
    report = {
        "scope": "two historical Chinese merge packets and one current Chinese source audit; not historical Mem0 replay",
        "source_sha256": sha(DATA), "inventory_sha256": sha(INVENTORY),
        "witnesses_sha256": sha(WITNESSES),
        "max_calls": 6, "max_tokens_per_call": 2200, "temperature": 0.0,
        "sdk_max_retries": 0, "stop_on_first_failure": True,
        "requests": [], "calls": [],
        "limits": ["Full source sessions are supplied: this tests source interpretation, not source retrieval.",
                   "Human fact inventory and QA gold are never included in model requests.",
                   "No Memory/Graph/retrieval/answer/judge writes or calls; one response per arm is not an effect estimate."],
    }
    for index, (fixture_id, conv_id, sessions, pairs) in enumerate(fixtures):
        candidates = []
        for pair in pairs:
            for version in ("older_memory", "later_memory"):
                memory = pair[version]
                assert memory["content"] and any(ord(char) >= 128 for char in memory["content"])
                candidates.append({"case_id": pair["pair_id"], "memory_id": memory["id"],
                                   "write_session_id": memory["session_id"], "content": memory["content"]})
        transcript = source_lines(data[conv_id]["conversation"], sessions)
        fixture = {"fixture_id": fixture_id, "origin": "historical_chinese_memory_pair",
                   "case_ids": [pair["pair_id"] for pair in pairs], "candidates": candidates,
                   "source_sessions": sessions, "source_conversation": conv_id}
        report.setdefault("fixtures", []).append(fixture)
        suffix = "\n候选记忆：\n" + json.dumps(candidates, ensure_ascii=False) + "\n来源对话：\n" + transcript
        order = ("base", "source_policy") if index % 2 == 0 else ("source_policy", "base")
        for arm in order:
            user = BASE + (POLICY if arm == "source_policy" else "") + suffix
            report["requests"].append({"fixture_id": fixture_id, "arm": arm,
                                       "system": "你是中文事实核查与记忆归并器。仅输出有效 JSON。",
                                       "user": user, "user_sha256": hashlib.sha256(user.encode()).hexdigest()})

    road_candidate = {"case_id": ROAD_ID, "memory_id": road["id"],
                      "write_session_id": road["session_id"], "content": road["content"]}
    road_source = source_lines(data["conv-41"]["conversation"], ("session_11",))
    report["fixtures"].append({"fixture_id": "road_companion_current", "origin": "current_chinese_memory_source_audit",
                               "case_ids": [ROAD_ID], "candidates": [road_candidate],
                               "source_sessions": ["session_11"], "source_conversation": "conv-41"})
    suffix = "\n候选记忆：\n" + json.dumps([road_candidate], ensure_ascii=False) + "\n来源对话：\n" + road_source
    for arm in ("base", "source_policy"):
        user = BASE + (POLICY if arm == "source_policy" else "") + suffix
        report["requests"].append({"fixture_id": "road_companion_current", "arm": arm,
                                   "system": "你是中文事实核查与记忆归并器。仅输出有效 JSON。",
                                   "user": user, "user_sha256": hashlib.sha256(user.encode()).hexdigest()})
    assert len(report["requests"]) == report["max_calls"]
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("prepared 6 fixed requests; no model calls")


def run() -> None:
    report = json.loads(OUTPUT.read_text())
    if report["calls"]:
        raise SystemExit("refusing to rerun a started diagnostic")
    from schema_rsi.config import get_settings
    from schema_rsi.llm.chat import make_chat_client

    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    report["model"] = settings.llm.model
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0, timeout=60)
    def save():
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    save()
    for request in report["requests"]:
        call = {"call_index": len(report["calls"]) + 1,
                "fixture_id": request["fixture_id"], "arm": request["arm"]}
        assert call["call_index"] <= report["max_calls"]
        report["calls"].append(call)
        save()
        try:
            raw, usage = client.complete(system=request["system"], user=request["user"],
                                         max_tokens=report["max_tokens_per_call"], temperature=0.0)
            call.update(raw_response=raw, usage=usage)
            try:
                call["parsed_response"] = json.loads(raw)
                call["json_parse_ok"] = True
            except ValueError:
                call["json_parse_ok"] = False
            if not raw:
                raise RuntimeError("empty response")
            print(call["call_index"], call["fixture_id"], call["arm"], "received", flush=True)
        except Exception as exc:
            call.update(error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
            report["stopped_after_failure"] = True
            save()
            print(call["call_index"], call["error_type"], call["http_status"], flush=True)
            return
        save()
    report["completed_planned_calls"] = True
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    prepare() if args.prepare else run()
