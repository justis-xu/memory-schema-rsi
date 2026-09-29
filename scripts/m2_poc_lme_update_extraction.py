#!/usr/bin/env python3
"""Four-call Chinese temporal fact extraction POC on frozen LME source excerpts."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/analysis/m2_lme_update_operations_20260928.json"
OUT = ROOT / "results/analysis/m2_lme_update_extraction_20260929.json"

SELECTED = {
    "031748ae": {"answer_8748f791_1": [4, 5, 8, 9, 10, 11],
                 "answer_8748f791_2": [0, 1, 2, 3]},
    "f685340e": {"answer_25df025b_1": [0, 1, 6, 7],
                 "answer_25df025b_2": [0, 1]},
}
BASE = """从以下中文对话中提取用户自身的可核事实。只输出有效 JSON：
{"facts":[{"subject":"人物","relation":"关系","object":"值或对象","time_scope":"原话限定的时间或场次","status":"陈述、计划或纠正等状态","source_turn_ids":["源轮次ID"]}]}
每条 fact 只写一个事实；来源 ID 必须在输入里。助手发言只作上下文，不当作用户本人声明。不要使用外部知识。没有事实则 facts 为空数组。
"""
POLICY = """额外来源约束：
- 同场用户明确纠正前一句时，保留纠正关系，不把被纠正值当现行值；跨场新值仍须保留旧时间窗。
- 一次周日计划不能推出每周日惯例；频率和具体星期分别核来源，保留“计划”和“惯常”的区别。
- 一个事实若需多轮才能支持，引用全部必要轮次；不要把助手推测写成用户本人明确自述。
"""


def save(report):
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def prepare():
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")
    packet_bytes = PACKET.read_bytes()
    packet = json.loads(packet_bytes)
    by_id = {case["question_id"]: case for case in packet["cases"]}
    assert set(SELECTED) <= set(by_id)
    fixtures = []
    for question_id, sessions in SELECTED.items():
        source_sessions = {s["session_id"]: s for s in
                           by_id[question_id]["source"]["chinese"]["answer_sessions"]}
        lines = []
        refs = []
        for session_id, indices in sessions.items():
            session = source_sessions[session_id]
            lines.append(f"\n场次 {session_id}；日期 {session['date']}：")
            turns = {turn["index"]: turn for turn in session["turns"]}
            for index in indices:
                turn = turns[index]
                ref = f"{session_id}:turn:{index}"
                refs.append(ref)
                lines.append(f"[{ref}] {turn['role']}：{turn['content']}")
        fixture = {"question_id": question_id, "source_refs": refs,
                   "conversation_text": "\n".join(lines)}
        fixtures.append(fixture)
    report = {
        "scope": "两道固定中文知识更新病例、两个提取提示臂；非历史 Mem0 重放",
        "source_packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "selected_turns": SELECTED,
        "model_calls_max": 4,
        "max_tokens_per_call": 1200,
        "temperature": 0.0,
        "sdk_max_retries": 0,
        "stop_on_first_failure": True,
        "requests": [], "calls": [],
        "limits": ["问题和 gold、人工侧车均不进入请求。",
                   "所选为真实来源场次的定向摘录，不是完整历史或历史 Mem0 提取重放。",
                   "仅评提取来源保真与反伤，不调用答题或评分模型，不推断答题收益。"],
    }
    for index, fixture in enumerate(fixtures):
        order = ("base", "source_policy") if index == 0 else ("source_policy", "base")
        for arm in order:
            user = BASE + (POLICY if arm == "source_policy" else "") + fixture["conversation_text"]
            report["requests"].append({"question_id": fixture["question_id"],
                                       "arm": arm, "source_refs": fixture["source_refs"],
                                       "system": "你是中文事实提取器，只输出有效 JSON。",
                                       "user": user,
                                       "user_sha256": hashlib.sha256(user.encode()).hexdigest()})
    assert len(report["requests"]) == report["model_calls_max"]
    save(report)
    print("prepared 4 fixed requests; zero model calls")


def run():
    report = json.loads(OUT.read_text())
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
        call = {"call_index": len(report["calls"]) + 1,
                "question_id": request["question_id"], "arm": request["arm"]}
        assert call["call_index"] <= report["model_calls_max"]
        report["calls"].append(call)
        save(report)
        try:
            raw, usage = client.complete(system=request["system"], user=request["user"],
                                         max_tokens=report["max_tokens_per_call"],
                                         temperature=report["temperature"])
            call["raw_response"] = raw
            call["usage"] = usage
            try:
                call["parsed_response"] = json.loads(raw)
                call["json_parse_ok"] = True
            except ValueError:
                call["json_parse_ok"] = False
            if not raw:
                raise RuntimeError("empty response")
            print(call["call_index"], call["question_id"], call["arm"], "received", flush=True)
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
        parser.error("choose --prepare or --run")
    prepare() if args.prepare else run()
