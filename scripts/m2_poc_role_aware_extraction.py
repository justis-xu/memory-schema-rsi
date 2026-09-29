#!/usr/bin/env python3
"""Fixed four-call Chinese source-role extraction diagnostic; no writes to memory."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LME = ROOT / "results/analysis/m2_lme_update_operations_20260928.json"
LOCOMO = ROOT / "results/analysis/m2_role_label_source_sample_20260929.json"
OUT = ROOT / "results/analysis/m2_role_aware_extraction_20260929.json"
LME_TURNS = (4, 5, 6, 7, 8, 10, 11)

BASE = """从下面的中文对话提取关于参与者的可核事实。只输出有效 JSON：
{"facts":[{"claim":"一个原子事实，含必要限定语","subject":"人物","source_turn_ids":["输入中的ID"],"source_speaker":"原话说话者","status":"本人陈述/对方陈述/建议/计划/推断/纠正等"}]}
不要使用外部知识。每个事实只写一个主张；不确定的内容保持限定语。无事实输出空数组。
"""
POLICY = """额外来源规则：区分对话载体角色和说话者身份。真人对话中，即使载体角色是 assistant，具名的第二说话者仍是真人，保留他本人的有源事实。AI 助理回复中的建议、身份或报价，在用户未确认前不可写成用户事实；若提取，必须标为助理陈述或建议。一个结论涉及两人时，引用双方必要话轮。纠正后的新值与此前旧值分开，不把未明确说出的包含关系补出来。计划不写成已发生。
"""


def save(report):
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def prepare():
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")
    lp = json.loads(LME.read_text())
    case = next(c for c in lp["cases"] if c["question_id"] == "031748ae")
    session = next(s for s in case["source"]["chinese"]["answer_sessions"]
                   if s["session_id"] == "answer_8748f791_1")
    turns = {t["index"]: t for t in session["turns"]}
    lme_lines = ["场景：LongMemEval；user 是真人，assistant 是 AI 助理。", "日期：" + session["date"]]
    for i in LME_TURNS:
        t = turns[i]
        lme_lines.append(f"[answer_8748f791_1:turn:{i}] {t['role']}：{t['content']}")

    cp = json.loads(LOCOMO.read_text())
    conv = next(c for c in cp["cases"] if c["conversation"] == "conv-44")
    locomo_lines = ["场景：LoCoMo；speaker_a 和 speaker_b 是两位真人，适配器将 speaker_b 放在 assistant 载体角色。",
                    f"speaker_a={conv['speaker_a']}；speaker_b={conv['speaker_b']}；场次={conv['written_session']}"]
    for t in conv["source_witnesses"]:
        locomo_lines.append(f"[{t['dia_id']}] 载体角色={t['role_in_adapter']}；说话者={t['speaker']}：{t['text']}")

    fixtures = [("lme_engineer", "\n".join(lme_lines), "base", "source_policy"),
                ("locomo_two_humans", "\n".join(locomo_lines), "source_policy", "base")]
    report = {
        "scope": "两段固定中文真实来源；两提示臂；只测提取来源与反伤",
        "source_sha256": {"lme_packet": hashlib.sha256(LME.read_bytes()).hexdigest(),
                          "locomo_witness_packet": hashlib.sha256(LOCOMO.read_bytes()).hexdigest()},
        "selected_lme_turns": list(LME_TURNS), "selected_locomo_dia_ids": [t["dia_id"] for t in conv["source_witnesses"]],
        "model_calls_max": 4, "max_tokens_per_call": 1100, "temperature": 0.0,
        "sdk_max_retries": 0, "stop_on_first_failure": True,
        "requests": [], "calls": [],
        "limits": ["原题、gold、人工结论未进入模型请求。", "定向源片段，不是完整会话或历史写入重放。",
                   "不调用答题、裁判、图或记忆写入，提取结果不能声称答题收益。"],
    }
    for fixture, excerpt, *arms in fixtures:
        for arm in arms:
            user = BASE + (POLICY if arm == "source_policy" else "") + "\n来源对话：\n" + excerpt
            report["requests"].append({"fixture": fixture, "arm": arm,
                                       "system": "你是中文事实提取器，只输出有效 JSON。", "user": user,
                                       "user_sha256": hashlib.sha256(user.encode()).hexdigest()})
    assert len(report["requests"]) == 4
    save(report)
    print("prepared four requests, zero model calls")


def run():
    report = json.loads(OUT.read_text())
    if report["calls"]:
        raise SystemExit("refusing to rerun started diagnostic")
    from schema_rsi.config import get_settings
    from schema_rsi.llm.chat import make_chat_client
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    report["model"] = settings.llm.model
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0, timeout=45)
    save(report)
    for req in report["requests"]:
        call = {"call_index": len(report["calls"]) + 1, "fixture": req["fixture"], "arm": req["arm"]}
        assert call["call_index"] <= 4
        report["calls"].append(call)
        save(report)
        try:
            raw, usage = client.complete(system=req["system"], user=req["user"],
                                         max_tokens=report["max_tokens_per_call"], temperature=0.0)
            call["raw_response"] = raw
            call["usage"] = usage
            try:
                call["parsed_response"] = json.loads(raw)
                call["json_parse_ok"] = True
            except ValueError:
                call["json_parse_ok"] = False
            if not raw:
                raise RuntimeError("empty response")
            print(call["call_index"], call["fixture"], call["arm"], "received", flush=True)
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
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.prepare == args.run:
        p.error("choose --prepare or --run")
    prepare() if args.prepare else run()
