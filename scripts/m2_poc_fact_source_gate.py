#!/usr/bin/env python3
"""事实来源门离线 POC：对固定开发集 23 条候选记忆做逐子句来源判定。

输入: results/analysis/m2_fact_gate_devset_20261001.json（门只读 gate_inputs，绝不读 oracle）
输出: results/analysis/m2_fact_gate_run_20261001.json（原始判定+守卫修正+验收对照）
预算: 每候选 1 次判定调用，JSON 解析失败允许 1 次修复调用，总上限 60 次；
      0 embedding、0 图、0 答题/裁判。max_retries=0、timeout=45s，拒绝重跑。
判定只依据源话轮、候选正文与运行时 metadata；人工 oracle 仅在验收脚本中读取
（与本脚本物理分离），用于离线验收，不进入任何判定输入。
用法: python scripts/m2_poc_fact_source_gate.py [v1|v2]
v2 为唯一一次有界修订尝试（2026-10-01 冻结）：只加严 v1 四个失败根因相关规则；
累计预算（v1+v2）60 次封顶，无论成败跑完即停，不做第三次尝试。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEVSET = ROOT / "results/analysis/m2_fact_gate_devset_20261001.json"
OUT = ROOT / "results/analysis/m2_fact_gate_run_20261001.json"
MAX_CALLS = 60

SYSTEM = """你是记忆入库前的事实来源审计器。给你一份候选记忆正文和它写入时的全部源话轮，你要把候选正文拆成子句，逐子句判断来源支持，决定处理动作。只依据给出的源话轮判断，不使用任何外部知识。

判定规则：
1. 每个子句必须指出支持它的源话轮编号（source_refs，用 source_ref 或 turn_index）。找不到任何源话轮支持 → support=absent，status=withdraw。
2. 子句核心有源但有限定词、身份、数量、细节超出原话 → support=partial，status=weaken，并在 note 中写明超出部分；若可拆分，拆成多条子句分别判。
3. 子句完整受源话轮支持 → support=full，status=keep。
4. 说话者：assistant/助手话轮的内容若没有用户话轮确认，不能作为用户事实保留；候选记忆是用户记忆时，这类子句应 withdraw 或 weaken 并注明来源说话者。
5. 纠错：若同一事实被后面的源话轮纠正（如人数、日期），以纠正后为准并引用纠正轮；被纠正的旧表述不得作为现行事实保留。
6. channel：支持话轮是正常文字填 text；来自图片 query 填 image_query；来自图片描述填 blip_caption。图片渠道线索不能写成当事人的文字自述。
7. record_action：所有子句 keep → keep；需要拆分（部分撤回/削弱）→ split；核心无源 → withdraw。partial 子句存在时禁止整条 keep。

输出严格 JSON，不要 markdown 围栏：
{"clauses": [{"clause": "...", "support": "full|partial|absent", "status": "keep|weaken|withdraw", "source_refs": ["..."], "speaker": "...", "channel": "text|image_query|blip_caption", "note": "..."}], "record_action": "keep|split|withdraw"}"""

# v2 修订（冻结于 2026-10-01）：只针对 v1 的四个失败根因加严，其余不变。
SYSTEM_V2 = SYSTEM.replace(
    "4. 说话者：assistant/助手话轮的内容若没有用户话轮确认，不能作为用户事实保留；候选记忆是用户记忆时，这类子句应 withdraw 或 weaken 并注明来源说话者。",
    "4. 说话者：assistant/助手话轮的内容若没有用户话轮确认，不能作为用户事实保留；候选记忆是用户记忆时，这类子句应 withdraw 或 weaken 并注明来源说话者。助手对用户话轮的复述、总结或\"确认\"（如\"总结一下\"）一律不算用户确认：只要用户本人的话轮没有出现过该表述（包括其中的限定词、身份、数量），该子句对用户记忆候选不得判 full。").replace(
    "7. record_action：所有子句 keep → keep；需要拆分（部分撤回/削弱）→ split；核心无源 → withdraw。partial 子句存在时禁止整条 keep。",
    """7. record_action 必须与子句一致：任何子句 withdraw → 不得为 keep；全部子句 keep → 不得为 withdraw；需要拆分（部分撤回/削弱）→ split。partial 子句存在时禁止整条 keep。
8. 限定词：源话轮包含明确限定或缓和语（如"但我还好""可能""听说"）而候选正文写成无条件陈述 → support=partial，status=weaken，note 写明缺失的限定词。
9. 身份推断：同行者、同伴、关系人等身份只有源话轮明确指名才能保留；源话轮只说"我们"等未指明表述时，身份部分必须 withdraw（出行等核心事实可拆成单独子句保留），不能凭推断 weaken 保留。""")


def fmt_source_turns(gi: dict) -> str:
    lines = []
    if "source_turns" in gi:
        for t in gi["source_turns"]:
            ref = t.get("source_ref") or f"turn_{t['turn_index']}"
            extra = ""
            if t.get("image_query"):
                extra += f" [图片query: {t['image_query']}]"
            if t.get("blip_caption"):
                extra += f" [图片描述: {t['blip_caption']}]"
            lines.append(f"# {ref} | {t.get('role') or t.get('speaker')}: {t['content']}{extra}")
    else:
        for sid, s in gi["source_sessions"].items():
            lines.append(f"## 会话 {sid}（{s['date']}）")
            for t in s["turns"]:
                lines.append(f"# {t['source_ref']} | {t['speaker']}: {t['content']}")
    return "\n".join(lines)


def fmt_candidate(gi: dict) -> str:
    cm = gi["candidate_memory"]
    meta = [f"written_session: {cm.get('written_session_id')}"]
    if cm.get("written_session_date"):
        meta.append(f"written_date: {cm['written_session_date']}")
    if cm.get("attributed_to"):
        meta.append(f"attributed_to: {cm['attributed_to']}")
    return f"候选记忆正文：{cm['content']}\n（{'；'.join(meta)}）"


def strip_fence(text: str) -> str:
    m = re.fullmatch(r"```(?:json)?\s*\n(.*)\n\s*```", text, re.S)
    return m.group(1) if m else text


def parse_verdict(text: str) -> dict:
    data = json.loads(strip_fence(text))
    assert isinstance(data.get("clauses"), list) and data.get("record_action") in {
        "keep", "split", "withdraw"}
    for c in data["clauses"]:
        assert c["support"] in {"full", "partial", "absent"}
        assert c["status"] in {"keep", "weaken", "withdraw"}
    return data


def apply_guards(verdict: dict, valid_refs: set[str]) -> dict:
    """确定性守卫：absent 强制撤回；partial 禁止 keep；keep 与撤回子句并存降级 split；
    引用不存在的源话轮记为 hallucinated_ref。返回修正后的副本与修正日志。"""
    v = json.loads(json.dumps(verdict, ensure_ascii=False))
    log = []
    for c in v["clauses"]:
        bad = [r for r in c.get("source_refs", []) if r not in valid_refs]
        if bad:
            c["hallucinated_ref"] = bad
            log.append(f"引用不存在源: {bad}")
        if c["support"] == "absent" and c["status"] != "withdraw":
            log.append(f"absent→withdraw: {c['clause'][:30]}")
            c["status"] = "withdraw"
        if c["support"] == "partial" and c["status"] == "keep":
            log.append(f"partial keep→weaken: {c['clause'][:30]}")
            c["status"] = "weaken"
    if v["record_action"] == "keep" and any(c["status"] != "keep" for c in v["clauses"]):
        log.append("存在非 keep 子句，record_action keep→split")
        v["record_action"] = "split"
    if v["record_action"] == "keep" and any(c.get("hallucinated_ref") for c in v["clauses"]):
        log.append("存在无效引用，record_action keep→split")
        v["record_action"] = "split"
    v["guard_log"] = log
    return v


def valid_refs_of(gi: dict) -> set[str]:
    refs = set()
    if "source_turns" in gi:
        for t in gi["source_turns"]:
            refs.add(t.get("source_ref") or f"turn_{t['turn_index']}")
    else:
        for s in gi["source_sessions"].values():
            for t in s["turns"]:
                refs.add(t["source_ref"])
    return refs


def run() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    assert version in ("v1", "v2")
    system = SYSTEM if version == "v1" else SYSTEM_V2
    out = OUT if version == "v1" else ROOT / "results/analysis/m2_fact_gate_run_v2_20261001.json"
    if out.exists():
        raise SystemExit(f"拒绝重跑：{out} 已存在（单次运行规则）")
    # v2 为唯一一次修订尝试：总预算（含 v1 已耗）冻结 60 次，成败即停。
    cumulative_cap = MAX_CALLS
    spent_before = 0
    if version == "v2":
        v1 = json.loads((ROOT / "results/analysis/m2_fact_gate_run_20261001.json").read_text(encoding="utf-8"))
        spent_before = v1["budget"]["used_calls"]
    devset = json.loads(DEVSET.read_text(encoding="utf-8"))
    gate_inputs = devset["gate_inputs"]

    from schema_rsi.config import get_settings
    from schema_rsi.llm.chat import ChatClient

    s = get_settings()
    client = ChatClient(base_url=s.llm.base_url, api_key=s.llm.api_key,
                        model=s.llm.model, timeout=45)
    client._client = client._client.with_options(max_retries=0, timeout=45)

    calls = 0
    results = []
    run_report = {
        "scope": f"事实来源门离线 POC 运行账（{version}）：23 条候选逐子句判定，原始判定与守卫修正分开记录。",
        "version": version,
        "devset_sha256": __import__("hashlib").sha256(DEVSET.read_bytes()).hexdigest(),
        "model": s.llm.model,
        "budget": {"max_calls_cumulative": cumulative_cap, "spent_before": spent_before,
                   "used_calls_this_run": calls, "embedding": 0,
                   "graph": 0, "answer_or_judge": 0},
        "oracle_note": "本文件不含 oracle 对照；oracle 在 devset 中，判定阶段未读取。",
        "results": results,
    }
    save_report = lambda: out.write_text(
        json.dumps(run_report, ensure_ascii=False, indent=1), encoding="utf-8")
    save_report()
    for gi in gate_inputs:
        user = (f"源话轮（判定唯一依据）：\n{fmt_source_turns(gi)}\n\n{fmt_candidate(gi)}\n\n"
                "请输出逐子句判定 JSON。")
        raw_text, usage = client.complete(system=system, user=user, max_tokens=2048)
        calls += 1
        entry = {"input_id": gi["input_id"], "group": gi["group"],
                 "model": s.llm.model, "usage": usage, "calls": 1}
        try:
            verdict = parse_verdict(raw_text)
        except Exception as e:
            repair, _ = client.complete(
                system=system, user=user + "\n\n你上次的输出无法解析为要求的 JSON（"
                + str(e)[:120] + "）。重新输出严格 JSON，不要任何围栏或说明文字。",
                max_tokens=2048)
            calls += 1
            entry["repair"] = True
            raw_text = repair
            verdict = parse_verdict(raw_text)
        assert spent_before + calls <= cumulative_cap, "超出累计调用预算"
        entry["raw_verdict"] = verdict
        entry["raw_text_head"] = raw_text[:200]
        entry["gated"] = apply_guards(verdict, valid_refs_of(gi))
        results.append(entry)
        run_report["budget"]["used_calls_this_run"] = calls
        save_report()
        print(f"  {gi['input_id']}: {entry['gated']['record_action']} "
              f"({len(verdict['clauses'])} clauses, calls={calls})", flush=True)

    run_report["budget"]["used_calls_this_run"] = calls
    save_report()
    print(f"✓ 运行账写入 {out.relative_to(ROOT)}，本次调用 {calls} 次（累计 {spent_before + calls}）")
    return 0


if __name__ == "__main__":
    sys.exit(run())
