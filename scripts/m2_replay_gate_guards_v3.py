#!/usr/bin/env python3
"""阶段 1a：离线守卫 v3 重放（零模型调用）。

在 10/1 事实门 v1/v2 的 raw 输出上验证三个确定性防线，不改任何模型判定原文：
1. 限定词预拆分：括号限定强制独立检查，逐字校验 cited ref 话轮；无法逐字对应 → 撤回该限定。
2. 日期锚政策：时间子句与批次/写入日期锚年月一致 → 视为锚支持（oracle 口径），absent→full。
3. 引文覆盖率度量：每子句对 cited ref 的最长逐字连续覆盖，量化"强制逐字引用契约"会拦下多少子句（1b 必要性判据）。

产出: results/analysis/m2_gate_guards_v3_replay_20261001.json
验收复用 scripts/m2_evaluate_fact_gate.py 的 evaluate()（gated_v2 归一化幂等，不撤销 v3 语义修复）。
停止门在本脚本内按"守卫后最终动作"重判（部署形态=模型+守卫）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from m2_evaluate_fact_gate import DEVSET, evaluate  # noqa: E402

RUNS = {
    "v1": ROOT / "results/analysis/m2_fact_gate_run_20261001.json",
    "v2": ROOT / "results/analysis/m2_fact_gate_run_v2_20261001.json",
}
OUT = ROOT / "results/analysis/m2_gate_guards_v3_replay_20261001.json"

PUNCT = re.compile(r"[\s，。、；：！？“”«»\"'（）()【】\[\]·—\-…,.:;!?\n]")


def core(text: str) -> str:
    return PUNCT.sub("", text)


def turns_by_ref(gi: dict) -> dict[str, str]:
    pool = {}
    for t in gi.get("source_turns", []):
        pool[t.get("source_ref") or f"turn_{t['turn_index']}"] = t["content"]
    for s in gi.get("source_sessions", {}).values():
        for t in s["turns"]:
            pool[t["source_ref"]] = t["content"]
    return pool


def longest_verbatim_span(needle: str, haytexts: list[str]) -> int:
    """needle 在任一 hay 中的最长连续逐字子串长度（去标点后）。"""
    n = core(needle)
    best = 0
    for h in haytexts:
        hc = core(h)
        # 双指针滑窗：对每个起点扩展
        for i in range(len(n)):
            j = len(n)
            while j > i:
                if n[i:j] in hc:
                    best = max(best, j - i)
                    break
                j -= 1
            if best >= len(n) - i:
                break
    return best


def anchor_date(gi: dict) -> str | None:
    cm = gi["candidate_memory"]
    return cm.get("written_session_date") or None


TIME_PAT = re.compile(r"20\d{2}|\d{1,2}\s*月|[上下]个月?|[上下]周|去年|今年|明年|最近|截至|上周日|每周")

def anchor_covers_time(clause: str, anchor: str | None) -> bool:
    """时间子句（含时间表述才适用）的年月是否能由日期锚直接支持。宽松规则：子句中出现的
    年份与锚一致，且出现的月份与锚一致（数值比较，容忍前导零）。"""
    if not anchor or not TIME_PAT.search(clause):
        return False
    years = set(re.findall(r"(20\d{2})", clause))
    ay = re.search(r"(20\d{2})", anchor)
    months_clause = {int(m) for m in re.findall(r"(?:^|\D)(\d{1,2})\s*月", clause)}
    am = re.search(r"[/-](\d{1,2})[/-]|/(\d{1,2})\s|\s(\d{1,2})/", anchor)
    am_val = next((int(g) for g in (am.groups() if am else []) if g), None)
    ok_year = (not years) or (ay and years == {ay.group(1)})
    ok_month = (not months_clause) or (am_val and months_clause == {am_val})
    return ok_year and ok_month


QUAL_WORDS = ("含", "包括", "其中", "即")

def split_qualifier(clause: str) -> tuple[str, list[str]]:
    """抽出括号限定，返回 (去括号主干, [限定片段])。"""
    quals = re.findall(r"[（(]([^（）()]+)[)）]", clause)
    main = re.sub(r"[（(][^（）()]*[)）]", "", clause)
    return main, quals


def guard_v3(verdict: dict, gi: dict) -> tuple[dict, list[str]]:
    v = json.loads(json.dumps(verdict, ensure_ascii=False))
    turns = turns_by_ref(gi)
    anchor = anchor_date(gi)
    log = []
    for c in v["clauses"]:
        refs = c.get("source_refs", [])
        valid = [turns[r] for r in refs if r in turns]
        # 日期锚政策
        if c["status"] == "withdraw" and not refs and anchor_covers_time(c["clause"], anchor):
            c.update(support="full", status="keep", anchored_by="session_date")
            log.append(f"日期锚恢复: {c['clause'][:24]}")
            continue
        # 限定词逐字校验
        main, quals = split_qualifier(c["clause"])
        if quals and valid:
            bad = []
            for q in quals:
                span = longest_verbatim_span(q, valid)
                if span < max(2, len(core(q)) * 0.6):
                    bad.append(q)
            if bad:
                for q in bad:
                    # 限定片段无逐字来源 → 限定撤回：主干降级 weaken
                    if c["status"] == "keep":
                        c["support"] = "partial" if c["support"] == "full" else c["support"]
                        c["status"] = "weaken"
                        c["note"] = (c.get("note", "") + f" [v3守卫: 限定'{q}'在引用话轮无逐字对应]").strip()
                        log.append(f"限定撤回: '{q}' @ {c['clause'][:20]}")
                c["withdrawn_qualifiers"] = bad
    # 动作与子句一致
    actions = [c["status"] for c in v["clauses"]]
    if all(a == "keep" for a in actions) and v["record_action"] != "keep":
        v["record_action"] = "keep"
        log.append("动作对齐: 全keep→keep")
    if v["record_action"] == "keep" and any(a != "keep" for a in actions):
        v["record_action"] = "split"
        log.append("动作对齐: keep→split")
    if v["record_action"] == "withdraw" and all(a in {"keep", "weaken"} for a in actions):
        v["record_action"] = "keep" if all(a == "keep" for a in actions) else "split"
        log.append("动作对齐: withdraw→与子句一致")
    return v, log


def quote_coverage(verdict: dict, gi: dict) -> list[dict]:
    turns = turns_by_ref(gi)
    rows = []
    for c in verdict["clauses"]:
        refs = c.get("source_refs", [])
        valid = [turns[r] for r in refs if r in turns]
        n = core(c["clause"])
        span = longest_verbatim_span(c["clause"], valid) if valid else 0
        rows.append({
            "clause_head": c["clause"][:30],
            "len_core": len(n),
            "longest_span": span,
            "coverage": round(span / len(n), 3) if n else 1.0,
            "quotable_strict": span >= 8 or (n and span / len(n) >= 0.6),
        })
    return rows


def main() -> int:
    devset = json.loads(DEVSET.read_text(encoding="utf-8"))
    inputs = {g["input_id"]: g for g in devset["gate_inputs"]}
    report = {"scope": "阶段1a：守卫 v3 离线重放（零调用）。限定词逐字校验+日期锚政策+引文覆盖率度量。"}
    coverage_summary = {}
    for tag, run_path in RUNS.items():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        synth_results = []
        v3_logs, coverage_rows = {}, {}
        for r in run["results"]:
            gi = inputs[r["input_id"]]
            v3, log = guard_v3(r["raw_verdict"], gi)
            v3_logs[r["input_id"]] = log
            coverage_rows[r["input_id"]] = quote_coverage(r["raw_verdict"], gi)
            synth_results.append({"input_id": r["input_id"], "raw_verdict": r["raw_verdict"],
                                  "gated": v3, "group": r["group"]})
        synth_run = {"results": synth_results}
        evaluation = evaluate(devset, synth_run)
        sg = evaluation["stop_gate"]
        # 部署形态停止门：按守卫后最终动作重判 fail-1（raw keep 且守卫后仍 keep 才算触发）
        partial_ids = ("A1", "A3", "A4")
        final_keeps = [iid for iid in partial_ids
                       if next(x for x in synth_results if x["input_id"] == iid)["gated"]["record_action"] == "keep"]
        c_withdrawn = [iid for iid in (f"C{i}" for i in range(10))
                       if next(x for x in synth_results if x["input_id"] == iid)["gated"]["record_action"] == "withdraw"]
        deployed = {
            "fail_1_final_action_keep": final_keeps,
            "fail_2_class_rejection": c_withdrawn,
            "verdict": "FAIL" if (final_keeps or len(c_withdrawn) >= 9) else "PASS",
        }
        all_rows = [row for rows in coverage_rows.values() for row in rows]
        coverage_summary[tag] = {
            "clauses": len(all_rows),
            "strict_quotable": sum(1 for r in all_rows if r["quotable_strict"]),
            "mean_coverage": round(sum(r["coverage"] for r in all_rows) / len(all_rows), 3),
            "zero_span": sum(1 for r in all_rows if r["longest_span"] == 0),
        }
        report[tag] = {
            "acceptance": {c["case"]: c["pass"] for c in evaluation["checks"]},
            "acceptance_detail": evaluation["checks"],
            "stop_gate_raw_form": sg,
            "stop_gate_deployed_form": deployed,
            "guard_v3_logs": {k: v for k, v in v3_logs.items() if v},
            "coverage": coverage_rows,
        }
        print(f"[{tag}] 验收 {evaluation['summary']['passed']}/{evaluation['summary']['total']}"
              f" 停止门(部署形态)={deployed['verdict']} fail1={final_keeps or '无'}")
    report["coverage_summary"] = coverage_summary
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
