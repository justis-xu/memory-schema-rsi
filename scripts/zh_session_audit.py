#!/usr/bin/env python
"""强模型审计造真值：中文轨道 conv 的每个 session，用 glm-5.3 找当前记忆漏提取的事实。

背景：上一版英文 POC（results/laya_poc/poc_report.md）的真值来自
attribution_split_A.json（关键词归因+人工校准），中文轨道没有对应文件。本脚本
改用强模型逐 session 审计：glm-5.3（非 flash；base_url/api_key 取自 .env 的
LLM_BASE_URL/LLM_API_KEY，model 固定 'glm-5.3'，temperature=0，按项目惯例
禁思考，见 src/schema_rsi/llm/chat.py:44），输入 = 该 session 的中文原文
turns + 该 session_id 的中文记忆（从 {prefix}:locomo:{conv} 按
metadata.session_id 分组），要求只输出 {"missing_facts": [...]}，每条是
『原文中明确存在、值得提取、但当前记忆里没有』的具体事实，中文短语形式。

标签修正规则（上次英文 POC 的教训，内置）：每条 missing_fact 取其『连续 ≥4 字
且最长』、逐字出现在该 session 自己原文里的关键短语做锚定；找不到这种短语
（典型：QA gold 泄漏 / 跨 session 误归因）→ 丢弃并计入 dropped。

输出 results/laya_poc_zh/audit.json：{"sessions": [{"session_id",
"missing_facts": [...], "dropped": [...]}]}（失败 session 额外带 error 键）。
stdout 打印：正例 session 数（missing_facts 非空）/ 有效漏点总条数 / 丢弃条数。

串行、并发=1（另一窗口可能还在跑终测，不抢配额）；单次失败（含 JSON 解析失败）
重试 2 次后记录 error 继续；逐 session 落盘，中断重跑安全（重跑覆盖全文件）。
只读数据集与 {prefix}: 向量库；唯一写入是 --out。

用法：
  .venv/bin/python scripts/zh_session_audit.py --config config/locomo_zh.yaml \
      --prefix zhfull --conv conv-42 [--limit 2] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 关 mem0 PostHog 噪声，保 stdout 干净

AUDIT_MODEL = "glm-5.3"  # 非 flash 强模型（提取主库是 glm-5.3-flash）
MIN_KEYPHRASE = 4  # 标签修正：关键短语最少连续字数（中文 4 字撞库概率已很低）

SYSTEM_PROMPT = (
    "你是记忆提取审计员。对照单段对话原文与当前已提取的记忆，"
    "找出原文中明确存在、值得提取、但当前记忆遗漏的具体事实。"
    "只输出 JSON，不要输出任何解释。"
)

USER_PROMPT_TMPL = """【对话原文（{session_id}）】
{turns}

【当前已提取记忆】
{memories}

严格对照上面的原文，列出当前记忆遗漏的具体事实。要求：
- 每条是原文中明确存在的信息（人物/时间/地点/数字/事件/偏好等），用中文短语表述
- 措辞尽量贴近原文用字，不要改写、不要推断、不要联想
- 当前记忆已覆盖的信息不要列
- 没有遗漏就输出空数组
只输出 JSON：{{"missing_facts": ["…", "…"]}}"""


def build_user_prompt(session: dict, memory_texts: list[str]) -> str:
    turns = "\n".join(f"{t['speaker']}: {t['content']}" for t in session["turns"])
    mems = "\n".join(f"- {m}" for m in memory_texts) if memory_texts else "（该 session 无提取记忆）"
    return USER_PROMPT_TMPL.format(session_id=session["session_id"], turns=turns, memories=mems)


def session_text(session: dict) -> str:
    """该 session 自己的原文（锚定用，与 prompt 里的原文段逐字一致）。"""
    return "\n".join(f"{t['speaker']}: {t['content']}" for t in session["turns"])


# ---- glm-5.3 调用（串行，单失败重试 2 次） -----------------------------------
def audit_session(client, session: dict, memory_texts: list[str], retries: int = 2) -> list[str]:
    """调 glm-5.3 拿 missing_facts（含 JSON 解析；重试耗尽抛错由调用方记录）。"""
    last = ""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=AUDIT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(session, memory_texts)},
                ],
                temperature=0.0,
                max_tokens=2048,
                extra_body={"thinking": {"type": "disabled"}},  # 禁思考，项目惯例
            )
            text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
            return parse_missing_facts(text)
        except Exception as e:  # API 错误与解析失败同样退避重试
            last = str(e)[:200]
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"glm-5.3 audit failed: {last}")


def parse_missing_facts(text: str) -> list[str]:
    """容忍 ```json 围栏/前后杂文，取出 {"missing_facts": [...]}。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"输出中无 JSON 对象: {text[:120]!r}")
    obj = json.loads(m.group(0))
    if isinstance(obj.get("missing_facts"), list):
        return [str(x).strip() for x in obj["missing_facts"] if str(x).strip()]
    raise ValueError(f"missing_facts 不是列表: {text[:120]!r}")


# ---- 标签修正：关键短语锚定 ---------------------------------------------------
def ground_keyphrase(fact: str, text: str, min_len: int = MIN_KEYPHRASE) -> str | None:
    """fact 中『连续 ≥min_len 且最长』、逐字出现在 text 里的子串；找不到返回 None。

    从最长往短扫：返回的第一个命中即最长锚定短语。
    """
    n = len(fact)
    for length in range(n, min_len - 1, -1):
        for start in range(0, n - length + 1):
            seg = fact[start : start + length]
            if seg in text:
                return seg
    return None


# ---- 主流程 -----------------------------------------------------------------
def _write_out(out_path: Path, sessions: list[dict]) -> None:
    out_path.write_text(
        json.dumps({"sessions": sessions}, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/locomo_zh.yaml",
                    help="配置轨道（默认中文 config/locomo_zh.yaml）")
    ap.add_argument("--prefix", default="zhfull", help="记忆 user_id 前缀（中文轨道 zhfull）")
    ap.add_argument("--conv", default="conv-42")
    ap.add_argument("--limit", type=int, default=None, help="只审计前 N 个 session（冒烟用）")
    ap.add_argument("--out", default="results/laya_poc_zh/audit.json")
    ap.add_argument("--verbose", action="store_true", help="逐条打印锚定/丢弃明细（冒烟核验用）")
    args = ap.parse_args()

    from openai import OpenAI

    from schema_rsi.benchmarks.locomo import LocomoDataset
    from schema_rsi.config import get_settings
    from schema_rsi.memory import Mem0Backend

    settings = get_settings(args.config)
    if not (settings.llm.base_url and settings.llm.api_key):
        raise SystemExit("[FATAL] .env 缺 LLM_BASE_URL/LLM_API_KEY（audit 需要 glm-5.3 端点）")
    client = OpenAI(base_url=settings.llm.base_url, api_key=settings.llm.api_key, timeout=180.0)

    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    case = next((c for c in ds.cases if c.metadata.get("conversation_id") == args.conv), None)
    if case is None:
        raise SystemExit(f"[FATAL] 对话 {args.conv} 不在数据集 {settings.locomo_path}")
    sessions = case.history
    if args.limit:
        sessions = sessions[: args.limit]

    uid = f"{args.prefix}:locomo:{args.conv}"
    backend = Mem0Backend(settings)  # 只读；中文库 data/vector_store_zh/
    by_sid: dict[str, list[str]] = {}
    for m in backend.get_all_memories(user_id=uid):
        by_sid.setdefault(str((m.metadata or {}).get("session_id")), []).append(m.content)
    print(f"[audit] {args.conv}: sessions={len(sessions)}  user_id={uid}  "
          f"记忆 {sum(len(v) for v in by_sid.values())} 条 / 覆盖 {len(by_sid)} 个 session")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_sessions: list[dict] = []
    n_pos = n_kept = n_dropped = 0
    for i, sess in enumerate(sessions, 1):
        sid = sess["session_id"]
        mems = by_sid.get(sid, [])
        text = session_text(sess)
        entry: dict = {"session_id": sid, "missing_facts": [], "dropped": []}
        try:
            facts = audit_session(client, sess, mems)
        except Exception as e:  # 重试耗尽：记录 error，继续下一个 session
            entry["error"] = str(e)[:300]
            print(f"  [{i}/{len(sessions)}] {sid:11} ERROR: {str(e)[:120]}")
            out_sessions.append(entry)
            _write_out(out_path, out_sessions)
            continue
        for fact in dict.fromkeys(facts):  # 保序去重
            key = ground_keyphrase(fact, text)
            if key is None:
                entry["dropped"].append(fact)
                if args.verbose:
                    print(f"      DROP  {fact}")
            else:
                entry["missing_facts"].append(fact)
                if args.verbose:
                    print(f"      KEEP  {fact}   [锚定: {key}]")
        if entry["missing_facts"]:
            n_pos += 1
        n_kept += len(entry["missing_facts"])
        n_dropped += len(entry["dropped"])
        print(f"  [{i}/{len(sessions)}] {sid:11} turns={len(sess['turns']):3} mems={len(mems):2} "
              f"漏点={len(entry['missing_facts']):2} 丢弃={len(entry['dropped']):2}")
        out_sessions.append(entry)
        _write_out(out_path, out_sessions)

    summary = {
        "conv": args.conv,
        "model": AUDIT_MODEL,
        "n_sessions": len(out_sessions),
        "n_sessions_with_missing": n_pos,  # 正例 session 数
        "n_kept_facts": n_kept,  # 有效漏点总条数
        "n_dropped_facts": n_dropped,  # 丢弃条数
        "n_errors": sum(1 for s in out_sessions if s.get("error")),
        "output_file": str(out_path.relative_to(PROJECT_ROOT)),
    }
    print("\n=== AUDIT SUMMARY (JSON) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("=== END SUMMARY ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
