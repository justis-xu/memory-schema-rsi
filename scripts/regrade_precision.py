#!/usr/bin/env python
"""对已落盘的 LoCoMo 结果做精准度重判（不重新回答，零答题成本）。

两阶段：
1. 确定性严格检查（免费）：temporal 日期/时长精确匹配；其余题型实词+数字齐全
2. LLM 严格判分（glm-5.3）：只对宽松 judge 判 CORRECT 的题分级 exact/partial/wrong
   （宽松判 WRONG 的题在精准口径下不可能 exact，直接记 wrong）

用法：
  .venv/bin/python scripts/regrade_precision.py \
      base=results/full_locomo_20260923_083352.jsonl \
      graph=results/full_locomo_graph_scoped_20260924_020330.jsonl \
      [--pair base graph] [--judge-model glm-5.3] [--no-judge] [--limit 50]

输出：
  results/precision_regrade_<label>.jsonl   每题 {case_id, category, lenient,
                                             det:{passed,rule,reason}, judge:{...}, final, hedges}
  results/precision_summary_<ts>.json       汇总 + 配对翻转
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation.precision import find_hedges, strict_check  # noqa: E402
from schema_rsi.evaluation.precision_judge import (  # noqa: E402
    DEFAULT_PRECISION_MODEL,
    PrecisionJudge,
    PrecisionJudgeError,
)
from schema_rsi.llm.rate import AdaptiveLimiter, is_rate_limit_error  # noqa: E402

J_CATS = ("multi-hop", "temporal", "open-domain", "single-hop")
ALL_CATS = J_CATS + ("adversarial",)


def load_run(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def grade_run(
    label: str,
    path: str,
    judge: PrecisionJudge | None,
    limiter: AdaptiveLimiter | None,
    limit: int | None,
    judge_all: bool = False,
) -> tuple[list[dict], list[dict]]:
    """返回 (graded_rows, failures)。"""
    recs = load_run(path)
    if limit:
        recs = recs[:limit]
    rows: list[dict] = []

    def build_row(r: dict) -> dict:
        cat = (r.get("metadata") or {}).get("category")
        lenient = bool((r.get("metrics") or {}).get("judge_correct"))
        det = strict_check(cat, r.get("expected_answer") or "", r.get("predicted_answer") or "")
        return {
            "case_id": r["case_id"],
            "category": cat,
            "lenient": lenient,
            "det": det,
            "hedges": find_hedges(r.get("predicted_answer") or ""),
            "question": r["question"],
            "gold": r.get("expected_answer") or "",
            "pred": (r.get("predicted_answer") or "")[:400],
            "judge": None,
            "final": "wrong" if not lenient else None,
        }, r

    prepared = []
    for r in recs:
        row, raw = build_row(r)
        rows.append(row)
        if judge is not None and (judge_all or row["lenient"]):
            prepared.append(row)

    failures: list[dict] = []
    if judge is None or not prepared:
        return rows, failures

    def work(row: dict) -> dict:
        assert limiter is not None
        limiter.acquire()
        try:
            for _ in range(4):
                try:
                    out = judge.grade(row["category"], row["question"], row["gold"], row["pred"])
                    limiter.maybe_increase()
                    return out
                except PrecisionJudgeError as e:
                    if is_rate_limit_error(e):
                        limiter.on_rate_limited()
                        time.sleep(4)
                    else:
                        time.sleep(2)
            return {"grade": "unknown", "missing": [], "reason": "exhausted retries", "cached": False}
        finally:
            limiter.release()

    done = 0
    with ThreadPoolExecutor(max_workers=limiter._ceiling if limiter else 1) as pool:  # noqa: SLF001
        futs = {pool.submit(work, row): row for row in prepared}
        for fut in as_completed(futs):
            row = futs[fut]
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                out = {"grade": "unknown", "missing": [], "reason": str(e)[:120], "cached": False}
                failures.append({"case_id": row["case_id"], "error": str(e)[:160]})
            row["judge"] = {k: out.get(k) for k in ("grade", "missing", "reason")}
            if row["lenient"] or judge_all:
                row["final"] = out.get("grade", "unknown")
            done += 1
            if done % 200 == 0:
                judge.flush()
                print(f"  [{label}] strict-judged {done}/{len(prepared)} (conc={limiter.limit})", flush=True)
    judge.flush()
    return rows, failures


def summarize(rows: list[dict], label: str) -> dict:
    summary: dict = {"label": label, "n": len(rows), "per_category": {}, "adversarial": {}}
    for cat in ALL_CATS:
        sub = [r for r in rows if r["category"] == cat]
        if not sub:
            continue
        c = Counter(r["final"] for r in sub)
        n = len(sub)
        entry = {
            "n": n,
            "lenient_correct": sum(r["lenient"] for r in sub),
            "exact": c["exact"],
            "partial": c["partial"],
            "wrong": c["wrong"],
            "unknown": c["unknown"] + sum(1 for r in sub if r["final"] is None),
            "lenient_rate": round(sum(r["lenient"] for r in sub) / n, 4),
            "precise_rate": round(c["exact"] / n, 4),
        }
        if cat == "adversarial":
            entry["interpretation"] = {
                "exact": "干净拒答（无编造）", "partial": "拒答但附带猜测", "wrong": "编造答案"
            }
            summary["adversarial"] = entry
        else:
            summary["per_category"][cat] = entry
    j = [r for r in rows if r["category"] in J_CATS]
    if j:
        cj = Counter(r["final"] for r in j)
        lc = sum(r["lenient"] for r in j)
        summary["J_score"] = {
            "n": len(j),
            "lenient_official": round(lc / len(j), 4),
            "precise_exact_only": round(cj["exact"] / len(j), 4),
            "exact_plus_partial": round((cj["exact"] + cj["partial"]) / len(j), 4),
            "vague_share_of_lenient_correct": round(cj["partial"] / lc, 4) if lc else None,
            "counts": dict(cj),
        }
    # 模糊表达统计（所有已作答题）
    answered = [r for r in rows if (r["pred"] or "").strip()]
    summary["hedge_stats"] = {
        "answered": len(answered),
        "with_hedge_words": sum(1 for r in answered if r["hedges"]),
        "hedge_word_counter": dict(Counter(h for r in answered for h in r["hedges"]).most_common(10)),
    }
    # 确定性检查与 LLM 严格判分的一致率（可用子集）
    both = [
        r for r in rows
        if r["det"]["passed"] is not None and r["judge"] and r["judge"].get("grade") in {"exact", "partial", "wrong"}
    ]
    if both:
        agree = sum(
            1 for r in both
            if (r["det"]["passed"] and r["judge"]["grade"] == "exact")
            or (not r["det"]["passed"] and r["judge"]["grade"] in {"partial", "wrong"})
        )
        summary["det_vs_judge"] = {"n": len(both), "agree": agree, "agreement": round(agree / len(both), 4)}
    return summary


def paired(base_rows: list[dict], graph_rows: list[dict]) -> dict:
    b = {r["case_id"]: r for r in base_rows}
    g = {r["case_id"]: r for r in graph_rows}
    common = sorted(set(b) & set(g))

    def block(cats, name):
        ids = [i for i in common if b[i]["category"] in cats]
        trans = Counter((b[i]["final"], g[i]["final"]) for i in ids)
        # exact 翻转（McNemar，exact vs 非_exact）
        bb = sum(1 for i in ids if b[i]["final"] == "exact" and g[i]["final"] != "exact")
        gg = sum(1 for i in ids if b[i]["final"] != "exact" and g[i]["final"] == "exact")
        n2 = bb + gg
        p = 1.0
        if n2:
            tail = sum(math.comb(n2, k) for k in range(0, min(bb, gg) + 1)) / 2**n2
            p = min(1.0, 2 * tail)
        ex_b = sum(1 for i in ids if b[i]["final"] == "exact")
        ex_g = sum(1 for i in ids if g[i]["final"] == "exact")
        return {
            "name": name,
            "n": len(ids),
            "exact_base": ex_b,
            "exact_graph": ex_g,
            "base_only_exact": bb,
            "graph_only_exact": gg,
            "mcnemar_p": round(p, 4),
            "transitions": {f"{k[0]}->{k[1]}": v for k, v in sorted(trans.items()) if v >= 3},
        }

    return {
        "J_categories": block(set(J_CATS), "J (cats 1-4) exact 翻转"),
        "adversarial": block({"adversarial"}, "adversarial 干净拒答（exact=干净拒答）翻转"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path 形式")
    ap.add_argument("--pair", nargs="+", metavar="LABEL",
                    help="配对对比的 run label；需恰好 2 个且都已加载才执行配对分析（单 run 传 1 个仅命名，跳过配对）")
    ap.add_argument("--judge-model", default=DEFAULT_PRECISION_MODEL)
    ap.add_argument("--no-judge", action="store_true", help="只跑确定性阶段")
    ap.add_argument("--judge-all", action="store_true",
                    help="全量严格模式：不用宽松结果预筛，所有题都过严格判分（含宽判错的题）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-conc", type=int, default=None,
                    help="默认按模型自动：glm-5.3 非 flash=2，flash=8")
    ap.add_argument("--ceiling", type=int, default=None)
    args = ap.parse_args()

    get_settings()  # 触发 .env 加载
    runs = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        runs[label] = path

    # 兼容单 run + --pair 只给 1 个 label 的调用：视其为该 run 的命名
    # （使输出 precision_regrade_<label>.jsonl 与调用方期望一致，也避免覆盖既有同名产物；
    #   配对分析仍需 2 个 label，此处不做配对）
    if args.pair and len(args.pair) == 1 and len(runs) == 1 and args.pair[0] not in runs:
        old = next(iter(runs))
        runs[args.pair[0]] = runs.pop(old)
        print(f"（--pair={args.pair[0]}：单 run，将其 label 从 {old!r} 命名为 {args.pair[0]!r}；配对分析需 2 个 label，跳过）")

    # 判题并发：glm-5.3（非 flash）质量档限流严，最多 2 并发；flash 可高
    is_flash = "flash" in args.judge_model
    start_conc = args.start_conc if args.start_conc is not None else (8 if is_flash else 2)
    ceiling = args.ceiling if args.ceiling is not None else (12 if is_flash else 2)

    judge = None
    limiter = None
    if not args.no_judge:
        judge = PrecisionJudge(model=args.judge_model)
        limiter = AdaptiveLimiter(start=start_conc, ceiling=ceiling)
        print(f"precision judge model={args.judge_model} conc={start_conc}->{ceiling}, "
              f"cached={len(judge._cache)}")  # noqa: SLF001

    ts = time.strftime("%Y%m%d_%H%M%S")
    all_rows: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for label, path in runs.items():
        print(f"\n===== regrade [{label}] {path}")
        rows, failures = grade_run(label, path, judge, limiter, args.limit, judge_all=args.judge_all)
        all_rows[label] = rows
        out_path = Path("results") / f"precision_regrade_{label}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize(rows, label)
        if failures:
            s["judge_failures"] = failures[:20]
            print(f"  !! {len(failures)} 题判分失败（记 unknown）")
        summaries[label] = s
        print(json.dumps(s, ensure_ascii=False, indent=2))

    if args.pair:
        if len(args.pair) == 2 and all(l in all_rows for l in args.pair):
            p = paired(all_rows[args.pair[0]], all_rows[args.pair[1]])
            summaries["paired"] = p
            print(f"\n===== paired {args.pair[0]} vs {args.pair[1]}")
            print(json.dumps(p, ensure_ascii=False, indent=2))
        else:
            print(f"\n（--pair={args.pair}：配对分析需恰好 2 个已加载的 label，本次跳过 paired 分析）")

    out = Path("results") / f"precision_summary_{ts}.json"
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
