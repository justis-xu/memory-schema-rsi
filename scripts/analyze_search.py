#!/usr/bin/env python
"""探索轮次分析：排名 + 漂移校正 + bootstrap 稳定性选择（ESPO 式）。

用法：.venv/bin/python scripts/analyze_search.py [--bootstrap]
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROWS = Path("results/search_rounds.jsonl")
ROUND_DIR = Path("data/search/rounds")


def load_rounds() -> list[dict]:
    return [json.loads(l) for l in open(ROWS)] if ROWS.exists() else []


def load_grades(rid: str) -> dict[str, str]:
    p = ROUND_DIR / f"{rid}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["case_id"]: json.loads(l)["final"] for l in open(p)}


def main() -> None:
    rounds = load_rounds()
    if not rounds:
        print("无轮次结果"); return
    bootstrap = "--bootstrap" in sys.argv
    append_ledger = "--append-ledger" in sys.argv

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _report(rounds, bootstrap)
    text = buf.getvalue()
    print(text)

    if append_ledger:
        import time as _time

        stamp = _time.strftime("%Y-%m-%d %H:%M")
        with open("docs/search_ledger.md", "a", encoding="utf-8") as fh:
            fh.write(f"\n## 自动追加（{stamp}，{len(rounds)} 轮累计）\n\n```\n{text.strip()}\n```\n")
        print(f"→ 已追加到 docs/search_ledger.md")


def _report(rounds: list[dict], bootstrap: bool) -> None:
    calibs = [r for r in rounds if "calib" in r["id"]]
    calib_exact = [r["exact"] for r in calibs]
    noise = (max(calib_exact) - min(calib_exact)) / 2 if len(calib_exact) >= 2 else None
    print(f"校准轮 exact: {calib_exact} → 噪声半径 ≈ ±{noise:.0f}" if noise is not None
          else f"校准轮: {calib_exact}（仅一次，漂移未量化）")

    print(f"\n{'round':<20} {'exact':>5} {'Δref':>5} {'↑':>3} {'↓':>3} {'partial':>7} {'wrong':>5}  graphs/preset")
    for r in sorted(rounds, key=lambda x: -x["exact"]):
        g = ",".join(r["graphs"]) or "无图"
        print(f"{r['id']:<20} {r['exact']:>5} {r['delta_exact']:>+5} {r['flips_up']:>3} {r['flips_down']:>3} "
              f"{r['grades'].get('partial', 0):>7} {r['grades'].get('wrong', 0):>5}  {g} | {r['answer_preset']}")

    # 分题型（各轮 exact 数对比，找谁在哪些题型强）
    cats = ["multi-hop", "temporal", "single-hop", "open-domain"]
    print("\n分题型 exact：")
    print(f"{'round':<20}" + "".join(f"{c:>12}" for c in cats))
    for r in sorted(rounds, key=lambda x: -x["exact"]):
        row = [r["per_category"].get(c, {}).get("exact", 0) for c in cats]
        print(f"{r['id']:<20}" + "".join(f"{v:>12}" for v in row))

    if bootstrap and len(rounds) >= 3:
        # ESPO 式稳定性选择：B=20 有放回重采样，统计各轮在 exact 数上胜出的次数
        grades = {r["id"]: load_grades(r["id"]) for r in rounds}
        ids_by_round = {rid: [c for c, g in gmap.items() if g != "unknown"]
                        for rid, gmap in grades.items()}
        B = 20
        rng = random.Random(11)
        wins = Counter()
        best_point = max(rounds, key=lambda r: r["exact"])["id"]
        for _ in range(B):
            sample = rng.choices(sorted(ids_by_round.get(best_point, [])), k=len(ids_by_round.get(best_point, [])))
            champ, champ_ex = None, -1
            for r in rounds:
                gmap = grades[r["id"]]
                ex = sum(1 for c in sample if gmap.get(c) == "exact")
                if ex > champ_ex:
                    champ, champ_ex = r["id"], ex
            wins[champ] += 1
        print(f"\nbootstrap 稳定性（B={B}，胜出次数）：{dict(wins.most_common())}")


if __name__ == "__main__":
    main()
