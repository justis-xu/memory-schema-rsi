#!/usr/bin/env python3
"""Summarize only paired, successfully judged vector-free replacement cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--expected", type=int, default=24)
    args = ap.parse_args()
    latest = {}
    malformed = 0
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            latest[row["id"]] = row
    paired = [row for row in latest.values()
              if all("correct" in row.get("arms", {}).get(arm, {}) and
                     not str(row["arms"][arm].get("judge_note", "")).startswith("judge error")
                     for arm in ("keyword", "graph"))]
    outcomes = Counter()
    by_category = defaultdict(Counter)
    for row in paired:
        label = f"keyword_{int(row['arms']['keyword']['correct'])}_graph_{int(row['arms']['graph']['correct'])}"
        outcomes[label] += 1
        by_category[row["category"]][label] += 1
    print(json.dumps({"paired": len(paired), "expected": args.expected,
                      "unpaired_or_missing": args.expected - len(paired), "malformed_lines": malformed,
                      "outcomes": outcomes, "by_category": by_category}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
