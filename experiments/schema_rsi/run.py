#!/usr/bin/env python3
"""Run the isolated Schema RSI experiment on a frozen JSON snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from schema_rsi_lab import Dataset, EvolutionEngine


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE / "fixtures" / "tiny.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--deep-rounds", type=int, default=3)
    args = parser.parse_args()

    output = args.output or HERE / "runs" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    data = Dataset.load(args.dataset)
    report = EvolutionEngine(
        data, output, context_k=args.context_k, deep_rounds=args.deep_rounds,
    ).run()
    print(json.dumps({
        "output": str(output),
        "dataset_sha256": report["dataset_sha256"],
        "schema_id": report["schema_id"],
        "accepted_changes": report["accepted_changes"],
        "validation": report["validation"],
        "test": report["test"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
