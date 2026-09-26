#!/usr/bin/env python3
"""Turn a frozen embedding snapshot into sparse candidate edges and latent hubs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from schema_rsi_lab import Dataset
from schema_rsi_lab.pool import build_pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON with memories, cases, and optional ann_edges")
    parser.add_argument("--output", type=Path, required=True, help="new snapshot path; existing files are refused")
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--max-local-user-memories", type=int, default=2000)
    parser.add_argument("--hub-bits", type=int, default=4)
    parser.add_argument("--hub-tables", type=int, default=2)
    parser.add_argument("--hub-cap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    snapshot = build_pool(
        raw["memories"], raw["cases"], ann_edges=raw.get("ann_edges"),
        knn_k=args.knn_k, max_local_user_memories=args.max_local_user_memories,
        hub_bits=args.hub_bits, hub_tables=args.hub_tables,
        hub_cap=args.hub_cap, seed=args.seed,
    )
    data = Dataset.from_dict(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output), "sha256": data.sha256,
        "memories": data.memory_count, "nodes": len(data.nodes),
        "candidate_edges": len(data.edges), "cases": len(data.cases),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
