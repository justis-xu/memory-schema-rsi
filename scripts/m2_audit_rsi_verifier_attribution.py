#!/usr/bin/env python3
"""Audit archived RSI accepted repairs and current support-ID parser contract."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/schema_rsi")]

from schema_rsi.memory.base import MemoryRecord  # noqa: E402
from schema_rsi_lab.verify import verify_answer  # noqa: E402


FILES = {
    "original": ROOT / "experiments/schema_rsi/runs/conv26_verify_repair_final.jsonl",
    "optimized": ROOT / "experiments/schema_rsi/runs/conv26_source_optimized.jsonl",
}
OUT = ROOT / "results/analysis/m2_rsi_verifier_attribution_20260927.json"


class StubClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, **_kwargs):
        return json.dumps(self.payload, ensure_ascii=False), {}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    archived = {}
    for name, path in FILES.items():
        rows = [json.loads(line) for line in path.open()]
        accepted = []
        for row in rows:
            if not row.get("accepted"):
                continue
            verdict = row.get("second_verdict") or {}
            accepted.append({"case_id": row["id"], "verdict_status": verdict.get("status"),
                             "verdict_keys": sorted(verdict),
                             "support_ids_present": bool(row.get("support_ids")),
                             "final_correct_archived": row.get("final_correct")})
        archived[name] = {"path": str(path.relative_to(ROOT)),
                          "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "rows": len(rows), "accepted": accepted}
    rec = MemoryRecord(id="m1", content="梅拉妮最近和孩子们一起画画", metadata={})
    synthetic = []
    for indices in ([], [999], ["bad"], [1]):
        raw = {"status": "supported", "reason": "有证据", "support_indices": indices}
        verdict = verify_answer(StubClient(raw), "梅拉妮喜欢做什么？", "画画", [rec])
        synthetic.append({"support_indices": indices, "raw_status": raw["status"],
                          "parsed_status": verdict.status,
                          "parsed_support_ids": list(verdict.support_ids),
                          "current_accept_predicate": verdict.status == "supported"
                              and bool(verdict.support_ids),
                          "prior_accept_predicate": raw["status"] == "supported"})
    result = {
        "scope": "Historical English conv-26 RSI accepted-repair provenance plus deterministic current parser probe; no model calls or production writes",
        "archived": archived, "synthetic_parser_cases": synthetic,
        "limits": ["Older archived verdicts predate the support_indices contract; absence of IDs is not proof that the answers lacked source support.",
                   "Synthetic malformed verifier replies test control flow, not a measured model error rate.",
                   "A valid support ID only proves a memory was named, not that it genuinely supports the answer or maps to a source turn.",
                   "This experimental verifier is separate from the production Chinese evaluation pipeline."],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print({name: (row["rows"], len(row["accepted"])) for name, row in archived.items()},
          [(row["support_indices"], row["parsed_status"]) for row in synthetic])


if __name__ == "__main__":
    main()
