"""Promotion gate for an end-to-end graph-memory policy.

The evaluator is injected because accuracy requires real answer and judge calls.
Conversation-level splits prevent a policy from learning on one QA about a
conversation and claiming an independent test on another QA from the same one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from .lifecycle import LifecyclePolicy


def split_conversations(cases: list[dict]) -> dict[str, list[dict]]:
    conversations = sorted({str(c["conversation_id"]) for c in cases})
    if len(conversations) < 5:
        raise ValueError("need at least five independent conversations for train/validation/test")
    n = len(conversations)
    train_end = max(1, round(n * 0.6))
    valid_end = max(train_end + 1, round(n * 0.8))
    owners = {conv: ("train" if i < train_end else "validation" if i < valid_end else "test")
              for i, conv in enumerate(conversations)}
    result = {name: [case for case in cases if owners[str(case["conversation_id"])] == name]
              for name in ("train", "validation", "test")}
    if not all(result.values()):
        raise ValueError("all splits need cases")
    return result


def _summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty evaluation")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation case")
    fixed = sum(row["correct"] is True and row["baseline_correct"] is False for row in rows)
    harmed = sum(row["correct"] is False and row["baseline_correct"] is True for row in rows)
    return {"cases": len(rows), "fixed": fixed, "harmed": harmed, "net": fixed - harmed,
            "accuracy": sum(row["correct"] is True for row in rows) / len(rows),
            "mean_model_calls": sum(int(row.get("model_calls", 0)) for row in rows) / len(rows),
            "mean_edge_visits": sum(int(row.get("edge_visits", 0)) for row in rows) / len(rows)}


class LifecycleRSI:
    def __init__(self, cases: list[dict], output_dir: str | Path,
                 evaluator: Callable[[LifecyclePolicy, list[dict]], list[dict]]):
        self.splits = split_conversations(cases)
        self.output_dir = Path(output_dir)
        self.evaluator = evaluator
        self.active = LifecyclePolicy(verify_enabled=False, use_source=False, query_graph_slots=0)

    def _write(self, path: str, content: dict) -> None:
        target = self.output_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _append(self, path: str, content: dict) -> None:
        target = self.output_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(content, ensure_ascii=False, sort_keys=True) + "\n")

    def _evaluate(self, policy: LifecyclePolicy, split: str) -> tuple[dict, list[dict]]:
        requested = self.splits[split]
        rows = self.evaluator(policy, requested)
        if {str(row["id"]) for row in rows} != {str(case["id"]) for case in requested}:
            raise ValueError("evaluator did not return exactly the requested cases")
        expected_baseline = {str(c["id"]): c["baseline_correct"] for c in requested}
        if any(row["baseline_correct"] is not expected_baseline[str(row["id"])] for row in rows):
            raise ValueError("baseline verdict changed across paired trial")
        return _summary(rows), rows

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._write("manifest.json", {name: sorted({c["conversation_id"] for c in rows})
                                      for name, rows in self.splits.items()})
        proposals = [
            ("verify_only", replace(self.active, verify_enabled=True)),
            ("memory_graph", replace(self.active, verify_enabled=True, query_graph_slots=2)),
            ("memory_plus_source", replace(self.active, verify_enabled=True,
                                           query_graph_slots=2, use_source=True)),
            ("sparse_source", replace(self.active, verify_enabled=True,
                                      query_graph_slots=1, use_source=True,
                                      write_knn=3, max_visits=32, source_turn_slots=1)),
            ("verified_success_paths", replace(self.active, verify_enabled=True,
                                               query_graph_slots=1, use_source=True,
                                               use_success_paths=True,
                                               success_path_threshold=0.5,
                                               success_path_slots=2)),
        ]
        trained = []
        for name, policy in proposals:
            report, rows = self._evaluate(policy, "train")
            self._append("raw/trials.jsonl", {"proposal": name, "policy": asdict(policy),
                                             "train": report, "train_rows": rows})
            trained.append((report["net"], -report["mean_model_calls"], name, policy))
        trained.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
        impact = []
        for _, _, name, policy in trained[:2]:
            report, _ = self._evaluate(policy, "validation")
            accepted = report["fixed"] >= 1 and report["harmed"] == 0 and report["mean_model_calls"] <= 3
            event = {"proposal": name, "validation": report, "accepted": accepted,
                     "reason": "validation_gain_without_regression" if accepted else "validation_gate_rejected"}
            impact.append(event)
            self._append("wiki/impact.jsonl", event)
            if accepted:
                self.active = policy
                break
        self._write("active_policy.json", asdict(self.active))
        test, _ = self._evaluate(self.active, "test")
        result = {"active_policy": asdict(self.active), "test": test,
                  "validation_events": impact,
                  "split_conversations": {k: len({c["conversation_id"] for c in v})
                                          for k, v in self.splits.items()}}
        self._write("report.json", result)
        return result
