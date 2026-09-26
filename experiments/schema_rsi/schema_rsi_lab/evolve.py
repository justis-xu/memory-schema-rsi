"""Broad search, failure-directed repair, independent gating, and freezing."""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from .graph import CompiledGraph, compile_graph, evaluate_split
from .model import Channel, Dataset, Schema


@dataclass(frozen=True)
class Limits:
    min_validation_gain: float = 0.01
    max_mean_visits: float = 32.0
    max_arcs_per_memory: float = 12.0
    visit_penalty: float = 0.002
    arc_penalty: float = 0.001


class EvolutionEngine:
    """Deterministic proposer/curriculum with an objective validation gate.

    This is an algorithmic RSI prototype. It does not claim to reproduce the
    LLM agent systems in OaK, RSIAgent, or WikiSkill.
    """

    def __init__(
        self, data: Dataset, output_dir: str | Path, *, context_k: int = 4,
        deep_rounds: int = 3, limits: Limits | None = None,
    ) -> None:
        if context_k < 2 or deep_rounds < 0:
            raise ValueError("context_k must be >=2 and deep_rounds must be >=0")
        self.data = data
        self.output_dir = Path(output_dir)
        self.context_k = context_k
        self.seed_k = max(1, context_k // 2)
        self.deep_rounds = deep_rounds
        self.limits = limits or Limits()
        self._graphs: dict[str, CompiledGraph] = {}
        self._reports: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
        self._seen: set[str] = set()
        self._patterns: Counter[str] = Counter()
        self._missing_channels: Counter[str] = Counter()
        self._impact: list[dict] = []
        self.active = Schema(
            channels=(), seed_k=context_k, context_k=context_k, graph_slots=0,
        )

    def _candidate(self, channels: tuple[Channel, ...], **changes) -> Schema:
        args = {
            "channels": tuple(sorted(channels, key=lambda c: c.id)),
            "seed_k": self.seed_k,
            "context_k": self.context_k,
            "graph_slots": min(2, self.context_k - self.seed_k),
            "max_hops": 2,
            "max_visits": 32,
            "hop_discount": 0.8,
        }
        if self.active.channels:
            args.update(self.active.to_dict())
            args["channels"] = tuple(sorted(channels, key=lambda c: c.id))
        args.update(changes)
        return Schema(**args)

    def _graph(self, schema: Schema) -> CompiledGraph:
        if schema.id not in self._graphs:
            self._graphs[schema.id] = compile_graph(self.data, schema)
        return self._graphs[schema.id]

    def _evaluate(self, schema: Schema, split: str) -> tuple[dict, list[dict]]:
        key = (schema.id, split)
        if key not in self._reports:
            self._reports[key] = evaluate_split(self.data, schema, split, self._graph(schema))
        return self._reports[key]

    def _utility(self, report: dict) -> float:
        return (
            report["delta"]
            - self.limits.visit_penalty * report["mean_edge_visits"]
            - self.limits.arc_penalty * report["arcs"] / self.data.memory_count
        )

    def _broad(self) -> list[tuple[str, Schema]]:
        proposals: list[tuple[str, Schema]] = []
        slots = sorted({1, min(2, self.context_k - self.seed_k)})
        for channel in self.data.channels:
            for degree in (1, 3):
                for hops in (1, 2):
                    for graph_slots in slots:
                        schema = self._candidate(
                            (Channel(channel, max_degree=degree),),
                            graph_slots=graph_slots, max_hops=hops,
                        )
                        proposals.append((f"broad:{channel}:d{degree}:h{hops}:s{graph_slots}", schema))
        return proposals

    def _shortest_path_channels(self, source_ids: tuple[str, ...], target: str) -> tuple[str, ...] | None:
        """Diagnostic only; reads the fixed candidate pool, never query gold to build edges."""
        neighbors: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in self.data.edges:
            neighbors[edge.source].append((edge.target, edge.channel))
            neighbors[edge.target].append((edge.source, edge.channel))
        frontier = deque((source, ()) for source in source_ids)
        visited = set(source_ids)
        while frontier:
            node_id, channels = frontier.popleft()
            if len(channels) >= 2:
                continue
            for next_id, channel in neighbors.get(node_id, []):
                if next_id in visited:
                    continue
                path = (*channels, channel)
                if next_id == target:
                    return path
                visited.add(next_id)
                frontier.append((next_id, path))
        return None

    def _diagnose(self) -> dict:
        _, rows = self._evaluate(self.active, "train")
        case_by_id = {case.id: case for case in self.data.cases}
        round_patterns: Counter[str] = Counter()
        round_channels: Counter[str] = Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        enabled = {channel.id for channel in self.active.channels}
        for row in rows:
            case = case_by_id[row["case_id"]]
            missed = set(case.gold).difference(row["selected"])
            for gold_id in sorted(missed):
                if gold_id in row["baseline"]:
                    reason = "displaced_baseline_evidence"
                else:
                    path = self._shortest_path_channels(
                        case.base_ranked[:self.seed_k], gold_id,
                    )
                    if path is None:
                        reason = "candidate_pool_gap"
                    elif set(path).difference(enabled):
                        reason = "channel_missing"
                        round_channels.update(set(path).difference(enabled))
                    elif len(path) > self.active.max_hops:
                        reason = "hop_limit"
                    else:
                        reason = "budget_or_rank"
                round_patterns[reason] += 1
                if len(examples[reason]) < 5:
                    examples[reason].append(case.id)
        self._patterns.update(round_patterns)
        self._missing_channels.update(round_channels)
        diagnosis = {
            "round_patterns": dict(round_patterns),
            "missing_channels": dict(round_channels),
            "examples": dict(examples),
            "cumulative_patterns": dict(self._patterns),
            "cumulative_missing_channels": dict(self._missing_channels),
        }
        self._write_json("wiki/patterns.json", diagnosis)
        return diagnosis

    def _deep(self, diagnosis: dict) -> list[tuple[str, Schema]]:
        proposals: list[tuple[str, Schema]] = []
        enabled = {channel.id for channel in self.active.channels}
        for channel in sorted(self.data.channels, key=lambda c: -diagnosis["missing_channels"].get(c, 0)):
            if channel not in enabled:
                channels = (*self.active.channels, Channel(channel, max_degree=3))
                proposals.append((f"deep:add:{channel}", self._candidate(channels, max_hops=2)))
        for channel in self.active.channels:
            more_degree = replace(channel, max_degree=min(12, channel.max_degree + 2))
            channels = tuple(more_degree if c.id == channel.id else c for c in self.active.channels)
            proposals.append((f"deep:degree:{channel.id}", self._candidate(channels)))
            if channel.min_score > 0:
                lower = replace(channel, min_score=max(0.0, channel.min_score - 0.2))
                channels = tuple(lower if c.id == channel.id else c for c in self.active.channels)
                proposals.append((f"deep:threshold:{channel.id}", self._candidate(channels)))
            if channel.weight < 1.5:
                heavier = replace(channel, weight=round(channel.weight + 0.25, 2))
                channels = tuple(heavier if c.id == channel.id else c for c in self.active.channels)
                proposals.append((f"deep:weight:{channel.id}", self._candidate(channels)))
        if self.active.channels:
            if self.active.max_hops == 1:
                proposals.append(("deep:hops", replace(self.active, max_hops=2)))
            if self.active.graph_slots < self.context_k - self.active.seed_k:
                proposals.append(("deep:slots", replace(self.active, graph_slots=self.active.graph_slots + 1)))
            if self.active.max_visits < 64:
                proposals.append(("deep:visits", replace(self.active, max_visits=64)))
        return proposals

    def _append_jsonl(self, relative: str, value: dict) -> None:
        path = self.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    def _write_json(self, relative: str, value: dict) -> None:
        path = self.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _search_round(self, stage: str, proposals: list[tuple[str, Schema]]) -> bool:
        ranked: list[tuple[float, str, Schema]] = []
        for name, schema in proposals:
            if schema.id in self._seen or schema.id == self.active.id:
                continue
            self._seen.add(schema.id)
            train, rows = self._evaluate(schema, "train")
            self._append_jsonl("raw/trials.jsonl", {
                "stage": stage, "proposal": name, "schema_id": schema.id,
                "schema": schema.to_dict(), "train": train, "train_cases": rows,
            })
            ranked.append((self._utility(train), name, schema))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2].id))
        improved = False
        # Keep validation as a gate rather than a free search objective.
        for _, name, schema in ranked[:8]:
            train, _ = self._evaluate(schema, "train")
            validation, _ = self._evaluate(schema, "validation")
            active_train, _ = self._evaluate(self.active, "train")
            active_validation, _ = self._evaluate(self.active, "validation")
            density = validation["arcs"] / self.data.memory_count
            reason = "accepted"
            if validation["delta"] < self.limits.min_validation_gain:
                reason = "no_validation_gain"
            elif validation["mean_edge_visits"] > self.limits.max_mean_visits:
                reason = "visit_limit"
            elif density > self.limits.max_arcs_per_memory:
                reason = "arc_limit"
            elif self._utility(train) + 1e-9 < self._utility(active_train):
                reason = "train_regression"
            elif self._utility(validation) <= self._utility(active_validation) + 1e-9:
                reason = "validation_regression_or_tie"
            accepted = reason == "accepted"
            event = {
                "stage": stage, "proposal": name, "schema_id": schema.id,
                "train": train, "validation": validation,
                "accepted": accepted, "reason": reason,
                "previous_schema_id": self.active.id,
            }
            self._append_jsonl("wiki/impact.jsonl", event)
            self._impact.append(event)
            if accepted:
                self.active = schema
                self._write_json("active_schema.json", schema.to_dict())
                improved = True
        return improved

    def run(self) -> dict:
        """Create a fresh run directory; never mutate the repository's live evaluator."""
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._write_json("input.json", {
            "dataset_sha256": self.data.sha256,
            "node_count": len(self.data.nodes),
            "candidate_edge_count": len(self.data.edges),
            "case_count": len(self.data.cases),
            "context_k": self.context_k,
            "limits": self.limits.__dict__,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        self._write_json("active_schema.json", self.active.to_dict())
        self._search_round("broad", self._broad())
        diagnosis = self._diagnose()
        for index in range(self.deep_rounds):
            if not diagnosis["round_patterns"]:
                break
            if not self._search_round(f"deep:{index + 1}", self._deep(diagnosis)):
                break
            diagnosis = self._diagnose()

        # The only test-set read happens after the active schema is frozen.
        frozen = self.active
        train, _ = self._evaluate(frozen, "train")
        validation, _ = self._evaluate(frozen, "validation")
        test, test_cases = self._evaluate(frozen, "test")
        report = {
            "status": "frozen",
            "scope": "offline evidence-ID retrieval proxy; no end-to-end QA claim",
            "dataset_sha256": self.data.sha256,
            "schema_id": frozen.id,
            "schema": frozen.to_dict(),
            "train": train,
            "validation": validation,
            "test": test,
            "test_cases": test_cases,
            "accepted_changes": sum(event["accepted"] for event in self._impact),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_json("report.json", report)
        return report
