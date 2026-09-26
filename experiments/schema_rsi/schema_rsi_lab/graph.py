"""Compile sparse candidate edges and retrieve bounded graph neighborhoods."""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass

from .model import Case, Dataset, Schema


@dataclass(frozen=True)
class Arc:
    target: str
    channel: str
    score: float


@dataclass(frozen=True)
class CompiledGraph:
    adjacency: dict[str, tuple[Arc, ...]]
    arc_count: int


def compile_graph(data: Dataset, schema: Schema) -> CompiledGraph:
    """Top-degree selection is local to each node and opaque relation channel.

    Candidate edges are supplied once by an ANN/grouping stage. Each trial
    compiles only that sparse pool; no all-pairs similarity computation occurs.
    """
    settings = {channel.id: channel for channel in schema.channels}
    choices: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for edge in data.edges:
        spec = settings.get(edge.channel)
        if spec is None or edge.score < spec.min_score:
            continue
        for source, target in ((edge.source, edge.target), (edge.target, edge.source)):
            key = (source, edge.channel)
            choices[key][target] = max(choices[key].get(target, 0.0), edge.score)

    adjacency: dict[str, list[Arc]] = defaultdict(list)
    for (source, channel), targets in choices.items():
        spec = settings[channel]
        top = sorted(targets.items(), key=lambda pair: (-pair[1], pair[0]))[:spec.max_degree]
        adjacency[source].extend(Arc(target, channel, score * spec.weight) for target, score in top)
    frozen = {
        source: tuple(sorted(arcs, key=lambda arc: (-arc.score, arc.target, arc.channel)))
        for source, arcs in adjacency.items()
    }
    return CompiledGraph(frozen, sum(map(len, frozen.values())))


@dataclass(frozen=True)
class RetrievalTrace:
    selected: tuple[str, ...]
    graph_added: tuple[str, ...]
    edge_visits: int
    expanded_nodes: int


def retrieve(data: Dataset, graph: CompiledGraph, schema: Schema, case: Case) -> RetrievalTrace:
    """Best-first traversal; only memory nodes enter answer context.

    The baseline and graph arm receive the same context_k. Graph additions
    must come from outside baseline top-context, so the gain cannot be caused
    merely by providing more memories to the answerer.
    """
    baseline = case.base_ranked[:schema.context_k]
    if not schema.channels or schema.graph_slots == 0:
        return RetrievalTrace(baseline, (), 0, 0)

    seeds = case.base_ranked[:schema.seed_k]
    baseline_set = set(baseline)
    heap: list[tuple[float, int, str]] = []
    best: dict[tuple[str, int], float] = {}
    for rank, node_id in enumerate(seeds):
        score = 1.0 / (rank + 1)
        best[(node_id, 0)] = score
        heapq.heappush(heap, (-score, 0, node_id))

    added: list[str] = []
    visits = 0
    expanded = 0
    while heap and visits < schema.max_visits and len(added) < schema.graph_slots:
        neg_score, depth, node_id = heapq.heappop(heap)
        score = -neg_score
        if score < best.get((node_id, depth), 0.0):
            continue
        node = data.nodes[node_id]
        if node.kind == "M" and node_id not in baseline_set and node_id not in added:
            added.append(node_id)
            if len(added) >= schema.graph_slots:
                break
        if depth >= schema.max_hops:
            continue
        expanded += 1
        for arc in graph.adjacency.get(node_id, ()):
            if visits >= schema.max_visits:
                break
            visits += 1
            target = data.nodes[arc.target]
            if target.user_id != case.user_id:
                raise ValueError("compiled graph crossed user boundary")
            next_score = score * arc.score * (schema.hop_discount if depth else 1.0)
            key = (target.id, depth + 1)
            if next_score <= 0 or next_score <= best.get(key, 0.0):
                continue
            best[key] = next_score
            heapq.heappush(heap, (-next_score, depth + 1, target.id))

    selected = list(seeds)
    selected.extend(added)
    for node_id in baseline:
        if len(selected) >= schema.context_k:
            break
        if node_id not in selected:
            selected.append(node_id)
    return RetrievalTrace(tuple(selected), tuple(added), visits, expanded)


def evaluate_split(
    data: Dataset, schema: Schema, split: str, graph: CompiledGraph | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate evidence-ID recall and deterministic work; no model API calls."""
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    graph = graph or compile_graph(data, schema)
    rows: list[dict] = []
    elapsed_ms = 0.0
    for case in data.cases:
        if case.split != split:
            continue
        start = time.perf_counter()
        trace = retrieve(data, graph, schema, case)
        elapsed_ms += (time.perf_counter() - start) * 1000
        gold = set(case.gold)
        baseline = case.base_ranked[:schema.context_k]
        rows.append({
            "case_id": case.id,
            "selected": list(trace.selected),
            "baseline": list(baseline),
            "gold": list(case.gold),
            "recall": len(gold.intersection(trace.selected)) / len(gold),
            "baseline_recall": len(gold.intersection(baseline)) / len(gold),
            "graph_added": list(trace.graph_added),
            "edge_visits": trace.edge_visits,
            "expanded_nodes": trace.expanded_nodes,
        })
    if not rows:
        raise ValueError(f"no cases for split {split}")
    n = len(rows)
    report = {
        "split": split,
        "cases": n,
        "recall": sum(r["recall"] for r in rows) / n,
        "baseline_recall": sum(r["baseline_recall"] for r in rows) / n,
        "delta": sum(r["recall"] - r["baseline_recall"] for r in rows) / n,
        "mean_edge_visits": sum(r["edge_visits"] for r in rows) / n,
        "max_edge_visits": max(r["edge_visits"] for r in rows),
        "mean_graph_added": sum(len(r["graph_added"]) for r in rows) / n,
        "arcs": graph.arc_count,
        "mean_traversal_ms": elapsed_ms / n,
    }
    return report, rows
