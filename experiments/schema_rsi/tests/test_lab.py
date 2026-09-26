"""Checks for the experiment's evidence, isolation, and promotion rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schema_rsi_lab import Channel, Dataset, EvolutionEngine, Schema, evaluate_split  # noqa: E402
from schema_rsi_lab.pool import build_pool  # noqa: E402


FIXTURE = ROOT / "fixtures" / "tiny.json"


def test_two_hop_latent_hub_improves_evidence_at_fixed_budget():
    data = Dataset.load(FIXTURE)
    schema = Schema(channels=(Channel("r0"), Channel("r1")), seed_k=2, context_k=4, graph_slots=1)
    report, rows = evaluate_split(data, schema, "test")
    assert report["baseline_recall"] == 0.0
    assert report["recall"] == 1.0
    assert all(len(row["selected"]) == len(row["baseline"]) == 4 for row in rows)
    assert all(row["edge_visits"] <= schema.max_visits for row in rows)
    by_case = {case.id: case for case in data.cases}
    assert all(
        all(data.nodes[node_id].user_id == by_case[row["case_id"]].user_id for node_id in row["selected"])
        for row in rows
    )


def test_cross_user_candidate_edge_is_rejected(tmp_path):
    raw = json.loads(FIXTURE.read_text())
    raw["edges"].append({"source": "a1", "target": "b4", "channel": "r9", "score": 1.0})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="cross-user"):
        Dataset.load(path)


def test_validation_gate_rolls_back_and_test_gold_cannot_steer_search(tmp_path):
    base = json.loads(FIXTURE.read_text())
    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text(json.dumps(base))
    normal = EvolutionEngine(Dataset.load(ordinary), tmp_path / "normal").run()
    assert normal["accepted_changes"] >= 2

    changed_test = json.loads(FIXTURE.read_text())
    for case in changed_test["cases"]:
        if case["split"] == "test":
            case["gold"] = [case["base_ranked"][0]]
    changed_path = tmp_path / "changed-test.json"
    changed_path.write_text(json.dumps(changed_test))
    changed = EvolutionEngine(Dataset.load(changed_path), tmp_path / "changed").run()
    assert changed["schema_id"] == normal["schema_id"]
    assert changed["test"]["delta"] != normal["test"]["delta"]

    no_gain = json.loads(FIXTURE.read_text())
    for case in no_gain["cases"]:
        if case["split"] == "validation":
            case["gold"] = [case["base_ranked"][0]]
    no_gain_path = tmp_path / "no-gain.json"
    no_gain_path.write_text(json.dumps(no_gain))
    rejected = EvolutionEngine(Dataset.load(no_gain_path), tmp_path / "rejected").run()
    assert rejected["accepted_changes"] == 0
    assert rejected["schema"]["channels"] == ()


def test_run_directory_is_never_overwritten(tmp_path):
    engine = EvolutionEngine(Dataset.load(FIXTURE), tmp_path / "run", deep_rounds=0)
    engine.run()
    with pytest.raises(FileExistsError):
        EvolutionEngine(Dataset.load(FIXTURE), tmp_path / "run", deep_rounds=0).run()


def test_pool_builder_creates_opaque_hubs_without_gold_leakage():
    memories = [
        {"id": "a1", "user_id": "a", "embedding": [1.0, 0.0]},
        {"id": "a2", "user_id": "a", "embedding": [1.0, 0.0]},
        {"id": "a3", "user_id": "a", "embedding": [-1.0, 0.0]},
        {"id": "b1", "user_id": "b", "embedding": [1.0, 0.0]},
        {"id": "b2", "user_id": "b", "embedding": [1.0, 0.0]},
    ]
    cases = [
        {"id": "train", "user_id": "a", "split": "train", "base_ranked": ["a1", "a2"], "gold": ["a2"]},
        {"id": "validation", "user_id": "b", "split": "validation", "base_ranked": ["b1", "b2"], "gold": ["b2"]},
        {"id": "test", "user_id": "a", "split": "test", "base_ranked": ["a1", "a2"], "gold": ["a2"]},
    ]
    first = build_pool(memories, cases, knn_k=1, hub_bits=1, hub_tables=1)
    data = Dataset.from_dict(first)
    assert any(node.kind == "H" for node in data.nodes.values())
    assert {edge.channel for edge in data.edges} == {"r0", "r1"}
    assert all(data.nodes[e.source].user_id == data.nodes[e.target].user_id for e in data.edges)

    changed_cases = json.loads(json.dumps(cases))
    changed_cases[0]["gold"] = ["a1"]
    changed = build_pool(memories, changed_cases, knn_k=1, hub_bits=1, hub_tables=1)
    assert changed["nodes"] == first["nodes"]
    assert changed["edges"] == first["edges"]
    with pytest.raises(ValueError, match="precomputed ANN"):
        build_pool(memories, cases, max_local_user_memories=2)
