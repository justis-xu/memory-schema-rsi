"""Lifecycle contracts: prior-only reads, user isolation, source recovery and budget."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema_rsi_lab.lifecycle import LifecycleGraph, LifecyclePolicy, add_session_with_graph  # noqa: E402
from schema_rsi_lab.provenance import SourceGraph  # noqa: E402
from schema_rsi_lab.lifecycle_rsi import LifecycleRSI, split_conversations  # noqa: E402
from schema_rsi_lab.graph_only import GraphOnlyRetriever  # noqa: E402


def _memory(mid, user, text, vector, session="session_1"):
    return {"id": mid, "user_id": user, "content": text,
            "metadata": {"session_id": session}, "embedding": vector}


def test_ingest_reads_only_prior_memory_and_query_keeps_budget():
    graph = LifecycleGraph(LifecyclePolicy(write_min_cosine=0, hub_bits=1, hub_tables=1,
                                            query_seed_k=1, query_graph_slots=1,
                                            min_query_overlap=0))
    assert graph.before_extract("u", [{"role": "user", "content": "camping"}])["ids"] == []
    graph.upsert(_memory("a", "u", "family camping trip", [1, 0]))
    pre = graph.before_extract("u", [{"role": "user", "content": "family camping"}])
    assert pre["ids"] == ["a"]
    graph.upsert(_memory("b", "u", "family camping marshmallows", [0.9, 0.1]))
    graph.upsert(_memory("c", "u", "unrelated pottery", [0.7, 0.3]))
    result = graph.for_question("u", "family camping marshmallows", ["a", "c"], context_k=2)
    assert len(result["selected"]) == len(result["baseline"]) == 2
    assert result["selected"] == ["a", "b"]
    assert result["edge_visits"] <= graph.policy.max_visits


def test_cross_user_never_links_or_returns():
    graph = LifecycleGraph(LifecyclePolicy(hub_bits=1, hub_tables=1))
    graph.upsert(_memory("a", "u1", "shared theme", [1, 0]))
    graph.upsert(_memory("b", "u2", "shared theme", [1, 0]))
    assert graph.expand("u1", ["a", "b"], max_results=10)[0] == []
    assert graph.for_question("u1", "shared theme", ["b", "a"], context_k=2)["selected"] == ["a"]
    with pytest.raises(ValueError, match="cross-user"):
        graph._connect("a", "b", "r9", 0.9)


def test_source_turn_retains_caption_missing_from_fact():
    graph = LifecycleGraph()
    graph.upsert(_memory("m", "u", "Poetry reading was inspiring", [1, 0], "session_17"))
    sources = SourceGraph(graph, [{"session_id": "session_17", "date": "2023-10-06",
                                   "turns": [{"dia_id": "D17:19", "speaker": "Caroline",
                                              "content": "At the poetry reading I saw a sign: Trans Lives Matter"}]}])
    result = sources.for_question("u", "What did the signs at the poetry reading say?", ["m"])
    assert result["source_turns"][0]["id"] == "T:u|D17:19"
    assert "Trans Lives Matter" in result["source_turns"][0]["content"]


def test_source_index_does_not_cross_users_with_same_dia_id():
    graph = LifecycleGraph()
    graph.upsert(_memory("a", "u1", "poetry reading", [1, 0]))
    graph.upsert(_memory("b", "u2", "poetry reading", [1, 0]))
    sources = SourceGraph(graph, [])
    sources.append_session({"session_id": "session_1", "date": "2023-01-01",
                            "turns": [{"dia_id": "D1:1", "content": "secret blue poster"}]}, user_id="u1")
    sources.append_session({"session_id": "session_1", "date": "2023-01-01",
                            "turns": [{"dia_id": "D1:1", "content": "public red poster"}]}, user_id="u2")
    result = sources.for_question("u2", "What did the red poster say?", ["b"])
    assert [t["id"] for t in result["source_turns"]] == ["T:u2|D1:1"]


def test_mem0_hook_passes_prior_context_and_commits_source_after_fact():
    class Collection:
        def get(self, ids, include):
            import numpy as np
            return {"ids": ids, "embeddings": np.asarray([[1.0, 0.0]], dtype="float32")}

    class Raw:
        vector_store = type("Store", (), {"collection": Collection()})()

        def add(self, messages, *, user_id, metadata, prompt):
            assert user_id == "u"
            assert metadata["session_id"] == "session_1"
            assert "NEW MESSAGES only" in prompt
            return {"results": [{"id": "new", "memory": "poster says red", "event": "ADD"}]}

    backend = type("Backend", (), {"raw": Raw()})()
    graph = LifecycleGraph()
    sources = SourceGraph(graph, [])
    session = {"session_id": "session_1", "date": "2023-01-01",
               "turns": [{"dia_id": "D1:1", "content": "The poster says red"}]}
    records, pre = add_session_with_graph(
        backend, graph, "u", [{"role": "user", "content": "The poster says red"}],
        {"session_id": "session_1"}, source_graph=sources, source_session=session,
    )
    assert pre["ids"] == []
    assert records[0].metadata["session_id"] == "session_1"
    assert graph.memories["new"]["metadata"]["session_id"] == "session_1"
    assert sources.for_question("u", "What did the poster say?", ["new"])["source_turns"]


def test_lifecycle_rsi_splits_by_conversation_and_gates_policy(tmp_path):
    cases = [{"id": f"q{i}", "conversation_id": f"c{i}", "baseline_correct": False}
             for i in range(5)]
    splits = split_conversations(cases)
    assert {c["conversation_id"] for c in splits["train"]}.isdisjoint(
        {c["conversation_id"] for c in splits["validation"] + splits["test"]}
    )

    def evaluate(policy, requested):
        return [{"id": c["id"], "baseline_correct": c["baseline_correct"],
                 "correct": bool(policy.use_source and policy.verify_enabled),
                 "model_calls": 2 if policy.verify_enabled else 0,
                 "edge_visits": 10 if policy.query_graph_slots else 0}
                for c in requested]

    report = LifecycleRSI(cases, tmp_path / "run", evaluate).run()
    assert report["active_policy"]["use_source"] is True
    assert report["test"]["fixed"] == 1
    assert (tmp_path / "run" / "active_policy.json").exists()
    with pytest.raises(ValueError, match="five independent conversations"):
        split_conversations(cases[:4])


def test_vector_free_keyword_and_graph_arms_share_context_budget():
    graph = LifecycleGraph(LifecyclePolicy(write_min_cosine=0, hub_bits=1, hub_tables=1))
    graph.upsert(_memory("a", "u", "Melanie family hiking", [1, 0]))
    graph.upsert(_memory("b", "u", "Family camping marshmallows", [0.9, 0.1]))
    graph.upsert(_memory("c", "u", "Painted ceramic bowl", [-1, 0]))
    sources = SourceGraph(graph, [])
    retriever = GraphOnlyRetriever(graph, sources)
    plain = retriever.retrieve("u", "What did Melanie do while hiking with family?",
                               graph_enabled=False, context_k=2)
    expanded = retriever.retrieve("u", "What did Melanie do while hiking with family?",
                                  graph_enabled=True, context_k=2)
    assert len(plain["selected"]) <= 2
    assert len(expanded["selected"]) <= 2
    assert all(graph.owners[mid] == "u" for mid in expanded["selected"])
