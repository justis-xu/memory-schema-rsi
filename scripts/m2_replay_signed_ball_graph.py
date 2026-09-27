"""Replay current conv-43 cached S2 graph candidate generation without services."""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m2_audit_zh_graph_cache_alignment import current_memories, signature  # noqa: E402
from schema_rsi.evaluation.retriever import GraphRetriever  # noqa: E402
from schema_rsi.graph.builder import GraphBuilder  # noqa: E402
from schema_rsi.graph.variants import _kmeans, hops_for  # noqa: E402
from schema_rsi.memory.base import MemoryRecord  # noqa: E402


CASE = "locomo_conv-43_qa25"
USER = "zhfull:locomo:conv-43"
JOHN = "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"
TIM = "219b4c97-f2af-4734-a4f4-1d93ebe19f99"
OUT = ROOT / "results/analysis/m2_signed_ball_graph_replay_20260927.json"
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"


class LocalGraph:
    """Only the builder/retriever methods needed here; preserves insertion order."""

    def __init__(self):
        self._key_props = {}
        self.vertices = {}
        self.edges = []
        self._edge_keys = set()

    def reset_graph(self, *, drop_schema=True):
        self.vertices.clear()
        self.edges.clear()
        self._edge_keys.clear()
        if drop_schema:
            self._key_props.clear()

    def create_schema(self, schema):
        for vertex in schema.vertex_labels:
            self._key_props[vertex.name] = vertex.primary_key

    def upsert_vertex(self, label, key, properties=None):
        self.vertices[(label, key)] = dict(properties or {})
        return f"{label}:{key}"

    def upsert_edge(self, label, out_label, out_key, in_label, in_key, properties=None):
        out, target = (out_label, out_key), (in_label, in_key)
        assert out in self.vertices and target in self.vertices, (label, out, target)
        edge_key = (label, out, target)
        if edge_key not in self._edge_keys:
            self.edges.append(edge_key)
            self._edge_keys.add(edge_key)

    def get_neighbors(self, label, key, direction="BOTH", edge_labels=None, limit=50):
        center = (label, key)
        if center not in self.vertices:
            return []
        # HugeGraphStore requests limit*2 raw edges before label filtering.
        raw = [edge for edge in self.edges if
               (direction in ("OUT", "BOTH") and edge[1] == center) or
               (direction in ("IN", "BOTH") and edge[2] == center)][:limit * 2]
        found = {}
        for edge_label, out, target in raw:
            if edge_labels and edge_label not in edge_labels:
                continue
            other = target if out == center else out
            if other == center or other in found:
                continue
            if len(found) >= limit:
                break
            found[other] = {"label": other[0], "id": f"{other[0]}:{other[1]}",
                            "properties": self.vertices[other], "via_edge": edge_label}
        return list(found.values())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    db_before = sha(DB)
    all_memories = current_memories()
    subset = {mid: row for mid, row in all_memories.items() if row["user_id"] == USER}
    assert len(subset) == 211
    structured_path = ROOT / "data/graph_zh/structured_cache.json"
    topic_path = ROOT / "data/graph_zh/topic_cache.json"
    embedding_path = ROOT / "data/graph_zh/memory_embeddings.json"
    structured = json.loads(structured_path.read_text())
    topics = json.loads(topic_path.read_text())
    embeddings = json.loads(embedding_path.read_text())
    prior = json.loads((ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json").read_text())
    case = next(c for c in prior["cases"] if c["case_id"] == CASE)
    seeds = [MemoryRecord(id=row["id"], content=row["content"], metadata={"user_id": USER})
             for row in case["reranked"][:5]]
    exclude = {row["id"] for row in case["reranked"]}
    assert TIM in {rec.id for rec in seeds} and JOHN not in exclude
    records = {mid: MemoryRecord(id=mid, content=row["content"],
                                 metadata={"user_id": USER, "session_id": row["session_id"]})
               for mid, row in subset.items()}
    assert set(records) <= set(embeddings), "missing V8 embeddings; no model calls allowed"

    def one_run(order, with_topics, max_candidates=30, with_cluster=False):
        store = LocalGraph()
        builder = GraphBuilder(store)
        recs = [records[mid] for mid in order]
        builder.build_s1(recs, structured, reset=True)
        if with_topics:
            builder.build_concepts(recs, topics)
        if with_cluster:
            import numpy as np
            from schema_rsi.graph.schema import EdgeLabelSpec, GraphSchema, PropertySpec, VertexLabelSpec

            store.create_schema(GraphSchema(
                name="v8cluster", version="v8",
                vertex_labels=[VertexLabelSpec(name="V8Cluster", primary_key="cluster_key",
                                               properties=[PropertySpec("cluster_key", "TEXT")])],
                edge_labels=[EdgeLabelSpec(name="V8_IN_CLUSTER", source_labels=["Memory"],
                                           target_labels=["V8Cluster"])],
            ))
            k = max(4, min(12, len(recs) // 25))
            assignment, k = _kmeans(np.array([embeddings[rec.id] for rec in recs]), k)
            for cluster in range(k):
                key = f"{len(USER)}:{USER}:cl{cluster}"
                store.upsert_vertex("V8Cluster", key, {"cluster_key": key,
                                                       "name": f"cluster{cluster}", "user_id": USER})
            for rec, cluster in zip(recs, assignment):
                key = f"{len(USER)}:{USER}:cl{int(cluster)}"
                store.upsert_edge("V8_IN_CLUSTER", "Memory", rec.id, "V8Cluster", key)
        found = GraphRetriever(store).retrieve_fused(
            seeds, exclude_ids=exclude, per_node_limit=15, max_candidates=max_candidates,
            hops=hops_for(["s2", "v8cluster"]) if with_cluster else None)
        rank = next((i for i, row in enumerate(found, 1) if row["id"] == JOHN), None)
        return store, found, rank

    ids = sorted(records)
    store, candidates, baseline_rank = one_run(ids, True)
    _, s1_candidates, s1_rank = one_run(ids, False)
    _, s2_all, s2_full_rank = one_run(ids, True, max_candidates=500)
    _, s1_all, s1_full_rank = one_run(ids, False, max_candidates=500)
    _, s2v8_candidates, s2v8_rank = one_run(ids, True, with_cluster=True)
    _, s2v8_all, s2v8_full_rank = one_run(ids, True, max_candidates=500, with_cluster=True)
    ranks = []
    full_ranks = []
    for seed in range(50):
        shuffled = ids[:]
        random.Random(seed).shuffle(shuffled)
        ranks.append(one_run(shuffled, True)[2])
        full_ranks.append(one_run(shuffled, True, max_candidates=500)[2])
    reverse_rank = one_run(list(reversed(ids)), True)[2]
    reverse_v8_rank = one_run(list(reversed(ids)), True, with_cluster=True)[2]
    v8_random_ranks = []
    for seed in range(20):
        shuffled = ids[:]
        random.Random(seed).shuffle(shuffled)
        v8_random_ranks.append(one_run(shuffled, True, with_cluster=True)[2])
    signed_key = f"{len(USER)}:{USER}:signed basketball"
    linked = store.get_neighbors("Entity", signed_key, direction="IN",
                                 edge_labels=["MENTIONS"], limit=15)
    result = {
        "scope": "Current conv-43 Chroma memories and existing extraction caches; isolated in-process GraphBuilder S1/S2 + GraphRetriever; no HugeGraph, reranker, answerer or model calls",
        "case_id": CASE, "question": case["question"], "user_id": USER,
        "current_chroma_logic_sha256": signature(all_memories),
        "current_chroma_file_sha256_before": db_before,
        "current_chroma_file_sha256_after": sha(DB),
        "conv43_count": len(records), "structured_cache_count": len(set(records) & set(structured)),
        "topic_cache_count": len(set(records) & set(topics)),
        "structured_cache_sha256": sha(structured_path), "topic_cache_sha256": sha(topic_path),
        "embedding_cache_count": len(set(records) & set(embeddings)),
        "embedding_cache_sha256": sha(embedding_path),
        "prior_vector_db_sha256_before": prior.get("vector_db_sha256_before"),
        "seed_ids": [r.id for r in seeds], "excluded_vector_ids": sorted(exclude),
        "signed_ball_entity_neighbor_ids": [r["properties"]["memory_id"] for r in linked],
        "graph_counts": {"vertices": len(store.vertices), "edges": len(store.edges),
                         "labels": dict(Counter(label for label, _ in store.vertices))},
        "s2_sorted_order": {"john_rank": baseline_rank, "candidate_count": len(candidates),
                            "candidates": candidates},
        "s1_sorted_order": {"john_rank": s1_rank, "candidate_count": len(s1_candidates),
                            "uncapped_john_rank": s1_full_rank, "uncapped_candidate_count": len(s1_all)},
        "s2_sorted_order_uncapped_john_rank": s2_full_rank,
        "s2_sorted_order_uncapped_candidate_count": len(s2_all),
        "s2_reverse_order_john_rank": reverse_rank,
        "s2v8_sorted_order": {"john_rank": s2v8_rank, "candidate_count": len(s2v8_candidates),
                               "uncapped_john_rank": s2v8_full_rank,
                               "uncapped_candidate_count": len(s2v8_all)},
        "s2v8_reverse_order_john_rank": reverse_v8_rank,
        "s2v8_random_order_john_ranks": v8_random_ranks,
        "s2v8_random_order_john_present_count": sum(rank is not None for rank in v8_random_ranks),
        "s2_random_order_john_ranks": ranks,
        "s2_random_order_uncapped_john_ranks": full_ranks,
        "s2_random_order_john_present_count": sum(rank is not None for rank in ranks),
        "limits": ["Cache entries have no source content hash and missing IDs were not newly extracted.",
                   "LocalGraph preserves insertion order, while HugeGraph edge enumeration order is not verified.",
                   "S2+V8 replay uses cached embeddings and the production k-means helper but does not run the final fused reranker or answerer.",
                   "The frozen vector seed list came from a prior current-library probe; this is not a same-run live retrieval."],
    }
    assert result["current_chroma_file_sha256_before"] == result["current_chroma_file_sha256_after"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("s2_sorted_order", "s2_random_order_john_ranks", "s2_random_order_uncapped_john_ranks",
                       "s2v8_random_order_john_ranks", "excluded_vector_ids")},
                     ensure_ascii=False, indent=2))
    print("s2_sorted_order_john_rank", baseline_rank,
          "random_rank_counts", Counter(str(rank) for rank in ranks))


if __name__ == "__main__":
    main()
