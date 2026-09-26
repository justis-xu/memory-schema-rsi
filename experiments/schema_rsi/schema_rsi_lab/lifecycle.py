"""A graph policy spanning extraction context, graph writes, and QA retrieval.

Edges are operational channels: r0 vector neighbors, r1 latent hash buckets,
r2 nearby facts from one ingest session. No human ontology is assumed. The
local vector scan is guarded; a larger deployment must supply an ANN index.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

_WORDS = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the and for with from what when where who how did does was were are have has had this that their they them she her his him you your about into after before which why".split())


def _tokens(value: str) -> set[str]:
    return {w for w in _WORDS.findall(value.lower()) if len(w) > 2 and w not in _STOP}


def _overlap(query: str, document: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(document)) / len(q)


@dataclass(frozen=True)
class LifecyclePolicy:
    write_knn: int = 5
    write_min_cosine: float = 0.4
    hub_bits: int = 5
    hub_tables: int = 2
    max_hub_members: int = 24
    ingest_seed_k: int = 3
    ingest_graph_k: int = 3
    query_seed_k: int = 5
    query_graph_slots: int = 2
    max_hops: int = 2
    max_visits: int = 80
    min_query_overlap: float = 0.12
    min_path_score: float = 0.05
    source_session_seeds: int = 3
    source_query_terms: int = 3
    source_turn_slots: int = 2
    verify_enabled: bool = True
    use_source: bool = True
    use_success_paths: bool = False
    success_path_threshold: float = 0.5
    success_path_slots: int = 2
    local_scan_limit: int = 2000

    def __post_init__(self) -> None:
        if not 1 <= self.write_knn <= 128 or not 0 <= self.write_min_cosine <= 1:
            raise ValueError("invalid write policy")
        if not 1 <= self.hub_bits <= 16 or not 0 <= self.hub_tables <= 8:
            raise ValueError("invalid hub policy")
        if self.max_hub_members < 2 or self.local_scan_limit < 2:
            raise ValueError("invalid graph size limits")
        if min(self.ingest_seed_k, self.query_seed_k, self.max_hops, self.max_visits) < 1:
            raise ValueError("invalid traversal policy")
        if self.max_hops > 3:
            raise ValueError("max_hops must be <= 3")
        if not 0 <= self.min_query_overlap <= 1 or not 0 <= self.min_path_score <= 1:
            raise ValueError("invalid query gates")
        if min(self.source_session_seeds, self.source_query_terms) < 1 or self.source_turn_slots < 0:
            raise ValueError("invalid source retrieval budget")
        if not 0 <= self.success_path_threshold <= 1 or not 0 <= self.success_path_slots <= 5:
            raise ValueError("invalid successful-path policy")


class LifecycleGraph:
    def __init__(self, policy: LifecyclePolicy | None = None):
        self.policy = policy or LifecyclePolicy()
        self.memories: dict[str, dict] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.owners: dict[str, str] = {}
        self.arcs: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
        self._hub_members: dict[str, set[str]] = defaultdict(set)
        self._planes: dict[tuple[str, int, int], np.ndarray] = {}
        self.edge_writes = 0

    def _connect(self, left: str, right: str, channel: str, score: float) -> None:
        if left == right or self.owners[left] != self.owners[right]:
            raise ValueError("self/cross-user edge")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid edge score")
        # An arc key carries its channel; two channels may connect one pair.
        for source, target in ((left, right), (right, left)):
            key = f"{channel}:{target}"
            old = self.arcs[source].get(key)
            if old is None or score > old[1]:
                self.arcs[source][key] = (channel, score)
        self.edge_writes += 1

    def _planes_for(self, user: str, table: int, dimension: int) -> np.ndarray:
        key = (user, table, dimension)
        if key not in self._planes:
            seed = int.from_bytes(hashlib.sha256(f"{user}:{table}:latent-v1".encode()).digest()[:8], "big")
            self._planes[key] = np.random.default_rng(seed).standard_normal(
                (dimension, self.policy.hub_bits), dtype=np.float32
            )
        return self._planes[key]

    def upsert(self, record: dict) -> None:
        mid = str(record["id"])
        user = str(record["user_id"])
        if not mid or not user or not record.get("content"):
            raise ValueError("memory needs id, user and content")
        vector = np.asarray(record["embedding"], dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("invalid vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("zero vector")
        vector /= norm
        if mid in self.memories:
            self.delete(mid)
        peers = [other for other in self.memories if self.owners[other] == user]
        if len(peers) >= self.policy.local_scan_limit:
            raise ValueError("local scan limit reached; use an ANN backed writer")
        self.memories[mid] = record
        self.vectors[mid] = vector
        self.owners[mid] = user

        if peers:
            matrix = np.stack([self.vectors[other] for other in peers])
            similarity = matrix @ vector
            order = sorted(range(len(peers)), key=lambda i: (-float(similarity[i]), peers[i]))
            for i in order[: self.policy.write_knn]:
                cosine = min(1.0, max(-1.0, float(similarity[i])))
                if cosine >= self.policy.write_min_cosine:
                    self._connect(mid, peers[i], "r0", (cosine + 1) / 2)
            session = record.get("metadata", {}).get("session_id")
            same_session = [p for p in peers if self.memories[p].get("metadata", {}).get("session_id") == session]
            if session and same_session:
                for other in same_session[-2:]:
                    self._connect(mid, other, "r2", 0.7)

        for table in range(self.policy.hub_tables):
            signature = "".join("1" if bit else "0" for bit in vector @ self._planes_for(user, table, len(vector)) >= 0)
            name = f"H:{hashlib.sha256(f'{user}:{table}:{signature}'.encode()).hexdigest()[:20]}"
            members = self._hub_members[name]
            if len(members) >= self.policy.max_hub_members:
                continue
            self.owners[name] = user
            members.add(mid)
            self._connect(mid, name, "r1", 0.75)

    def delete(self, mid: str) -> None:
        if mid not in self.memories:
            return
        del self.memories[mid], self.vectors[mid], self.owners[mid]
        self.arcs.pop(mid, None)
        for source in list(self.arcs):
            for key in list(self.arcs[source]):
                if key.endswith(f":{mid}"):
                    del self.arcs[source][key]
        for hub in list(self._hub_members):
            self._hub_members[hub].discard(mid)
            if not self._hub_members[hub]:
                del self._hub_members[hub]
                self.owners.pop(hub, None)
                self.arcs.pop(hub, None)

    def expand(self, user: str, seeds: list[str], *, max_results: int) -> tuple[list[tuple[str, float]], int]:
        queue: list[tuple[float, int, str]] = []
        best: dict[str, float] = {}
        for rank, mid in enumerate(seeds):
            if self.owners.get(mid) != user:
                continue
            score = 1 / (rank + 1)
            heapq.heappush(queue, (-score, 0, mid))
            best[mid] = score
        candidates: dict[str, float] = {}
        visits = 0
        while queue and visits < self.policy.max_visits:
            neg, depth, source = heapq.heappop(queue)
            score = -neg
            if score + 1e-9 < best.get(source, 0) or depth >= self.policy.max_hops:
                continue
            for key, (channel, weight) in sorted(self.arcs.get(source, {}).items()):
                if visits >= self.policy.max_visits:
                    break
                visits += 1
                target = key[len(channel) + 1:]
                if self.owners.get(target) != user:
                    raise ValueError("cross-user graph traversal")
                next_score = score * weight * (0.72 if depth else 1.0)
                if target in self.memories and target not in seeds:
                    candidates[target] = max(candidates.get(target, 0), next_score)
                if depth + 1 < self.policy.max_hops and next_score > best.get(target, 0):
                    best[target] = next_score
                    heapq.heappush(queue, (-next_score, depth + 1, target))
        return sorted(candidates.items(), key=lambda pair: (-pair[1], pair[0]))[:max_results], visits

    def before_extract(self, user: str, messages: list[dict]) -> dict:
        """Use the new raw conversation to read the *prior* graph, before writing facts."""
        query = " ".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")
        lexical = sorted(
            ((mid, _overlap(query, r["content"])) for mid, r in self.memories.items() if self.owners[mid] == user),
            key=lambda pair: (-pair[1], pair[0]),
        )
        seeds = [mid for mid, score in lexical[: self.policy.ingest_seed_k] if score > 0]
        neighbors, visits = self.expand(user, seeds, max_results=self.policy.ingest_graph_k * 4)
        selected = seeds + [mid for mid, _ in neighbors if mid not in seeds][: self.policy.ingest_graph_k]
        return {"ids": selected, "memories": [self.memories[mid]["content"] for mid in selected],
                "graph_added": max(0, len(selected) - len(seeds)), "edge_visits": visits}

    def for_question(self, user: str, question: str, ranked: list[str], *, context_k: int = 15) -> dict:
        """Candidate graph evidence competes for the same answer-context budget."""
        baseline = [mid for mid in ranked[:context_k] if self.owners.get(mid) == user]
        seeds = baseline[: self.policy.query_seed_k]
        neighbors, visits = self.expand(user, seeds, max_results=40)
        candidates = []
        for mid, path_score in neighbors:
            if mid in baseline or path_score < self.policy.min_path_score:
                continue
            lexical = _overlap(question, self.memories[mid]["content"])
            if lexical < self.policy.min_query_overlap:
                continue
            candidates.append((mid, 0.65 * lexical + 0.35 * path_score, lexical, path_score))
        candidates.sort(key=lambda row: (-row[1], row[0]))
        selected = baseline.copy()
        added = []
        slots = min(self.policy.query_graph_slots, max(0, context_k - len(seeds)))
        for mid, score, lexical, path_score in candidates[:slots]:
            if len(selected) >= context_k:
                selected.pop()
            selected.insert(min(len(seeds) + len(added), len(selected)), mid)
            added.append({"id": mid, "score": round(score, 4), "lexical": round(lexical, 4),
                          "path": round(path_score, 4)})
        return {"selected": selected, "baseline": baseline, "graph_added": added,
                "edge_visits": visits, "candidate_count": len(candidates)}

    def stats(self) -> dict:
        return {"memory_nodes": len(self.memories), "latent_hubs": len(self._hub_members),
                "directed_arcs": sum(map(len, self.arcs.values())), "edge_writes": self.edge_writes,
                "policy": asdict(self.policy)}


def add_session_with_graph(backend, graph: LifecycleGraph, user_id: str,
                           messages: list[dict], metadata: dict | None = None,
                           *, source_graph=None, source_session: dict | None = None) -> tuple[list, dict]:
    """Operational Mem0 hook: read graph, extract with context, then commit graph.

    Use a separate Mem0 collection for experiments. The reference facts are
    explicitly marked as prior context so they cannot become new memories by
    themselves. Graph writes happen only after Mem0 returned durable IDs.
    """
    from schema_rsi.memory.mem0_backend import _as_items, _to_record

    pre = graph.before_extract(user_id, messages)
    prior = "\n".join(f"- {fact}" for fact in pre["memories"])
    instructions = (
        "Extract facts from NEW MESSAGES only. PRIOR GRAPH CONTEXT below is for "
        "identity and contradiction resolution; do not save it as a new fact "
        "unless the new messages themselves support the fact.\n"
        f"PRIOR GRAPH CONTEXT:\n{prior or '(none)'}"
    )
    meta = {k: v for k, v in (metadata or {}).items() if k not in ("user_id", "agent_id", "run_id")}
    result = backend.raw.add(messages, user_id=user_id, metadata=meta, prompt=instructions)
    items = _as_items(result)
    records = [_to_record(item, default_user_id=user_id) for item in items]
    for rec in records:
        for key, value in meta.items():
            rec.metadata.setdefault(key, value)
        rec.metadata["event"] = next((item.get("event") for item in items if str(item.get("id")) == rec.id), None)
    if source_graph is not None and source_session is not None:
        # Preserve raw turns (including image captions) even if Mem0 extracts
        # no fact. They can later be reached by the source inverted index.
        source_graph.append_session(source_session, user_id=user_id)
    if records:
        vector_rows = backend.raw.vector_store.collection.get(ids=[r.id for r in records], include=["embeddings"])
        vectors = {mid: v.tolist() for mid, v in zip(vector_rows["ids"], vector_rows["embeddings"])}
        for rec in records:
            if rec.id not in vectors:
                raise RuntimeError(f"Mem0 committed {rec.id} but vector cannot be read")
            graph.upsert({"id": rec.id, "user_id": user_id, "content": rec.content,
                          "metadata": rec.metadata, "embedding": vectors[rec.id]})
            if source_graph is not None:
                source_graph.link_memory(rec.id)
    return records, pre
