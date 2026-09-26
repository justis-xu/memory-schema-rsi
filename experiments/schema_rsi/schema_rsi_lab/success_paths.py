"""Opaque query-to-evidence bundles learned from verified answer episodes.

Each episode is an experience node connected to a set of memory/source nodes. It is
only admitted with explicit attestation from the caller. The graph itself does
not depend on human-readable edge or relation names.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .graph_only import _terms
from .provenance import SourceGraph


def operation(question: str) -> str:
    q = question.lower().strip()
    for pattern, name in ((r"\bhow (many|much)\b", "count"),
                          (r"\bhow long\b", "duration"),
                          (r"\b(when|what year|what date)\b", "time"),
                          (r"\b(where|which place|which city)\b", "place"),
                          (r"\b(who|whose)\b", "person"),
                          (r"\bwhy\b", "cause")):
        if re.search(pattern, q):
            return name
    return "other"


def _similarity(left: set[str], right: set[str], idf: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    intersection = sum(idf.get(term, 0) for term in left & right)
    union = sum(idf.get(term, 0) for term in left | right)
    return intersection / union if union else 0.0


def budgeted_route(keyword: list[str], path: list[str], *, slots: int, context_k: int = 15) -> list[str]:
    """Give path turns up to `slots` while preserving the same context budget."""
    selected = keyword[:5]
    for tid in path:
        if tid not in selected and sum(x.startswith("T:") for x in selected) < 5:
            selected.append(tid)
        if len(selected) >= 5 + slots:
            break
    for item in keyword[5:]:
        if item in selected or item.startswith("T:") and sum(x.startswith("T:") for x in selected) >= 5:
            continue
        selected.append(item)
        if len(selected) >= context_k:
            break
    return selected


class SuccessPathGraph:
    def __init__(self, sources: SourceGraph):
        self.sources = sources
        self.episodes: dict[str, dict] = {}
        self.by_user: dict[str, set[str]] = defaultdict(set)
        self.df: dict[str, Counter] = defaultdict(Counter)
        self.postings: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def record(self, *, user: str, episode_id: str, question: str,
               support_ids: list[str], attested: bool,
               attestation_source: str) -> str:
        """Commit only an independently attested question/support bundle."""
        if not attested:
            raise ValueError("unverified support cannot become a success path")
        if not attestation_source:
            raise ValueError("attestation source is required")
        key = f"{user}|{episode_id}"
        if key in self.episodes:
            raise ValueError("duplicate experience episode")
        valid = []
        for tid in support_ids:
            if tid.startswith("T:"):
                owner = self.sources.turns.get(tid[2:], {}).get("user_id")
            else:
                owner = self.sources.graph.owners.get(tid) if tid in self.sources.graph.memories else None
            if owner is None:
                raise ValueError("success path needs existing evidence nodes")
            if owner != user:
                raise ValueError("cross-user success path")
            if tid not in valid:
                valid.append(tid)
        if not valid:
            raise ValueError("success path needs supporting evidence")
        terms = _terms(question)
        if not terms:
            raise ValueError("success path needs a searchable question")
        node = "E:" + hashlib.sha256(f"{user}|{episode_id}".encode()).hexdigest()[:24]
        self.episodes[key] = {"node": node, "user": user, "episode_id": episode_id,
                              "question": question, "terms": terms,
                              "operation": operation(question), "support_ids": valid,
                              "attestation_source": attestation_source}
        self.by_user[user].add(key)
        self.df[user].update(terms)
        for term in terms:
            self.postings[user][term].add(key)
        return node

    def query(self, *, user: str, question: str, threshold: float = 0.5,
              same_operation: bool = False, exclude_episode: str | None = None,
              max_peers: int = 3) -> dict:
        ids = self.by_user.get(user, set())
        qterms = _terms(question)
        peers = []
        candidates = set().union(*(self.postings[user].get(term, set()) for term in qterms)) if qterms else set()
        relevant_terms = set(qterms)
        for key in candidates:
            relevant_terms.update(self.episodes[key]["terms"])
        # Unseen query terms still carry weight in the denominator. Ignoring
        # them would overstate similarity to a vaguely related old question.
        idf = {term: math.log1p((len(ids) + 1) / (self.df[user].get(term, 0) + 1))
               for term in relevant_terms}
        for key in candidates:
            if key == f"{user}|{exclude_episode}":
                continue
            episode = self.episodes[key]
            if same_operation and episode["operation"] != operation(question):
                continue
            score = _similarity(qterms, episode["terms"], idf)
            if score >= threshold:
                peers.append((score, key))
        peers.sort(key=lambda pair: (-pair[0], pair[1]))
        weights: dict[str, float] = defaultdict(float)
        for score, key in peers[:max_peers]:
            for tid in self.episodes[key]["support_ids"]:
                weights[tid] += score
        route = [tid for tid, _ in sorted(weights.items(), key=lambda pair: (-pair[1], pair[0]))]
        return {"support_ids": route,
                "peer_ids": [self.episodes[key]["episode_id"] for _, key in peers[:max_peers]],
                "max_similarity": peers[0][0] if peers else 0}

    def save(self, path: str | Path) -> None:
        """Atomically persist verified episodes; source and fact nodes stay elsewhere."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"user": item["user"], "episode_id": item["episode_id"],
                 "question": item["question"], "support_ids": item["support_ids"],
                 "attestation_source": item["attestation_source"]}
                for _, item in sorted(self.episodes.items())]
        payload = {"schema_version": 1, "episodes": rows}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                             prefix=f".{target.name}.", suffix=".tmp",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path, sources: SourceGraph) -> SuccessPathGraph:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("episodes"), list):
            raise ValueError("unsupported success-path snapshot")
        result = cls(sources)
        for row in payload["episodes"]:
            result.record(user=row["user"], episode_id=row["episode_id"],
                          question=row["question"], support_ids=row["support_ids"],
                          attested=True, attestation_source=row["attestation_source"])
        return result


def record_verified_episode(paths: SuccessPathGraph, *, user: str, episode_id: str,
                            question: str, verdict: dict, feedback_correct: bool,
                            feedback_source: str) -> str | None:
    """Commit support selected by the verifier only after independent feedback.

    `feedback_correct` must come from a separate outcome signal. A verifier's
    own supported verdict alone does not establish that the answer was right.
    """
    support = verdict.get("support_ids") or []
    if not feedback_correct or verdict.get("status") != "supported" or not support:
        return None
    if not feedback_source:
        raise ValueError("independent feedback source is required")
    return paths.record(user=user, episode_id=episode_id, question=question,
                        support_ids=list(support), attested=True,
                        attestation_source=f"verifier+{feedback_source}")
