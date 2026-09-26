"""Bounded cross-session source-turn neighborhood from rare shared terms.

This topology uses opaque weighted links. It does not require relation labels
or entity extraction, and each turn keeps only a small number of neighbors.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict

from .graph_only import GraphOnlyRetriever, _terms
from .provenance import SourceGraph
from .success_paths import budgeted_route


def _turn_terms(text: str) -> set[str]:
    terms = set()
    for term in _terms(text):
        if len(term) > 5 and term.endswith("ing"):
            term = term[:-3]
        elif len(term) > 4 and term.endswith("ed"):
            term = term[:-2]
        terms.add(term)
    return terms


class TurnGraph:
    def __init__(self, sources: SourceGraph, *, max_neighbors: int = 6,
                 min_shared: int = 2, max_term_fraction: float = 0.2):
        self.sources = sources
        self.max_neighbors = max_neighbors
        self.min_shared = min_shared
        self.max_term_fraction = max_term_fraction
        self.docs: dict[str, set[str]] = {}
        self.postings: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.by_user: dict[str, list[str]] = defaultdict(list)
        for raw_id, turn in sources.turns.items():
            tid = f"T:{raw_id}"
            user = turn["user_id"]
            terms = _turn_terms(turn["content"] + " " + str(turn.get("speaker") or ""))
            self.docs[tid] = terms
            self.by_user[user].append(tid)
            for term in terms:
                self.postings[user][term].add(tid)
        self.arcs: dict[str, list[tuple[str, float]]] = {}
        for user, ids in self.by_user.items():
            n = len(ids)
            df = {term: len(posting) for term, posting in self.postings[user].items()}
            idf = {term: math.log1p((n + 1) / (count + 1)) for term, count in df.items()}
            weights = {tid: sum(idf[t] for t in self.docs[tid]) for tid in ids}
            for tid in ids:
                shared: dict[str, set[str]] = defaultdict(set)
                source_session = sources.turns[tid[2:]]["session_id"]
                for term in self.docs[tid]:
                    if df[term] > max(2, n * max_term_fraction):
                        continue
                    for other in self.postings[user][term]:
                        if other != tid and sources.turns[other[2:]]["session_id"] != source_session:
                            shared[other].add(term)
                neighbors = []
                for other, overlap in shared.items():
                    if len(overlap) < min_shared:
                        continue
                    intersection = sum(idf[term] for term in overlap)
                    union = weights[tid] + weights[other] - intersection
                    score = intersection / union if union else 0
                    neighbors.append((other, score))
                neighbors.sort(key=lambda item: (-item[1], item[0]))
                self.arcs[tid] = neighbors[:max_neighbors]

    def retrieve(self, retriever: GraphOnlyRetriever, user: str, question: str,
                 *, slots: int = 2, context_k: int = 15,
                 min_score: float = 0.2, preserve_sources: bool = True,
                 recurrence_only: bool = False) -> dict:
        started = time.perf_counter()
        keyword = retriever.retrieve(user, question, graph_enabled=False,
                                     context_k=context_k)["selected"]
        if recurrence_only and not re.search(r"\b(how many times|how often|number of times)\b",
                                              question.lower()):
            return {"selected": keyword, "keyword": keyword, "added": [],
                    "seed_count": 0, "candidate_count": 0,
                    "local_ms": round((time.perf_counter() - started) * 1000, 4)}
        seeds = []
        for item in keyword[:8]:
            if item.startswith("T:"):
                seeds.append(item)
            else:
                seeds.extend(f"T:{tid}" for tid, _ in self.sources.memory_to_turns.get(item, []))
        seeds = list(dict.fromkeys(seeds))[:8]
        qterms = _turn_terms(question)
        if recurrence_only:
            qterms -= {"many", "time", "taken", "take", "often", "number"}
        scores = {}
        for rank, seed in enumerate(seeds):
            seed_terms = self.docs.get(seed, set())
            for candidate, edge in self.arcs.get(seed, []):
                if candidate in keyword:
                    continue
                candidate_terms = self.docs[candidate]
                lexical = len(qterms & candidate_terms) / len(qterms) if qterms else 0
                new_terms = len((qterms & candidate_terms) - seed_terms) / len(qterms) if qterms else 0
                score = (0.45 * edge + 0.4 * lexical + 0.15 * new_terms) / (1 + 0.08 * rank)
                scores[candidate] = max(score, scores.get(candidate, 0))
        route = [mid for mid, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                 if score >= min_score]
        if preserve_sources:
            selected = keyword.copy()
            additions = []
            for mid in route:
                if len(additions) >= slots or sum(item.startswith("T:") for item in selected) >= 5:
                    break
                if len(selected) >= context_k:
                    # Never evict a directly found source turn merely to follow
                    # an unverified edge. The replacement pays a memory slot.
                    tail_memory = next((i for i in range(len(selected) - 1, -1, -1)
                                        if not selected[i].startswith("T:")), None)
                    if tail_memory is None:
                        break
                    selected.pop(tail_memory)
                selected.insert(min(5 + len(additions), len(selected)), mid)
                additions.append(mid)
        else:
            selected = budgeted_route(keyword, route, slots=slots, context_k=context_k)
            additions = [mid for mid in selected if mid not in keyword]
        return {"selected": selected, "keyword": keyword,
                "added": additions,
                "seed_count": len(seeds), "candidate_count": len(scores),
                "local_ms": round((time.perf_counter() - started) * 1000, 4)}

    def stats(self) -> dict:
        return {"turns": len(self.docs), "directed_arcs": sum(map(len, self.arcs.values())),
                "max_neighbors": self.max_neighbors, "min_shared": self.min_shared}
