"""Vector-free keyword seeds and graph-neighborhood answer contexts.

This is a replacement ablation: neither arm reads Chroma, reranker output, nor
the baseline memory ranking. The keyword arm uses an inverted index over fact
and source nodes; the graph arm adds bounded neighborhood expansion.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict

from .lifecycle import LifecycleGraph, _STOP
from .provenance import SourceGraph

_WORDS = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> set[str]:
    terms = set()
    for word in _WORDS.findall(text.lower()):
        if len(word) < 3 or word in _STOP:
            continue
        # Conservative plural normalization, kept identical for query/index.
        terms.add(word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word)
    return terms


class GraphOnlyRetriever:
    def __init__(self, graph: LifecycleGraph, sources: SourceGraph):
        self.graph = graph
        self.sources = sources
        self.docs: dict[str, dict] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        for mid, memory in graph.memories.items():
            self._add(mid, memory["user_id"], "memory", memory["content"])
        for tid, turn in sources.turns.items():
            self._add(f"T:{tid}", turn["user_id"], "turn", turn["content"])
        self.user_sizes = Counter(doc["user"] for doc in self.docs.values())
        self.turn_to_memories: dict[str, list[str]] = defaultdict(list)
        for mid, links in sources.memory_to_turns.items():
            for tid, _ in links:
                self.turn_to_memories[f"T:{tid}"].append(mid)

    def _add(self, doc_id: str, user: str, kind: str, text: str) -> None:
        terms = _terms(text)
        self.docs[doc_id] = {"user": user, "kind": kind, "text": text, "terms": terms}
        for term in terms:
            self.postings[term].add(doc_id)

    def keyword(self, user: str, question: str) -> list[tuple[str, float]]:
        q = _terms(question)
        scores: dict[str, float] = defaultdict(float)
        n = self.user_sizes[user]
        for term in q:
            posting = [mid for mid in self.postings.get(term, ()) if self.docs[mid]["user"] == user]
            if not posting:
                continue
            idf = math.log1p((n - len(posting) + 0.5) / (len(posting) + 0.5))
            for mid in posting:
                length_penalty = 1 / math.sqrt(max(1, len(self.docs[mid]["terms"])))
                scores[mid] += idf * length_penalty
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))

    @staticmethod
    def _budgeted(ids: list[str], k: int, max_turns: int = 5) -> list[str]:
        selected = []
        turns = 0
        for mid in ids:
            if mid.startswith("T:"):
                if turns >= max_turns:
                    continue
                turns += 1
            selected.append(mid)
            if len(selected) >= k:
                break
        return selected

    def retrieve(self, user: str, question: str, *, graph_enabled: bool, context_k: int = 15) -> dict:
        start = time.perf_counter()
        ranked = self.keyword(user, question)
        keyword_ids = self._budgeted([mid for mid, _ in ranked], context_k)
        if not graph_enabled or not keyword_ids:
            return {"selected": keyword_ids, "keyword": keyword_ids,
                    "graph_added": [], "edge_visits": 0,
                    "local_ms": round((time.perf_counter() - start) * 1000, 3)}

        # Seed from both directly matched memories and source turns' linked
        # memories. No vector result is consulted.
        seeds = []
        for mid in keyword_ids[:8]:
            related = self.turn_to_memories.get(mid, []) if mid.startswith("T:") else [mid]
            for rid in related:
                if rid not in seeds and self.graph.owners.get(rid) == user:
                    seeds.append(rid)
        seeds = seeds[:5]
        neighbors, visits = self.graph.expand(user, seeds, max_results=30)
        score_by_id = dict(ranked)
        max_keyword = ranked[0][1] or 1.0
        candidates = []
        for mid, path_score in neighbors:
            if mid in keyword_ids:
                continue
            lexical = score_by_id.get(mid, 0.0) / max_keyword
            if lexical < 0.12 and path_score < 0.55:
                continue
            candidates.append((mid, 0.55 * lexical + 0.45 * path_score))
        source = self.sources.for_question(user, question, seeds, max_turns=5)
        for turn in source["source_turns"]:
            mid = turn["id"]
            if mid in keyword_ids:
                continue
            lexical = score_by_id.get(mid, 0.0) / max_keyword
            candidates.append((mid, 0.65 * lexical + 0.35 * turn["path"]))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        additions = []
        for mid, score in candidates:
            if mid in additions:
                continue
            additions.append(mid)
            if len(additions) >= 3:
                break
        selected = self._budgeted(keyword_ids[:5] + additions + keyword_ids[5:], context_k)
        return {"selected": selected, "keyword": keyword_ids,
                "graph_added": [mid for mid in additions if mid in selected],
                "edge_visits": visits,
                "local_ms": round((time.perf_counter() - start) * 1000, 3)}

    def content(self, doc_id: str) -> str:
        if doc_id.startswith("T:"):
            turn = self.sources.turns[doc_id[2:]]
            return f"Source conversation on {turn['session_date']}: {turn['content']}"
        return self.docs[doc_id]["text"]
