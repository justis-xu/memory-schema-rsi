"""Conservative, no-gold source-turn candidates for a successful answer.

This is a locator, not an entailment verifier. Its output remains provisional
until a separate verifier checks that the turns actually support the answer.
"""

from __future__ import annotations

import math

from .graph_only import GraphOnlyRetriever, _terms

_UNINFORMATIVE = {"yes", "no", "unknown", "none", "unspecified", "unavailable"}


def mine_supports(retriever: GraphOnlyRetriever, user: str, question: str,
                  answer: str, *, max_turns: int = 2,
                  mode: str = "strict") -> list[dict]:
    if mode not in {"strict", "weighted"}:
        raise ValueError("unknown support miner mode")
    answer_terms = _terms(answer) - _UNINFORMATIVE
    query_terms = _terms(question)
    if not answer_terms or not query_terms or max_turns < 1:
        return []
    postings = retriever.postings
    source_postings = {term: {doc_id for doc_id in postings.get(term, set())
                              if doc_id.startswith("T:") and retriever.docs[doc_id]["user"] == user}
                       for term in answer_terms}
    if len(answer_terms) == 1 and min(map(len, source_postings.values())) > 8:
        return []
    if mode == "weighted":
        # Prefer answer terms that are rare in this user's source corpus.
        # Terms absent from the corpus cannot locate a turn; retaining only
        # present terms makes this an exploratory, lower-trust candidate mode.
        selected_terms = sorted((term for term in answer_terms if source_postings[term]),
                                key=lambda term: (len(source_postings[term]), term))[:6]
        if not selected_terms:
            return []
        idf = {term: math.log1p((retriever.user_sizes[user] + 1) /
                                 (len(source_postings[term]) + 1)) for term in selected_terms}
        denominator = sum(idf.values())
        candidates = set().union(*(source_postings[term] for term in selected_terms))
    else:
        selected_terms = sorted(answer_terms)
        idf = {}
        denominator = 1
        candidates = set().union(*source_postings.values())
    scored = []
    for doc_id in candidates:
        doc = retriever.docs[doc_id]
        answer_coverage = (sum(idf[term] for term in selected_terms if term in doc["terms"]) / denominator
                           if mode == "weighted" else
                           len(answer_terms & doc["terms"]) / len(answer_terms))
        query_coverage = len(query_terms & doc["terms"]) / len(query_terms)
        threshold = 0.45 if mode == "weighted" else 0.75
        if answer_coverage < threshold or query_coverage < 0.15:
            continue
        exact = answer.strip().lower() in doc["text"].lower()
        score = 0.7 * answer_coverage + 0.3 * query_coverage + (0.1 if exact else 0)
        scored.append((score, doc_id, answer_coverage, query_coverage, exact))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"id": doc_id, "score": round(score, 4),
             "answer_coverage": round(answer_coverage, 4),
             "question_coverage": round(query_coverage, 4), "exact_answer": exact}
            for score, doc_id, answer_coverage, query_coverage, exact in scored[:max_turns]]
