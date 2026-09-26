"""Immutable source-turn layer linked to compressed Mem0 facts.

The source is kept verbatim, including supplied image captions and session
dates. Memory-to-turn links are sparse lexical candidates within one session;
query-time traversal can recover details omitted by memory extraction.
"""

from __future__ import annotations

from collections import defaultdict

from .lifecycle import LifecycleGraph, _overlap, _tokens


class SourceGraph:
    def __init__(self, graph: LifecycleGraph, sessions: list[dict]):
        self.graph = graph
        self.turns: dict[str, dict] = {}
        self.by_session: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.term_index: dict[str, set[str]] = defaultdict(set)
        self.memory_to_turns: dict[str, list[tuple[str, float]]] = {}
        for session in sessions:
            self.append_session(session)
        for mid in graph.memories:
            self.link_memory(mid)

    def append_session(self, session: dict, *, user_id: str | None = None) -> None:
        owners = {r["user_id"] for r in self.graph.memories.values()}
        user = user_id or session.get("user_id") or (next(iter(owners)) if len(owners) == 1 else None)
        if not user:
            raise ValueError("source session needs an explicit user_id")
        sid = str(session["session_id"])
        for index, turn in enumerate(session["turns"]):
            # LoCoMo supplies dia_id; LongMemEval-S cleaned has no turn IDs.
            # Its immutable session position is sufficient for local links.
            raw_id = str(turn.get("dia_id") or f"{sid}#{index}")
            tid = f"{user}|{raw_id}"
            if tid in self.turns:
                continue
            self.turns[tid] = {**turn, "user_id": user, "session_id": sid,
                               "session_date": session.get("date")}
            self.by_session[(user, sid)].append(tid)
            for term in _tokens(turn["content"]):
                self.term_index[term].add(tid)

    def link_memory(self, mid: str) -> None:
        memory = self.graph.memories[mid]
        user = memory["user_id"]
        sid = str(memory.get("metadata", {}).get("session_id", ""))
        scores = [(tid, _overlap(memory["content"], self.turns[tid]["content"]))
                  for tid in self.by_session.get((user, sid), [])]
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        self.memory_to_turns[mid] = [(tid, score) for tid, score in scores[:3] if score > 0]

    def for_question(self, user_id: str, question: str, baseline_ids: list[str], *, max_turns: int | None = None) -> dict:
        max_turns = self.graph.policy.source_turn_slots if max_turns is None else max_turns
        anchor = {mid for mid in baseline_ids[:8] if self.graph.owners.get(mid) == user_id}
        # Local graph paths: memory -> linked source turn, or memory -> session
        # hub -> turn. This recovers both exact turn matches and omitted details.
        sessions = {str(self.graph.memories[mid].get("metadata", {}).get("session_id", ""))
                    for mid in baseline_ids[:self.graph.policy.source_session_seeds] if mid in anchor}
        linked = {tid: score for mid in anchor for tid, score in self.memory_to_turns.get(mid, [])}
        candidates = set(linked)
        for sid in sessions:
            candidates.update(self.by_session.get((user_id, sid), []))
        # Raw-source search seeds can recover information lost by extraction,
        # even when its session produced no graph-linked fact. The inverted
        # index bounds work to turns containing a query term.
        rare_terms = sorted(
            (term for term in _tokens(question) if self.term_index.get(term)),
            key=lambda term: (len(self.term_index[term]), term),
        )[:self.graph.policy.source_query_terms]
        for term in rare_terms:
            candidates.update(self.term_index.get(term, ()))
        ranked = []
        for tid in candidates:
            turn = self.turns[tid]
            if turn["user_id"] != user_id:
                continue
            lexical = _overlap(question, turn["content"])
            if lexical < 0.3:
                continue
            path = linked.get(tid, 0.25 if turn["session_id"] in sessions else 0)
            ranked.append((tid, 0.8 * lexical + 0.2 * path, lexical, path))
        ranked.sort(key=lambda row: (-row[1], row[0]))
        selected = []
        for tid, score, lexical, path in ranked[:max_turns]:
            turn = self.turns[tid]
            selected.append({"id": f"T:{tid}", "content": turn["content"],
                             "date": turn["session_date"], "speaker": turn.get("speaker"),
                             "image_urls": turn.get("image_urls", []),
                             "score": round(score, 4), "lexical": round(lexical, 4),
                             "path": round(path, 4)})
        return {"source_turns": selected, "candidate_turns": len(candidates),
                "source_nodes": len(self.turns), "source_links": sum(map(len, self.memory_to_turns.values()))}
