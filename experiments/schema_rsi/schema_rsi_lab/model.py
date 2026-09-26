"""Frozen input snapshot and the small schema search space.

Relation IDs (r0, r1, ...) have no natural-language meaning. Their candidate
edges come from a separate, frozen generator such as ANN or latent clustering.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Node:
    id: str
    user_id: str
    kind: str  # M = memory, H = opaque latent hub


@dataclass(frozen=True)
class CandidateEdge:
    source: str
    target: str
    channel: str
    score: float


@dataclass(frozen=True)
class Case:
    id: str
    user_id: str
    split: str
    base_ranked: tuple[str, ...]
    gold: tuple[str, ...]  # evidence memory IDs, never answer strings


@dataclass(frozen=True)
class Channel:
    id: str
    max_degree: int = 3
    min_score: float = 0.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or self.max_degree < 1:
            raise ValueError("channel needs a nonempty ID and positive degree")
        if not 0 <= self.min_score <= 1 or not 0 < self.weight <= 2:
            raise ValueError("channel score threshold/weight out of range")


@dataclass(frozen=True)
class Schema:
    channels: tuple[Channel, ...]
    seed_k: int = 2
    context_k: int = 4
    graph_slots: int = 1
    max_hops: int = 2
    max_visits: int = 32
    hop_discount: float = 0.8

    def __post_init__(self) -> None:
        if len({c.id for c in self.channels}) != len(self.channels):
            raise ValueError("duplicate channel ID")
        if self.context_k < 1 or not 1 <= self.seed_k <= self.context_k:
            raise ValueError("invalid context/seed budget")
        if not 0 <= self.graph_slots <= self.context_k - self.seed_k:
            raise ValueError("graph slots exceed context budget")
        if self.max_hops not in (1, 2) or self.max_visits < 1:
            raise ValueError("unsupported hop or visit budget")
        if not 0 < self.hop_discount <= 1:
            raise ValueError("hop_discount must be in (0, 1]")

    @property
    def id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Schema":
        return cls(
            channels=tuple(Channel(**c) for c in raw.get("channels", [])),
            seed_k=int(raw.get("seed_k", 2)),
            context_k=int(raw.get("context_k", 4)),
            graph_slots=int(raw.get("graph_slots", 1)),
            max_hops=int(raw.get("max_hops", 2)),
            max_visits=int(raw.get("max_visits", 32)),
            hop_discount=float(raw.get("hop_discount", 0.8)),
        )


@dataclass(frozen=True)
class Dataset:
    nodes: dict[str, Node]
    edges: tuple[CandidateEdge, ...]
    cases: tuple[Case, ...]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        if not isinstance(data, dict):
            raise ValueError("dataset must be a JSON object")
        nodes = [Node(**item) for item in data["nodes"]]
        edges = [CandidateEdge(**item) for item in data["edges"]]
        cases = [
            Case(
                id=item["id"], user_id=item["user_id"], split=item["split"],
                base_ranked=tuple(item["base_ranked"]), gold=tuple(item["gold"]),
            )
            for item in data["cases"]
        ]
        if len({n.id for n in nodes}) != len(nodes) or len({q.id for q in cases}) != len(cases):
            raise ValueError("duplicate node or case ID")
        by_id = {n.id: n for n in nodes}
        if not any(n.kind == "M" for n in nodes):
            raise ValueError("dataset has no memory nodes")
        for n in nodes:
            if not n.id or not n.user_id or n.kind not in ("M", "H"):
                raise ValueError(f"invalid node: {n}")
        for edge in edges:
            if edge.source not in by_id or edge.target not in by_id:
                raise ValueError(f"unknown edge endpoint: {edge}")
            if edge.source == edge.target or by_id[edge.source].user_id != by_id[edge.target].user_id:
                raise ValueError(f"self/cross-user edge: {edge}")
            if not edge.channel or not math.isfinite(edge.score) or not 0 <= edge.score <= 1:
                raise ValueError(f"invalid edge score/channel: {edge}")
        for case in cases:
            if case.split not in ("train", "validation", "test"):
                raise ValueError(f"invalid split: {case.split}")
            if not case.gold or not case.base_ranked:
                raise ValueError(f"case lacks gold evidence or base ranking: {case.id}")
            if len(set(case.base_ranked)) != len(case.base_ranked) or len(set(case.gold)) != len(case.gold):
                raise ValueError(f"duplicate memory ID in case: {case.id}")
            for node_id in (*case.base_ranked, *case.gold):
                node = by_id.get(node_id)
                if node is None or node.kind != "M" or node.user_id != case.user_id:
                    raise ValueError(f"case references non-memory/cross-user node: {case.id}:{node_id}")
        if {c.split for c in cases} != {"train", "validation", "test"}:
            raise ValueError("train, validation and test splits are all required")
        fingerprint = hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(by_id, tuple(edges), tuple(cases), fingerprint)

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(sorted({edge.channel for edge in self.edges}))

    @property
    def memory_count(self) -> int:
        return sum(node.kind == "M" for node in self.nodes.values())
