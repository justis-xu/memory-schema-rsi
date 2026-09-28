"""Explicit source identity gate for graph candidates, independent of transport.

Callers must establish snapshot completeness. Matching a source manifest does not
prove extraction semantics or that graph edges were actually built from it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

from schema_rsi.memory.base import MemoryRecord


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class SourceEntry:
    content: str
    metadata_json: str


@dataclass(frozen=True)
class SourceSnapshot:
    user_id: str
    complete: bool
    revision: str
    reference_date: str
    source_sha256: str
    records: Mapping[str, SourceEntry]

    @classmethod
    def from_records(cls, records: Sequence[MemoryRecord], *, user_id: str,
                     complete: bool, revision: str, reference_date: str) -> SourceSnapshot:
        """Freeze records; complete/revision are caller evidence, not inferred here."""
        date.fromisoformat(reference_date)
        if not user_id or not revision:
            raise ValueError("snapshot requires user_id and revision")
        entries = {}
        for record in records:
            if not record.id or record.id in entries:
                raise ValueError("source IDs must be nonempty and unique")
            if record.metadata.get("user_id") != user_id:
                raise ValueError("source owner mismatch")
            entries[record.id] = SourceEntry(record.content, _json(record.metadata))
        payload = {"user_id": user_id, "reference_date": reference_date,
                   "records": [{"id": mid, "content": entry.content,
                                "metadata": json.loads(entry.metadata_json)}
                               for mid, entry in sorted(entries.items())]}
        return cls(user_id, complete, revision, reference_date, _sha(_json(payload)),
                   MappingProxyType(entries))


def admit_graph_candidates(candidates: Sequence[dict], snapshot: SourceSnapshot,
                           *, user_id: str, graph_source_sha256: str | None = None) -> tuple[list[dict], dict]:
    """Reject stale identities; restore current source dates without reranking.

    With no graph manifest, admission establishes identity only. A manifest
    mismatch rejects the whole pool. This function never queries or writes a DB.
    """
    accepted = []
    decisions = []
    seen = set()
    pool_reason = ("incomplete_source_snapshot" if not snapshot.complete else
                   "snapshot_owner_mismatch" if snapshot.user_id != user_id else
                   "graph_source_version_mismatch" if graph_source_sha256 is not None
                   and graph_source_sha256 != snapshot.source_sha256 else None)
    for index, candidate in enumerate(candidates):
        mid = candidate.get("id")
        entry = snapshot.records.get(mid) if isinstance(mid, str) else None
        reason = pool_reason
        if reason is None:
            if not isinstance(mid, str) or not mid:
                reason = "invalid_id"
            elif candidate.get("user_id") != user_id:
                reason = "candidate_owner_mismatch"
            elif entry is None:
                reason = "absent_current_id"
            elif candidate.get("content") != entry.content:
                reason = "body_mismatch_requires_relation_rebuild"
            elif not entry.content:
                reason = "empty_source_content"
            else:
                metadata = json.loads(entry.metadata_json)
                expiration = metadata.get("expiration_date")
                if expiration:
                    try:
                        expired = date.fromisoformat(str(expiration)) < date.fromisoformat(snapshot.reference_date)
                        if expired:
                            reason = "expired_source"
                    except ValueError:
                        reason = "invalid_expiration_date"
                if reason is None and mid in seen:
                    reason = "duplicate_candidate"
        if reason is None:
            seen.add(mid)
            current = deepcopy(candidate)
            metadata = json.loads(entry.metadata_json)
            current.update(user_id=user_id, content=entry.content,
                           session_id=metadata.get("session_id"), session_date=metadata.get("session_date"))
            accepted.append(current)
        decisions.append({"index": index, "id": mid, "accepted": reason is None,
                          "reason": reason or "same_owner_id_body",
                          "candidate_content_sha256": _sha(candidate["content"]) if isinstance(candidate.get("content"), str) else None})
    trace = {"source_sha256": snapshot.source_sha256, "source_revision": snapshot.revision,
             "source_complete": snapshot.complete, "reference_date": snapshot.reference_date,
             "graph_source_sha256": graph_source_sha256,
             "relationship_provenance": "unverified" if graph_source_sha256 is None else "manifest_claim_only",
             "input_count": len(candidates), "accepted_count": len(accepted), "decisions": decisions}
    return accepted, trace
