"""Batch input provenance; this does not establish fact-level entailment."""
from __future__ import annotations

import hashlib
import json


def build_source_batch(*, benchmark: str, user_id: str, session: dict,
                       turns: list[dict], messages: list[dict]) -> dict:
    """Use history fields only. Turn references are local to the hashed batch.

    Gold fields, QA metadata, credentials and extraction settings are not inputs.
    The message copy records the actual date anchor and role/content sent.
    """
    sources = []
    for index, turn in enumerate(turns):
        source = {'source_ref': f'turn:{index}', 'turn_index': index,
                  'role': turn.get('role', 'user'), 'content': turn.get('content', '')}
        for key in ('dia_id', 'speaker', 'image_urls', 'blip_caption', 'image_query'):
            if turn.get(key) is not None:
                source[key] = turn[key]
        sources.append(source)
    payload = {'trace_version': 1, 'benchmark': benchmark, 'user_id': user_id,
               'session_id': str(session.get('session_id')),
               'session_date': session.get('date'), 'source_turns': sources,
               'input_messages': [{'role': m['role'], 'content': m['content']} for m in messages]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    # Round trip also detaches nested history values from a caller-owned sink.
    batch = json.loads(encoded)
    batch['batch_id'] = hashlib.sha256(encoded.encode()).hexdigest()
    batch['fact_support_status'] = 'not_provided'
    return batch


def snapshot_returned_records(records: list) -> list[dict]:
    """Freeze backend-returned text for later fact alignment, without full metadata."""
    snapshots = []
    for record in records:
        content = record.content or ''
        metadata = record.metadata or {}
        snapshots.append({
            'id': record.id,
            'content': content,
            'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'session_id': metadata.get('session_id'),
            'session_date': metadata.get('session_date'),
            'event': metadata.get('event'),
        })
    return snapshots
