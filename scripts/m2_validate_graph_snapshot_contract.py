#!/usr/bin/env python3
"""Offline proposed snapshot admission/publication contract, not production wiring."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from schema_rsi.graph.builder import GraphBuilder
from schema_rsi.graph.schema import make_schema_s1
from schema_rsi.graph.extractor_s1 import StructuredExtractor
from schema_rsi.evaluation.retriever import GraphRetriever
from m2_audit_graph_stale_incremental import NeighborRecorder, memory, extraction

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_graph_snapshot_contract_20260928.json'
KINDS = ('entities', 'events', 'preferences', 'relationships')
EMPTY = {k: [] for k in KINDS}
PROFILE = {'prompt_revision': 'synthetic-source-v1', 'model_revision': 'synthetic-model-v1',
           'schema': asdict(make_schema_s1()), 'compiler_revision': 'offline-contract-v1'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def source_fingerprint(rows, profile):
    return digest({'profile': profile, 'memories': sorted(
        [{'id': r.id, 'content': r.content, 'metadata': r.metadata} for r in rows], key=lambda r: r['id'])})


def envelopes(rows, values, profile):
    return {r.id: {'status': 'success', 'input_sha256': digest(asdict(r)),
                   'profile_sha256': digest(profile), 'raw': {**deepcopy(EMPTY), **values[r.id]}}
            for r in rows}


def admit(rows, responses, owner, profile):
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids) or any(not mid for mid in ids):
        raise ValueError('duplicate_or_empty_source_id')
    if any(r.metadata.get('user_id') != owner for r in rows):
        raise ValueError('source_owner_mismatch')
    if set(responses) != set(ids):
        raise ValueError('response_id_set_mismatch')
    normalized = {}
    states = {}
    required = {'entities': ('name',), 'events': ('name',),
                'preferences': ('subject', 'pref_type', 'object'),
                'relationships': ('from_entity', 'relation', 'to_entity')}
    for r in rows:
        item = responses[r.id]
        if item.get('status') != 'success':
            raise ValueError('extraction_failed')
        if item.get('input_sha256') != digest(asdict(r)) or item.get('profile_sha256') != digest(profile):
            raise ValueError('extraction_fingerprint_mismatch')
        raw = item.get('raw')
        if not isinstance(raw, dict) or set(raw) != set(KINDS):
            raise ValueError('invalid_structure_fields')
        for kind in KINDS:
            if not isinstance(raw[kind], list):
                raise ValueError('invalid_structure_type')
            for fact in raw[kind]:
                if not isinstance(fact, dict) or any(not isinstance(fact.get(f), str) or not fact[f].strip() for f in required[kind]):
                    raise ValueError('invalid_fact_fields')
        normalized[r.id] = StructuredExtractor._normalize(raw)
        states[r.id] = 'success_nonempty' if any(normalized[r.id].values()) else 'success_valid_empty'
    return normalized, states


class Registry:
    def __init__(self):
        self.active = None

    def publish(self, rows, responses, profile=PROFILE, fail_stage=False):
        normalized, states = admit(rows, responses, '合成用户', profile)
        stage = NeighborRecorder()
        if any(any(ext.values()) for ext in normalized.values()):
            GraphBuilder(stage).build_s1(rows, normalized, reset=False)
        else:
            # Explicit prototype branch: production build_s1 rejects valid all-empty.
            stage.create_schema(make_schema_s1())
            for r in rows:
                stage.upsert_vertex('Memory', r.id, {'memory_id': r.id, 'content': r.content, **r.metadata})
        if fail_stage:
            raise RuntimeError('synthetic_stage_failure_before_publish')
        assert {k for (label, k) in stage.vertices if label == 'Memory'} == {r.id for r in rows}
        manifest = {'source_sha256': source_fingerprint(rows, profile), 'states': states,
                    'extraction_sha256': digest(normalized)}
        # In-memory pointer switch only; no claim of database transaction atomicity.
        self.active = {'manifest': manifest, 'store': stage}

    def retrieve(self, rows, anchor, profile=PROFILE):
        if not self.active or self.active['manifest']['source_sha256'] != source_fingerprint(rows, profile):
            return {'route': 'vector_fallback_version_mismatch', 'candidates': []}
        candidates = GraphRetriever(self.active['store'], SimpleNamespace()).retrieve_fused(
            [anchor], exclude_ids={anchor.id})
        return {'route': 'active_graph', 'candidates': candidates}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    a = memory('anchor', '旧话题锚点')
    old = memory('deleted', '旧记录')
    changed = memory('changed', '旧话题内容')
    initial = [a, old, changed]
    now = [a, memory('changed', '新话题内容'), memory('replacement', '新记录')]
    ext = {**extraction([a], '旧话题'), **extraction(now[1:], '新话题')}
    valid = envelopes(now, ext, PROFILE)
    cases = []
    def run(name, rows=now, response=None, profile=PROFILE, fail_stage=False, accepted=False):
        reg = Registry()
        reg.publish(initial, envelopes(initial, extraction(initial, '旧话题'), PROFILE))
        previous = reg.active
        reason = None
        try:
            reg.publish(rows, valid if response is None else response, profile, fail_stage)
            actual = True
        except (ValueError, RuntimeError) as exc:
            actual = False
            reason = str(exc)
        assert actual == accepted, (name, reason)
        if not actual:
            assert reg.active is previous
        route = reg.retrieve(rows, rows[0], profile)
        cases.append({'name': name, 'accepted': actual, 'reason': reason,
                      'old_snapshot_preserved_on_rejection': reg.active is previous,
                      'read': route, 'manifest': reg.active['manifest'], 'graph': reg.active['store'].export()})
        return reg, route
    reg, read = run('delete_and_same_id_relation_replacement', accepted=True)
    assert {g['id'] for g in read['candidates']} == set()
    assert ('Memory', 'deleted') not in reg.active['store'].vertices
    new_read = reg.retrieve(now, now[1])
    assert {g['id'] for g in new_read['candidates']} == {'replacement'}
    empty_one = deepcopy(valid); empty_one['changed']['raw'] = deepcopy(EMPTY)
    reg, _ = run('one_valid_empty_withdraws_old_relations', response=empty_one, accepted=True)
    assert not any(e['from_key'] == 'changed' for e in reg.active['store'].edges)
    all_empty = envelopes(now, {r.id: EMPTY for r in now}, PROFILE)
    reg, read = run('all_valid_empty_publishes_memory_only', response=all_empty, accepted=True)
    assert not reg.active['store'].edges
    failed = deepcopy(valid); failed['changed']['status'] = 'failed'
    _, read = run('failed_current_extraction_preserves_old_but_disables_stale_read', response=failed)
    assert read['route'] == 'vector_fallback_version_mismatch'
    omitted = deepcopy(valid); omitted.pop('changed')
    run('missing_response_id', response=omitted)
    malformed = deepcopy(valid); malformed['changed']['raw']['entities'] = 'not_a_list'
    run('malformed_is_not_valid_empty', response=malformed)
    invalid_fact = deepcopy(valid); invalid_fact['changed']['raw']['entities'] = [{'type': 'person'}]
    run('invalid_fact_is_not_valid_empty', response=invalid_fact)
    stale = deepcopy(valid); stale['changed']['input_sha256'] = digest(asdict(changed))
    run('same_id_cached_old_content', response=stale)
    profile2 = {**PROFILE, 'prompt_revision': 'synthetic-source-v2'}
    run('changed_prompt_invalidates_cached_extraction', profile=profile2)
    run('duplicate_source_ids', rows=now + [now[1]])
    foreign = now + [memory('foreign', '别的用户', '其他用户')]
    run('foreign_owner_source', rows=foreign)
    run('stage_failure_before_pointer_switch', fail_stage=True)
    unchanged_failure = envelopes(initial, extraction(initial, '旧话题'), PROFILE)
    unchanged_failure['changed']['status'] = 'failed'
    _, read = run('failed_refresh_same_source_keeps_valid_previous_read', rows=initial, response=unchanged_failure)
    assert read['route'] == 'active_graph'
    assert {g['id'] for g in read['candidates']} == {'deleted', 'changed'}
    # Fingerprints must be stable under enumeration order and change on dates/model.
    assert source_fingerprint(now, PROFILE) == source_fingerprint(list(reversed(now)), PROFILE)
    dated = deepcopy(now); dated[0].metadata['session_date'] = '2026-09-28'
    assert source_fingerprint(dated, PROFILE) != source_fingerprint(now, PROFILE)
    assert source_fingerprint(now, {**PROFILE, 'model_revision': 'v2'}) != source_fingerprint(now, PROFILE)
    result = {'scope': '13 synthetic cases for proposed admission and isolated publication contract; zero external calls/real graph writes',
              'profile': PROFILE, 'cases': cases, 'fingerprint_controls_passed': ['order_invariant', 'date_sensitive', 'model_sensitive'],
              'limits': ['Not integrated into production extractor, build entry or pipeline.', 'All-empty handling is an explicit prototype branch.',
                         'Pointer atomicity is in-memory only; HugeGraph version namespaces/transactions not validated.',
                         'Current source snapshot completeness is an assumption, not established by a topK response.', 'No answer gain or semantic extraction quality measured.']}
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'cases': len(cases), 'accepted': sum(c['accepted'] for c in cases), 'rejected': sum(not c['accepted'] for c in cases)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
