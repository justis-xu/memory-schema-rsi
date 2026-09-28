#!/usr/bin/env python3
"""Current Chinese source plus fault controls for the callable admission API."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from schema_rsi.evaluation.graph_admission import SourceSnapshot, admit_graph_candidates
from schema_rsi.memory.base import MemoryRecord
from m2_audit_gina_preference_storage import read_store
from m2_audit_graph_stale_incremental import replay, memory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_graph_source_admission_20260928.json'


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    rows, before = read_store()
    owner = 'zhfull:locomo:conv-30'
    records = [MemoryRecord(r[0], r[1], {'user_id': r[2], 'session_id': r[3], 'session_date': r[4]}) for r in rows if r[2] == owner]
    snapshot = SourceSnapshot.from_records(records, user_id=owner, complete=True,
                                          revision=before['id_content_user_session_sha256'], reference_date='2026-09-28')
    target = next(r for r in records if r.id == 'e093b97c-78a8-4fb6-88db-285d47314f4f')
    candidate = {'id': target.id, 'content': target.content, 'user_id': owner,
                 'session_date': None, 'session_id': 'stale-session', 'via': ['entity:合成控制'], 'support': 2}
    cases = []
    def check(name, candidates, expected, source=snapshot, graph_sha=None, requested_owner=owner):
        frozen = deepcopy(candidates)
        accepted, trace = admit_graph_candidates(candidates, source, user_id=requested_owner, graph_source_sha256=graph_sha)
        assert [d['reason'] for d in trace['decisions']] == expected
        assert candidates == frozen
        cases.append({'name': name, 'accepted_ids': [g['id'] for g in accepted], 'trace': trace})
        return accepted, trace
    accepted, _ = check('current_pool_outside_candidate_restores_date', [candidate], ['same_owner_id_body'])
    # The candidate is deliberately outside a one-record vector pool; source is all 102.
    vector_pool_ids = {next(r.id for r in records if r.id != target.id)}
    assert target.id not in vector_pool_ids and accepted[0]['id'] == target.id
    assert accepted[0]['session_date'] == target.metadata['session_date']
    assert accepted[0]['session_id'] == target.metadata['session_id']
    assert accepted[0]['via'] == candidate['via'] and accepted[0]['support'] == 2
    check('deleted_id', [{**candidate, 'id': 'synthetic-deleted'}], ['absent_current_id'])
    check('same_id_changed_body', [{**candidate, 'content': '合成旧正文'}], ['body_mismatch_requires_relation_rebuild'])
    check('foreign_candidate', [{**candidate, 'user_id': 'other-user'}], ['candidate_owner_mismatch'])
    check('unknown_candidate_owner', [{k: v for k, v in candidate.items() if k != 'user_id'}], ['candidate_owner_mismatch'])
    check('incomplete_topk_source', [candidate], ['incomplete_source_snapshot'], source=replace(snapshot, complete=False))
    check('foreign_snapshot_request', [candidate], ['snapshot_owner_mismatch'], requested_owner='other-user')
    check('duplicate_pool_entry', [candidate, candidate], ['same_owner_id_body', 'duplicate_candidate'])
    check('stale_graph_manifest', [candidate], ['graph_source_version_mismatch'], graph_sha='synthetic-old-version')
    _, trace = check('matching_manifest_is_not_semantic_proof', [candidate], ['same_owner_id_body'], graph_sha=snapshot.source_sha256)
    assert trace['relationship_provenance'] == 'manifest_claim_only'
    # Snapshot does not change when the caller later mutates its records.
    old_hash = snapshot.source_sha256
    target.metadata['session_date'] = 'synthetic-change'
    target.content = 'synthetic-change'
    accepted, _ = check('frozen_source_survives_caller_mutation', [candidate], ['same_owner_id_body'])
    assert accepted[0]['content'] == candidate['content'] and snapshot.source_sha256 == old_hash
    for expiration, expected in [('2026-09-27', 'expired_source'), ('2026-09-28', 'same_owner_id_body'), ('invalid', 'invalid_expiration_date')]:
        record = MemoryRecord('expiry', '合成有效性控制', {'user_id': owner, 'expiration_date': expiration})
        source = SourceSnapshot.from_records([record], user_id=owner, complete=True, revision='synthetic', reference_date='2026-09-28')
        check('expiration_' + expiration, [{'id': record.id, 'content': record.content, 'user_id': owner}], [expected], source=source)
    # Feed candidates from the real stale-incremental GraphRetriever replay into this API.
    prior = replay()
    current = [memory(r['id'], r['content']) for r in prior['current_source']]
    source = SourceSnapshot.from_records(current, user_id='合成用户', complete=True, revision='synthetic-current', reference_date='2026-09-28')
    check('actual_retriever_deleted_candidate_rejected', prior['fused_candidates'], ['absent_current_id'], source=source, requested_owner='合成用户')
    accepted, trace = check('actual_retriever_old_relation_stays_unverified', prior['bypass_candidates'],
                            ['absent_current_id', 'same_owner_id_body'], source=source, requested_owner='合成用户')
    assert accepted[0]['id'] == 'changed' and trace['relationship_provenance'] == 'unverified'
    _, after = read_store(); assert before == after
    result = {'scope': 'Callable source admission API with 102 actual Chinese Memory records and synthetic candidate/expiry/version controls; real GraphRetriever recording replay; zero model/service calls',
              'cases': cases, 'actual_source_count': len(snapshot.records), 'actual_source_user': owner,
              'logical_source_before': before, 'logical_source_after': after,
              'vector_pool_control_ids': sorted(vector_pool_ids),
              'limits': ['Not yet wired into EvaluationPipeline, so existing runs are unchanged.',
                         'complete/revision are caller assertions; source reader must establish them.',
                         'Identity gate cannot detect same-body stale relationships without verified graph provenance.',
                         'Manifest string equality is only a claim; edge construction evidence still required.',
                         'No candidate ranking or answer quality evaluated.']}
    paths = ['src/schema_rsi/evaluation/graph_admission.py', 'scripts/m2_verify_graph_source_admission.py']
    result['code_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'checks': len(cases), 'actual_source_count': len(snapshot.records), 'logical_source_unchanged': before == after}, ensure_ascii=False))


if __name__ == '__main__':
    main()
