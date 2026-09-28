#!/usr/bin/env python3
"""Current source witnesses for six late-only facts; no semantic automation."""
import hashlib
import json
from pathlib import Path
from m2_audit_gina_preference_storage import read_store
from m2_audit_graph_positive_mechanisms import PATHS

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/analysis/m2_carryover_fact_inventory_20260926.json'
OUT = ROOT / 'results/analysis/m2_carryover_current_witnesses_20260928.json'
# Manually confirmed content witnesses. These are not exhaustive entailment labels.
WITNESSES = {
    'conv-41_051a88fcd77c:4': ['d9d93932-9b77-4c7c-a3ac-a1b4dcbf40c7', '2fb7ce03-d35c-4b96-b3c1-cc914e5afa94'],
    'conv-44_b514edf1ce25:4': ['99549aa7-4b96-4b21-828b-dd5725ddcb04', '987421f4-89ba-4a49-958d-4185ac1732d4', '92291bd1-360e-46d3-952b-8d6c66b1e988'],
    'conv-43_de89be96dda6:3': ['3325426c-88b6-46e2-b7bb-d1303bce5de1'],
    'conv-26_444350391e64:2': ['137a6569-6e2e-4878-97d5-71289efc1f4a'],
    'conv-43_c95588df7268:4': ['6719cb5f-eeff-4b60-acdd-aa7fa29a134f', 'bd5e05ac-8fff-4146-9701-0796790c5f10', 'd3a79915-0f69-4ded-b392-3c5ad36f8eed'],
    'conv-47_12946f1346c5:3': ['59009b00-ef26-4701-bebd-b25b9dbd99ef'],
}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    inventory = json.loads(INPUT.read_text())
    rows, before = read_store()
    current = {r[0]: {'id': r[0], 'content': r[1], 'user_id': r[2],
                      'session_id': r[3], 'session_date': r[4]} for r in rows}
    pairs = []
    retirement = set()
    for p in inventory['pairs']:
        states = {}
        for key in ('older_memory', 'later_memory'):
            old = p[key]
            now = current.get(old['id'])
            states[key] = {'historical': old, 'current': now,
                           'status': 'absent_id' if now is None else
                           ('same_body' if now['content'] == old['content'] else 'changed_body')}
        if all(v['status'] == 'same_body' for v in states.values()):
            retirement.add(p['later_memory']['id'])
        pairs.append({'pair_id': p['pair_id'], **states})
    details = []
    for p in inventory['pairs']:
        for f in p['facts']:
            if f['fact_id'] not in WITNESSES:
                continue
            witnesses = [current[mid] for mid in WITNESSES[f['fact_id']]]
            assert all(m['user_id'] == 'zhfull:locomo:' + p['pair_id'].split('_')[0] for m in witnesses)
            details.append({'historical_fact_annotation': f,
                            'witness_scope': 'Manually confirmed current content carriers; not an exhaustive entailment search',
                            'coverage_qualification': 'Emotion witness supports feeling bad but omits source qualification that she is okay'
                            if f['fact_id'] == 'conv-26_444350391e64:2' else
                            ('Only creates-new-world component is unique; earlier memory already has writing joy'
                             if f['fact_id'] == 'conv-43_de89be96dda6:3' else 'See source and carrier text for scope'),
                            'current_witnesses': witnesses,
                            'confirmed_witness_ids_after_later_id_retirement': [m['id'] for m in witnesses if m['id'] not in retirement]})
    assert len(details) == 6
    sample = next(s for s in json.loads(PATHS['chinese_data'].read_text()) if s['sample_id'] == 'conv-41')
    road = current['509b61c0-35da-4fe7-94a0-e6924cf50f07']
    _, after = read_store()
    assert before == after
    report = {'scope': 'Current-state migration and witness loss control for fixed 12 pairs; zero model calls or writes',
              'pairs': pairs, 'later_ids_in_offline_retirement_control': sorted(retirement),
              'six_late_only_fact_witnesses': details,
              'current_owner_snapshots': {owner: [m for m in current.values() if m['user_id'] == owner]
                                          for owner in sorted({m['user_id'] for d in details for m in d['current_witnesses']})},
              'road_trip_companion': {'current_memory': road,
                                      'source_session11': sample['conversation']['session_11'],
                                      'manual_judgment': 'friends is not specified in this source session; we does not identify companions'},
              'logical_store_before': before, 'logical_store_after': after,
              'input_sha256': {'historical_inventory': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                               'chinese_data': hashlib.sha256(PATHS['chinese_data'].read_bytes()).hexdigest()},
              'limits': ['Witness labels are manual, not automatic matching or proof of no other carriers.',
                         'Retirement is a set operation only; no records, graph edges or indexes changed.',
                         'Source absence is scoped to the designated source session, not the whole conversation.',
                         'No retrieval cost or answer gain/harm measured.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'same_body_pairs': len(retirement),
                      'facts_losing_all_confirmed_witnesses': [d['historical_fact_annotation']['fact_id'] for d in details
                                                             if not d['confirmed_witness_ids_after_later_id_retirement']],
                      'unchanged': before == after}, ensure_ascii=False))


if __name__ == '__main__':
    main()
