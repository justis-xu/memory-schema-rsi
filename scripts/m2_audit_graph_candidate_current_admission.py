#!/usr/bin/env python3
"""Historical graph candidate/current source identity audit, no live graph claim."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from m2_audit_gina_preference_storage import read_store
from m2_audit_graph_positive_mechanisms import PATHS, jsonl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_graph_candidate_current_admission_20260928.json'


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    rows, before = read_store()
    current = {r[0]: r for r in rows}
    bodies = defaultdict(list)
    for r in rows:
        bodies[(r[2], r[1])].append(r[0])
    historical = jsonl(PATHS['graph'])
    unique = {}
    by = defaultdict(lambda: {'candidate_occurrences': Counter(), 'selected_occurrences': Counter(),
                              'visible_vector_ids': set(), 'candidate_ids': set(), 'cases': 0})
    categories = defaultdict(Counter)
    for cid, row in historical.items():
        owner = row['metadata']['user_id']
        assert owner == 'zhfull:locomo:' + cid.removeprefix('locomo_').split('_qa')[0]
        group = by[owner]; group['cases'] += 1
        group['visible_vector_ids'].update(r['id'] for r in row['retrieved_memories'])
        selected = set(row['metadata'].get('graph_context_ids') or [])
        candidates = row.get('graph_memories', [])
        assert selected <= {r['id'] for r in candidates}
        for g in candidates:
            mid = g['id']; cur = current.get(mid)
            status = 'absent_current_id' if cur is None else (
                'owner_mismatch' if cur[2] != owner else (
                    'body_mismatch' if cur[1] != g['content'] else 'same_owner_id_body'))
            aliases = bodies[(owner, g['content'])] if cur is None else []
            group['candidate_occurrences'][status] += 1
            group['candidate_ids'].add(mid)
            if mid not in unique:
                unique[mid] = {'id': mid, 'user_id': owner, 'status': status,
                               'historical_body_sha256': hashlib.sha256(g['content'].encode()).hexdigest(),
                               'current_equal_body_ids': sorted(aliases), 'candidate_occurrences': 0, 'selected_case_ids': []}
            item = unique[mid]
            assert item['user_id'] == owner and item['status'] == status
            assert item['historical_body_sha256'] == hashlib.sha256(g['content'].encode()).hexdigest()
            item['candidate_occurrences'] += 1
            if mid in selected:
                item['selected_case_ids'].append(cid)
                group['selected_occurrences'][status] += 1
                categories[row['metadata']['category']][status] += 1
    summary = []
    for owner, group in sorted(by.items()):
        visible = group.pop('visible_vector_ids') | group['candidate_ids']
        ids = group.pop('candidate_ids')
        current_ids = {r[0] for r in rows if r[2] == owner}
        summary.append({'user_id': owner, **group, 'current_source_count': len(current_ids),
                        'unique_graph_ids': len(ids), 'matching_graph_ids': len(ids & current_ids),
                        'visible_vector_graph_union_ids': len(visible),
                        'visible_union_current_shared_ids': len(visible & current_ids),
                        'absent_graph_ids_with_equal_current_body': sum(bool(unique[mid]['current_equal_body_ids']) for mid in ids if unique[mid]['status'] == 'absent_current_id')})
    _, after = read_store(); assert before == after
    totals = Counter(); selected_totals = Counter()
    for group in summary:
        totals.update(group['candidate_occurrences']); selected_totals.update(group['selected_occurrences'])
    result = {'scope': 'One historical Chinese fused run (all saved categories) compared with current complete SQLite source IDs/bodies; not historical-time validity or live Graph audit',
              'historical_path': str(PATHS['graph'].relative_to(ROOT)), 'historical_sha256': hashlib.sha256(PATHS['graph'].read_bytes()).hexdigest(),
              'historical_cases': len(historical), 'logical_source_before': before, 'logical_source_after': after,
              'candidate_occurrences': dict(totals), 'selected_occurrences': dict(selected_totals),
              'unique_graph_id_status': dict(Counter(u['status'] for u in unique.values())),
              'absent_unique_equal_body_counts': dict(Counter('equal_body_present' if u['current_equal_body_ids'] else 'no_equal_body' for u in unique.values() if u['status'] == 'absent_current_id')),
              'selected_by_category': dict(categories), 'by_user': summary, 'unique_candidates': list(unique.values()),
              'limits': ['Absent now does not mean invalid when originally answered.', 'Equal body under another ID is not verified provenance/relationship equivalence.',
                         'Same owner/ID/body does not establish graph edges or complete historical source-version identity.',
                         'No reranking or replacement context computed; candidate admission counts are not answer effects.',
                         'All categories are included; counts must not be mixed with prior non-adversarial/common-case denominators.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('historical_cases', 'candidate_occurrences', 'selected_occurrences', 'unique_graph_id_status', 'absent_unique_equal_body_counts')}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
