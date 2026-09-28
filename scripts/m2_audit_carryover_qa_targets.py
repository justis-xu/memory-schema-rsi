#!/usr/bin/env python3
"""Map two carryover source anchors to real QA and historical slot controls."""
import hashlib
import json
from pathlib import Path
from m2_audit_graph_positive_mechanisms import PATHS, source_packet, jsonl, context
from m2_audit_gina_preference_storage import read_store

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_carryover_qa_targets_20260928.json'
ANCHORS = {'conv-43': 'D15:3', 'conv-26': 'D17:10'}
OLDER = {'ba8d6bda-96c4-4304-be57-cdbbfefdb84f', '8642315d-3c4c-4c9e-a333-7f29093b90fc'}
LATER = {'3325426c-88b6-46e2-b7bb-d1303bce5de1', '137a6569-6e2e-4878-97d5-71289efc1f4a'}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    datasets = {lang: {s['sample_id']: s for s in json.loads(PATHS[lang + '_data'].read_text())}
                for lang in ('english', 'chinese')}
    runs = {key: jsonl(PATHS[key]) for key in ('base', 'graph', 'base_grade', 'graph_grade')}
    rows, before = read_store()
    current = {r[0]: r for r in rows}
    cases = []
    for conv, eid in ANCHORS.items():
        sample = datasets['chinese'][conv]
        for i, qa in enumerate(sample['qa']):
            if eid not in qa.get('evidence', []):
                continue
            cid = f'locomo_{conv}_qa{i}'
            archive = {}
            for arm in ('base', 'graph'):
                if cid not in runs[arm]:
                    continue
                selected = context(runs[arm][cid])
                archive[arm] = {
                    'original_context': selected,
                    'predicted_answer': runs[arm][cid]['predicted_answer'],
                    'grade': runs[arm + '_grade'].get(cid, {}).get('final'),
                    'remove_older_without_refill': [m for m in selected if m['id'] not in OLDER],
                    'remove_later_without_refill': [m for m in selected if m['id'] not in LATER],
                    'removed_older_ids': [m['id'] for m in selected if m['id'] in OLDER],
                    'removed_later_ids': [m['id'] for m in selected if m['id'] in LATER],
                    'pair_memories_match_current': all(m['id'] in current and current[m['id']][1] == m['content']
                                                     for m in selected if m['id'] in OLDER | LATER),
                }
            cases.append({'case_id': cid, 'category': qa['category'],
                          'source': {lang: source_packet(datasets[lang][conv], i) for lang in datasets},
                          'historical_context_controls': archive})
    assert len(cases) == 8 and sum(c['category'] != 5 for c in cases) == 6
    diagnostics = [
        {'kind': 'manual source-grounded diagnostic, not official QA',
         'conversation': 'conv-43', 'source_turn_ids': ['D15:3'],
         'question': '蒂姆在2023年10月21日谈到写作时，写作除了带来快乐，还能让他做到什么？',
         'required_elements': ['创造全新的世界'],
         'purpose': 'Test preservation of the subclause not required by the anchor-linked official QA.'},
        {'kind': 'manual source-grounded diagnostic, not official QA',
         'conversation': 'conv-26', 'source_turn_ids': ['D17:8', 'D17:9', 'D17:10'],
         'question': '2023年10月13日，梅拉妮说暂停陶艺让她感觉如何？她对这个感受作了什么补充？',
         'required_elements': ['确实难受', '但她还好'],
         'purpose': 'Test emotion and its qualification together; the current late summary omits the qualification.'},
    ]
    for d in diagnostics:
        sample = datasets['chinese'][d['conversation']]
        d['sources'] = []
        for eid in d['source_turn_ids']:
            ts = sample['conversation']['session_' + eid[1:].split(':')[0]]
            d['sources'].append(next(t for t in ts if t['dia_id'] == eid))
    _, after = read_store()
    assert before == after
    report = {
        'scope': 'All eight QA explicitly citing two source anchors; six non-adversarial; no model calls',
        'selection': {'anchors': ANCHORS, 'rule': 'QA evidence explicitly contains anchor, no outcome selection'},
        'cases': cases, 'unexecuted_diagnostic_questions': diagnostics,
        'logical_store_before': before, 'logical_store_after': after,
        'input_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in PATHS.items()},
        'manual_judgment': 'None of the six non-adversarial gold answers directly requires creating a new world or feeling bad but okay; qa136 requires reading and painting from the older memory.',
        'limits': ['Explicit evidence selection does not prove no other QA indirectly uses these facts.',
                   'Deletion is a fixed-context ID subtraction with no refill, rerank or answer generation.',
                   'Removing a direct carrier does not prove the model cannot infer or guess the answer.',
                   'Diagnostic questions are manually constructed from source, not held-out benchmark evidence.',
                   'No automatic deduplication strategy or current retrieval/answer gain was tested.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected_ids': [c['case_id'] for c in cases], 'non_adversarial': 6,
                      'store_unchanged': before == after}, ensure_ascii=False))


if __name__ == '__main__':
    main()
