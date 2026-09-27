#!/usr/bin/env python3
"""Read-only cohort of literal speaker-name swaps in Chinese LoCoMo QA."""

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json')
OUT = ROOT / 'results/analysis/m2_zh_speaker_swap_cohort_20260927.json'
ARCHIVES = {
    'no_graph': ROOT / 'results_zh/zhfull_locomo_20260925_115806.jsonl',
    'fused_no_jev': ROOT / 'results_zh/zhfull_fused_locomo_20260925_161049.jsonl',
    'jev_decider': ROOT / 'results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl',
    'jev_laya': ROOT / 'results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl',
}
SOURCE_CASES = {
    ('conv-26', 91, 157), ('conv-26', 128, 182),
    ('conv-48', 147, 216), ('conv-48', 163, 226), ('conv-48', 180, 233),
}


def speakers(sample):
    return {
        t['speaker']
        for key, rows in sample['conversation'].items()
        if key.startswith('session_') and not key.endswith('_date_time')
        for t in rows
    }


def source_window(sample, evidence):
    found = {}
    for session, turns in sample['conversation'].items():
        if not session.startswith('session_') or session.endswith('_date_time'):
            continue
        for index, turn in enumerate(turns):
            if turn['dia_id'] in evidence:
                found[turn['dia_id']] = {
                    'session_id': session,
                    'session_date': sample['conversation'][session + '_date_time'],
                    'window': [
                        {'dia_id': t['dia_id'], 'speaker': t['speaker'], 'text': t['text']}
                        for t in turns[max(0, index - 1):index + 2]
                    ],
                }
    assert set(found) == set(evidence)
    return found


def main():
    samples = json.loads(DATASET.read_text())
    pairs = []
    canonical = {}
    raw_pair_rows = 0
    for sample in samples:
        names = speakers(sample)
        if len(names) != 2:
            continue
        for i, positive in enumerate(sample['qa']):
            if positive['category'] != 4:
                continue
            for j, negative in enumerate(sample['qa']):
                if negative['category'] != 5:
                    continue
                if not set(positive.get('evidence', [])) & set(negative.get('evidence', [])):
                    continue
                edits = [e for e in SequenceMatcher(
                    None, positive['question'], negative['question']).get_opcodes()
                    if e[0] != 'equal']
                if len(edits) != 1 or edits[0][0] != 'replace':
                    continue
                _, a, b, c, d = edits[0]
                old = positive['question'][a:b]
                new = negative['question'][c:d]
                if old == new or {old, new} != names:
                    continue
                raw_pair_rows += 1
                conv = sample['sample_id']
                key = (conv, positive['question'], negative['question'],
                       tuple(sorted(set(positive.get('evidence', [])) & set(negative.get('evidence', [])))))
                if key in canonical:
                    existing = canonical[key]
                    assert existing['positive_answer'] == positive.get('answer')
                    existing.setdefault('duplicate_positive_ids', []).append(f'locomo_{conv}_qa{i}')
                    continue
                record = {
                    'conversation': conv, 'positive_index': i, 'adversarial_index': j,
                    'positive_id': f'locomo_{conv}_qa{i}',
                    'adversarial_id': f'locomo_{conv}_qa{j}',
                    'positive_question': positive['question'],
                    'adversarial_question': negative['question'],
                    'name_replacement': {'from': old, 'to': new},
                    'positive_answer': positive.get('answer'),
                    'adversarial_answer_field': negative.get('adversarial_answer'),
                    'shared_evidence': sorted(set(positive.get('evidence', [])) & set(negative.get('evidence', []))),
                    'arms': {},
                }
                if (conv, i, j) in SOURCE_CASES:
                    record['source_window'] = source_window(sample, record['shared_evidence'])
                pairs.append(record)
                canonical[key] = record
    assert raw_pair_rows == 77 and len(pairs) == 76
    assert len({p['adversarial_id'] for p in pairs}) == len(pairs)
    archive_info = {}
    summary = {}
    for arm, path in ARCHIVES.items():
        needed = {cid for pair in pairs for cid in (pair['positive_id'], pair['adversarial_id'])}
        rows = {}
        for line in path.open():
            row = json.loads(line)
            if row['case_id'] in needed:
                assert row['case_id'] not in rows
                rows[row['case_id']] = row
        archive_info[arm] = {'path': str(path.relative_to(ROOT)),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        overlaps = []
        retained = []
        changed = []
        for pair in pairs:
            if pair['positive_id'] not in rows or pair['adversarial_id'] not in rows:
                continue
            pos = rows[pair['positive_id']]
            neg = rows[pair['adversarial_id']]
            pids = pos['metadata']['answer_context_ids']
            nids = neg['metadata']['answer_context_ids']
            assert len(pids) == len(nids) == 15
            topid = pos['retrieved_memories'][0]['id']
            neg_topid = neg['retrieved_memories'][0]['id']
            n_pool = {m['id'] for m in neg['retrieved_memories'] + neg.get('graph_memories', [])}
            observation = {
                'final_context_overlap': len(set(pids) & set(nids)),
                'positive_top_retrieved_id': topid,
                'adversarial_top_retrieved_id': neg_topid,
                'positive_top_changed': topid != neg_topid,
                'positive_top_in_adversarial_pool': topid in n_pool,
                'positive_top_in_adversarial_final': topid in nids,
                'positive_final_ids': pids,
                'adversarial_final_ids': nids,
                'positive_predicted_answer': pos['predicted_answer'],
                'adversarial_predicted_answer': neg['predicted_answer'],
            }
            if 'source_window' in pair:
                for label, row in [('positive', pos), ('adversarial', neg)]:
                    contents = {m['id']: m['content'] for m in row['retrieved_memories'] + row.get('graph_memories', [])}
                    observation[label + '_final_context'] = [
                        {'id': mid, 'content': contents[mid]} for mid in row['metadata']['answer_context_ids']
                    ]
            pair['arms'][arm] = observation
            overlaps.append(observation['final_context_overlap'])
            retained.append(observation['positive_top_in_adversarial_final'])
            changed.append(observation['positive_top_changed'])
        summary[arm] = {
            'paired_cases': len(overlaps),
            'overlap_distribution': dict(sorted(Counter(overlaps).items())),
            'median_overlap': statistics.median(overlaps),
            'overlap_at_most_1': sum(x <= 1 for x in overlaps),
            'overlap_at_most_3': sum(x <= 3 for x in overlaps),
            'positive_top_retrieved_changed': sum(changed),
            'positive_top_retained_in_adversarial_final': sum(retained),
        }
    assert len([p for p in pairs if 'source_window' in p]) == len(SOURCE_CASES)
    result = {
        'scope': 'All exact one-speaker-name category-4/category-5 QA swaps with shared evidence in Chinese LoCoMo; four old independent runs; read-only',
        'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        'dataset_category_counts': dict(sorted(Counter(
            question['category'] for sample in samples for question in sample['qa']
        ).items())),
        'raw_index_pair_rows': raw_pair_rows,
        'unique_question_pairs': len(pairs),
        'selection_rule': 'same conversation, category 4 vs 5, overlapping evidence ID, exactly one SequenceMatcher replace, replacement strings equal the two dialogue speaker names',
        'archives': archive_info, 'summary': summary, 'pairs': pairs,
        'limits': [
            'The category-5 adversarial_answer is an attack target, not a true gold answer.',
            'The positive question top retrieved memory is an ID-retention proxy, not automatically a verified source fact.',
            'The four runs share questions and near-identical memory content; arm counts are not independent samples.',
            'Source windows were audited only for five selected mechanism cases.',
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('pairs', len(pairs), 'coverage', {k: v['paired_cases'] for k, v in summary.items()})


if __name__ == '__main__':
    main()
