#!/usr/bin/env python3
"""Read-only audit of a one-name LoCoMo Chinese QA pair across four archives."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json')
OUT = ROOT / 'results/analysis/m2_zh_adversarial_entity_pair_20260927.json'
ARCHIVES = {
    'no_graph': 'results_zh/zhfull_locomo_20260925_115806.jsonl',
    'fused_no_jev': 'results_zh/zhfull_fused_locomo_20260925_161049.jsonl',
    'jev_decider': 'results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl',
    'jev_laya': 'results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl',
}
CASES = ('locomo_conv-48_qa180', 'locomo_conv-48_qa233')
TARGET_ID = '663e599c-0540-4a4c-9054-05001d54f70f'
DISTRACTOR_ID = '30adf307-75c4-4816-8cb9-0cecd0c60ea3'


def rank(items, mid):
    return next((i for i, item in enumerate(items, 1) if item['id'] == mid), None)


def main():
    data = next(x for x in json.loads(DATASET.read_text()) if x['sample_id'] == 'conv-48')
    qa = data['qa']
    assert qa[180]['question'].replace('黛博拉', '乔琳') == qa[233]['question']
    turns = {}
    for row in data['conversation']['session_28']:
        if row['dia_id'] in ('D28:6', 'D28:7'):
            turns[row['dia_id']] = {'speaker': row['speaker'], 'text': row['text']}
    assert set(turns) == {'D28:6', 'D28:7'}
    out = {
        'scope': 'One-name substitution pair from Chinese LoCoMo, four archived independent runs, no new calls',
        'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        'source_turns': turns,
        'question_pair': {
            'locomo_conv-48_qa180': qa[180],
            'locomo_conv-48_qa233': qa[233],
        },
        'tracked_memories': {'target_id': TARGET_ID, 'distractor_id': DISTRACTOR_ID},
        'archives': {},
        'limits': [
            'The adversarial question changes the requested person, so the same factual answer is no longer valid.',
            'Archived retrieved_memories do not preserve the full pre-rerank vector candidate pool.',
            'Separate runs do not isolate Graph, Jev, or answer generation causally.',
        ],
    }
    for arm, rel in ARCHIVES.items():
        path = ROOT / rel
        entries = {}
        for line in path.open():
            row = json.loads(line)
            cid = row['case_id']
            if cid not in CASES:
                continue
            assert cid not in entries
            meta = row['metadata']
            vector = row['retrieved_memories']
            graph = row.get('graph_memories') or []
            merged = {m['id']: m for m in vector + graph}
            selected = meta['answer_context_ids']
            assert len(selected) == 15
            assert set(selected) <= set(merged)
            entries[cid] = {
                'question': row['question'],
                'predicted_answer': row['predicted_answer'],
                'jev': meta.get('jev_stop'),
                'tracked_ranks': {
                    label: {
                        'vector': rank(vector, mid), 'graph': rank(graph, mid),
                        'final': selected.index(mid) + 1 if mid in selected else None,
                    }
                    for label, mid in [('target', TARGET_ID), ('distractor', DISTRACTOR_ID)]
                },
                'final_context': [
                    {'slot': i, 'id': mid, 'content': merged[mid]['content']}
                    for i, mid in enumerate(selected, 1)
                ],
            }
        assert set(entries) == set(CASES)
        shared = sorted(
            {x['id'] for x in entries[CASES[0]]['final_context']}
            & {x['id'] for x in entries[CASES[1]]['final_context']}
        )
        out['archives'][arm] = {
            'path': rel, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'shared_final_context_ids': shared, 'cases': entries,
        }
    assert all(
        out['archives'][arm]['cases']['locomo_conv-48_qa180']['tracked_ranks']['target']['vector'] == 1
        and out['archives'][arm]['cases']['locomo_conv-48_qa233']['tracked_ranks']['distractor']['vector'] == 1
        and out['archives'][arm]['cases']['locomo_conv-48_qa233']['tracked_ranks']['target']['vector'] is None
        and out['archives'][arm]['cases']['locomo_conv-48_qa233']['tracked_ranks']['target']['graph'] is None
        for arm in ARCHIVES
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print('audited', len(ARCHIVES), 'archived arms and', len(CASES), 'questions')


if __name__ == '__main__':
    main()
