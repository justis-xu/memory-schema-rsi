#!/usr/bin/env python3
"""Read-only packet for three graph exact flips with necessary facts already in base."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh')
PATHS = {
    'base': ROOT / 'results_zh/zhfull_locomo_20260925_164645.jsonl',
    'graph': ROOT / 'results_zh/zhfull_fused_locomo_20260925_172126.jsonl',
    'base_grade': ROOT / 'results/precision_regrade_base.jsonl',
    'graph_grade': ROOT / 'results/precision_regrade_graph.jsonl',
    'english': DATA / 'locomo10.json',
    'chinese': DATA / 'locomo10_zh.json',
}
OUT = ROOT / 'results/analysis/m2_graph_existing_evidence_flips_20260928.json'
CASE_IDS = ['locomo_conv-41_qa84', 'locomo_conv-26_qa135', 'locomo_conv-44_qa41']
EXTRA_SOURCE_IDS = {'locomo_conv-44_qa41': ['D10:7']}


def read_jsonl(path):
    return {row['case_id']: row for row in map(json.loads, path.open())}


def context(row):
    records = {m['id']: m for m in row['retrieved_memories'] + row.get('graph_memories', [])}
    graph_ids = set(row['metadata'].get('graph_context_ids') or [])
    ids = row['metadata']['answer_context_ids']
    assert len(ids) == len(set(ids)) == 15
    return [{'rank': i, 'id': mid, 'route': 'graph' if mid in graph_ids else 'vector',
             'content': records[mid]['content'],
             'session_id': (records[mid].get('metadata') or {}).get('session_id'),
             'session_date': (records[mid].get('metadata') or {}).get('session_date')}
            for i, mid in enumerate(ids, 1)]


def source_turn(sample, dia_id):
    session = 'session_' + dia_id.split(':')[0][1:]
    turns = sample['conversation'][session]
    turn = next(t for t in turns if t['dia_id'] == dia_id)
    return {'dia_id': dia_id, 'session': session,
            'session_date': sample['conversation'][session + '_date_time'],
            'speaker': turn['speaker'], 'text': turn['text'],
            'query': turn.get('query'), 'blip_caption': turn.get('blip_caption')}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    base, graph, bg, gg = (read_jsonl(PATHS[k]) for k in ('base', 'graph', 'base_grade', 'graph_grade'))
    datasets = {k: {s['sample_id']: s for s in json.loads(PATHS[k].read_text())}
                for k in ('english', 'chinese')}
    cases = []
    for cid in CASE_IDS:
        conv, index = cid.removeprefix('locomo_').split('_qa')
        index = int(index)
        b, g = base[cid], graph[cid]
        bc, gc = context(b), context(g)
        b_ids, g_ids = {m['id'] for m in bc}, {m['id'] for m in gc}
        n_same_prefix = next((i for i, (x, y) in enumerate(zip(bc, gc)) if x['id'] != y['id']), 15)
        source = {}
        for lang, samples in datasets.items():
            sample = samples[conv]
            qa = sample['qa'][index]
            source[lang] = {'question': qa['question'], 'answer': qa.get('answer'),
                            'evidence_ids': qa['evidence'],
                            'turns': [source_turn(sample, eid) for eid in
                                      dict.fromkeys(qa['evidence'] + EXTRA_SOURCE_IDS.get(cid, []))]}
        assert source['english']['evidence_ids'] == source['chinese']['evidence_ids']
        cases.append({'case_id': cid, 'source': source,
                      'base_grade': bg[cid]['final'], 'graph_grade': gg[cid]['final'],
                      'base_judge': bg[cid].get('judge'), 'graph_judge': gg[cid].get('judge'),
                      'base_answer': b['predicted_answer'], 'graph_answer': g['predicted_answer'],
                      'same_ranked_prefix_count': n_same_prefix,
                      'shared_id_count': len(b_ids & g_ids),
                      'base_context': bc, 'graph_context': gc,
                      'gained': [m for m in gc if m['id'] not in b_ids],
                      'lost': [m for m in bc if m['id'] not in g_ids]})
    report = {'scope': 'Post-hoc three-case diagnostic of historical Chinese graph-only exact flips; independent runs; no model calls',
              'input_paths': {k: str(v) for k, v in PATHS.items()},
              'input_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in PATHS.items()},
              'cases': cases,
              'limits': ['Three hand-picked cases do not estimate the frequency of this mechanism in the 57 eligible flips.',
                         'Changed tail context and independent answer generations cannot be separated retrospectively.',
                         'Source turns are evidence, not proof that every memory extraction or gold answer is correct.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{'case_id': x['case_id'], 'same_prefix': x['same_ranked_prefix_count'],
                       'shared_ids': x['shared_id_count'], 'gained': len(x['gained'])}
                      for x in cases], ensure_ascii=False))


if __name__ == '__main__':
    main()
