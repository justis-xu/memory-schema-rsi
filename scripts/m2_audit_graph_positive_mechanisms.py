#!/usr/bin/env python3
"""Read-only mechanism packet for targeted Chinese graph-only exact flips."""

from collections import Counter
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
    'english_data': DATA / 'locomo10.json',
    'chinese_data': DATA / 'locomo10_zh.json',
}
OUT = ROOT / 'results/analysis/m2_graph_positive_mechanisms_20260927.json'
PREVIOUSLY_AUDITED = {
    'locomo_conv-47_qa96', 'locomo_conv-48_qa39', 'locomo_conv-43_qa25',
    'locomo_conv-41_qa86', 'locomo_conv-30_qa80',
}
# Chosen after a concise screen of 57 eligible positive flips to contrast
# candidate fact gain, already-present evidence, judge drift, and inference.
CASE_IDS = [
    'locomo_conv-49_qa85', 'locomo_conv-42_qa198',
    'locomo_conv-41_qa6', 'locomo_conv-47_qa72', 'locomo_conv-49_qa72',
    'locomo_conv-50_qa124', 'locomo_conv-42_qa118', 'locomo_conv-47_qa6',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return {row['case_id']: row for row in map(json.loads, path.open())}


def context(row):
    records = {m['id']: m for m in row['retrieved_memories'] + row.get('graph_memories', [])}
    ids = row['metadata']['answer_context_ids']
    graph_ids = set(row['metadata'].get('graph_context_ids') or [])
    assert len(ids) == len(set(ids)) == 15
    return [{'rank': i, 'id': mid, 'content': records[mid]['content'],
             'route': 'graph' if mid in graph_ids else 'vector',
             'session_date': (records[mid].get('metadata') or {}).get('session_date')}
            for i, mid in enumerate(ids, 1)]


def source_packet(sample, qa_index):
    qa = sample['qa'][qa_index]
    evidence = []
    for dia_id in qa['evidence']:
        session = 'session_' + dia_id.split(':')[0][1:]
        turns = sample['conversation'][session]
        i = next(i for i, turn in enumerate(turns) if turn['dia_id'] == dia_id)
        evidence.append({
            'dia_id': dia_id, 'session': session,
            'session_date': sample['conversation'][session + '_date_time'],
            'window': [{k: turn.get(k) for k in ('dia_id', 'speaker', 'text', 'query', 'blip_caption')}
                       for turn in turns[max(0, i - 3):i + 4]],
        })
    return {'question': qa['question'], 'answer': qa.get('answer'), 'evidence': evidence}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    base, graph, bg, gg = (jsonl(PATHS[k]) for k in ('base', 'graph', 'base_grade', 'graph_grade'))
    dataset = {lang: {s['sample_id']: s for s in json.loads(PATHS[lang + '_data'].read_text())}
               for lang in ('english', 'chinese')}
    eligible = []
    counts = Counter()
    for cid in sorted(base.keys() & graph.keys() & bg.keys() & gg.keys()):
        if cid in PREVIOUSLY_AUDITED or bg[cid]['category'] == 'adversarial':
            continue
        if bg[cid]['final'] == 'exact' or gg[cid]['final'] != 'exact':
            continue
        bc, gc = context(base[cid]), context(graph[cid])
        bi, gi = {m['id'] for m in bc}, {m['id'] for m in gc}
        if bi == gi or not any(m['route'] == 'graph' for m in gc):
            continue
        eligible.append(cid)
        gained = [m for m in gc if m['id'] not in bi]
        counts['graph_gained'] += sum(m['route'] == 'graph' for m in gained)
        counts['vector_gained'] += sum(m['route'] == 'vector' for m in gained)
        counts['cases_with_vector_gain'] += any(m['route'] == 'vector' for m in gained)
    assert len(eligible) == 57 and set(CASE_IDS) <= set(eligible)
    cases = []
    for cid in CASE_IDS:
        b, g = base[cid], graph[cid]
        bc, gc = context(b), context(g)
        bi, gi = {m['id'] for m in bc}, {m['id'] for m in gc}
        conv, index = cid.removeprefix('locomo_').split('_qa')
        index = int(index)
        sources = {lang: source_packet(dataset[lang][conv], index) for lang in ('english', 'chinese')}
        assert [e['dia_id'] for e in sources['english']['evidence']] == [e['dia_id'] for e in sources['chinese']['evidence']]
        cases.append({
            'case_id': cid, 'source': sources,
            'base_grade': bg[cid]['final'], 'graph_grade': gg[cid]['final'],
            'base_judge': bg[cid].get('judge'), 'graph_judge': gg[cid].get('judge'),
            'base_answer': b['predicted_answer'], 'graph_answer': g['predicted_answer'],
            'base_context': bc, 'graph_context': gc,
            'gained': [m for m in gc if m['id'] not in bi],
            'lost': [m for m in bc if m['id'] not in gi],
        })
    result = {
        'scope': 'Targeted eight-case mechanism review after screening 57 eligible Chinese graph-only exact flips; not random or prevalence sample; no new model calls',
        'eligible_rule': 'Non-adversarial, exclude earlier five mechanism cases, base non-exact, graph exact, graph candidate final, final ID sets differ',
        'eligible_case_ids': eligible,
        'eligible_counts': {'cases': len(eligible), **counts},
        'selected_case_ids': CASE_IDS,
        'input_paths': {k: str(v) for k, v in PATHS.items()},
        'input_sha256': {k: sha(v) for k, v in PATHS.items()},
        'cases': cases,
        'limits': ['Targeted selection enriches mechanisms and cannot estimate frequencies among the 57.',
                   'Separate historical runs confound candidate effects with generation and judge variation.',
                   'QA evidence may omit neighboring turns; each packet includes a three-turn source window.'],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'eligible_counts': result['eligible_counts'],
                      'selected_case_ids': CASE_IDS}, ensure_ascii=False))


if __name__ == '__main__':
    main()
