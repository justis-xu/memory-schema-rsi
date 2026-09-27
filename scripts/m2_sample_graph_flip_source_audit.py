#!/usr/bin/env python3
"""Deterministically sample archived Chinese graph flips for bilingual source audit."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh')
BASE = ROOT / 'results_zh/zhfull_locomo_20260925_164645.jsonl'
GRAPH = ROOT / 'results_zh/zhfull_fused_locomo_20260925_172126.jsonl'
BASE_GRADE = ROOT / 'results/precision_regrade_base.jsonl'
GRAPH_GRADE = ROOT / 'results/precision_regrade_graph.jsonl'
OUT = ROOT / 'results/analysis/m2_graph_flip_source_sample_20260927.json'
SEED = 'm2-graph-source-sample-v1:'
PREVIOUSLY_AUDITED = {
    'locomo_conv-47_qa96', 'locomo_conv-48_qa39', 'locomo_conv-43_qa25',
    'locomo_conv-41_qa86', 'locomo_conv-30_qa80',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return {row['case_id']: row for row in map(json.loads, path.open())}


def context(row):
    records = {m['id']: m for m in row['retrieved_memories'] + row.get('graph_memories', [])}
    graph_ids = set(row['metadata'].get('graph_context_ids') or [])
    ids = row['metadata']['answer_context_ids']
    assert len(ids) == len(set(ids)) == 15
    assert set(ids) <= records.keys()
    return [{'rank': rank, 'id': mid, 'content': records[mid]['content'],
             'route': 'graph' if mid in graph_ids else 'vector',
             'session_date': (records[mid].get('metadata') or {}).get('session_date')}
            for rank, mid in enumerate(ids, 1)]


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    paths = {'base_run': BASE, 'graph_run': GRAPH,
             'base_grade': BASE_GRADE, 'graph_grade': GRAPH_GRADE,
             'english_data': DATA / 'locomo10.json', 'chinese_data': DATA / 'locomo10_zh.json'}
    base, graph, bg, gg = (jsonl(p) for p in (BASE, GRAPH, BASE_GRADE, GRAPH_GRADE))
    datasets = {lang: {s['sample_id']: s for s in json.loads(paths[lang + '_data'].read_text())}
                for lang in ('english', 'chinese')}
    bins = defaultdict(list)
    for cid in sorted(base.keys() & graph.keys() & bg.keys() & gg.keys()):
        if cid in PREVIOUSLY_AUDITED or bg[cid]['category'] == 'adversarial':
            continue
        bc, gc = context(base[cid]), context(graph[cid])
        bi, gi = {m['id'] for m in bc}, {m['id'] for m in gc}
        if bi == gi or not any(m['route'] == 'graph' for m in gc):
            continue
        b_exact = bg[cid]['final'] == 'exact'
        g_exact = gg[cid]['final'] == 'exact'
        stratum = 'graph_only_exact' if g_exact and not b_exact else (
            'base_only_exact' if b_exact and not g_exact else 'same_exact_status')
        digest = hashlib.sha256((SEED + cid).encode()).hexdigest()
        bins[stratum].append((digest, cid))

    cases = []
    for stratum in ('graph_only_exact', 'base_only_exact', 'same_exact_status'):
        assert len(bins[stratum]) >= 3
        for digest, cid in sorted(bins[stratum])[:3]:
            b, g = base[cid], graph[cid]
            bc, gc = context(b), context(g)
            bi, gi = {m['id'] for m in bc}, {m['id'] for m in gc}
            conv, index = cid.removeprefix('locomo_').split('_qa')
            index = int(index)
            assert datasets['english'][conv]['qa'][index]['evidence'] == datasets['chinese'][conv]['qa'][index]['evidence']
            qa = {}
            for lang in ('english', 'chinese'):
                sample = datasets[lang][conv]
                q = sample['qa'][index]
                sources = []
                for dia_id in q['evidence']:
                    session = 'session_' + dia_id.split(':')[0][1:]
                    turn = next(t for t in sample['conversation'][session] if t['dia_id'] == dia_id)
                    sources.append({'session_id': session, 'session_date': sample['conversation'][session + '_date_time'],
                                    'dia_id': dia_id, 'speaker': turn['speaker'], 'text': turn['text'],
                                    'query': turn.get('query'), 'blip_caption': turn.get('blip_caption')})
                qa[lang] = {'question': q['question'], 'answer': q.get('answer'),
                            'evidence': sources}
            cases.append({
                'stratum': stratum, 'selection_sha256': digest, 'case_id': cid,
                'qa': qa, 'base_grade': bg[cid]['final'], 'graph_grade': gg[cid]['final'],
                'base_judge': bg[cid].get('judge'), 'graph_judge': gg[cid].get('judge'),
                'base_answer': b['predicted_answer'], 'graph_answer': g['predicted_answer'],
                'base_context': bc, 'graph_context': gc,
                'gained_ids': [m['id'] for m in gc if m['id'] not in bi],
                'lost_ids': [m['id'] for m in bc if m['id'] not in gi],
            })
    result = {
        'scope': 'Three hash-first Chinese non-adversarial cases from each graph-only exact, base-only exact, and unchanged exact-status stratum; graph ID selected and final context IDs changed; no new model calls',
        'selection_seed': SEED, 'previously_audited_exclusions': sorted(PREVIOUSLY_AUDITED),
        'eligible_stratum_counts': {k: len(v) for k, v in bins.items()},
        'input_paths': {k: str(v) for k, v in paths.items()},
        'input_sha256': {k: sha(v) for k, v in paths.items()},
        'cases': cases,
        'limits': ['Stratified sample is not an estimate of full graph effect.',
                   'Historical arms are separate runs; answer and judge variation remains.',
                   'QA evidence may be incomplete or incorrect and requires bilingual/manual audit.'],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'eligible_strata': result['eligible_stratum_counts'],
                      'selected': [c['case_id'] for c in cases]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
