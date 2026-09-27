#!/usr/bin/env python3
"""Fixed sample of Chinese base-only exact flips after graph candidates enter context."""

import hashlib
import json
from pathlib import Path

from m2_audit_graph_positive_mechanisms import PATHS, context, jsonl, source_packet


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_graph_negative_fact_loss_sample_20260928.json'
SEED = 'm2_graph_harm_sample_20260928|'
SAMPLE_SIZE = 12
PREVIOUSLY_AUDITED = {
    'locomo_conv-47_qa96', 'locomo_conv-48_qa39', 'locomo_conv-43_qa25',
    'locomo_conv-41_qa86', 'locomo_conv-30_qa80',
}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    base, graph, bg, gg = (jsonl(PATHS[k]) for k in ('base', 'graph', 'base_grade', 'graph_grade'))
    datasets = {lang: {s['sample_id']: s for s in json.loads(PATHS[lang + '_data'].read_text())}
                for lang in ('english', 'chinese')}
    eligible = []
    for cid in sorted(base.keys() & graph.keys() & bg.keys() & gg.keys()):
        if cid in PREVIOUSLY_AUDITED or bg[cid]['category'] == 'adversarial':
            continue
        if bg[cid]['final'] != 'exact' or gg[cid]['final'] == 'exact':
            continue
        bc, gc = context(base[cid]), context(graph[cid])
        if {m['id'] for m in bc} == {m['id'] for m in gc}:
            continue
        if not any(m['route'] == 'graph' for m in gc):
            continue
        eligible.append(cid)
    assert len(eligible) == 45
    selected = sorted(eligible, key=lambda cid: hashlib.sha256((SEED + cid).encode()).hexdigest())[:SAMPLE_SIZE]
    cases = []
    for cid in selected:
        b, g = base[cid], graph[cid]
        bc, gc = context(b), context(g)
        b_ids, g_ids = {m['id'] for m in bc}, {m['id'] for m in gc}
        conv, index = cid.removeprefix('locomo_').split('_qa')
        source = {lang: source_packet(datasets[lang][conv], int(index)) for lang in ('english', 'chinese')}
        assert [x['dia_id'] for x in source['english']['evidence']] == [x['dia_id'] for x in source['chinese']['evidence']]
        cases.append({'case_id': cid, 'source': source,
                      'base_answer': b['predicted_answer'], 'graph_answer': g['predicted_answer'],
                      'base_grade': bg[cid]['final'], 'graph_grade': gg[cid]['final'],
                      'base_judge': bg[cid].get('judge'), 'graph_judge': gg[cid].get('judge'),
                      'base_context': bc, 'graph_context': gc,
                      'gained': [m for m in gc if m['id'] not in b_ids],
                      'lost': [m for m in bc if m['id'] not in g_ids]})
    report = {
        'scope': 'Fixed hash 12/45 Chinese base-only exact flips with graph candidate in final context and changed IDs; historical independent runs; no new calls',
        'selection': {'seed': SEED, 'size': SAMPLE_SIZE, 'eligible_count': len(eligible),
                      'eligible_ids': eligible, 'selected_ids': selected,
                      'excluded_prior_cases': sorted(PREVIOUSLY_AUDITED)},
        'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in PATHS.items()},
        'cases': cases,
        'limits': ['QA evidence may omit or misstate supporting turns; check source and nearby dialogue.',
                   'Separate runs confound context change with answer and judge variation.',
                   'Twelve score-selected cases cannot estimate overall graph harm.'],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected_ids': selected, 'count': len(cases)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
