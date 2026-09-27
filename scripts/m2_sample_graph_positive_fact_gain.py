#!/usr/bin/env python3
"""Fixed hash sample from the 57 historical Chinese graph-only exact flips."""

import hashlib
import json
from pathlib import Path

from m2_audit_graph_positive_mechanisms import PATHS, context, jsonl, source_packet


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'results/analysis/m2_graph_positive_mechanisms_20260927.json'
OUT = ROOT / 'results/analysis/m2_graph_positive_fact_gain_sample_20260928.json'
SEED = 'm2_graph_sample_20260928|'
SAMPLE_SIZE = 12


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    previous = json.loads(PREVIOUS.read_text())
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == previous['input_sha256'][name]
               for name, path in PATHS.items())
    eligible = previous['eligible_case_ids']
    assert len(eligible) == len(set(eligible)) == 57
    selected = sorted(eligible, key=lambda cid: hashlib.sha256((SEED + cid).encode()).hexdigest())[:SAMPLE_SIZE]
    base, graph, bg, gg = (jsonl(PATHS[k]) for k in ('base', 'graph', 'base_grade', 'graph_grade'))
    datasets = {lang: {s['sample_id']: s for s in json.loads(PATHS[lang + '_data'].read_text())}
                for lang in ('english', 'chinese')}
    cases = []
    for cid in selected:
        b, g = base[cid], graph[cid]
        bc, gc = context(b), context(g)
        b_ids, g_ids = {m['id'] for m in bc}, {m['id'] for m in gc}
        assert bg[cid]['final'] != 'exact' and gg[cid]['final'] == 'exact'
        assert b_ids != g_ids and any(m['route'] == 'graph' for m in gc)
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
    result = {'scope': 'Fixed SHA256 sample without inspecting content, 12 of 57 eligible historical Chinese graph-only exact flips; no new model calls',
              'selection': {'seed': SEED, 'size': SAMPLE_SIZE, 'eligible_count': len(eligible),
                            'selected_ids': selected},
              'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in PATHS.items()},
              'cases': cases,
              'limits': ['Gold evidence can omit neighboring or unrelated source turns.',
                         'Separate historical answer runs and changed contexts do not identify graph candidate causal effect.',
                         'Twelve cases are too few for a precise rate estimate.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected_ids': selected, 'n': len(cases)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
