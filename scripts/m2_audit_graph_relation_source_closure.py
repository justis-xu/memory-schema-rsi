#!/usr/bin/env python3
"""Source closure packets for four targeted historical Chinese positive flips."""
import hashlib
import json
from pathlib import Path
from m2_audit_graph_positive_mechanisms import PATHS, context, jsonl, source_packet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_graph_relation_source_closure_20260928.json'
CASE_IDS = ['locomo_conv-43_qa166', 'locomo_conv-48_qa172',
            'locomo_conv-47_qa104', 'locomo_conv-50_qa99']


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    b, g, bg, gg = (jsonl(PATHS[k]) for k in ('base', 'graph', 'base_grade', 'graph_grade'))
    ds = {lang: {s['sample_id']: s for s in json.loads(PATHS[lang + '_data'].read_text())}
          for lang in ('english', 'chinese')}
    cases = []
    for cid in CASE_IDS:
        conv, idx = cid.removeprefix('locomo_').split('_qa')
        bc, gc = context(b[cid]), context(g[cid])
        bi, gi = {m['id'] for m in bc}, {m['id'] for m in gc}
        assert bg[cid]['final'] != 'exact' and gg[cid]['final'] == 'exact'
        assert any(m['route'] == 'graph' for m in gc) and bi != gi
        cases.append({'case_id': cid,
                      'source': {lang: source_packet(ds[lang][conv], int(idx)) for lang in ds},
                      'base_answer': b[cid]['predicted_answer'], 'graph_answer': g[cid]['predicted_answer'],
                      'base_grade': bg[cid]['final'], 'graph_grade': gg[cid]['final'],
                      'base_judge': bg[cid].get('judge'), 'graph_judge': gg[cid].get('judge'),
                      'base_context': bc, 'graph_context': gc,
                      'gained': [m for m in gc if m['id'] not in bi],
                      'lost': [m for m in bc if m['id'] not in gi]})
    # Full lexical source screen, with no claim that keyword matching establishes facts.
    john_potter_screen = {}
    for lang, names in [('english', ('harry potter',)), ('chinese', ('哈利',))]:
        s = ds[lang]['conv-43']
        john_potter_screen[lang] = [
            {'session': k, 'date': s['conversation'][k + '_date_time'], **t}
            for k, turns in s['conversation'].items() if isinstance(turns, list)
            for t in turns if any(n in t.get('text', '').lower() for n in names)]
    same_evidence_qa = {lang: [{'qa_index': i, **q} for i, q in enumerate(ds[lang]['conv-43']['qa'])
                              if 'D27:19' in q['evidence']] for lang in ds}
    report = {'scope': 'Four cases chosen after screening unaudited positives for identity, relation, plan and location changes; not a representative sample; zero new model calls',
              'case_ids': CASE_IDS, 'input_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in PATHS.items()},
              'cases': cases, 'conv43_potter_keyword_screen': john_potter_screen,
              'conv43_qa_sharing_D27_19': same_evidence_qa,
              'limits': ['Independent runs cannot isolate graph causality.',
                         'Keyword screen is not a complete semantic proof of absence.',
                         'Historical memories may not match the current Chinese library.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'cases': CASE_IDS, 'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
