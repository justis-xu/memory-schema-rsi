#!/usr/bin/env python3
"""Second deterministic preference packet; read sources and archived runs only."""
import hashlib
import json
from m2_sample_preference_relation_strength import ROOT, SCREEN, SEED, PATHS
from m2_audit_graph_positive_mechanisms import source_packet, jsonl, context

OUT = ROOT / 'results/analysis/m2_preference_source_channels_sample_20260928.json'


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    screen = json.loads(SCREEN.read_text())
    eligible = [r['case_id'] for r in screen['screened']
                if not r['all_valid_evidence_other_speaker'] and not r['missing_evidence_ids']]
    ranked = sorted(eligible, key=lambda c: hashlib.sha256((SEED + c).encode()).hexdigest())
    excluded = {'locomo_conv-43_qa96': 'Earlier theme-song/film source bridge audit'}
    selected = [c for c in ranked[10:] if c not in excluded][:10]
    assert len(eligible) == 43 and len(selected) == 10
    ds = {lang: {s['sample_id']: s for s in json.loads(PATHS[lang + '_data'].read_text())}
          for lang in ('english', 'chinese')}
    runs = {k: jsonl(PATHS[k]) for k in ('base', 'graph', 'base_grade', 'graph_grade')}
    cases = []
    for cid in selected:
        conv, idx = cid.removeprefix('locomo_').split('_qa')
        source = {lang: source_packet(ds[lang][conv], int(idx)) for lang in ds}
        history = {}
        for arm in ('base', 'graph'):
            if cid in runs[arm] and cid in runs[arm + '_grade']:
                history[arm] = {'answer': runs[arm][cid]['predicted_answer'],
                                'context': context(runs[arm][cid]),
                                'grade': runs[arm + '_grade'][cid]['final']}
        cases.append({'case_id': cid, 'source': source, 'historical_runs': history})
    previous = json.loads((ROOT / 'results/analysis/m2_preference_relation_strength_sample_20260928.json').read_text())
    overlaps = []
    for c in cases:
        conv = c['case_id'].split('_qa')[0]
        evidence = {e['dia_id'] for e in c['source']['chinese']['evidence']}
        for p in previous['cases']:
            if p['case_id'].split('_qa')[0] != conv:
                continue
            shared = evidence & {e['dia_id'] for e in p['source']['chinese']['evidence']}
            if shared:
                overlaps.append({'case_id': c['case_id'], 'previous_case_id': p['case_id'],
                                 'shared_designated_evidence': sorted(shared)})
    result = {'scope': 'Second fixed source review, not a population or causal estimate; zero model calls',
              'selection': {'seed': SEED, 'ranked_eligible_ids': ranked,
                            'previous_first_ten': ranked[:10], 'explicit_exclusions': excluded,
                            'selected_ids': selected},
              'input_paths': {k: str(v) for k, v in PATHS.items()},
              'input_sha256': {**{k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in PATHS.items()},
                               'screen': hashlib.sha256(SCREEN.read_bytes()).hexdigest()},
              'cases': cases, 'designated_evidence_overlap_with_previous_batch': overlaps,
              'limits': ['Windows include three turns on either side, not full-conversation semantic audit.',
                         'Questions share source sessions and are not independent samples.',
                         'Historical runs are separate; answer differences are not isolated Graph effects.',
                         'Manual support judgments must be read separately from raw packets.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected_ids': selected, 'overlaps': overlaps}, ensure_ascii=False))


if __name__ == '__main__':
    main()
