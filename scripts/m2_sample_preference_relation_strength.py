#!/usr/bin/env python3
"""Fixed source-strength sample from complete, unflagged literal preference rows."""
import hashlib
import json
from pathlib import Path
from m2_audit_graph_positive_mechanisms import PATHS, context, jsonl, source_packet

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / 'results/analysis/m2_preference_source_subject_screen_20260928.json'
OUT = ROOT / 'results/analysis/m2_preference_relation_strength_sample_20260928.json'
SEED = 'm2_preference_relation_sample_20260928|'


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    screen = json.loads(SCREEN.read_text())
    eligible = [r['case_id'] for r in screen['screened']
                if not r['all_valid_evidence_other_speaker'] and not r['missing_evidence_ids']]
    assert len(eligible) == 43
    selected = sorted(eligible, key=lambda c: hashlib.sha256((SEED + c).encode()).hexdigest())[:10]
    ds = {lang: {s['sample_id']: s for s in json.loads(PATHS[lang + '_data'].read_text())}
          for lang in ('english', 'chinese')}
    runs = {k: jsonl(PATHS[k]) for k in ('base', 'graph', 'base_grade', 'graph_grade')}
    cases = []
    for cid in selected:
        conv, idx = cid.removeprefix('locomo_').split('_qa')
        sources = {lang: source_packet(ds[lang][conv], int(idx)) for lang in ds}
        history = {}
        for arm in ('base', 'graph'):
            if cid not in runs[arm] or cid not in runs[arm + '_grade']:
                continue
            history[arm] = {'answer': runs[arm][cid]['predicted_answer'],
                            'context': context(runs[arm][cid]),
                            'grade': runs[arm + '_grade'][cid]['final'],
                            'judge': runs[arm + '_grade'][cid].get('judge')}
        cases.append({'case_id': cid, 'source': sources, 'historical_runs': history})
    duplicate_groups = {}
    for c in cases:
        q = c['source']['chinese']
        key = json.dumps([q['question'], q['answer'], [e['dia_id'] for e in q['evidence']]], ensure_ascii=False)
        duplicate_groups.setdefault(key, []).append(c['case_id'])
    report = {'scope': 'Fixed hash ten of 43 complete, unflagged literal-target Chinese preference questions; independent historical runs; zero new model calls',
              'selection': {'seed': SEED, 'eligible_ids': sorted(eligible), 'selected_ids': selected},
              'cases': cases, 'exact_question_gold_evidence_duplicates_in_sample': [v for v in duplicate_groups.values() if len(v) > 1],
              'input_sha256': {**{k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in PATHS.items()},
                               'screen': hashlib.sha256(SCREEN.read_bytes()).hexdigest()},
              'limits': ['Sample is not an overall error or effect estimate.',
                         'Same question/source appears twice and is not independent evidence.',
                         'Current database and source-grounded automatic extraction were not tested.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected_ids': selected, 'duplicates': report['exact_question_gold_evidence_duplicates_in_sample']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
