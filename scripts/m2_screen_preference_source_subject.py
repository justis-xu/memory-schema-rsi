#!/usr/bin/env python3
"""Literal Chinese preference screen; speaker mismatch flags require manual context."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from m2_audit_graph_positive_mechanisms import PATHS, source_packet, jsonl, context

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_preference_source_subject_screen_20260928.json'
TERMS = ['最喜欢', '最爱', '喜欢什么', '喜欢哪些']


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    datasets = {lang: {s['sample_id']: s for s in json.loads(PATHS[lang + '_data'].read_text())}
                for lang in ('english', 'chinese')}
    counts = Counter()
    screened, candidates = [], []
    for conv, sample in datasets['chinese'].items():
        turns = {t['dia_id']: t for v in sample['conversation'].values()
                 if isinstance(v, list) for t in v}
        names = sorted({t['speaker'] for t in turns.values()})
        for i, qa in enumerate(sample['qa']):
            if qa['category'] == 5 or not any(term in qa['question'] for term in TERMS):
                continue
            counts['non_adversarial_lexical_preference_questions'] += 1
            targets = [name for name in names if name in qa['question']]
            if len(targets) != 1:
                counts['excluded_non_unique_literal_target'] += 1
                continue
            counts['unique_literal_target_questions'] += 1
            missing = [eid for eid in qa['evidence'] if eid not in turns]
            evidence = [{k: turns[eid].get(k) for k in ('dia_id', 'speaker', 'text')}
                        for eid in qa['evidence'] if eid in turns]
            # Never classify partial/missing pointers as an all-other-speaker flag.
            flag = bool(evidence) and not missing and all(t['speaker'] != targets[0] for t in evidence)
            row = {'case_id': f'locomo_{conv}_qa{i}', 'qa_index': i, 'question': qa['question'],
                   'gold': qa['answer'], 'target': targets[0], 'evidence': evidence,
                   'missing_evidence_ids': missing, 'all_valid_evidence_other_speaker': flag}
            screened.append(row)
            counts['rows_with_missing_evidence'] += bool(missing)
            if flag:
                counts['all_valid_evidence_other_speaker'] += 1
                candidates.append({**row, 'source': {lang: source_packet(datasets[lang][conv], i)
                                                    for lang in datasets}})
    runs = {name: jsonl(PATHS[name]) for name in ('base', 'graph', 'base_grade', 'graph_grade')}
    for candidate in candidates:
        cid = candidate['case_id']
        candidate['historical_runs'] = {
            arm: {'answer': runs[arm][cid]['predicted_answer'],
                  'grade': runs[arm + '_grade'][cid]['final'],
                  'judge': runs[arm + '_grade'][cid].get('judge'),
                  'context': context(runs[arm][cid])}
            for arm in ('base', 'graph') if cid in runs[arm] and cid in runs[arm + '_grade']}
    assert len(screened) == 46 and len(candidates) == 2
    report = {'scope': 'All non-adversarial Chinese QA matching fixed preference substrings and exactly one literal conversation speaker; not all preferences or all dataset QA',
              'rule': {'terms': TERMS, 'flag': 'All evidence IDs valid and every evidence speaker differs from the single literal target'},
              'counts': dict(counts), 'screened': screened, 'candidates': candidates,
              'input_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in PATHS.items()},
              'limits': ['Speaker mismatch is a diagnostic flag, never a gold-error verdict.',
                         'Questions with aliases, implicit subjects, or other preference wording are excluded.',
                         'Unflagged questions may still have wrong relations, incomplete gold, or source problems.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
