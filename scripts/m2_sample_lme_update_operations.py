#!/usr/bin/env python3
"""Fixed five Chinese knowledge-update cases; bilingual source sessions only."""
import gc
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/xu/git/eval-datasets/longmemeval-zh')
OUT = ROOT / 'results/analysis/m2_lme_update_operations_20260928.json'
SEED = 'm2_lme_update_source_20260928|'
PATHS = {'chinese': DATA / 'longmemeval_s_cleaned_zh.json',
         'english': DATA / 'longmemeval_s_cleaned.json'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def packet(row):
    sources = []
    for sid in row['answer_session_ids']:
        idx = row['haystack_session_ids'].index(sid)
        sources.append({'session_id': sid, 'date': row['haystack_dates'][idx],
                        'turns': [{'index': i, **t} for i, t in enumerate(row['haystack_sessions'][idx])]})
    return {k: row[k] for k in ('question_id', 'question_type', 'question', 'answer', 'question_date')} | {
        'answer_sessions': sources, 'background_session_count': len(row['haystack_session_ids'])}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    zh = json.loads(PATHS['chinese'].read_text())
    eligible = [r['question_id'] for r in zh if r['question_type'] == 'knowledge-update']
    ranked = sorted(eligible, key=lambda qid: hashlib.sha256((SEED + qid).encode()).hexdigest())
    selected = ranked[:5]
    previous_path = ROOT / 'results/analysis/m2_lme_question_time_and_source_20260927.json'
    previous = {r['question_id'] for r in json.loads(previous_path.read_text())['fixed_source_sample']}
    assert len(eligible) == 68 and not set(selected) & previous
    cases = {r['question_id']: {'chinese': packet(r)} for r in zh if r['question_id'] in selected}
    del zh
    gc.collect()
    en = json.loads(PATHS['english'].read_text())
    for r in en:
        if r['question_id'] in cases:
            cases[r['question_id']]['english'] = packet(r)
    del en
    gc.collect()
    for value in cases.values():
        assert {s['session_id'] for s in value['chinese']['answer_sessions']} == {
            s['session_id'] for s in value['english']['answer_sessions']}
    # Frozen historical judgments. The f685340e "weekly Sundays" inference was
    # later corrected by m2_lme_clause_support_zh.md; keep this map unchanged so
    # the archived packet remains reproducible.
    judgments = {
        '031748ae': {'operation': 'in_session_correction_then_cross_session_update',
                     'required_values': ['initially leads 4 engineers', 'now leads 5 engineers'],
                     'risk': 'Earlier uncorrected group-size statements and assistant including-self inference are distinct from user correction.'},
        '0ddfec37': {'operation': 'historical_window_despite_later_increment',
                     'required_values': ['15 in first three months'],
                     'risk': 'Later 20 is an added count over a later period, not a replacement value for the first three months.'},
        'f685340e': {'operation': 'old_and_current_schedule_with_context',
                     'required_values': ['weekly Sundays previously', 'every other Sunday now'],
                     'risk': 'Deleting old schedule loses half of the requested comparison.'},
        '50635ada': {'operation': 'previous_status_query',
                     'required_values': ['Premier Silver before current Premier Gold'],
                     'risk': 'Latest status alone cannot answer previous status. Earlier source says eligible for Silver, not explicitly attained; gold assumes previous membership status.',
                     'source_strength': 'Earlier Silver eligibility is explicit; attained previous Silver status needs review.'},
        '71315a70': {'operation': 'latest_cumulative_estimate_same_project',
                     'required_values': ['10-12 hours'],
                     'risk': '5-6 and 10-12 are cumulative estimates for the same project; do not sum them or make the range exact.'},
    }
    report = {'scope': 'Fixed five of 68 retained Chinese knowledge-update cases; no model calls',
              'selection': {'seed': SEED, 'ranked_eligible_ids': ranked, 'selected_ids': selected,
                            'previous_source_sample_overlap': []},
              'cases': [{'question_id': qid, 'source': cases[qid],
                         'manual_operation_judgment': judgments[qid]} for qid in selected],
              'input_sha256': {k: sha(p) for k, p in PATHS.items()},
              'input_paths': {k: str(p) for k, p in PATHS.items()},
              'limits': ['Answer-session IDs and has_answer are oracle annotations for analysis, not runtime retrieval inputs.',
                         'Only designated source sessions were semantically reviewed, not all background sessions.',
                         'Operation categories are manual judgments, not an automatic temporal resolver.',
                         'No Chinese LongMemEval model results or component effects were measured.',
                         'Five selected questions cannot estimate category frequencies or benchmark error rates.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected_ids': selected, 'previous_overlap': []}, ensure_ascii=False))


if __name__ == '__main__':
    main()
