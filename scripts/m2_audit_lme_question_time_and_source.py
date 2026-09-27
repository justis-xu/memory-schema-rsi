#!/usr/bin/env python3
"""Read-only paired LongMemEval-S audit of question timestamps and fixed source sample."""

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/xu/git/memory-prompt/eval-datasets/longmemeval-zh')
EN = DATA / 'longmemeval_s_cleaned.json'
ZH = DATA / 'longmemeval_s_cleaned_zh.json'
OUT = ROOT / 'results/analysis/m2_lme_question_time_and_source_20260927.json'
FMT = '%Y/%m/%d (%a) %H:%M'


def parse(value):
    return datetime.strptime(value, FMT)


def time_audit(items):
    counts = Counter()
    affected = []
    for q in items:
        counts['questions'] += 1
        kind = q['question_type']
        counts[f'type_{kind}'] += 1
        query_time = parse(q['question_date'])
        later = [(sid, stamp) for sid, stamp in zip(q['haystack_session_ids'], q['haystack_dates'])
                 if parse(stamp) > query_time]
        answers = set(q['answer_session_ids'])
        answer_later = [(sid, stamp) for sid, stamp in later if sid in answers]
        if later:
            counts['questions_with_later_haystack'] += 1
            counts['later_haystack_sessions'] += len(later)
            counts[f'{kind}_questions_with_later_haystack'] += 1
        if answer_later:
            counts['questions_with_later_answer_session'] += 1
            counts['later_answer_sessions'] += len(answer_later)
            counts[f'{kind}_questions_with_later_answer_session'] += 1
            affected.append({'question_id': q['question_id'], 'question_type': kind,
                             'question_date': q['question_date'],
                             'later_answer_sessions': [
                                 {'session_id': sid, 'date': stamp} for sid, stamp in answer_later]})
        if any(parse(stamp).date() > query_time.date() for _, stamp in later):
            counts['questions_with_later_calendar_day'] += 1
    return dict(counts), affected


def answer_sources(q):
    positions = {sid: i for i, sid in enumerate(q['haystack_session_ids'])}
    result = []
    for sid in q['answer_session_ids']:
        i = positions[sid]
        marked = [{'turn_index': j, 'role': turn['role'], 'text': turn['content']}
                  for j, turn in enumerate(q['haystack_sessions'][i])
                  if turn.get('has_answer') is True]
        result.append({'session_id': sid, 'date': q['haystack_dates'][i], 'marked_turns': marked})
    return result


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    english = json.loads(EN.read_text())
    chinese = json.loads(ZH.read_text())
    by_id = {q['question_id']: q for q in english}
    assert len(english) == 500 and len(chinese) == 470 and len(by_id) == 500
    assert all(q['question_id'] in by_id for q in chinese)
    en_counts, en_affected = time_audit(english)
    zh_counts, zh_affected = time_audit(chinese)
    sample = []
    for kind in ('temporal-reasoning', 'knowledge-update'):
        for shortened, take in ((True, 4), (False, 2)):
            pool = [q for q in chinese if q['question_type'] == kind and
                    (len(q['haystack_session_ids']) < len(by_id[q['question_id']]['haystack_session_ids'])) == shortened]
            pool.sort(key=lambda q: hashlib.sha256(q['question_id'].encode()).hexdigest())
            for zh in pool[:take]:
                en = by_id[zh['question_id']]
                assert en['answer_session_ids'] == zh['answer_session_ids']
                assert en['question_date'] == zh['question_date']
                removed = [sid for sid in en['haystack_session_ids'] if sid not in zh['haystack_session_ids']]
                sample.append({'question_id': zh['question_id'], 'question_type': kind,
                               'has_removed_background': shortened, 'removed_background_ids': removed,
                               'question_date': zh['question_date'],
                               'english': {'question': en['question'], 'answer': en['answer'],
                                           'answer_sources': answer_sources(en)},
                               'chinese': {'question': zh['question'], 'answer': zh['answer'],
                                           'answer_sources': answer_sources(zh)}})
    assert len(sample) == 12
    report = {'scope': 'English 500 and Chinese 470 LongMemEval-S; literal timestamp comparison and fixed hash-order 12-case bilingual source sample; no model calls',
              'input_files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (EN, ZH)},
              'english_timing_counts': en_counts, 'chinese_timing_counts': zh_counts,
              'chinese_later_answer_cases': zh_affected,
              'english_later_answer_cases': en_affected,
              'fixed_source_sample': sample,
              'limits': ['Later timestamps all require interpretation of question_date time semantics; a same-day later session is not necessarily later by the benchmark intended date-only granularity.',
                         'has_answer marks source turns, not semantic entailment of the gold answer.',
                         'The 12 cases are a reproducible diagnostic sample, not an estimate of error prevalence.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'english': en_counts, 'chinese': zh_counts,
                      'sample_ids': [x['question_id'] for x in sample]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
