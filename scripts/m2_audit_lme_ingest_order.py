#!/usr/bin/env python3
"""Actual adapter/ingest with recording backend; no Mem0/Chroma initialization."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from schema_rsi.benchmarks.base import parse_session_date
from schema_rsi.benchmarks.longmemeval import _parse_item
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from m2_sample_lme_update_operations import PATHS, sha, ROOT

OUT = ROOT / 'results/analysis/m2_lme_ingest_order_20260928.json'
SAMPLE = ROOT / 'results/analysis/m2_lme_update_operations_20260928.json'


class Recorder:
    def __init__(self):
        self.calls = []

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append({'user_id': user_id, 'messages': messages, 'metadata': metadata})
        return []


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    sample = json.loads(SAMPLE.read_text())
    selected = set(sample['selection']['selected_ids'])
    ds = json.loads(PATHS['chinese'].read_text())
    order_rows = []
    for r in ds:
        dates = [parse_session_date(d) for d in r['haystack_dates']]
        assert all(dates)
        inversions = [i for i in range(1, len(dates)) if dates[i] < dates[i - 1]]
        day_inversions = [i for i in inversions if dates[i][:10] < dates[i - 1][:10]]
        if inversions:
            order_rows.append({'question_id': r['question_id'], 'question_type': r['question_type'],
                               'adjacent_timestamp_inversion_count': len(inversions),
                               'adjacent_day_inversion_count': len(day_inversions)})
    probes = []
    for i, r in enumerate(ds):
        if r['question_id'] not in selected:
            continue
        case = _parse_item(r, i)
        recorder = Recorder()
        # Bypass constructor: it initializes retrieval clients and Chroma.
        # Only the real ingest method is called, with an explicit isolated user ID.
        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        pipeline.backend = recorder
        pipeline.ingest_case(case, user_id='offline_recording:' + case.case_id)
        assert len(recorder.calls) == len(case.history)
        answer_order = []
        traces = []
        for pos, (session, call) in enumerate(zip(case.history, recorder.calls)):
            original = r['haystack_sessions'][pos]
            messages = call['messages']
            body = [m for m in messages if m['role'] != 'system']
            assert body == [{'role': t['role'], 'content': t['content']} for t in original]
            assert call['metadata']['session_id'] == r['haystack_session_ids'][pos]
            assert call['metadata']['session_date'] == r['haystack_dates'][pos]
            trace = {'call_index': pos, 'session_id': session['session_id'],
                     'date': session['date'], 'message_count': len(messages),
                     'date_anchor_present': bool(messages and messages[0]['role'] == 'system'),
                     'metadata_keys': sorted(call['metadata']),
                     'message_keys': sorted({k for m in messages for k in m}),
                     'ordered_message_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}
            if session['session_id'] in r['answer_session_ids']:
                answer_order.append({'call_index': pos, 'session_id': session['session_id'], 'date': session['date']})
                trace['source_position_checks'] = [
                    {'raw_turn_index': j, 'role': t['role'],
                     'content_sha256': hashlib.sha256(t['content'].encode()).hexdigest(),
                     'received_message_index': j + int(trace['date_anchor_present']),
                     'source_pointer_in_message': False, 'gold_annotation_in_message': False}
                    for j, t in enumerate(original)]
            traces.append(trace)
        probes.append({'question_id': case.case_id, 'session_calls': len(traces),
                       'history_turn_count': case.history_stats()['turns'],
                       'answer_source_call_order': answer_order,
                       'messages_preserve_role_content_and_order': True,
                       'calls_preserve_file_session_order': True,
                       'source_turn_ids_or_fact_links_sent': False, 'calls': traces})
    assert len(probes) == 5
    report = {'scope': 'Chinese 470-case file order plus actual adapter/ingest replay on five complete histories; recording backend only',
              'order_counts': {'cases_with_timestamp_inversions': len(order_rows),
                               'adjacent_timestamp_inversions': sum(r['adjacent_timestamp_inversion_count'] for r in order_rows),
                               'cases_with_day_inversions': sum(r['adjacent_day_inversion_count'] > 0 for r in order_rows),
                               'by_type': dict(Counter(r['question_type'] for r in order_rows))},
              'cases_with_inversions': order_rows, 'five_ingest_probes': probes,
              'input_sha256': {'chinese': sha(PATHS['chinese']), 'sample': sha(SAMPLE),
                               'adapter': sha(ROOT / 'src/schema_rsi/benchmarks/longmemeval.py'),
                               'pipeline': sha(ROOT / 'src/schema_rsi/evaluation/pipeline.py')},
              'limits': ['Same-day timestamp order semantics are unresolved; no claim of causal leakage or performance harm.',
                         'Recorder returns zero records and does not execute extraction, storage or retrieval.',
                         'Answer annotations select trace detail for offline inspection only; actual ingest receives no gold fields.',
                         'Ordered role/content remain reconstructible from frozen input and call position, but no explicit fact support IDs exist.',
                         'No production code or history ordering was changed.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'order_counts': report['order_counts'],
                      'probes': [{'id': p['question_id'], 'calls': p['session_calls'], 'turns': p['history_turn_count']} for p in probes]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
