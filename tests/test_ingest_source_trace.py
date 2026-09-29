"""Input identity, gold exclusion and write/trace failure boundaries."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from schema_rsi.benchmarks.base import BenchmarkCase
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from schema_rsi.memory.base import MemoryRecord


class Backend:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append(copy.deepcopy((user_id, messages, metadata)))
        if self.fail:
            raise RuntimeError('backend failure')
        return [MemoryRecord('m1', '工程师人数已更正', {'user_id': user_id})]


def case():
    return BenchmarkCase('c', 'longmemeval_s', [{'session_id': 's', 'date': '2023/05/11 (Thu) 02:02',
        'turns': [{'role': 'user', 'content': '先说5人', 'has_answer': True},
                  {'role': 'user', 'content': '纠正为4人', 'has_answer': True}]}],
        'gold question sentinel', 'gold answer sentinel', evidence=['gold evidence sentinel'],
        metadata={'raw': {'api_key': 'credential sentinel'}})


def run(c, backend, sink=None, max_turns=None, **trace_ids):
    pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
    pipeline.backend = backend
    return pipeline.ingest_case(c, user_id='u', source_trace_sink=sink,
                                max_turns=max_turns, **trace_ids)


def test_source_trace_preserves_request_and_excludes_gold():
    c = case()
    plain, traced, events = Backend(), Backend(), []
    assert run(c, plain) == run(c, traced, events.append) == 1
    assert plain.calls == traced.calls
    batch = events[0]['batch']
    assert [t['source_ref'] for t in batch['source_turns']] == ['turn:0', 'turn:1']
    assert batch['input_messages'] == traced.calls[0][1]
    assert events[1]['returned_record_ids'] == ['m1']
    assert events[1]['fact_support_status'] == 'not_provided'
    assert 'sentinel' not in str(events) and 'has_answer' not in str(events)


def test_batch_identity_changes_on_source_and_owner_but_not_gold():
    c, original = case(), []
    run(c, Backend(), original.append)
    base = original[0]['batch']['batch_id']
    c.answer, c.question, c.evidence = 'different', 'different', ['different']
    changed = []
    run(c, Backend(), changed.append)
    assert changed[0]['batch']['batch_id'] == base
    c.history[0]['turns'][1]['content'] = '纠正为6人'
    changed = []
    run(c, Backend(), changed.append)
    assert changed[0]['batch']['batch_id'] != base
    from schema_rsi.evaluation.source_trace import build_source_batch
    b = original[0]['batch']
    owner = build_source_batch(benchmark=c.benchmark, user_id='other', session=c.history[0],
                              turns=c.history[0]['turns'], messages=b['input_messages'])
    assert owner['batch_id'] != changed[0]['batch']['batch_id']


def test_same_batch_is_distinguishable_across_runs_and_attempts():
    runs = []
    for run_id, attempt_id in [('run-a', 'first'), ('run-a', 'retry'), ('run-b', 'first')]:
        backend, events = Backend(), []
        run(case(), backend, events.append, trace_run_id=run_id,
            trace_attempt_id=attempt_id)
        assert [(e['run_id'], e['attempt_id']) for e in events] == [(run_id, attempt_id)] * 2
        assert events[0]['batch']['batch_id'] == events[1]['batch_id']
        runs.append((backend.calls, events[0]['batch']['batch_id']))
    assert runs[0][0] == runs[1][0] == runs[2][0]
    assert runs[0][1] == runs[1][1] == runs[2][1]


@pytest.mark.parametrize('trace_ids', [
    {'trace_run_id': ''}, {'trace_attempt_id': '  '}, {'trace_run_id': 7},
])
def test_invalid_trace_identity_fails_before_backend_call(trace_ids):
    backend = Backend()
    with pytest.raises(ValueError, match='must be a nonempty string'):
        run(case(), backend, [].append, **trace_ids)
    assert backend.calls == []


def test_truncation_and_sink_mutation_do_not_change_backend_input():
    c, b, events = case(), Backend(), []
    def sink(event):
        events.append(copy.deepcopy(event))
        if event['event'] == 'batch_input':
            event['batch']['source_turns'][0]['content'] = 'mutated'
            event['batch']['input_messages'][1]['content'] = 'mutated'
    run(c, b, sink, max_turns=1)
    assert len(events[0]['batch']['source_turns']) == 1
    assert b.calls[0][1][1]['content'] == '先说5人'
    assert c.history[0]['turns'][0]['content'] == '先说5人'


def test_backend_failure_is_unknown_write_completion():
    events = []
    assert run(case(), Backend(fail=True), events.append) == 0
    assert events[1]['status'] == 'backend_error'
    assert events[1]['write_completion'] == 'unknown'
    assert 'backend failure' not in str(events)


@pytest.mark.parametrize('failing_event,expected_calls', [('batch_input', 0), ('batch_outcome', 1)])
def test_sink_failure_propagates_without_backend_retry(failing_event, expected_calls):
    b = Backend()
    def sink(event):
        if event['event'] == failing_event:
            raise OSError('sink unavailable')
    with pytest.raises(OSError, match='sink unavailable'):
        run(case(), b, sink)
    assert len(b.calls) == expected_calls
