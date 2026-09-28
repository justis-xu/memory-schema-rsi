"""Verify optional trace on real Chinese histories without model/store calls."""
import copy
import json
from schema_rsi.benchmarks.longmemeval import _parse_item
from schema_rsi.benchmarks.locomo import _parse_conversation
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from m2_sample_lme_update_operations import PATHS, ROOT, sha
from m2_audit_graph_positive_mechanisms import PATHS as LOCOMO_PATHS

OUT = ROOT / 'results/analysis/m2_ingest_source_trace_20260928.json'


class Recorder:
    def __init__(self):
        self.calls = []

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append(copy.deepcopy((user_id, messages, metadata)))
        return []


def replay(case, backend, sink=None):
    p = EvaluationPipeline.__new__(EvaluationPipeline)
    p.backend = backend
    return p.ingest_case(case, user_id='offline:' + case.case_id, source_trace_sink=sink)


def verify(case):
    plain, traced, events = Recorder(), Recorder(), []
    assert replay(case, plain) == replay(case, traced, events.append) == 0
    assert plain.calls == traced.calls
    assert len(events) == 2 * len(traced.calls)
    for i, call in enumerate(traced.calls):
        before, after = events[2 * i:2 * i + 2]
        b = before['batch']
        assert b['input_messages'] == call[1]
        assert after['batch_id'] == b['batch_id']
        assert after['status'] == 'backend_returned'
        assert after['returned_record_ids'] == []
        assert len(b['source_turns']) == len(case.history[i]['turns'])
        assert all('has_answer' not in t for t in b['source_turns'])
    return {'case_id': case.case_id, 'batch_count': len(traced.calls),
            'source_turn_count': sum(len(e['batch']['source_turns']) for e in events if e['event'] == 'batch_input'),
            'unchanged_backend_requests': True, 'matched_input_outcome_events': True,
            'batch_ids': [e['batch']['batch_id'] for e in events if e['event'] == 'batch_input']}, events


def main():
    if OUT.exists():
        raise SystemExit(f'refusing overwrite: {OUT}')
    sample = json.loads((ROOT / 'results/analysis/m2_lme_update_operations_20260928.json').read_text())
    ids = set(sample['selection']['selected_ids'])
    data = json.loads(PATHS['chinese'].read_text())
    rows = [verify(_parse_item(r, i))[0] for i, r in enumerate(data) if r['question_id'] in ids]
    del data
    source = next(r for r in json.loads(LOCOMO_PATHS['chinese_data'].read_text()) if r['sample_id'] == 'conv-42')
    row, events = verify(_parse_conversation(source, 0)[91])
    target = next(t for e in events if e['event'] == 'batch_input'
                  for t in e['batch']['source_turns'] if t.get('dia_id') == 'D9:14')
    assert target['image_query'] == 'fantasy novels dragon cover series'
    assert target['blip_caption'] and target['image_urls']
    result = {'scope': 'Five real Chinese LME histories and one full LoCoMo conversation with optional tracing; recording backend returns empty, no extraction',
              'longmemeval': rows, 'locomo': row, 'image_channel_example': target,
              'checks': {'all_backend_requests_unchanged': True, 'all_batch_events_paired': True,
                         'gold_has_answer_not_copied': True, 'locomo_image_fields_retained': True},
              'input_sha256': {'lme_chinese': sha(PATHS['chinese']), 'locomo_chinese': sha(LOCOMO_PATHS['chinese_data'])},
              'limits': ['Recorder outputs are empty; nonempty IDs and failure boundaries are covered by synthetic unit tests.',
                         'Batch association is not per-fact semantic support.',
                         'Caller must configure and persist the sink; existing runners do not enable it automatically.',
                         'No model calls, real database writes or answer effects measured.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'lme_batches': sum(r['batch_count'] for r in rows), 'lme_turns': sum(r['source_turn_count'] for r in rows),
                      'locomo_batches': row['batch_count'], 'locomo_turns': row['source_turn_count'], 'checks': result['checks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
