"""The opt-in single-conversation runner persists paired source/output events."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from ingest_conv import file_sha256, ingest_with_trace, safe_ingest_config
from schema_rsi.benchmarks.base import BenchmarkCase
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from schema_rsi.memory.base import MemoryRecord


class Backend:
    def __init__(self):
        self.calls = []

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append(copy.deepcopy((user_id, messages, metadata)))
        return [MemoryRecord('m-1', '用户带领四名工程师',
                             {'session_id': metadata['session_id'],
                              'session_date': metadata['session_date'],
                              'event': 'ADD'})]


def pipeline(backend):
    result = EvaluationPipeline.__new__(EvaluationPipeline)
    result.backend = backend
    return result


def case():
    return BenchmarkCase('c', 'longmemeval_s',
                         [{'session_id': 's-1', 'date': '2023/05/11 (Thu) 02:02',
                           'turns': [{'role': 'user', 'content': '我带领四名工程师'}]}],
                         'gold question sentinel', 'gold answer sentinel')


def test_opt_in_trace_persists_frozen_source_and_returned_text(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{"中文":"源话"}')
    source_sha = file_sha256(source)
    plain, traced = Backend(), Backend()
    pipeline(plain).ingest_case(case(), user_id='offline:u')
    trace = tmp_path / 'events.jsonl'
    assert ingest_with_trace(pipeline(traced), case(), 'offline:u',
                             trace_path=trace, run_id='run-1', attempt_id='first',
                             source_path=source, source_sha256=source_sha,
                             config_sha256='config-fingerprint',
                             effective_config={'llm_model': 'offline'}) == 1
    assert plain.calls == traced.calls
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [event['event'] for event in events] == [
        'run_started', 'batch_input', 'batch_outcome', 'run_completed']
    assert events[0]['source_file_sha256'] == source_sha
    assert events[0]['effective_config'] == {'llm_model': 'offline'}
    assert events[1]['batch']['batch_id'] == events[2]['batch_id']
    assert events[1]['run_id'] == events[2]['run_id'] == 'run-1'
    assert events[2]['returned_records'][0]['content'] == '用户带领四名工程师'
    assert events[3]['source_file_unchanged'] is True
    assert 'gold question sentinel' not in str(events)
    assert 'gold answer sentinel' not in str(events)


def test_existing_trace_file_blocks_backend_before_write(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    trace = tmp_path / 'events.jsonl'
    trace.write_text('existing')
    backend = Backend()
    with pytest.raises(FileExistsError):
        ingest_with_trace(pipeline(backend), case(), 'offline:u',
                          trace_path=trace, run_id='run-1', attempt_id='first',
                          source_path=source, source_sha256=file_sha256(source),
                          config_sha256='config-fingerprint',
                          effective_config={'llm_model': 'offline'})
    assert backend.calls == []
    assert trace.read_text() == 'existing'


def test_effective_config_snapshot_excludes_credentials_and_raw_endpoints():
    cfg = {
        'llm': {'config': {'model': 'glm-test', 'temperature': 0.0,
                           'max_tokens': 4000, 'disable_thinking': True,
                           'api_key': 'secret-llm', 'openai_base_url': 'https://private-llm'}},
        'embedder': {'config': {'model': 'embedding-test',
                                'api_key': 'secret-embed',
                                'openai_base_url': 'https://private-embed'}},
        'vector_store': {'config': {'collection_name': 'isolated'}},
        'custom_instructions': '中文提取',
    }
    snapshot = safe_ingest_config(cfg)
    assert snapshot['llm_model'] == 'glm-test'
    assert snapshot['embedding_model'] == 'embedding-test'
    assert snapshot['collection_name'] == 'isolated'
    assert 'secret' not in str(snapshot) and 'https://' not in str(snapshot)
    assert '中文提取' not in str(snapshot)
