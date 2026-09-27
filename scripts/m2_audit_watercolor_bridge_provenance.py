#!/usr/bin/env python3
"""Trace archived watercolor Graph candidate to source and current Chinese Memory."""

import hashlib
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
EN = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10.json')
ZH = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json')
ARCHIVE = ROOT / 'results_zh/zhfull_fused_locomo_20260925_172126.jsonl'
DB = ROOT / 'data/vector_store_zh/chroma/chroma.sqlite3'
OUT = ROOT / 'results/analysis/m2_watercolor_bridge_provenance_20260927.json'
CASE = 'locomo_conv-49_qa85'
TARGET = 'dbc78f1e-fe6b-4b63-b279-d6cba1c6e2db'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    sample = next(s for s in json.loads(path.read_text()) if s['sample_id'] == 'conv-49')
    turns = sample['conversation']['session_1']
    return {'session_date': sample['conversation']['session_1_date_time'],
            'turns': [{k: t.get(k) for k in ('dia_id', 'speaker', 'text')}
                      for t in turns if t['dia_id'] in {'D1:14', 'D1:15', 'D1:16', 'D1:17'}]}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    row = next(r for r in map(json.loads, ARCHIVE.open()) if r['case_id'] == CASE)
    graph = next(m for m in row['graph_memories'] if m['id'] == TARGET)
    assert TARGET in row['metadata']['graph_context_ids']
    anchor = graph['anchor_memory_id']
    archived_anchor = None
    archived_target_vector = None
    for record in map(json.loads, ARCHIVE.open()):
        for mem in record['retrieved_memories']:
            if mem['id'] == anchor:
                archived_anchor = mem
            if mem['id'] == TARGET:
                archived_target_vector = mem
    assert archived_anchor and archived_target_vector
    assert archived_target_vector['metadata']['session_id'] == 'session_1'
    assert archived_anchor['metadata']['session_id'] == 'session_2'
    assert archived_target_vector['content'] == graph['content']
    assert archived_target_vector['metadata']['user_id'] == graph['user_id']
    connection = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    records = []
    for (mid,) in connection.execute("""
        SELECT e.embedding_id FROM embeddings e
        JOIN embedding_metadata m ON m.id=e.id
        WHERE m.key='data' AND m.string_value LIKE '%几年前开始画水彩画%'
    """):
        meta = {k: v for k, v in connection.execute(
            "SELECT key,string_value FROM embedding_metadata WHERE id=(SELECT id FROM embeddings WHERE embedding_id=?)",
            (mid,))}
        if meta.get('user_id') == 'zhfull:locomo:conv-49':
            records.append({'id': mid, 'content': meta.get('data'), 'metadata': meta})
    connection.close()
    assert len(records) == 1 and records[0]['metadata']['session_id'] == 'session_1'
    report = {
        'scope': 'Read-only old Graph/vector source trace and current Chinese Chroma metadata; no model/retrieval/graph calls',
        'paths': {'english_data': str(EN), 'chinese_data': str(ZH),
                  'archive': str(ARCHIVE), 'current_chroma': str(DB)},
        'sha256': {k: sha(v) for k, v in [('english_data', EN), ('chinese_data', ZH),
                                        ('archive', ARCHIVE), ('current_chroma', DB)]},
        'case_id': CASE, 'question': row['question'],
        'english_source': source(EN), 'chinese_source': source(ZH),
        'archived_graph_candidate': graph,
        'archived_vector_same_id': archived_target_vector,
        'archived_first_anchor_vector': archived_anchor,
        'current_chinese_same_relation_memories': records,
        'limits': [
            'Archived Graph via list is aggregated and anchor_memory_id stores the first anchor; it is not a per-edge traversal trace.',
            'Current Chinese memory has a new ID and is not the same frozen run as the archived graph candidate.',
            'Current memory presence does not establish current vector top-15 exposure or answer gain.',
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'old_candidate': TARGET, 'first_anchor': anchor,
                      'current_chinese_memory': records[0]['id']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
