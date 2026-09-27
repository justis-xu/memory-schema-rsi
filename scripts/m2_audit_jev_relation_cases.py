#!/usr/bin/env python3
"""Read-only, source-bound Jev comparison for three fixed Chinese cases."""

import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json')
ARCHIVES = {
    'decider': ROOT / 'results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl',
    'laya': ROOT / 'results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl',
}
VERSION_ONLY_ARCHIVES = {
    'no_graph': ROOT / 'results_zh/zhfull_locomo_20260925_115806.jsonl',
    'fused_no_jev': ROOT / 'results_zh/zhfull_fused_locomo_20260925_161049.jsonl',
}
OUT = ROOT / 'results/analysis/m2_jev_relation_cases_20260927.json'
HISTORICAL_DEB_ID = '663e599c-0540-4a4c-9054-05001d54f70f'
CURRENT_DEB_ID = '53330642-0df8-47e7-9a80-7730d6b95834'
SOURCE_IDS = {
    'locomo_conv-26_qa111': ('D8:5', 'D8:6'),
    'locomo_conv-26_qa157': ('D4:2', 'D4:3'),
    'locomo_conv-48_qa233': ('D28:6', 'D28:7'),
}


def source_turns(sample, ids):
    found = {}
    for session, rows in sample['conversation'].items():
        if not session.startswith('session_') or session.endswith('_date_time'):
            continue
        for turn in rows:
            if turn['dia_id'] in ids:
                found[turn['dia_id']] = {
                    'session_id': session,
                    'session_date': sample['conversation'][session + '_date_time'],
                    'speaker': turn['speaker'], 'text': turn['text'],
                }
    assert set(found) == set(ids)
    return found


def main():
    dataset = {x['sample_id']: x for x in json.loads(DATASET.read_text())}
    result = {
        'scope': 'Three previously source-audited Chinese cases, two archived Jev services; read-only; no new model calls',
        'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        'archives': {}, 'cases': {}, 'qa233_memory_version_trace': {},
        'limits': [
            'Cases were selected from a prior RSI attribution probe, not sampled for Jev accuracy.',
            'Expanded first-stage ordered contexts are absent from historical logs and are not reconstructed.',
            'The original service request bodies and unrounded probabilities are absent from these archives.',
            'Historical arms are separate runs, not a within-run causal comparison.',
        ],
    }
    for cid, ids in SOURCE_IDS.items():
        conv, index = cid.removeprefix('locomo_').split('_qa')
        qa = dataset[conv]['qa'][int(index)]
        result['cases'][cid] = {
            'question': qa['question'], 'category': qa['category'],
            'source_turns': source_turns(dataset[conv], ids), 'arms': {},
        }
    for arm, path in ARCHIVES.items():
        result['archives'][arm] = {'path': str(path.relative_to(ROOT)),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        seen = set()
        old_hits = []
        for line in path.open():
            row = json.loads(line)
            old_vector = any(item['id'] == HISTORICAL_DEB_ID for item in row['retrieved_memories'])
            old_graph = any(item['id'] == HISTORICAL_DEB_ID for item in row['graph_memories'])
            if old_vector or old_graph:
                old_hits.append({'case_id': row['case_id'], 'vector': old_vector,
                                 'graph': old_graph,
                                 'selected': HISTORICAL_DEB_ID in (row['metadata'].get('answer_context_ids') or [])})
            cid = row['case_id']
            if cid not in SOURCE_IDS:
                continue
            assert cid not in seen
            seen.add(cid)
            meta = row['metadata']
            jev = meta['jev_stop']
            records = {}
            graph_ids = set()
            for item in row['retrieved_memories']:
                records[item['id']] = item
            for item in row['graph_memories']:
                graph_ids.add(item['id'])
                if item['id'] in records:
                    assert records[item['id']]['content'] == item['content']
                else:
                    records[item['id']] = item
            selected = []
            for index, mid in enumerate(meta['answer_context_ids'], 1):
                item = records[mid]
                selected.append({'slot': index, 'id': mid, 'content': item['content'],
                                 'graph_candidate': mid in graph_ids})
            assert len(selected) == 15
            result['cases'][cid]['arms'][arm] = {
                'jev': jev, 'predicted_answer': row['predicted_answer'],
                'final_context': selected,
            }
        assert seen == set(SOURCE_IDS)
        result['qa233_memory_version_trace'][arm] = {
            'historical_source_memory_id': HISTORICAL_DEB_ID,
            'archived_candidate_hits': old_hits,
        }
    for arm, path in VERSION_ONLY_ARCHIVES.items():
        hits = []
        for line in path.open():
            row = json.loads(line)
            in_vector = any(item['id'] == HISTORICAL_DEB_ID for item in row['retrieved_memories'])
            in_graph = any(item['id'] == HISTORICAL_DEB_ID for item in row.get('graph_memories', []))
            if in_vector or in_graph:
                hits.append({'case_id': row['case_id'], 'vector': in_vector,
                             'graph': in_graph,
                             'selected': HISTORICAL_DEB_ID in (row['metadata'].get('answer_context_ids') or [])})
        result['qa233_memory_version_trace'][arm] = {
            'archive': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'historical_source_memory_id': HISTORICAL_DEB_ID,
            'archived_candidate_hits': hits,
        }
    hpath = ROOT / 'data/vector_store_zh/mem0_history_zh.db'
    db = sqlite3.connect(f'file:{hpath}?mode=ro', uri=True)
    for label, mid in [('historical', HISTORICAL_DEB_ID), ('current', CURRENT_DEB_ID)]:
        history = db.execute(
            'select event,created_at,updated_at,is_deleted from history where memory_id=? '
            'order by coalesce(updated_at,created_at)', (mid,)
        ).fetchall()
        result['qa233_memory_version_trace'][label + '_memory'] = {
            'id': mid,
            'history': [dict(zip(('event', 'created_at', 'updated_at', 'is_deleted'), h))
                        for h in history],
        }
    archived_old_content = {
        item['content']
        for line in ARCHIVES['laya'].open() for row in [json.loads(line)]
        for item in row['retrieved_memories'] + row['graph_memories']
        if item['id'] == HISTORICAL_DEB_ID
    }
    assert len(archived_old_content) == 1
    result['qa233_memory_version_trace']['historical_memory']['content'] = archived_old_content.pop()
    vpath = ROOT / 'data/vector_store_zh/chroma/chroma.sqlite3'
    vector = sqlite3.connect(f'file:{vpath}?mode=ro', uri=True)
    current_content = vector.execute(
        "select m.string_value from embeddings e join embedding_metadata m on m.id=e.id "
        "where e.embedding_id=? and m.key='data'", (CURRENT_DEB_ID,)
    ).fetchone()
    assert current_content
    result['qa233_memory_version_trace']['current_memory']['content'] = current_content[0]
    assert [h['event'] for h in result['qa233_memory_version_trace']['historical_memory']['history']] == ['ADD', 'DELETE']
    assert [h['event'] for h in result['qa233_memory_version_trace']['current_memory']['history']] == ['ADD']
    for arm in (*ARCHIVES, *VERSION_ONLY_ARCHIVES):
        hits = result['qa233_memory_version_trace'][arm]['archived_candidate_hits']
        assert hits and not any(h['case_id'] == 'locomo_conv-48_qa233' for h in hits)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('cases', len(result['cases']), 'arms', list(result['archives']))


if __name__ == '__main__':
    main()
