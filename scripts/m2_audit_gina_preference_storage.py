#!/usr/bin/env python3
"""Read-only SQLite/source audit of Gina's agreement-qualified dance preference."""
import hashlib
import json
import sqlite3
from pathlib import Path
from m2_audit_graph_positive_mechanisms import PATHS, source_packet, jsonl, context

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/vector_store_zh/chroma/chroma.sqlite3'
HISTORY = ROOT / 'data/vector_store_zh/mem0_history_zh.db'
ATOMIC = ROOT / 'data/extraction/atomic_cache.json'
STORAGE = ROOT / 'third_party/mem0-src/mem0/memory/storage.py'
OUT = ROOT / 'results/analysis/m2_gina_preference_storage_20260928.json'
USER = 'zhfull:locomo:conv-30'
CID = 'locomo_conv-30_qa39'


def read_store():
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    rows = con.execute("""SELECT e.embedding_id,
        max(CASE WHEN m.key='data' THEN m.string_value END),
        max(CASE WHEN m.key='user_id' THEN m.string_value END),
        max(CASE WHEN m.key='session_id' THEN m.string_value END),
        max(CASE WHEN m.key='session_date' THEN m.string_value END)
        FROM embeddings e LEFT JOIN embedding_metadata m ON m.id=e.id
        GROUP BY e.id ORDER BY e.embedding_id""").fetchall()
    con.close()
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode())
    return rows, {'embedding_count': len(rows), 'id_content_user_session_sha256': h.hexdigest()}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    rows, before = read_store()
    current = [{'id': r[0], 'content': r[1], 'user_id': r[2], 'session_id': r[3], 'session_date': r[4]}
               for r in rows if r[2] == USER]
    ids = {r['id'] for r in current}
    con = sqlite3.connect(f'file:{HISTORY}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    history = [dict(r) for r in con.execute('SELECT memory_id,old_memory,new_memory,event,created_at,updated_at,is_deleted FROM history')]
    associated = [r for r in history if r['memory_id'] in ids]
    lexical = [r for r in history if any(n in ((r['old_memory'] or '') + (r['new_memory'] or '')) for n in ('Gina','吉娜'))
               and any(n in ((r['old_memory'] or '') + (r['new_memory'] or '')) for n in ('contemporary','现代舞','当代舞','favorite','最爱'))]
    messages = [dict(r) for r in con.execute('SELECT role,created_at,length(content) AS content_length,CASE WHEN role=\'system\' THEN content ELSE NULL END AS system_content FROM messages WHERE session_scope=?', ('user_id=' + USER,))]
    con.close()
    old = {}
    histories = {}
    for arm in ('base','graph'):
        run = jsonl(PATHS[arm])
        for cid, result in run.items():
            if not cid.startswith('locomo_conv-30_qa'):
                continue
            for m in result['retrieved_memories'] + result.get('graph_memories', []):
                if m['id'] in old:
                    assert old[m['id']]['content'] == m['content']
                old[m['id']] = {'id': m['id'], 'content': m['content']}
        histories[arm] = {'answer': run[CID]['predicted_answer'], 'context': context(run[CID])}
    cur = {r['id']: r for r in current}
    shared = sorted(ids & old.keys())
    comparison = {'historical_visible_ids': len(old), 'current_ids': len(ids), 'shared_ids': len(shared),
                  'shared_content_mismatch_ids': [mid for mid in shared if cur[mid]['content'] != old[mid]['content']],
                  'current_only_ids': sorted(ids - old.keys()), 'historical_only_ids': sorted(old.keys() - ids)}
    atomic = json.loads(ATOMIC.read_text())['full:locomo:conv-30:session_1']
    sources = {lang: source_packet(next(s for s in json.loads(PATHS[lang + '_data'].read_text()) if s['sample_id'] == 'conv-30'), 39)
               for lang in ('english','chinese')}
    _, after = read_store()
    assert before == after
    report = {'scope': 'Current Chinese conv30 complete logical Memory snapshot, associated ADD history, local English atomic cache and historical visible candidates; zero model calls',
              'case_id': CID, 'source': sources, 'current_memories': current,
              'current_session1_memories': [r for r in current if r['session_id'] == 'session_1'],
              'associated_current_id_history': associated, 'global_history_gina_dance_or_favorite_lexical_hits': lexical,
              'messages_retained_for_user': messages, 'english_atomic_session1': atomic,
              'historical_current_comparison': comparison, 'historical_runs': histories,
              'logical_store_before': before, 'logical_store_after': after, 'logical_store_unchanged': True,
              'input_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in {**PATHS,'atomic':ATOMIC,'history':HISTORY,'storage_code':STORAGE}.items()},
              'limits': ['ADD history is not raw extraction request/response.',
                         'Messages retain recent input only, not session1 transcript.',
                         'English atomic cache differs in pipeline/language/version; no Chinese automatic capability claim.',
                         'Historical candidate union is not proof of the entire historical database.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'current_count':len(current),'session1_count':len(report['current_session1_memories']),'comparison':comparison,'history_events':len(associated),'message_count':len(messages)},ensure_ascii=False))


if __name__ == '__main__':
    main()
