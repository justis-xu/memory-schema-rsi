#!/usr/bin/env python3
"""Read-only versioned storage audit for Nate's question-qualified game preference."""
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from m2_audit_gina_preference_storage import read_store, DB, HISTORY, ATOMIC
from m2_audit_graph_positive_mechanisms import PATHS, source_packet, jsonl, context

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_nate_preference_storage_20260928.json'
USER = 'zhfull:locomo:conv-42'
CID = 'locomo_conv-42_qa95'
TERMS = ('xenoblade', 'xeonoblade', '异度')


def title_match(text):
    return any(t in (text or '').lower() for t in TERMS)


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    rows, before = read_store()
    current = [{'id':r[0], 'content':r[1], 'user_id':r[2], 'session_id':r[3], 'session_date':r[4]}
               for r in rows if r[2] == USER]
    all_titles = [{'id':r[0], 'content':r[1], 'user_id':r[2], 'session_id':r[3], 'session_date':r[4]}
                  for r in rows if title_match(r[1])]
    ids = {r['id'] for r in current}
    con = sqlite3.connect(f'file:{HISTORY}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    history = [dict(r) for r in con.execute('SELECT memory_id,old_memory,new_memory,event,created_at,updated_at,is_deleted FROM history')]
    associated = [r for r in history if r['memory_id'] in ids]
    title_history = [r for r in history if title_match((r['old_memory'] or '') + (r['new_memory'] or ''))]
    messages = [dict(r) for r in con.execute("SELECT role,created_at,length(content) AS content_length,CASE WHEN role='system' THEN content ELSE NULL END AS system_content FROM messages WHERE session_scope=?", ('user_id=' + USER,))]
    con.close()
    old, runs = {}, {}
    for arm in ('base','graph'):
        run = jsonl(PATHS[arm])
        for cid, result in run.items():
            if cid.startswith('locomo_conv-42_qa'):
                for m in result['retrieved_memories'] + result.get('graph_memories', []):
                    if m['id'] in old:
                        assert old[m['id']]['content'] == m['content']
                    old[m['id']] = {'id': m['id'], 'content':m['content']}
        grade = jsonl(PATHS[arm + '_grade'])[CID]
        runs[arm] = {'answer':run[CID]['predicted_answer'], 'context':context(run[CID]),
                     'grade':grade['final'], 'judge':grade.get('judge')}
    cur = {r['id']:r for r in current}
    shared = sorted(ids & old.keys())
    comparison = {'historical_visible_ids':len(old), 'current_ids':len(ids), 'shared_ids':len(shared),
                  'shared_content_mismatch_ids':[mid for mid in shared if cur[mid]['content']!=old[mid]['content']],
                  'historical_only_ids':sorted(old.keys()-ids), 'current_only_ids':sorted(ids-old.keys())}
    atomic = json.loads(ATOMIC.read_text())['full:locomo:conv-42:session_27']
    source = {lang:source_packet(next(s for s in json.loads(PATHS[lang+'_data'].read_text()) if s['sample_id']=='conv-42'),95)
              for lang in ('english','chinese')}
    _, after = read_store()
    assert before == after
    report = {'scope':'One question-qualified relation across visible historical and current Chinese Memory plus local English atomic cache; read-only, zero model calls',
              'case_id':CID, 'source':source, 'current_user_memories':current,
              'current_session27_memories':[r for r in current if r['session_id']=='session_27'],
              'current_all_users_title_hits':all_titles, 'history_title_hits':title_history,
              'associated_current_id_history':associated, 'history_event_counts':dict(Counter(r['event'] for r in associated)),
              'messages_retained_for_user':messages, 'english_atomic_session27':atomic,
              'historical_visible_title_hits':[r for r in old.values() if title_match(r['content'])],
              'historical_current_comparison':comparison, 'historical_runs':runs,
              'logical_store_before':before, 'logical_store_after':after, 'logical_store_unchanged':True,
              'input_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in {**PATHS,'atomic':ATOMIC,'history':HISTORY}.items()},
              'limits':['Title search may miss aliases or implicit references.',
                        'Different IDs and user namespaces are not one frozen experiment.',
                        'History ADD/DELETE records lack raw extraction response and per-fact source mapping.',
                        'Cache output is not proof of storage, retrieval or answer gain.']}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'current_count':len(current),'session27_count':len(report['current_session27_memories']),
                      'historical_count':len(old),'shared_count':len(shared),'title_hits':len(all_titles),
                      'history_events':report['history_event_counts'],'atomic_count':len(atomic)},ensure_ascii=False))


if __name__ == '__main__':
    main()
