#!/usr/bin/env python3
"""Read-only audit of two anonymous book series and their source channels."""
import hashlib
import json
import sqlite3
from pathlib import Path
from m2_audit_graph_positive_mechanisms import PATHS, jsonl, context
from m2_audit_gina_preference_storage import read_store, HISTORY, DB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_nate_book_disambiguation_20260928.json'
USER = 'zhfull:locomo:conv-42'
CID = 'locomo_conv-42_qa91'
ANCHORS = {'fantasy_series': ['D9:13', 'D9:14', 'D9:15'],
           'space_opera_series': ['D19:16', 'D19:17', 'D19:18', 'D19:19', 'D19:20']}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    rows, before = read_store()
    current = [{'id': r[0], 'content': r[1], 'user_id': r[2],
                'session_id': r[3], 'session_date': r[4]}
               for r in rows if r[2] == USER]
    relevant = [r for r in current if r['session_id'] in ('session_9', 'session_19')
                and any(w in r['content'] for w in ('系列', '小说'))]
    sources = {}
    for lang in ('english', 'chinese'):
        s = next(s for s in json.loads(PATHS[lang + '_data'].read_text())
                 if s['sample_id'] == 'conv-42')
        turns = {t['dia_id']: t for k, v in s['conversation'].items()
                 if isinstance(v, list) for t in v}
        sources[lang] = {name: [{'turn': turns[eid], 'session_date': s['conversation'][
                                  'session_' + eid[1:].split(':')[0] + '_date_time']}
                              for eid in ids] for name, ids in ANCHORS.items()}
    runs = {k: jsonl(PATHS[k]) for k in ('base', 'graph', 'base_grade', 'graph_grade')}
    archive = {}
    for arm in ('base', 'graph'):
        record = runs[arm][CID]
        archive[arm] = {'answer': record['predicted_answer'],
                        'grade': runs[arm + '_grade'][CID]['final'],
                        'retrieved_memories': record['retrieved_memories'],
                        'graph_memories': record.get('graph_memories', []),
                        'context': context(record)}
    ids = {r['id'] for r in current}
    candidate_ids = {r['id'] for arm in ('base', 'graph')
                     for r in runs[arm][CID]['retrieved_memories'] + runs[arm][CID].get('graph_memories', [])}
    con = sqlite3.connect(f'file:{HISTORY}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    selected_ids = {r['id'] for r in relevant}
    history = [dict(r) for r in con.execute(
        'SELECT memory_id,old_memory,new_memory,event,created_at,updated_at,is_deleted FROM history')
               if r['memory_id'] in selected_ids]
    con.close()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    keys = [r[0] for r in con.execute('''SELECT DISTINCT m.key FROM embedding_metadata m
        WHERE m.id IN (SELECT id FROM embedding_metadata WHERE key='user_id' AND string_value=?)
        ORDER BY m.key''', (USER,))]
    attribution = {mid: [dict(key=r[0], value=r[1]) for r in con.execute(
        "SELECT key,string_value FROM embedding_metadata WHERE id=(SELECT id FROM embeddings WHERE embedding_id=?) AND key='attributed_to'",
        (mid,))] for mid in sorted(selected_ids)}
    con.close()
    atomic_path = ROOT / 'data/extraction/atomic_cache.json'
    atomic = json.loads(atomic_path.read_text())
    code_paths = {name: ROOT / rel for name, rel in {
        'adapter': 'src/schema_rsi/benchmarks/locomo.py',
        'ingest': 'src/schema_rsi/evaluation/pipeline.py',
        'atomic': 'src/schema_rsi/extraction/atomic.py'}.items()}
    _, after = read_store()
    assert before == after
    report = {
        'scope': 'Two targeted anonymous book series in Chinese conv42; no inference or database writes',
        'case_id': CID, 'sources': sources, 'current_owner_memories': current,
        'relevant_current_memories': relevant, 'relevant_add_history': history,
        'current_owner_metadata_keys': keys, 'historical_runs': archive,
        'relevant_attribution_metadata': attribution,
        'current_vs_historical_case_candidate_ids': {
            'current_owner_count': len(ids), 'historical_case_candidates': len(candidate_ids),
            'shared_ids': sorted(ids & candidate_ids)},
        'english_atomic_sessions': {str(n): atomic.get(f'full:locomo:conv-42:session_{n}') for n in (9, 19)},
        'source_channel_judgments': {
            'fantasy_series': {'likes': 'D9:14 text', 'features': 'adventure/magic/characters in text',
                               'dragon': 'D9:14 query says dragon cover; caption does not identify dragon',
                               'favorite': 'Not explicitly established by these source turns',
                               'plot_about_dragons': 'Not established by cover query alone'},
            'space_opera_series': {'favorite': 'D19:19 text, refers back to D19:17 series',
                                   'genre': 'space opera in D19:17 query only',
                                   'features': 'battles/characters/action/plot in text'}},
        'logical_store_before': before, 'logical_store_after': after,
        'logical_store_unchanged': True,
        'input_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest()
                         for k, p in {**PATHS, 'english_atomic_cache': atomic_path,
                                      'history': HISTORY, **code_paths}.items()},
        'limits': ['Source-channel labels are manual judgments, not automatic extraction output.',
                   'No original images were inspected.',
                   'ADD history does not retain the raw extraction response or input-to-fact alignment.',
                   'Current source snapshot is a different version from historical candidates.',
                   'English atomic cache is a different extraction pipeline; not a matched language trial.',
                   'No current retrieval, answer or overall quality effect was measured.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'current_count': len(current), 'relevant_count': len(relevant),
                      'historical_candidate_overlap': len(ids & candidate_ids),
                      'metadata_keys': keys, 'unchanged': before == after}, ensure_ascii=False))


if __name__ == '__main__':
    main()
