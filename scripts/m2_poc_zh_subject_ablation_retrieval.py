#!/usr/bin/env python3
"""Frozen current-Chroma retrieval probe; three calls, no answer generation."""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.llm.rerank import RerankClient  # noqa: E402


OUT = ROOT / 'results/analysis/m2_zh_subject_ablation_retrieval_20260927.json'
USER = 'zhfull:locomo:conv-48'
TARGET = '53330642-0df8-47e7-9a80-7730d6b95834'
DISTRACTOR = 'd2e2e49f-f93f-4dd4-a604-2b8e5e0397f3'
QUERIES = (
    ('original', '乔琳和妈妈的老朋友们做了什么？'),
    ('subject_masked', '妈妈的老朋友们做了什么？'),
    ('correct_subject_oracle', '黛博拉和妈妈的老朋友们做了什么？'),
)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pos(ids, mid):
    return ids.index(mid) + 1 if mid in ids else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--salvage-vector', action='store_true')
    parser.add_argument('--rerank-salvage', action='store_true')
    parser.add_argument('--verify-storage', action='store_true')
    args = parser.parse_args()
    settings = get_settings(str(ROOT / 'config/locomo_zh.yaml'))
    db = settings.resolve_path(settings.mem0['vector_store_path']) / 'chroma.sqlite3'
    prepared = {
        'scope': 'One fixed current Chinese conversation, three query variants; vector top45 then rerank top15; no answerer or judge',
        'questions': [{'arm': arm, 'query': query} for arm, query in QUERIES],
        'user_id': USER, 'target_id': TARGET, 'distractor_id': DISTRACTOR,
        'max_embedding_calls': 3, 'max_logical_rerank_calls': 3,
        'max_rerank_http_attempts': 6, 'calls_completed': 0,
        'embedding_model': settings.embedding.model, 'rerank_model': settings.rerank.model,
        'chroma_sqlite_sha256_before': file_sha(db),
        'results': [],
        'limits': [
            'Current memory IDs differ from the 2026-09-25 archive, so this does not reconstruct that historical counterfactual.',
            'The corrected-person query is an oracle upper bound and must not be used as a deployable rewrite.',
            'The name-masked query can surface another person; its hits are evidence to check, not an answer to the original question.',
            'Only main Chroma collection is present; no atomic pool or Graph is queried.',
        ],
    }
    if args.verify_storage:
        result = json.loads(OUT.read_text())
        vector_db = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        check = {
            'sqlite_quick_check': vector_db.execute('pragma quick_check').fetchone()[0],
            'conv48_user_metadata_rows': vector_db.execute(
                "select count(*) from embedding_metadata where key='user_id' and string_value=?",
                (USER,),
            ).fetchone()[0],
            'tracked_contents': {},
        }
        for label, mid in [('target', TARGET), ('distractor', DISTRACTOR)]:
            row = vector_db.execute(
                "select m.string_value from embeddings e join embedding_metadata m on m.id=e.id "
                "where e.embedding_id=? and m.key='data'", (mid,)
            ).fetchone()
            check['tracked_contents'][label] = {'id': mid, 'content': row[0] if row else None}
        result['post_run_storage_check'] = check
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print('storage', check['sqlite_quick_check'], check['conv48_user_metadata_rows'])
        return
    if args.rerank_salvage:
        result = json.loads(OUT.read_text())
        assert result.get('vector_salvage_completed')
        assert len(result['vector_salvage']) == 2 and not result.get('rerank_salvage')
        vector_db = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        result['initial_rerank_input_issue'] = (
            'Chroma documents are null for this Mem0 collection; content is in embedding_metadata.data. '
            'The first rerank request received null documents, so its RerankError does not diagnose service availability.'
        )
        rerank = RerankClient(settings.rerank.base_url, settings.rerank.api_key,
                              settings.rerank.model, timeout=30)
        result['rerank_salvage'] = []
        for entry in result['vector_salvage']:
            for item in entry['vector_top45']:
                content = vector_db.execute(
                    "select m.string_value from embeddings e join embedding_metadata m on e.id=m.id "
                    "where e.embedding_id=? and m.key='data'", (item['id'],)
                ).fetchone()
                assert content and content[0]
                item['content'] = content[0]
            try:
                scored = rerank.rerank(entry['query'],
                                        [item['content'] for item in entry['vector_top45']],
                                        top_n=15)
                positions = [x['index'] for x in scored]
                assert len(positions) == len(set(positions)) == 15
                ids = [item['id'] for item in entry['vector_top45']]
                selected = [ids[i] for i in positions]
                row = {
                    'arm': entry['arm'], 'query': entry['query'],
                    'rerank_top15': [
                        {'rank': rank, 'id': ids[x['index']],
                         'content': entry['vector_top45'][x['index']]['content'],
                         'score': x['score'], 'vector_rank': x['index'] + 1}
                        for rank, x in enumerate(scored, 1)
                    ],
                    'tracked_rerank_ranks': {'target': pos(selected, TARGET),
                                             'distractor': pos(selected, DISTRACTOR)},
                    'rerank_endpoint_style': rerank.last_endpoint,
                }
            except Exception as exc:
                row = {'arm': entry['arm'], 'query': entry['query'],
                       'error_type': type(exc).__name__}
            result['rerank_salvage'].append(row)
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print(entry['arm'], row.get('tracked_rerank_ranks', row.get('error_type')), flush=True)
            if 'error_type' in row:
                break
        result['logical_rerank_calls_total'] = 1 + len(result['rerank_salvage'])
        assert result['logical_rerank_calls_total'] <= result['max_logical_rerank_calls']
        result['rerank_salvage_completed'] = len(result['rerank_salvage']) == 2 and all(
            'rerank_top15' in x for x in result['rerank_salvage'])
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        return
    if args.salvage_vector:
        if not OUT.exists():
            raise SystemExit('missing stopped result')
        result = json.loads(OUT.read_text())
        assert result['calls_completed'] == 1 and result['stopped_early']
        assert result['results'][0]['arm'] == 'original'
        assert not result.get('vector_salvage')
        import chromadb  # noqa: E402
        from openai import OpenAI  # noqa: E402
        client = chromadb.PersistentClient(path=str(db.parent))
        collection = client.get_collection(settings.mem0['collection_name'])
        embed = OpenAI(base_url=settings.embedding.base_url,
                       api_key=settings.embedding.api_key, max_retries=0, timeout=20)
        result['vector_salvage'] = []
        for arm, query in QUERIES[:2]:
            try:
                vector = embed.embeddings.create(model=settings.embedding.model,
                                                 input=[query]).data[0].embedding
                hits = collection.query(query_embeddings=[vector], n_results=45,
                                        where={'user_id': USER},
                                        include=['documents', 'metadatas', 'distances'])
                ids = hits['ids'][0]
                assert len(ids) == 45
                metas = hits['metadatas'][0]
                row = {
                    'arm': arm, 'query': query,
                    'vector_top45': [
                        {'rank': i, 'id': mid, 'content': meta['data'], 'distance': dist,
                         'session_id': meta.get('session_id'), 'session_date': meta.get('session_date')}
                        for i, (mid, meta, dist) in enumerate(
                            zip(ids, metas, hits['distances'][0]), 1)
                    ],
                    'tracked_vector_ranks': {'target': pos(ids, TARGET),
                                             'distractor': pos(ids, DISTRACTOR)},
                }
            except Exception as exc:
                row = {'arm': arm, 'query': query, 'error_type': type(exc).__name__}
            result['vector_salvage'].append(row)
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print(arm, row.get('tracked_vector_ranks', row.get('error_type')), flush=True)
            if 'error_type' in row:
                break
        result['embedding_calls_total'] = 1 + len(result['vector_salvage'])
        assert result['embedding_calls_total'] <= result['max_embedding_calls']
        result['vector_salvage_completed'] = len(result['vector_salvage']) == 2 and all(
            'vector_top45' in x for x in result['vector_salvage'])
        result['chroma_sqlite_sha256_after_salvage'] = file_sha(db)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        return
    if not args.run:
        if OUT.exists():
            raise SystemExit(f'refusing to overwrite {OUT}')
        OUT.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n')
        print('prepared 3 query arms; max 3 embeddings and 3 logical reranks')
        return
    if not OUT.exists():
        raise SystemExit('prepare first')
    result = json.loads(OUT.read_text())
    assert result['calls_completed'] == 0
    assert result['questions'] == prepared['questions']
    assert result['chroma_sqlite_sha256_before'] == file_sha(db)
    assert settings.rerank_enabled

    import chromadb  # noqa: E402
    from openai import OpenAI  # noqa: E402

    client = chromadb.PersistentClient(path=str(db.parent))
    collection = client.get_collection(settings.mem0['collection_name'])
    assert collection.count() >= 200
    embed = OpenAI(base_url=settings.embedding.base_url,
                   api_key=settings.embedding.api_key, max_retries=0, timeout=20)
    rerank = RerankClient(settings.rerank.base_url, settings.rerank.api_key,
                          settings.rerank.model, timeout=30)
    for arm, query in QUERIES:
        try:
            vector = embed.embeddings.create(model=settings.embedding.model,
                                             input=[query]).data[0].embedding
            hits = collection.query(query_embeddings=[vector], n_results=45,
                                    where={'user_id': USER},
                                    include=['documents', 'metadatas', 'distances'])
            ids = hits['ids'][0]
            metas = hits['metadatas'][0]
            distances = hits['distances'][0]
            docs = [meta['data'] for meta in metas]
            assert len(ids) == len(docs) == len(metas) == len(distances) == 45
            scored = rerank.rerank(query, docs, top_n=15)
            positions = [x['index'] for x in scored]
            assert len(positions) == len(set(positions)) == 15
            selected = [ids[i] for i in positions]
            row = {
                'arm': arm, 'query': query,
                'vector_top45': [
                    {'rank': i, 'id': mid, 'content': doc, 'distance': dist,
                     'session_id': meta.get('session_id'), 'session_date': meta.get('session_date')}
                    for i, (mid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), 1)
                ],
                'rerank_top15': [
                    {'rank': rank, 'id': ids[x['index']], 'content': docs[x['index']],
                     'score': x['score'], 'vector_rank': x['index'] + 1}
                    for rank, x in enumerate(scored, 1)
                ],
                'tracked_ranks': {
                    label: {'vector': pos(ids, mid), 'rerank': pos(selected, mid)}
                    for label, mid in [('target', TARGET), ('distractor', DISTRACTOR)]
                },
                'rerank_endpoint_style': rerank.last_endpoint,
            }
        except Exception as exc:
            row = {'arm': arm, 'query': query, 'error_type': type(exc).__name__}
            result['stopped_early'] = True
        result['results'].append(row)
        result['calls_completed'] += 1
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        if result.get('stopped_early'):
            print('stopped at', arm, row['error_type'])
            break
        print(arm, row['tracked_ranks'], flush=True)
    result['chroma_sqlite_sha256_after'] = file_sha(db)
    result['completed'] = result['calls_completed'] == 3 and not result.get('stopped_early')
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
