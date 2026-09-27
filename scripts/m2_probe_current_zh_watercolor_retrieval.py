#!/usr/bin/env python3
"""One-question current Chinese vector/rerank exposure probe, with store fingerprint."""

import hashlib
import json
from pathlib import Path
import sqlite3

from schema_rsi.config import get_settings
from schema_rsi.llm.rerank import RerankClient


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json')
DB = ROOT / 'data/vector_store_zh/chroma/chroma.sqlite3'
OUT = ROOT / 'results/analysis/m2_current_zh_watercolor_retrieval_20260927.json'
USER = 'zhfull:locomo:conv-49'
TARGET = 'e3762a39-169b-45be-9aa6-c8053f882d16'


def logical_signature():
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    h = hashlib.sha256()
    count = 0
    for row in con.execute("""
        SELECT e.embedding_id,
          max(CASE WHEN m.key='data' THEN m.string_value END),
          max(CASE WHEN m.key='user_id' THEN m.string_value END),
          max(CASE WHEN m.key='session_id' THEN m.string_value END),
          max(CASE WHEN m.key='session_date' THEN m.string_value END)
        FROM embeddings e LEFT JOIN embedding_metadata m ON m.id=e.id
        GROUP BY e.id ORDER BY e.embedding_id
    """):
        h.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode())
        count += 1
    con.close()
    return {'embedding_count': count, 'id_content_user_session_sha256': h.hexdigest()}


def rank(ids, mid):
    return ids.index(mid) + 1 if mid in ids else None


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    sample = next(s for s in json.loads(DATA.read_text()) if s['sample_id'] == 'conv-49')
    question = sample['qa'][85]['question']
    settings = get_settings(str(ROOT / 'config/locomo_zh.yaml'))
    before = logical_signature()
    report = {
        'scope': 'Current pure-Chinese one-question vector top45 and rerank top15; no Graph, answer, or judge',
        'case_id': 'locomo_conv-49_qa85', 'question': question,
        'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'user_id': USER, 'target_memory_id': TARGET,
        'vector_k': 45, 'final_k': 15,
        'max_embedding_calls': 1, 'max_logical_rerank_calls': 1,
        'max_rerank_http_attempts': 2,
        'embedding_model': settings.embedding.model, 'rerank_model': settings.rerank.model,
        'logical_store_before': before,
        'calls': {'embedding': 0, 'logical_rerank': 0},
        'limits': [
            'Current memory IDs and corpus differ from historical mixed-language archives.',
            'Candidate rank is not answer quality or proof of Graph necessity.',
            'Chroma may change SQLite bytes despite unchanged logical ID/content/user/session fingerprint.',
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    try:
        import chromadb
        from openai import OpenAI

        chroma = chromadb.PersistentClient(path=str(DB.parent))
        collection = chroma.get_collection(settings.mem0['collection_name'])
        embed = OpenAI(base_url=settings.embedding.base_url,
                       api_key=settings.embedding.api_key, timeout=20, max_retries=0)
        report['calls']['embedding'] = 1
        vector = embed.embeddings.create(model=settings.embedding.model, input=[question]).data[0].embedding
        hits = collection.query(query_embeddings=[vector], n_results=45,
                                where={'user_id': USER}, include=['metadatas', 'distances'])
        ids = hits['ids'][0]
        metas = hits['metadatas'][0]
        distances = hits['distances'][0]
        assert len(ids) == len(metas) == len(distances) == 45
        assert all(m.get('user_id') == USER and m.get('data') for m in metas)
        report['vector_top45'] = [
            {'rank': i, 'id': mid, 'content': meta['data'],
             'session_id': meta.get('session_id'), 'session_date': meta.get('session_date'),
             'distance': distance}
            for i, (mid, meta, distance) in enumerate(zip(ids, metas, distances), 1)
        ]
        report['target_vector_rank'] = rank(ids, TARGET)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

        reranker = RerankClient(settings.rerank.base_url, settings.rerank.api_key,
                                settings.rerank.model, timeout=30)
        report['calls']['logical_rerank'] = 1
        scored = reranker.rerank(question, [m['data'] for m in metas], top_n=15)
        indices = [item['index'] for item in scored]
        assert len(indices) == len(set(indices)) == 15
        assert all(0 <= i < 45 for i in indices)
        report['rerank_top15'] = [
            {**report['vector_top45'][item['index']], 'rank': j,
             'vector_rank': item['index'] + 1, 'rerank_score': item['score']}
            for j, item in enumerate(scored, 1)
        ]
        report['rerank_endpoint_style'] = reranker.last_endpoint
        report['target_final_rank'] = rank([m['id'] for m in report['rerank_top15']], TARGET)
        print('target vector', report['target_vector_rank'], 'final', report['target_final_rank'], flush=True)
    except Exception as exc:
        report['stopped_after_failure'] = True
        report['error_type'] = type(exc).__name__
        report['http_status'] = getattr(exc, 'status_code', None)
        print('stopped', report['error_type'], report['http_status'], flush=True)
    finally:
        report['logical_store_after'] = logical_signature()
        report['logical_store_unchanged'] = before == report['logical_store_after']
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('logical_store_unchanged', report['logical_store_unchanged'], flush=True)


if __name__ == '__main__':
    main()
