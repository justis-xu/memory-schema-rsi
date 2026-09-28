#!/usr/bin/env python3
"""Read-only collection inventory and real Mem0/Chroma list limit fault probes."""
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from mem0 import Memory
from mem0.memory.main import _payload_is_expired
from mem0.vector_stores.chroma import ChromaDB
from schema_rsi.memory.mem0_backend import Mem0Backend, _GET_ALL_LIMIT, _to_record

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/vector_store_zh/chroma/chroma.sqlite3'
OUT = ROOT / 'results/analysis/m2_memory_snapshot_completeness_20260928.json'


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def snapshot():
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    con.execute('BEGIN')
    rows = con.execute('''SELECT c.name,e.embedding_id,m.key,m.string_value,m.int_value,m.float_value,m.bool_value
        FROM embeddings e JOIN segments s ON e.segment_id=s.id JOIN collections c ON s.collection=c.id
        LEFT JOIN embedding_metadata m ON e.id=m.id ORDER BY c.name,e.embedding_id,m.key''').fetchall()
    con.close()
    grouped = {}
    for collection, mid, key, *values in rows:
        payload = grouped.setdefault((collection, mid), {})
        if key is not None:
            payload[key] = next((v for v in values if v is not None), None)
    logical = [{'collection': c, 'id': mid, 'metadata': p} for (c, mid), p in sorted(grouped.items())]
    return logical


class CollectionStub:
    def __init__(self, payloads):
        self.payloads = payloads
        self.limits = []

    def get(self, where, limit):
        self.limits.append(limit)
        rows = self.payloads[:limit]
        return {'ids': [str(i) for i in range(len(rows))], 'metadatas': rows}


def limit_probe(payloads):
    col = CollectionStub(payloads)
    vector = object.__new__(ChromaDB)
    vector.collection = col
    raw = object.__new__(Memory)
    raw.vector_store = vector
    # Avoid SDK telemetry/init; execute the real formatting/filter/limit implementation.
    raw.get_all = lambda *, filters, top_k: {'results': raw._get_all_from_vector_store(
        filters, max(top_k * 4, 60), False, top_k)}
    backend = object.__new__(Mem0Backend)
    backend._memory = raw
    returned = backend.get_all_memories('synthetic-owner')
    return {'input_count': len(payloads), 'nonexpired_input_count': sum(not _payload_is_expired(p) for p in payloads),
            'collection_fetch_limits': col.limits, 'returned_count': len(returned),
            'last_returned_id': returned[-1].id if returned else None}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    before = snapshot()
    main_rows = [r for r in before if r['collection'] == 'schema_rsi_memories_zh']
    owners = Counter(r['metadata'].get('user_id') for r in main_rows)
    user_stats = []
    for owner, count in sorted(owners.items(), key=lambda x: str(x[0])):
        rows = [r for r in main_rows if r['metadata'].get('user_id') == owner]
        user_stats.append({'user_id': owner, 'physical_count': count,
                           'expiration_field_count': sum(bool(r['metadata'].get('expiration_date')) for r in rows),
                           'expired_count_now': sum(_payload_is_expired(r['metadata']) for r in rows),
                           'below_mem0_output_limit': count < _GET_ALL_LIMIT,
                           'ids_sha256': sha(sorted(r['id'] for r in rows)), 'records_sha256': sha(rows)})
    live = {'user_id': 'synthetic-owner', 'data': '合成有效记忆'}
    expired = {**live, 'expiration_date': '2000-01-01'}
    probes = {'1001_live': limit_probe([live] * 1001),
              '4000_expired_then_10_live': limit_probe([expired] * 4000 + [live] * 10)}
    assert probes['1001_live']['returned_count'] == 1000
    assert probes['4000_expired_then_10_live']['returned_count'] == 0
    synthetic_item = {'id': 'probe', 'memory': '合成记录', 'user_id': 'synthetic-owner',
                      'expiration_date': '2030-01-01', 'updated_at': '2026-01-01', 'created_at': '2025-01-01',
                      'hash': 'synthetic-hash', 'metadata': {'session_date': '2025-01-01'}}
    converted = _to_record(synthetic_item)
    after = snapshot()
    assert before == after
    zh = [u for u in user_stats if str(u['user_id']).startswith('zhfull:locomo:')]
    result = {'scope': 'Read-only SQLite all collection metadata inventory; fake collection executes real Mem0Backend and SDK list/filter implementation; zero network/model calls',
              'runtime_memory_class_path': inspect.getfile(Memory), 'mem0_output_limit': _GET_ALL_LIMIT,
              'collections': dict(Counter(r['collection'] for r in before)), 'main_collection_total': len(main_rows),
              'users': user_stats, 'zhfull_locomo_users': len(zh), 'zhfull_locomo_count': sum(u['physical_count'] for u in zh),
              'zhfull_max_user_count': max(u['physical_count'] for u in zh),
              'two_read_logical_sha256': [sha(before), sha(after)], 'two_reads_equal': True,
              'synthetic_limit_probes': probes, 'synthetic_promoted_fields_conversion': {'input': synthetic_item, 'output_metadata': converted.metadata,
                  'not_preserved_as_metadata': [k for k in ('expiration_date', 'updated_at', 'created_at', 'hash') if k not in converted.metadata]},
              'limits': ['SQLite metadata inventory is not Chroma service/client integration or index consistency proof.',
                         'Two equal reads establish endpoint stability only, not absence of transient writes or a publish/read lock.',
                         'Synthetic limit probes bypass telemetry/public get_all but use its verified fetch/output formulas.',
                         'No evidence that historical runs actually hit the cap or expiration prefix.']}
    paths = ['src/schema_rsi/memory/mem0_backend.py', 'src/schema_rsi/memory/chroma_direct.py',
             'third_party/mem0-src/mem0/memory/main.py', 'third_party/mem0-src/mem0/vector_stores/chroma.py', 'scripts/build_graph_zh.py']
    result['code_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('collections', 'main_collection_total', 'zhfull_locomo_users', 'zhfull_locomo_count', 'zhfull_max_user_count', 'synthetic_limit_probes', 'two_reads_equal')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
