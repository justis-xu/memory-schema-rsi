#!/usr/bin/env python3
"""Actual current source/cache coverage and isolated S1 recording replay."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from schema_rsi.graph.builder import GraphBuilder
from schema_rsi.memory.base import MemoryRecord
from m2_audit_gina_preference_storage import read_store
from m2_audit_graph_stale_incremental import NeighborRecorder

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/graph_zh/structured_cache.json'
OUT = ROOT / 'results/analysis/m2_current_cache_graph_readiness_20260928.json'
KINDS = ('entities', 'events', 'preferences', 'relationships')


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    rows, before = read_store()
    cache = json.loads(CACHE.read_text())
    reports = []
    for owner in sorted({r[2] for r in rows if str(r[2]).startswith('zhfull:locomo:')}):
        selected = [r for r in rows if r[2] == owner]
        memories = [MemoryRecord(r[0], r[1], {'user_id': r[2], 'session_id': r[3], 'session_date': r[4]}) for r in selected]
        hit = [r.id for r in memories if r.id in cache]
        miss = [r.id for r in memories if r.id not in cache]
        report = {'user_id': owner, 'current_source_count': len(memories), 'cache_hits': len(hit), 'cache_misses': len(miss),
                  'current_missing_records': [{'id': r.id, 'content': r.content, 'session_id': r.metadata['session_id']} for r in memories if r.id in miss],
                  'cached_nonempty_count': sum(any(cache[mid].get(kind) for kind in KINDS) for mid in hit),
                  'cache_extraction_provenance': 'unknown_no_body_model_prompt_fingerprint'}
        store = NeighborRecorder()
        try:
            graph_report = GraphBuilder(store).build_s1(memories, {mid: cache[mid] for mid in hit}, reset=False)
            outgoing = Counter(edge['from_key'] for edge in store.edges if edge['from_label'] == 'Memory')
            stored_memory_ids = {key for (label, key) in store.vertices if label == 'Memory'}
            isolated = [r.id for r in memories if not outgoing[r.id]]
            assert stored_memory_ids == {r.id for r in memories}
            assert set(miss) <= set(isolated)
            report['offline_build'] = {'accepted': True, 'label_counts': graph_report.label_counts,
                                       'edge_counts': dict(Counter(edge['label'] for edge in store.edges)),
                                       'memory_id_coverage_complete': True, 'memory_without_outgoing_derived_edges': isolated,
                                       'unprocessed_memory_without_edges_count': len(miss),
                                       'cached_memory_without_edges_count': len(set(isolated) & set(hit)),
                                       'recorded_graph_sha256': hashlib.sha256(json.dumps(store.export(), ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        except ValueError as exc:
            report['offline_build'] = {'accepted': False, 'reason': str(exc), 'vertices_written': len(store.vertices)}
        reports.append(report)
    _, after = read_store(); assert before == after
    result = {'scope': 'Current 10-user Chinese source plus actual local S1 cache; real GraphBuilder with isolated recording store; no model/live graph writes',
              'logical_source_before': before, 'logical_source_after': after, 'cache_total_records': len(cache),
              'cache_sha256': hashlib.sha256(CACHE.read_bytes()).hexdigest(),
              'current_main_source_count': sum(r['current_source_count'] for r in reports),
              'current_cache_hits': sum(r['cache_hits'] for r in reports), 'current_cache_misses': sum(r['cache_misses'] for r in reports),
              'by_user': reports,
              'limits': ['Existing cache has no per-record input/model/prompt fingerprints; hits do not prove semantic validity.',
                         'Partial-cache projection is a diagnostic, not a graph published as ready.',
                         'A legitimate empty extraction can also produce no edges; missing and valid-empty statuses must remain distinct.',
                         'No topic/embedding/cluster/model extraction or answer experiment executed.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{'user_id': r['user_id'], 'source': r['current_source_count'], 'hits': r['cache_hits'],
                       'accepted': r['offline_build']['accepted'], 'unprocessed_without_edges': r['offline_build'].get('unprocessed_memory_without_edges_count'),
                       'cached_without_edges': r['offline_build'].get('cached_memory_without_edges_count')} for r in reports], ensure_ascii=False))


if __name__ == '__main__':
    main()
