#!/usr/bin/env python3
"""Read-only live HugeGraph Memory inventory against current Chinese source."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from schema_rsi.config import get_settings
from schema_rsi.graph.hugegraph_store import HugeGraphStore
from m2_audit_gina_preference_storage import read_store

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_live_graph_source_version_20260928.json'


def main():
    target = OUT if '--global' not in sys.argv else ROOT / 'results/analysis/m2_live_graph_global_state_20260928.json'
    if target.exists():
        raise SystemExit(f'refusing to overwrite: {target}')
    settings = get_settings('config/locomo_zh.yaml')
    rows, before = read_store()
    owners = sorted({r[2] for r in rows if str(r[2]).startswith('zhfull:locomo:')})
    store = HugeGraphStore(settings, config=replace(settings.hugegraph, timeout=5))
    if '--global' in sys.argv:
        queries = {'vertex_label_counts': 'g.V().groupCount().by(label)',
                   'memory_owner_counts': "g.V().hasLabel('Memory').groupCount().by('user_id')",
                   'edge_label_counts': 'g.E().groupCount().by(label)'}
        results = {}
        for name, query in queries.items():
            try:
                results[name] = store.run_gremlin(query).get('result', {}).get('data', [])
            except Exception as exc:
                results[name] = {'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
        result = {'scope': 'Supplemental read-only whole-graph label/owner aggregation; no graph writes or model calls',
                  'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'queries': queries, 'results': results,
                  'limits': ['Current configured graph instance only, not all other instances/hosts.', 'No reset cause or historical state inferred.']}
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(results, ensure_ascii=False))
        return
    reports = []
    for owner in owners:
        current = {r[0]: r for r in rows if r[2] == owner}
        report = {'user_id': owner, 'current_source_count': len(current)}
        try:
            result = store._request('GET', f'/graphs/{store.graph}/graph/vertices',
                                    params={'properties': json.dumps({'user_id': owner}), 'limit': 10000})
            vertices = [v for v in result.get('vertices', []) if v.get('label') == 'Memory']
            # Read-only Gremlin query; no mutations, and owner comes from audited source.
            query = "g.V().hasLabel('Memory').has('user_id'," + json.dumps(owner) + ").count()"
            count = store.run_gremlin(query).get('result', {}).get('data', [])
            total = int(count[0]) if count else None
            graph = {v['properties'].get('memory_id'): v['properties'] for v in vertices}
            ids = set(graph); source_ids = set(current)
            shared = sorted(ids & source_ids)
            report.update({'graph_count': total, 'enumerated_count': len(vertices),
                           'enumeration_count_matches': total == len(vertices) == len(graph),
                           'current_shared_count': len(shared), 'current_missing_ids': sorted(source_ids - ids),
                           'graph_only_ids': sorted(ids - source_ids),
                           'shared_body_mismatch_ids': [mid for mid in shared if graph[mid].get('content') != current[mid][1]],
                           'shared_date_mismatch_ids': [mid for mid in shared if (graph[mid].get('session_date') or '') != (current[mid][4] or '')],
                           'vertices': [{'id': mid, 'content_sha256': hashlib.sha256(str(p.get('content', '')).encode()).hexdigest(),
                                         'user_id': p.get('user_id'), 'session_id': p.get('session_id'), 'session_date': p.get('session_date')}
                                        for mid, p in sorted(graph.items(), key=lambda item: str(item[0]))]})
        except Exception as exc:
            # Never serialize raw HTTP body, connection URL or authentication state.
            report.update({'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None),
                           'enumeration_count_matches': False})
        reports.append(report)
    _, after = read_store()
    assert before == after
    result = {'scope': 'Configured live graph read-only Memory inventories for 10 Chinese users; current SQLite source comparison; no graph/source writes or model calls',
              'captured_at_utc': datetime.now(timezone.utc).isoformat(),
              'logical_source_before': before, 'logical_source_after': after,
              'by_user': reports,
              'limits': ['Separate REST/Gremlin/user reads are not one graph transaction; graph may change between them.',
                         'Memory identity/body/date alignment does not validate edges, extraction versions or answer gain.',
                         'Count equality is enumeration evidence at these reads, not a build provenance manifest.',
                         'Service address and auth settings deliberately omitted.']}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k not in ('vertices', 'current_missing_ids', 'graph_only_ids', 'shared_body_mismatch_ids', 'shared_date_mismatch_ids')}
                      | {k + '_count': len(r[k]) for k in ('current_missing_ids', 'graph_only_ids', 'shared_body_mismatch_ids', 'shared_date_mismatch_ids') if k in r}
                     for r in reports], ensure_ascii=False))


if __name__ == '__main__':
    main()
