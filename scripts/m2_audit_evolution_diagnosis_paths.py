#!/usr/bin/env python3
"""Order, compiled-path and bounded-search controls for actual RSI diagnosis."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/schema_rsi'))
from schema_rsi_lab.model import Dataset, Schema, Channel
from schema_rsi_lab.evolve import EvolutionEngine


def probe(kind):
    nodes = [{'id': 'm' + str(i), 'user_id': 'synthetic', 'kind': 'M'} for i in range(1, 6)]
    nodes += [{'id': h, 'user_id': 'synthetic', 'kind': 'H'} for h in ('h1', 'h2')]
    def edge(a, b, channel='r1', score=0.8):
        return {'source': a, 'target': b, 'channel': channel, 'score': score}
    if kind in ('disabled_first', 'enabled_first'):
        edges = [edge('m1', 'm5', 'r0'), edge('m1', 'm5')]
        if kind == 'enabled_first':
            edges.reverse()
    elif kind == 'enabled_two_hop_alternative':
        edges = [edge('m1', 'm5', 'r0'), edge('m1', 'h1'), edge('h1', 'm5')]
    elif kind == 'threshold_pruned':
        edges = [edge('m1', 'm5', score=0.4)]
    else:
        edges = [edge('m1', 'h1'), edge('h1', 'h2'), edge('h2', 'm5')]
    raw = {'nodes': nodes, 'edges': edges,
           'cases': [{'id': split + '_case', 'user_id': 'synthetic', 'split': split,
                      'base_ranked': ['m1', 'm2', 'm3', 'm4', 'm5'], 'gold': ['m5']}
                     for split in ('train', 'validation', 'test')]}
    data = Dataset.from_dict(raw)
    with tempfile.TemporaryDirectory(prefix='m2_diagnosis_') as folder:
        engine = EvolutionEngine(data, Path(folder), context_k=4)
        engine.active = Schema(channels=(Channel('r1', min_score=0.5 if kind == 'threshold_pruned' else 0),),
                               context_k=4, seed_k=2, graph_slots=1, max_hops=1, max_visits=1)
        _, rows = engine._evaluate(engine.active, 'train')
        diagnosis = engine._diagnose()
        graph = engine._graph(engine.active)
        compiled = {node: [{'target': a.target, 'channel': a.channel, 'score': a.score} for a in arcs]
                    for node, arcs in sorted(graph.adjacency.items())}
        return {'input': raw, 'schema': engine.active.to_dict(), 'compiled': compiled,
                'train_rows': rows, 'diagnosis': diagnosis}


def main():
    phase = sys.argv[1]
    if phase not in ('before', 'after'):
        raise SystemExit('before|after required')
    out = ROOT / f'results/analysis/m2_evolution_diagnosis_{phase}_20260928.json'
    if out.exists():
        raise SystemExit('refusing overwrite')
    results = {k: probe(k) for k in ('disabled_first', 'enabled_first', 'enabled_two_hop_alternative', 'threshold_pruned', 'three_hop_pool_path')}
    a, b = results['disabled_first'], results['enabled_first']
    assert a['compiled'] == b['compiled'] and a['train_rows'] == b['train_rows']
    if phase == 'before':
        assert a['diagnosis']['round_patterns'] == {'channel_missing': 1}
        assert b['diagnosis']['round_patterns'] == {'budget_or_rank': 1}
    else:
        assert a['diagnosis']['round_patterns'] == b['diagnosis']['round_patterns'] == {'budget_or_rank': 1}
        assert results['enabled_two_hop_alternative']['diagnosis']['round_patterns'] == {'hop_limit': 1}
        assert results['threshold_pruned']['diagnosis']['round_patterns'] == {'compile_filter_or_degree': 1}
        assert results['three_hop_pool_path']['diagnosis']['path_horizon'] == 2
    result = {'scope': 'Five synthetic real EvolutionEngine diagnosis controls, no model calls or live graph writes',
              'phase': phase, 'results': results,
              'code_sha256': hashlib.sha256((ROOT / 'experiments/schema_rsi/schema_rsi_lab/evolve.py').read_bytes()).hexdigest(),
              'limits': ['No semantic source labels or real benchmark frequency measured.',
                         'Available structural path does not prove fact support or that parameter changes improve answers.',
                         'Three-hop fixture is outside the supported two-hop schema search space.']}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v['diagnosis']['round_patterns'] for k, v in results.items()}))


if __name__ == '__main__':
    main()
