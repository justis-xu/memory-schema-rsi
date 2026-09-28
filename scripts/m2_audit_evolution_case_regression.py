#!/usr/bin/env python3
"""Historical accepted-schema transitions and actual mean-gate harm control."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/schema_rsi'))
from schema_rsi_lab.model import Dataset, Schema, Channel
from schema_rsi_lab.graph import evaluate_split
from schema_rsi_lab.evolve import EvolutionEngine

OUT = ROOT / 'results/analysis/m2_evolution_case_regression_20260928.json'
FIXTURE = ROOT / 'experiments/schema_rsi/fixtures/tiny.json'


def paired(before, after):
    old = {r['case_id']: r for r in before}
    assert set(old) == {r['case_id'] for r in after}
    changes = []
    for row in after:
        previous = old[row['case_id']]
        gold = set(row['gold'])
        changes.append({'case_id': row['case_id'], 'before_recall': previous['recall'], 'after_recall': row['recall'],
                        'lost_gold_ids': sorted(gold & (set(previous['selected']) - set(row['selected']))),
                        'gained_gold_ids': sorted(gold & (set(row['selected']) - set(previous['selected']))),
                        'before_selected': previous['selected'], 'after_selected': row['selected']})
    return {'gained_cases': sum(r['after_recall'] > r['before_recall'] for r in changes),
            'harmed_cases': sum(r['after_recall'] < r['before_recall'] for r in changes),
            'cases_losing_any_gold_id': sum(bool(r['lost_gold_ids']) for r in changes), 'rows': changes}


def synthetic_data():
    raw = {'nodes': [], 'edges': [], 'cases': []}
    for user, split in [('t', 'train'), ('v', 'validation'), ('x', 'test')]:
        ids = [user + str(i) for i in range(1, 7)]
        raw['nodes'] += [{'id': mid, 'user_id': user, 'kind': 'M'} for mid in ids]
        raw['edges'].append({'source': ids[0], 'target': ids[4], 'channel': 'r0', 'score': 1.0})
        for suffix, target in [('gain_a', ids[4]), ('gain_b', ids[4]), ('harm', ids[3])]:
            raw['cases'].append({'id': split + '_' + suffix, 'user_id': user, 'split': split,
                                 'base_ranked': ids, 'gold': [target]})
    return raw


def main():
    logging_only = '--log-control' in sys.argv
    target = OUT if not logging_only else ROOT / 'results/analysis/m2_evolution_case_impact_logging_20260928.json'
    if target.exists():
        raise SystemExit('refusing overwrite')
    data = Dataset.load(FIXTURE)
    archives = []
    for run in ([] if logging_only else sorted((ROOT / 'experiments/schema_rsi/runs').glob('run-*'))):
        impact_path = run / 'wiki/impact.jsonl'
        if not impact_path.exists():
            continue
        manifest = json.loads((run / 'input.json').read_text())
        assert data.sha256 == manifest['dataset_sha256']
        trials = [json.loads(line) for line in (run / 'raw/trials.jsonl').read_text().splitlines()]
        schemas = {r['schema_id']: Schema.from_dict(r['schema']) for r in trials}
        baseline = Schema(channels=(), seed_k=manifest['context_k'], context_k=manifest['context_k'], graph_slots=0)
        schemas[baseline.id] = baseline
        transitions = []
        for event in map(json.loads, impact_path.read_text().splitlines()):
            if not event['accepted']:
                continue
            previous = schemas[event['previous_schema_id']]
            candidate = schemas[event['schema_id']]
            transition = {'stage': event['stage'], 'schema_id': candidate.id, 'previous_schema_id': previous.id, 'splits': {}}
            for split in ('train', 'validation'):
                _, old_rows = evaluate_split(data, previous, split)
                report, new_rows = evaluate_split(data, candidate, split)
                assert abs(report['recall'] - event[split]['recall']) < 1e-12
                assert abs(report['delta'] - event[split]['delta']) < 1e-12
                if split == 'train':
                    archived_rows = next(r['train_cases'] for r in trials if r['schema_id'] == candidate.id)
                    assert {r['case_id']: r['selected'] for r in new_rows} == {r['case_id']: r['selected'] for r in archived_rows}
                transition['splits'][split] = paired(old_rows, new_rows)
            transitions.append(transition)
        archives.append({'run': str(run.relative_to(ROOT)), 'dataset_sha256': data.sha256, 'accepted_transitions': transitions,
                         'artifact_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (impact_path, run / 'raw/trials.jsonl', run / 'input.json')}})
    raw = synthetic_data(); controlled = Dataset.from_dict(raw)
    schema = Schema(channels=(Channel('r0', max_degree=1),), seed_k=2, context_k=4, graph_slots=1, max_hops=1)
    with tempfile.TemporaryDirectory(prefix='m2_evolution_gate_') as folder:
        engine = EvolutionEngine(controlled, Path(folder), context_k=4)
        original = engine.active
        accepted = engine._search_round('synthetic_control', [('mean_gain_with_one_harm', schema)])
        assert accepted and engine.active.id == schema.id
        _, before_rows = engine._evaluate(original, 'validation')
        validation_report, after_rows = engine._evaluate(schema, 'validation')
        comparison = paired(before_rows, after_rows)
        assert comparison['gained_cases'] == 2 and comparison['harmed_cases'] == 1
        control = {'input': raw, 'accepted': accepted, 'event': engine._impact[0], 'validation': validation_report, 'paired_validation': comparison}
    result = {'scope': 'Accepted historical tiny-fixture transitions plus one synthetic real EvolutionEngine gate control; no model/answer calls',
              'historical': archives, 'synthetic_control': control,
              'limits': ['Historical validation case details reconstructed with matching fixture SHA and current deterministic evaluator, not original archived per-case validation logs.',
                         'Synthetic gains/harm ratio is constructed, not benchmark frequency.', 'Gold-ID loss is not necessarily an answer error.',
                         'No production gate changed; mean-gain objective and zero-harm policy are distinct choices.']}
    if logging_only:
        logged = control['event']['case_impact']['validation']
        assert logged['harmed_case_ids'] == ['validation_harm']
        assert logged['cases_losing_gold'] == [{'case_id': 'validation_harm', 'lost_gold_ids': ['v4']}]
        result['scope'] = 'Added acceptance-event case-impact logging; same synthetic gate control, no historical replay'
        result['before_result_sha256'] = hashlib.sha256(OUT.read_bytes()).hexdigest()
    result['evolution_code_sha256'] = hashlib.sha256((ROOT / 'experiments/schema_rsi/schema_rsi_lab/evolve.py').read_bytes()).hexdigest()
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'historical': [{'run': a['run'], 'changes': [{s: {k: v for k, v in p.items() if k != 'rows'} for s, p in t['splits'].items()} for t in a['accepted_transitions']]} for a in archives],
                      'synthetic_accepted': accepted, 'synthetic_paired': {k: v for k, v in comparison.items() if k != 'rows'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
