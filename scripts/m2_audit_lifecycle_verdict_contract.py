#!/usr/bin/env python3
"""Fault injection for LifecycleRSI evaluator completeness; no model calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/schema_rsi'))
from schema_rsi_lab.lifecycle_rsi import LifecycleRSI

CODE = ROOT / 'experiments/schema_rsi/schema_rsi_lab/lifecycle_rsi.py'
CASES = [{'id': f'q{i}_{j}', 'conversation_id': f'c{i}', 'baseline_correct': bool(j)}
         for i in range(5) for j in range(2)]


def probe(mode):
    def evaluator(policy, requested):
        rows = []
        for case in requested:
            correct = True if not case['baseline_correct'] else (False if mode == 'explicit_harm' else True)
            if case['baseline_correct'] and mode == 'unknown_verdict':
                correct = None
            row = {'id': case['id'], 'baseline_correct': case['baseline_correct'], 'correct': correct,
                   'model_calls': 2, 'edge_visits': 10}
            if mode == 'missing_call_count':
                row.pop('model_calls')
            elif mode == 'negative_call_count':
                row['model_calls'] = -1
            elif mode == 'string_verdict' and case['baseline_correct']:
                row['correct'] = 'false'
            elif mode == 'missing_visit_count':
                row.pop('edge_visits')
            rows.append(row)
        return rows
    with tempfile.TemporaryDirectory(prefix='m2_lifecycle_contract_') as folder:
        try:
            result = LifecycleRSI(CASES, Path(folder) / 'run', evaluator).run()
            return {'completed': True, 'policy_promoted': result['active_policy']['verify_enabled'],
                    'validation_events': result['validation_events'], 'test': result['test']}
        except ValueError as exc:
            return {'completed': False, 'error_type': type(exc).__name__, 'reason': str(exc)}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('before', 'after'):
        raise SystemExit('usage: script before|after')
    phase = sys.argv[1]
    out = ROOT / f'results/analysis/m2_lifecycle_verdict_contract_{phase}_20260928.json'
    if out.exists():
        raise SystemExit('refusing overwrite')
    modes = ('valid_complete', 'explicit_harm', 'unknown_verdict', 'string_verdict',
             'missing_call_count', 'negative_call_count', 'missing_visit_count')
    results = {mode: probe(mode) for mode in modes}
    assert results['valid_complete']['policy_promoted'] is True
    assert results['explicit_harm']['policy_promoted'] is False
    for mode in modes[2:]:
        if phase == 'before':
            assert results[mode]['policy_promoted'] is True
        else:
            assert results[mode]['completed'] is False
    report = {'scope': 'Seven synthetic evaluator response modes through actual LifecycleRSI; five independent conversations, paired two cases each; no model/API calls',
              'phase': phase, 'code_sha256': hashlib.sha256(CODE.read_bytes()).hexdigest(), 'cases': CASES, 'results': results,
              'limits': ['Injected evaluator outcomes/costs, not observed model failures or historical rates.',
                         'No Chinese answer or learned experience effect measured.', 'Does not evaluate independent EvolutionEngine evidence-recall gate.']}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
