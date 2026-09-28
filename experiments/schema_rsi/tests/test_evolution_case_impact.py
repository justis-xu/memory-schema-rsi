"""Mean-gain promotion must expose displaced evidence in its event log."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from schema_rsi_lab.evolve import EvolutionEngine
from schema_rsi_lab.model import Dataset, Schema, Channel
from m2_audit_evolution_case_regression import synthetic_data


def test_mean_gain_promotion_reports_harmed_case_and_lost_evidence(tmp_path):
    engine = EvolutionEngine(Dataset.from_dict(synthetic_data()), tmp_path / 'run', context_k=4)
    schema = Schema(channels=(Channel('r0', max_degree=1),), seed_k=2, context_k=4,
                    graph_slots=1, max_hops=1)
    assert engine._search_round('control', [('mean_gain_with_harm', schema)])
    event = engine._impact[0]
    assert event['accepted'] is True
    impact = event['case_impact']['validation']
    assert set(impact['gained_case_ids']) == {'validation_gain_a', 'validation_gain_b'}
    assert impact['harmed_case_ids'] == ['validation_harm']
    assert impact['cases_losing_gold'] == [{'case_id': 'validation_harm', 'lost_gold_ids': ['v4']}]
