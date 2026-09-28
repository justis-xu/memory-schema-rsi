"""Structural diagnostic must not depend on raw candidate edge ordering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from m2_audit_evolution_diagnosis_paths import probe


def test_enabled_route_prevents_order_dependent_missing_channel():
    first = probe('disabled_first')
    second = probe('enabled_first')
    assert first['compiled'] == second['compiled']
    assert first['train_rows'] == second['train_rows']
    assert first['diagnosis']['round_patterns'] == second['diagnosis']['round_patterns'] == {'budget_or_rank': 1}


def test_available_two_hop_route_and_compilation_filter_are_distinct():
    assert probe('enabled_two_hop_alternative')['diagnosis']['round_patterns'] == {'hop_limit': 1}
    assert probe('threshold_pruned')['diagnosis']['round_patterns'] == {'compile_filter_or_degree': 1}
