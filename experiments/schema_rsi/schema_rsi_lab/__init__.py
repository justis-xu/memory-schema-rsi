"""Isolated, offline prototype for evolving opaque memory-graph schemas."""

from .model import CandidateEdge, Case, Dataset, Node, Schema, Channel
from .graph import evaluate_split
from .evolve import EvolutionEngine

__all__ = [
    "CandidateEdge", "Case", "Dataset", "Node", "Schema", "Channel",
    "evaluate_split", "EvolutionEngine",
]
