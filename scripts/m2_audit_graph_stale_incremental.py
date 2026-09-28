#!/usr/bin/env python3
"""Synthetic incremental build/retrieval replay; no network or database writes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from schema_rsi.graph.builder import GraphBuilder
from schema_rsi.evaluation.retriever import GraphRetriever
from schema_rsi.evaluation.pipeline import EvaluationPipeline, _graph_memory_record
from schema_rsi.memory.base import MemoryRecord
from m2_audit_preference_graph_projection import Recorder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_graph_stale_incremental_20260928.json'


class NeighborRecorder(Recorder):
    def __init__(self):
        super().__init__()
        self._key_props = {'Memory': 'memory_id', 'Entity': 'entity_id', 'Event': 'event_id'}

    def get_neighbors(self, label, key, *, direction, edge_labels, limit):
        found = []
        seen = set()
        for edge in self.edges:
            if edge['label'] not in edge_labels:
                continue
            origin = (edge['from_label'], edge['from_key'])
            target = (edge['to_label'], edge['to_key'])
            other = target if direction == 'OUT' and origin == (label, key) else (
                origin if direction == 'IN' and target == (label, key) else None)
            if other and other not in seen:
                seen.add(other)
                found.append({'label': other[0], 'properties': self.vertices[other]})
        return found[:limit]


def memory(mid, text, owner='合成用户'):
    return MemoryRecord(mid, text, {'user_id': owner, 'session_id': 'synthetic'})


def extraction(rows, entity):
    return {r.id: {'entities': [{'name': entity, 'type': 'other'}]} for r in rows}


def replay():
    store = NeighborRecorder()
    builder = GraphBuilder(store)
    anchor = memory('anchor', '讨论旧话题。')
    deleted = memory('deleted', '已经从当前记忆清单删除的旧事实。')
    changed = memory('changed', '原来讨论旧话题。')
    builder.build_s1([anchor, deleted, changed], extraction([anchor, deleted, changed], '旧话题'), reset=False)
    changed_now = memory('changed', '现在只讨论新话题。')
    replacement = memory('replacement', '新增的替代记忆。')
    current = [anchor, changed_now, replacement]
    builder.build_s1([anchor], extraction([anchor], '旧话题'), reset=False)
    builder.build_s1([changed_now, replacement], extraction([changed_now, replacement], '新话题'), reset=False)
    # Deliberately reachable cross-owner control: ownership must still reject it.
    store.upsert_vertex('Memory', 'other-owner', {'memory_id': 'other-owner', 'content': '其他用户事实', 'user_id': '另一个用户'})
    old_entity = next(k for (label, k), props in store.vertices.items() if label == 'Entity' and props['name'] == '旧话题')
    store.upsert_edge('MENTIONS', 'Memory', 'other-owner', 'Entity', old_entity)
    graph = GraphRetriever(store, SimpleNamespace())
    fused = graph.retrieve_fused([anchor], exclude_ids={r.id for r in current})
    bypass = graph.retrieve([anchor])
    source_ids = {r.id for r in current}
    # Run the real Jev expansion pool path with a deterministic ranking stub.
    # Selecting graph first proves admissibility, not realistic ranking/answer harm.
    class RankingStub:
        def rerank_pool(self, question, records, top_n):
            return sorted(records, key=lambda r: r.id != 'deleted')[:top_n]
    pipeline = object.__new__(EvaluationPipeline)
    pipeline.settings = SimpleNamespace(evaluation={'graph_seed_k': 1})
    pipeline.graph_retriever = graph
    pipeline.retriever = RankingStub()
    selected, expanded = pipeline._evidence_expand(SimpleNamespace(question='合成问题'), '合成用户', [anchor], [], 15, 30, 1)
    observations = {
        'deleted_vertex_retained': ('Memory', 'deleted') in store.vertices,
        'deleted_id_absent_from_current_source': 'deleted' not in source_ids,
        'deleted_id_fused_candidate': 'deleted' in {g['id'] for g in fused},
        'deleted_id_bypass_candidate': 'deleted' in {g['id'] for g in bypass},
        'same_id_new_content_visible_via_old_entity': any(g['id'] == 'changed' and g['content'] == changed_now.content and g['via'] == 'entity:旧话题' for g in bypass),
        'cross_owner_rejected_both_paths': all(g['id'] != 'other-owner' for g in fused + bypass),
        'deleted_id_enters_real_expansion_context_with_stub_ranker': selected[0].id == 'deleted',
    }
    assert all(observations.values()), observations
    return {'current_source': [{'id': r.id, 'content': r.content} for r in current],
            'graph': store.export(), 'fused_candidates': fused, 'bypass_candidates': bypass,
            'expansion_candidates': expanded, 'expansion_context_ids': [r.id for r in selected],
            'converted_fused_records': [{'id': _graph_memory_record(g).id, 'content': _graph_memory_record(g).content} for g in fused],
            'observations': observations}


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    result = replay()
    result['scope'] = 'Synthetic actual GraphBuilder/GraphRetriever/Jev expansion replay; fake graph transport and deterministic ranking; zero model/service calls.'
    result['limits'] = ['Not a live HugeGraph snapshot or historical failure rate.', 'Stub ranker demonstrates acceptance only, not actual candidate ranking or answer harm.', 'No model extraction, embedding or cluster reassignment executed.']
    paths = ['src/schema_rsi/graph/builder.py', 'src/schema_rsi/graph/hugegraph_store.py', 'src/schema_rsi/evaluation/retriever.py', 'src/schema_rsi/evaluation/pipeline.py', 'scripts/build_graph_zh.py', 'src/schema_rsi/graph/variants.py']
    result['code_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['observations'], ensure_ascii=False))


if __name__ == '__main__':
    main()
