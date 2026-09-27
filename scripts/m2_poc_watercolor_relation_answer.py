#!/usr/bin/env python3
"""Frozen-context Chinese answer POC: source relation versus placebo and harm control."""

import hashlib
import json
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.evaluation.prompts_official import ANSWER_GENERATION_PROMPT
from schema_rsi.llm.chat import make_chat_client


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'results/analysis/m2_graph_positive_mechanisms_20260927.json'
GRAPH_RUN = ROOT / 'results_zh/zhfull_fused_locomo_20260925_172126.jsonl'
OUT = ROOT / 'results/analysis/m2_watercolor_relation_answer_poc_20260927.json'
SYSTEM = 'You are answering a question using retrieved memories from past conversations.'
WATERCOLOR = 'locomo_conv-49_qa85'
REPAIR = 'locomo_conv-50_qa124'
TARGET_ID = 'dbc78f1e-fe6b-4b63-b279-d6cba1c6e2db'
PLACEBO_ID = 'b8fc79ba-90b9-4cd6-8d7b-f3ba8b34e5ae'
DISTRACTOR_ID = '6eb8af39-c052-4b17-bab0-d40254862427'


def prompt(question, memories):
    lines = 'Memories in fixed retrieval order:\n' + '\n'.join(
        f"{i}. {m['content']}" for i, m in enumerate(memories, 1))
    return ANSWER_GENERATION_PROMPT.format(reference_date='2023', memories=lines, question=question)


def find_memory(rows, mid):
    matches = [m for m in rows if m['id'] == mid]
    assert len(matches) == 1
    return matches[0]


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite prior calls: {OUT}')
    packet = json.loads(PACKET.read_text())
    by_case = {c['case_id']: c for c in packet['cases']}
    wc, rep = by_case[WATERCOLOR], by_case[REPAIR]
    base_wc = wc['base_context']
    target = find_memory(wc['graph_context'], TARGET_ID)
    placebo = find_memory(by_case['locomo_conv-49_qa72']['gained'], PLACEBO_ID)
    assert target['route'] == placebo['route'] == 'graph'
    assert all(PLACEBO_ID not in [m['id'] for m in row['base_context']]
               for row in (wc, rep))

    graph_row = next(r for r in map(json.loads, GRAPH_RUN.open()) if r['case_id'] == REPAIR)
    distractor = find_memory(graph_row['graph_memories'], DISTRACTOR_ID)
    base_rep = rep['base_context']

    def replace(base, rank, mem):
        result = [dict(m) for m in base]
        result[rank - 1] = dict(mem, rank=rank)
        assert len(result) == 15
        assert sum(a['id'] != b['id'] for a, b in zip(base, result)) == 1
        return result

    arms = {
        WATERCOLOR: {'base': base_wc, 'target': replace(base_wc, 15, target),
                     'placebo': replace(base_wc, 15, placebo)},
        REPAIR: {'base': base_rep, 'distractor': replace(base_rep, 12, distractor)},
    }
    orders = {WATERCOLOR: ['base', 'target', 'placebo', 'placebo', 'target', 'base'],
              REPAIR: ['base', 'distractor', 'distractor', 'base']}
    settings = get_settings(str(ROOT / 'config/locomo_zh.yaml'))
    result = {
        'scope': 'Old mixed-language frozen 15-memory answer POC; one slot changed per arm; zero judge calls',
        'source_packet': str(PACKET.relative_to(ROOT)),
        'source_packet_sha256': hashlib.sha256(PACKET.read_bytes()).hexdigest(),
        'graph_run_sha256': hashlib.sha256(GRAPH_RUN.read_bytes()).hexdigest(),
        'model': settings.llm.model, 'temperature': 0.0,
        'max_calls': 10, 'max_completion_tokens_per_call': 1200,
        'prompt_template': 'official LoCoMo answer prompt, numbered fixed order, dates omitted for content-only swap',
        'cases': [],
        'limits': [
            'Retrospectively selected cases, not a prevalence or overall graph effect estimate.',
            'Old mixed-language memories, not current pure-Chinese memory store.',
            'A one-slot swap tests candidate content, not graph traversal, retrieval, or reranking.',
            'Two responses per arm are a diagnostic stability check, not a variance estimate.',
        ],
    }
    for cid in (WATERCOLOR, REPAIR):
        case = by_case[cid]
        prompts = {arm: prompt(case['source']['chinese']['question'], ctx)
                   for arm, ctx in arms[cid].items()}
        result['cases'].append({
            'case_id': cid, 'question': case['source']['chinese']['question'],
            'gold_for_manual_label_only': case['source']['chinese']['answer'],
            'source_evidence_ids': [e['dia_id'] for e in case['source']['chinese']['evidence']],
            'replace_rank': 15 if cid == WATERCOLOR else 12,
            'removed_memory': (base_wc if cid == WATERCOLOR else base_rep)[14 if cid == WATERCOLOR else 11],
            'arms': {arm: {'added_memory': (ctx[14] if cid == WATERCOLOR else ctx[11]) if arm != 'base' else None,
                           'context_ids': [m['id'] for m in ctx],
                           'prompt_sha256': hashlib.sha256(prompts[arm].encode()).hexdigest()}
                     for arm, ctx in arms[cid].items()},
            'order': orders[cid], 'calls': [],
        })
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    calls = 0
    for case in result['cases']:
        cid = case['case_id']
        for arm in case['order']:
            assert calls < result['max_calls']
            calls += 1
            try:
                answer, usage = client.complete(system=SYSTEM, user=prompt(
                    case['question'], arms[cid][arm]), max_tokens=1200, temperature=0.0)
                if not answer:
                    raise RuntimeError('empty answer')
                call = {'call_index': calls, 'arm': arm, 'answer': answer, 'usage': usage}
                print(cid, calls, arm, answer[:160].replace('\n', ' '), flush=True)
            except Exception as exc:
                call = {'call_index': calls, 'arm': arm, 'error_type': type(exc).__name__,
                        'http_status': getattr(exc, 'status_code', None)}
                case['calls'].append(call)
                result['stopped_after_failure'] = True
                OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                print(cid, calls, arm, call['error_type'], call['http_status'], flush=True)
                return
            case['calls'].append(call)
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert calls == 10
    result['stopped_after_failure'] = False
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
