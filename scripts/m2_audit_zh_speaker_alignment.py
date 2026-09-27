#!/usr/bin/env python3
"""Read-only alignment of LoCoMo English/Chinese turn speakers and archived exposure."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh')
OUT = ROOT / 'results/analysis/m2_zh_speaker_alignment_20260927.json'
ARCHIVES = {
    'no_graph': ROOT / 'results_zh/zhfull_locomo_20260925_115806.jsonl',
    'fused_no_jev': ROOT / 'results_zh/zhfull_fused_locomo_20260925_161049.jsonl',
    'jev_decider': ROOT / 'results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl',
    'jev_laya': ROOT / 'results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl',
}
SESSION = re.compile(r'session_\d+$')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    paths = {lang: DATA / filename for lang, filename in
             [('english', 'locomo10.json'), ('chinese', 'locomo10_zh.json')]}
    sets = {lang: {s['sample_id']: s for s in json.loads(path.read_text())}
            for lang, path in paths.items()}
    assert sets['english'].keys() == sets['chinese'].keys()
    per_conversation = {}
    mismatches = []
    turn_count = 0
    affected_sessions = set()
    for cid in sorted(sets['english']):
        en, zh = sets['english'][cid], sets['chinese'][cid]
        en_sessions = {k for k in en['conversation'] if SESSION.fullmatch(k)}
        zh_sessions = {k for k in zh['conversation'] if SESSION.fullmatch(k)}
        assert en_sessions == zh_sessions
        counts = defaultdict(Counter)
        rows = []
        for session in sorted(en_sessions, key=lambda k: int(k.split('_')[1])):
            a, b = en['conversation'][session], zh['conversation'][session]
            assert len(a) == len(b)
            assert en['conversation'][session + '_date_time'] == zh['conversation'][session + '_date_time']
            for x, y in zip(a, b):
                assert x['dia_id'] == y['dia_id']
                counts[x['speaker']][y['speaker']] += 1
                rows.append((session, x, y))
                turn_count += 1
        assert len(counts) == 2
        mapping = {speaker: row.most_common(1)[0][0] for speaker, row in counts.items()}
        assert len(set(mapping.values())) == 2
        per_conversation[cid] = {
            'turns': len(rows), 'speaker_mapping': mapping,
            'aligned_name_pair_counts': {speaker: dict(c) for speaker, c in counts.items()},
        }
        for session, x, y in rows:
            if y['speaker'] == mapping[x['speaker']]:
                continue
            affected_sessions.add((cid, session))
            mismatches.append({
                'conversation': cid, 'session': session, 'dia_id': x['dia_id'],
                'english_speaker': x['speaker'], 'expected_chinese_speaker': mapping[x['speaker']],
                'actual_chinese_speaker': y['speaker'],
                'english_text': x['text'], 'chinese_text': y['text'],
            })
    assert turn_count == 5882 and len(mismatches) == 3

    qa_evidence = []
    for item in mismatches:
        cid = item['conversation']
        for i, qa in enumerate(sets['chinese'][cid]['qa']):
            if item['dia_id'] in (qa.get('evidence') or []):
                qa_evidence.append({'conversation': cid, 'qa_index': i,
                                    'dia_id': item['dia_id'], 'question': qa['question']})

    exposure = {}
    for arm, path in ARCHIVES.items():
        unique_memories = {}
        selected_counts = Counter()
        case_counts = Counter()
        for line in path.open():
            row = json.loads(line)
            final_ids = set(row['metadata'].get('answer_context_ids', []))
            seen_session = set()
            for mem in row['retrieved_memories']:
                meta = mem.get('metadata') or {}
                key = (meta.get('user_id', '').removeprefix('zhfull:locomo:'), meta.get('session_id'))
                if key not in affected_sessions:
                    continue
                mid = mem['id']
                record = unique_memories.setdefault(mid, {
                    'id': mid, 'conversation': key[0], 'session': key[1],
                    'content': mem['content'], 'metadata': meta,
                })
                assert record['content'] == mem['content']
                if mid in final_ids:
                    selected_counts[mid] += 1
                    seen_session.add(key)
            for key in seen_session:
                case_counts['/'.join(key)] += 1
        exposure[arm] = {
            'archive_path': str(path.relative_to(ROOT)), 'archive_sha256': sha(path),
            'unique_retrieved_memories_from_affected_sessions': list(unique_memories.values()),
            'final_selection_counts_by_memory_id': dict(selected_counts),
            'cases_with_affected_session_memory_in_final_by_session': dict(case_counts),
        }
    result = {
        'scope': 'Exact aligned dia_id speaker labels across all English/Chinese LoCoMo turns; archived vector metadata exposure only; no model calls',
        'dataset_paths': {k: str(v) for k, v in paths.items()},
        'dataset_sha256': {k: sha(v) for k, v in paths.items()},
        'conversations': len(per_conversation), 'turns': turn_count,
        'mismatch_count': len(mismatches), 'affected_sessions': sorted([list(k) for k in affected_sessions]),
        'per_conversation': per_conversation, 'mismatches': mismatches,
        'qa_with_mismatched_turn_in_evidence': qa_evidence,
        'historical_vector_exposure': exposure,
        'limits': [
            'Majority speaker mapping checks speaker labels, not semantic translation quality.',
            'Session-level metadata cannot attribute an extracted memory fact to a particular turn.',
            'Graph memories lack session metadata and are excluded from exposure counts.',
            'Final-selection counts are retrieval exposures, not answer or score impact.',
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'turns': turn_count, 'mismatches': len(mismatches),
                      'affected_sessions': result['affected_sessions'],
                      'archive_unique_memories': {k: len(v['unique_retrieved_memories_from_affected_sessions'])
                                                  for k, v in exposure.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
