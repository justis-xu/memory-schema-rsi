#!/usr/bin/env python3
"""Read-only English/Chinese LoCoMo QA evidence-ID integrity audit."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh')
PATHS = {'english': DATA / 'locomo10.json', 'chinese': DATA / 'locomo10_zh.json'}
OUT = ROOT / 'results/analysis/m2_locomo_evidence_id_integrity_20260927.json'
VALID = re.compile(r'D\d+:\d+$')


def normalize(entry, available):
    if entry in available:
        return 'valid', [entry]
    if re.fullmatch(r'D\d+:\d+(?:[;\s]+D\d+:\d+)+', entry):
        parts = re.split(r'[;\s]+', entry)
        if all(part in available for part in parts):
            return 'joined_valid_ids', parts
    match = re.fullmatch(r'D:(\d+):(\d+)', entry)
    if match:
        candidate = f'D{int(match.group(1))}:{int(match.group(2))}'
        if candidate in available:
            return 'extra_colon', [candidate]
    match = re.fullmatch(r'D(\d+):(\d+)', entry)
    if match:
        candidate = f'D{int(match.group(1))}:{int(match.group(2))}'
        if candidate != entry and candidate in available:
            return 'leading_zero', [candidate]
    return 'unresolved', []


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    sets = {lang: {s['sample_id']: s for s in json.loads(path.read_text())}
            for lang, path in PATHS.items()}
    assert sets['english'].keys() == sets['chinese'].keys()
    counts = Counter()
    cases = []
    for cid in sorted(sets['english']):
        en, zh = sets['english'][cid], sets['chinese'][cid]
        assert len(en['qa']) == len(zh['qa'])
        turns = {}
        for session, rows in zh['conversation'].items():
            if not (session.startswith('session_') and isinstance(rows, list)):
                continue
            for turn in rows:
                assert turn['dia_id'] not in turns
                turns[turn['dia_id']] = (session, turn)
        en_ids = {t['dia_id'] for session, rows in en['conversation'].items()
                  if session.startswith('session_') and isinstance(rows, list) for t in rows}
        assert en_ids == set(turns)
        for index, (a, b) in enumerate(zip(en['qa'], zh['qa'])):
            assert a.get('evidence') == b.get('evidence')
            counts['qa'] += 1
            entries = a.get('evidence') or []
            counts['raw_entries'] += len(entries)
            observations = []
            for entry in entries:
                status, resolved = normalize(entry, turns)
                counts[status] += 1
                counts['resolved_ids'] += len(resolved)
                if status != 'valid':
                    observations.append({'raw': entry, 'status': status, 'resolved_ids': resolved})
            if not observations:
                continue
            ids = [mid for obs in observations for mid in obs['resolved_ids']]
            cases.append({
                'case_id': f'locomo_{cid}_qa{index}', 'conversation': cid, 'qa_index': index,
                'category': a['category'], 'raw_evidence': entries,
                'observations': observations,
                'english': {'question': a['question'], 'answer': a.get('answer')},
                'chinese': {'question': b['question'], 'answer': b.get('answer')},
                'resolved_source_turns': [
                    {'dia_id': mid, 'session': turns[mid][0],
                     'session_date': zh['conversation'][turns[mid][0] + '_date_time'],
                     'speaker': turns[mid][1]['speaker'], 'text': turns[mid][1]['text'],
                     'image_query': turns[mid][1].get('query'),
                     'blip_caption': turns[mid][1].get('blip_caption')}
                    for mid in ids
                ],
            })
    assert counts['qa'] == 1986 and len(cases) == 9
    assert counts['joined_valid_ids'] == 4 and counts['extra_colon'] == 1
    assert counts['leading_zero'] == 1 and counts['unresolved'] == 3
    report = {
        'scope': 'All English/Chinese LoCoMo QA evidence entries checked against actual dia_id, with only syntax-safe normalization; no model calls or dataset edits',
        'dataset_paths': {k: str(v) for k, v in PATHS.items()},
        'dataset_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in PATHS.items()},
        'counts': dict(counts), 'affected_qa': len(cases), 'cases': cases,
        'limits': [
            'Syntax-safe normalization is an audit suggestion, not a mutation of benchmark files.',
            'Unresolved evidence entries require source review and must not be guessed from nearby IDs.',
            'Valid dia_id syntax and existence do not establish that the turn supports every answer fact.',
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'counts': report['counts'], 'affected_qa': len(cases)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
