#!/usr/bin/env python3
"""Versioned small source-relation diagnostic, retaining prior failed run."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
from urllib.parse import urlparse

from m2_poc_preference_source_extraction import fixtures, BASE, POLICY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_source_relations_v2_poc_20260928.json'
PACKET = ROOT / 'results/analysis/m2_current_cache_missing_example_20260928.json'
BASE_V2 = BASE + '\n每条fact另加source_channel，取text或image_caption。输入caption只是附带描述，不是已经查看的图片。\n'
POLICY_V2 = POLICY + '\n- 图片caption支持的细节标image_caption；不能写成说话者亲口自述，也不能声称已查看图片。问答中的肯定偏好须引用前问与回答。\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(f'refusing overwrite: {OUT}')
    packet = json.loads(PACKET.read_text())
    animal = {'fixture_id': 'animal_question_caption_real', 'origin': 'real_dataset_snippet',
              'session_date': packet['session_date'], 'turns': packet['turns'],
              'manual_acceptance_zh': '梅拉妮喜欢画动物引用D13:9/10，最近画马引用D13:8；木板墙只能标caption来源，不得称亲口自述或已看图片'}
    fs = [animal] + fixtures()
    report = {'scope': 'Three real Chinese source snippets and five synthetic controls; new shared JSON contract with text/caption channel, not historical Mem0 replay',
              'max_calls': 16, 'max_tokens_per_call': 1200, 'temperature': 0.0, 'retry_count': 0,
              'stop_on_first_failure': True, 'fixtures': fs, 'requests': [], 'calls': [],
              'base_template': BASE_V2, 'policy_addition': POLICY_V2,
              'prior_run_retained': 'results/analysis/m2_preference_source_extraction_poc_20260928.json',
              'source_packet_sha256': hashlib.sha256(PACKET.read_bytes()).hexdigest(),
              'limits': ['No Memory/Graph/retrieval/answer/judge writes or calls.', 'One output per arm cannot establish stability or answer gain.',
                         'No images are fetched or shown; image_caption is text evidence only.', 'Synthetic controls are not prevalence observations.']}
    for i, fixture in enumerate(fs):
        lines = []
        for turn in fixture['turns']:
            lines.append(f"[{turn['dia_id']}] {turn['speaker']}: {turn['text']}")
            if turn.get('blip_caption'):
                lines.append(f"[{turn['dia_id']} image_caption] {turn['blip_caption']}")
        suffix = '\n会话日期：' + fixture['session_date'] + '\n对话：\n' + '\n'.join(lines)
        for arm in (('base', 'source_policy') if i % 2 == 0 else ('source_policy', 'base')):
            user = BASE_V2 + (POLICY_V2 if arm == 'source_policy' else '') + suffix
            report['requests'].append({'fixture_id': fixture['fixture_id'], 'arm': arm,
                                       'system': '你是事实提取器。仅输出有效JSON。', 'user': user,
                                       'user_sha256': hashlib.sha256(user.encode()).hexdigest()})
    def save():
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    save()
    if not args.run:
        print('prepared 16 exact requests; no model calls')
        return
    from schema_rsi.config import get_settings
    from schema_rsi.llm.chat import make_chat_client
    settings = get_settings('config/locomo_zh.yaml')
    parsed = urlparse(settings.llm.base_url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)), timeout=3):
            report['tcp_probe'] = 'reachable'
    except OSError as exc:
        report['tcp_probe'] = 'unreachable'
        report['probe_error_type'] = type(exc).__name__
        save()
        return
    report['model'] = settings.llm.model
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0, timeout=30)
    # Authentication/capability preflight; never persist models response or connection details.
    try:
        client._client.models.list()
        report['models_preflight'] = 'success'
    except Exception as exc:
        report['models_preflight'] = {'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
        save()
        return
    save()
    for request in report['requests']:
        call = {'call_index': len(report['calls']) + 1, 'fixture_id': request['fixture_id'], 'arm': request['arm']}
        assert call['call_index'] <= report['max_calls']
        report['calls'].append(call)
        save()
        try:
            text, usage = client.complete(system=request['system'], user=request['user'], max_tokens=1200, temperature=0.0)
            call.update(raw_response=text, usage=usage)
            try:
                call['parsed_response'] = json.loads(text)
                call['json_parse_ok'] = True
            except ValueError:
                call['json_parse_ok'] = False
            if not text:
                raise RuntimeError('empty response')
            print(call['call_index'], call['fixture_id'], call['arm'], 'received', flush=True)
        except Exception as exc:
            call.update(error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
            report['stopped_after_failure'] = True
            save()
            print(call['call_index'], call['error_type'], call['http_status'], flush=True)
            return
        save()
    report['completed_planned_calls'] = True
    save()


if __name__ == '__main__':
    main()
