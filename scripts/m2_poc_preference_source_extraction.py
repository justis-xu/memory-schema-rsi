#!/usr/bin/env python3
"""Small Chinese extraction-only diagnostic with real snippets and synthetic harm controls."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json')
FIXTURES = ROOT / 'results/analysis/m2_preference_extraction_fixtures_20260928.json'
OUT = ROOT / 'results/analysis/m2_preference_source_extraction_poc_20260928.json'
BASE = '''从对话提取人物事实，用简体中文。只输出JSON对象，格式：
{"facts":[{"subject":"人物","relation":"关系","object":"对象","support_type":"支持方式","source_turn_ids":["轮次ID"]}]}
每条写一个关系，保留人名、作品名、否定、日期等信息。没有可提取事实则facts为空数组。
支持方式可写direct、agreement、question_context、inference。source_turn_ids必须使用输入中的轮次ID。
'''
POLICY = '''额外来源约束：
- 联合阅读提问、回答、反问和赞同。明确“我也是”可承接上一人的偏好，但必须保留各自主体，引用两轮。
- 若只用近期经历回答“最爱”提问，保留“在被问最爱时介绍”的语境，标question_context并引用提问和回答；不升级唯一或永久最爱。
- 只有提问而未回答不能形成该偏好事实；明确否定不能转为肯定。
- 常玩、职业主打、读过、做过不自动等于最爱。不要把另一人的偏好转移给本人。
- 仅提取来源所支持的关系，不用外部知识补排名或属性。
'''


def fixtures():
    ds = {s['sample_id']:s for s in json.loads(DATA.read_text())}
    specs = [('gina_agreement_real','conv-30','session_1',['D1:8','D1:9']),
             ('nate_question_context_real','conv-42','session_27',['D27:22','D27:23'])]
    result=[]
    for fid,conv,session,ids in specs:
        turns=[{k:t[k] for k in ('dia_id','speaker','text')} for t in ds[conv]['conversation'][session] if t['dia_id'] in ids]
        assert [t['dia_id'] for t in turns]==ids
        result.append({'fixture_id':fid,'origin':'real_dataset_snippet','conversation':conv,
                       'session_date':ds[conv]['conversation'][session+'_date_time'],'turns':turns})
    controls=[
        ('question_only_synthetic',[('乔安娜','你最喜欢的游戏是什么？'),('内特','这次先不聊游戏。')], '不得生成内特最爱游戏事实'),
        ('disagreement_synthetic',[('乔恩','现代舞是我的最爱。你呢？'),('吉娜','我不是，我最爱的是街舞。')], '保留乔恩现代舞/吉娜街舞；不得吉娜现代舞最爱'),
        ('usual_not_favorite_synthetic',[('内特','我职业上通常玩CS:GO，但没有说它是我的最爱。最近我在玩异度神剑，觉得很好玩。')], '保存职业常玩与近期游玩；不得无条件最爱'),
        ('made_not_favorite_synthetic',[('内特','昨天我做了莓果巧克力蛋糕，还没有吃过它。')], '保存做过与未吃；不得喜欢或最爱'),
        ('explicit_favorite_synthetic',[('约翰','我最喜欢的披萨是夏威夷披萨。')], '应保留约翰最爱夏威夷披萨及直接来源'),
    ]
    for fid,turns,expect in controls:
        result.append({'fixture_id':fid,'origin':'synthetic_counterfactual_control_not_dataset',
                       'session_date':'unspecified','turns':[{'dia_id':f'S{i}','speaker':name,'text':t} for i,(name,t) in enumerate(turns,1)],
                       'manual_acceptance_zh':expect})
    result[0]['manual_acceptance_zh']='保留乔恩和吉娜最爱现代舞，吉娜赞同须引用D1:8和D1:9，不错人'
    result[1]['manual_acceptance_zh']='保留内特被问最爱时介绍异度神剑及近期玩/推荐，引用D27:22/23；不升级唯一最爱'
    return result


def prompts(f):
    # Manual acceptance fields and dataset QA gold are never sent to the model.
    transcript='\n'.join(f"[{t['dia_id']}] {t['speaker']}: {t['text']}" for t in f['turns'])
    suffix=f"\n会话日期：{f['session_date']}\n对话：\n{transcript}"
    return {'base':BASE+suffix,'source_policy':BASE+POLICY+suffix}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if FIXTURES.exists() or OUT.exists():
        raise SystemExit('refusing to overwrite fixtures or prior run')
    fs=fixtures()
    fixture_packet={'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'fixtures':fs,
                    'scope':'Two real source snippets, five synthetic controls; not benchmark prevalence'}
    FIXTURES.write_text(json.dumps(fixture_packet,ensure_ascii=False,indent=2)+'\n')
    report={'scope':'Extraction-only Chinese prompt diagnostic; new common JSON contract, not faithful replay of historical Mem0 extraction',
            'max_calls':14,'max_tokens_per_call':1200,'temperature':0.0,'retry_count':0,'stop_on_first_failure':True,
            'fixture_sha256':hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
            'base_template':BASE,'source_policy_addition':POLICY,'requests':[], 'calls':[],
            'limits':['One response per arm does not establish stability or causal answer gain.',
                      'Synthetic controls are not real dataset observations.',
                      'Prompt arms share a new structured output contract; no historical baseline reproduction.',
                      'No Memory/Graph writes or retrieval/answer/judge calls.']}
    for i,f in enumerate(fs):
        pp=prompts(f)
        for arm in (['base','source_policy'] if i%2==0 else ['source_policy','base']):
            report['requests'].append({'fixture_id':f['fixture_id'],'arm':arm,'system':'你是事实提取器。仅输出有效JSON。',
                                       'user':pp[arm],'user_sha256':hashlib.sha256(pp[arm].encode()).hexdigest()})
    def save():OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    save()
    if not args.run:
        print('prepared 7 fixtures and 14 exact requests; no calls');return
    from schema_rsi.config import get_settings
    from schema_rsi.llm.chat import make_chat_client
    s=get_settings(str(ROOT/'config/locomo_zh.yaml'));report['model']=s.llm.model
    client=make_chat_client(s);client._client=client._client.with_options(max_retries=0,timeout=30)
    for req in report['requests']:
        call={'call_index':len(report['calls'])+1,'fixture_id':req['fixture_id'],'arm':req['arm']}
        assert call['call_index']<=report['max_calls']
        report['calls'].append(call);save()
        try:
            text,usage=client.complete(system=req['system'],user=req['user'],max_tokens=1200,temperature=0.0)
            call.update({'raw_response':text,'usage':usage})
            try:call['parsed_response']=json.loads(text)
            except ValueError:call['json_parse_ok']=False
            else:call['json_parse_ok']=True
            if not text:raise RuntimeError('empty response')
            print(call['call_index'],call['fixture_id'],call['arm'],'response received',flush=True)
        except Exception as exc:
            call.update({'error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)})
            report['stopped_after_failure']=True;save()
            print(call['call_index'],call['error_type'],call['http_status'],flush=True);return
        save()
    report['completed_planned_calls']=True;save()


if __name__=='__main__':main()
