#!/usr/bin/env python3
"""Offline recording-store replay of S1 preference projection; no service calls."""
import hashlib
import json
from pathlib import Path
from schema_rsi.graph.builder import GraphBuilder
from schema_rsi.graph.extractor_s1 import StructuredExtractor
from schema_rsi.memory.base import MemoryRecord
from m2_audit_gina_preference_storage import read_store

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/graph_zh/structured_cache.json'
OUT = ROOT / 'results/analysis/m2_preference_graph_projection_20260928.json'


class Recorder:
    def __init__(self):self.vertices={};self.edges=[]
    def reset_graph(self, **kwargs):self.vertices={};self.edges=[]
    def create_schema(self,schema):self.schema=schema
    def upsert_vertex(self,label,key,props):self.vertices[(label,key)]=dict(props)
    def upsert_edge(self,label,fl,fk,tl,tk,props=None):self.edges.append({'label':label,'from_label':fl,'from_key':fk,'to_label':tl,'to_key':tk,'properties':props or {}})
    def export(self):return {'vertices':[{'label':l,'key':k,'properties':p} for (l,k),p in self.vertices.items()],'edges':self.edges}


def record(rows,extract):
    store=Recorder()
    report=GraphBuilder(store).build_s1(rows,extract)
    return {'counts':report.label_counts,**store.export()}


def main():
    if OUT.exists():raise SystemExit(f'refusing to overwrite: {OUT}')
    cache=json.loads(CACHE.read_text())
    gina_path=ROOT/'results/analysis/m2_gina_preference_storage_20260928.json'
    nate_path=ROOT/'results/analysis/m2_nate_preference_storage_20260928.json'
    gina=json.loads(gina_path.read_text());nate=json.loads(nate_path.read_text())
    targets=['e093b97c-78a8-4fb6-88db-285d47314f4f','7063f8d7-4ba9-408d-acbb-1771d448e885','0877eae7-ffe9-44cf-b7ac-97a7384e84a0','c6aaa7b8-b700-423c-99f6-9f47c6a11ad7']
    current={r['id']:r for r in gina['current_memories']+nate['current_user_memories']}
    live_rows, signature = read_store()
    live = {r[0]:r for r in live_rows if r[2] in ('zhfull:locomo:conv-30','zhfull:locomo:conv-42')}
    assert set(live) == set(current)
    assert all(live[mid][1] == rec['content'] for mid,rec in current.items())
    old={r['id']:r for r in nate['historical_visible_title_hits']}
    rows=[]
    for mid in targets:
        r=current.get(mid) or old[mid]
        user=r.get('user_id') or 'zhfull:locomo:conv-42'
        rows.append(MemoryRecord(mid,r['content'],{'user_id':user,'session_id':r.get('session_id','session_27'),'session_date':r.get('session_date','')}))
    coverage=[]
    for user,mems in [('zhfull:locomo:conv-30',gina['current_memories']),('zhfull:locomo:conv-42',nate['current_user_memories'])]:
        hits=[m['id'] for m in mems if m['id'] in cache]
        coverage.append({'user_id':user,'current_memory_count':len(mems),'cache_id_hits':len(hits),'cache_id_misses':len(mems)-len(hits)})
    actual_projection=record(rows,{mid:cache.get(mid,{}) for mid in targets})
    # Synthetic same-key collision: no inference calls; tests current field handling only.
    synthetic=[];raw={}
    for mid,strength,turns in [('synthetic_strong','explicit_favorite',['S1']),('synthetic_weak','recently_read',['S2'])]:
        synthetic.append(MemoryRecord(mid,'合成记录，仅验证结构字段。',{'user_id':'synthetic-owner','session_id':'s'}))
        raw[mid]={'entities':[{'name':'测试人物','type':'person'}], 'events':[], 'relationships':[],
                  'preferences':[{'subject':'测试人物','pref_type':'like','object':'测试作品',
                                  'support_strength':strength,'source_turn_ids':turns,'source_date':'2026-01-01'}]}
    normalized={mid:StructuredExtractor._normalize(ext) for mid,ext in raw.items()}
    collision=record(synthetic,normalized)
    result={'scope':'Two-case cache freshness and four Memory S1 recording-store projection, plus synthetic field/collision probe; zero external calls and no real graph writes',
            'live_store_logical_signature':signature,'snapshot_id_content_reconfirmed':True,
            'cache_coverage_for_current_ids':coverage,
            'selected_memories':[{'id':r.id,'content':r.content,'metadata':r.metadata,'cached_extraction':cache.get(r.id)} for r in rows],
            'actual_cached_projection':actual_projection,
            'synthetic_raw_extraction':raw,'synthetic_normalized_extraction':normalized,'synthetic_projection':collision,
            'observations':{'current_nate_id_cached':targets[-1] in cache,
                            'normalized_preferences_drop_support_and_source_fields':all(set(ext['preferences'][0])=={'subject','pref_type','object'} for ext in normalized.values()),
                            'synthetic_different_strength_preferences_merge_to_single_node':collision['counts']['Preference']==1},
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CACHE,gina_path,nate_path,ROOT/'src/schema_rsi/graph/builder.py',ROOT/'src/schema_rsi/graph/extractor_s1.py',ROOT/'src/schema_rsi/graph/schema.py',ROOT/'src/schema_rsi/evaluation/retriever.py']},
            'limits':['Replay uses recording store, not the current live graph snapshot.',
                      'Cache existence does not establish original input body/model/prompt identity.',
                      'Synthetic collision is a code-contract result, not observed benchmark harm.',
                      'No retrieval, extraction or answer quality measured.']}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'coverage':coverage,'actual_counts':actual_projection['counts'],'synthetic_counts':collision['counts'],'observations':result['observations']},ensure_ascii=False))


if __name__=='__main__':main()
