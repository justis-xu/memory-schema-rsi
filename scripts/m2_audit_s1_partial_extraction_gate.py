#!/usr/bin/env python3
"""Offline fault injection for actual StructuredExtractor and S1 construction guard."""
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from schema_rsi.graph.extractor_s1 import StructuredExtractor
from schema_rsi.graph.builder import GraphBuilder
from schema_rsi.memory.base import MemoryRecord
from m2_audit_preference_graph_projection import Recorder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/m2_s1_partial_extraction_gate_20260928.json'
EMPTY={'entities':[],'events':[],'preferences':[],'relationships':[]}
NONEMPTY={'entities':[{'name':'测试人物','type':'person'}],'events':[],'preferences':[],'relationships':[]}
ROWS=[MemoryRecord(mid,'合成记忆：只验证控制流程。',{'user_id':'synthetic-owner','session_id':'s'}) for mid in ('A','B','C')]


class Client:
    def __init__(self,response=None):self.calls=0;self.response=response
    def complete(self,**kwargs):
        self.calls+=1
        if self.response is None:raise RuntimeError('synthetic extraction failure; no network')
        return json.dumps(self.response,ensure_ascii=False),{}


class GuardRecorder(Recorder):
    def __init__(self):
        super().__init__();self.reset_calls=0;self.vertices[('Memory','old-sentinel')]={'memory_id':'old-sentinel'}
    def reset_graph(self,**kwargs):
        self.reset_calls+=1;super().reset_graph(**kwargs)


def build(ext,reset):
    store=GuardRecorder()
    result={'reset_requested':reset}
    try:
        r=GraphBuilder(store).build_s1(ROWS,ext,reset=reset)
        result.update({'accepted':True,'report_counts':r.label_counts})
    except Exception as exc:
        result.update({'accepted':False,'error_type':type(exc).__name__,'reason':str(exc)})
    result.update({'reset_calls':store.reset_calls,'old_sentinel_retained':('Memory','old-sentinel') in store.vertices,
                   'recorded_graph':store.export()})
    return result


def extract(cache,client,batch_size):
    with tempfile.TemporaryDirectory(prefix='m2_s1_gate_') as folder:
        path=Path(folder)/'cache.json';path.write_text(json.dumps(cache))
        settings=SimpleNamespace(raw={'graph':{'structured_cache':str(path),'extract_batch_size':batch_size}},resolve_path=lambda p:Path(p))
        extractor=StructuredExtractor(settings,client=client)
        result=extractor.extract(ROWS)
        saved=json.loads(path.read_text())
        return {'input_cache':cache,'output':result,'saved_cache':saved,'fake_client_calls':client.calls,
                'returned_id_count':len(result),'nonempty_id_count':sum(any(r.get(k) for k in EMPTY) for r in result.values()),
                'build_reset_true':build(result,True),'build_reset_false':build(result,False)}


def main():
    if OUT.exists():raise SystemExit(f'refusing to overwrite: {OUT}')
    scenarios={
        'mixed_cache_and_missing_request_failure':extract({'A':NONEMPTY,'B':EMPTY},Client(),1),
        'valid_json_omits_two_requested_ids':extract({},Client({'A':NONEMPTY}),3),
        'all_request_failures':extract({},Client(),1),
        'valid_complete_empty_structures':extract({},Client({'A':EMPTY,'B':EMPTY,'C':EMPTY}),3),
    }
    # Replay only the literal skip predicate, never call build_graph_zh.main or real stores.
    old_ids=['old-A','old-B','old-C'];new_ids=['new-A','new-B','new-C']
    before={'Memory':3,'V8Cluster':1};force=False
    skip=not force and before.get('Memory')==len(new_ids) and before.get('V8Cluster',0)>0
    report={'scope':'Four synthetic failure/empty-response scenarios through actual extractor/build_s1, in temporary cache and recording store; zero model/service calls',
            'scenarios':scenarios,'same_count_changed_ids_skip_predicate':{'old_ids':old_ids,'new_ids':new_ids,'before_counts':before,'force':force,'would_skip':skip},
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/schema_rsi/graph/extractor_s1.py',ROOT/'src/schema_rsi/graph/builder.py',ROOT/'scripts/build_graph_zh.py',ROOT/'scripts/eval_s1_ab.py',ROOT/'experiments/schema_rsi/schema_rsi_lab/graph.py',ROOT/'experiments/schema_rsi/schema_rsi_lab/evolve.py']},
            'limits':['Injected responses/failures are synthetic, not observed historical failure rates.',
                      'Recorded graph is not a HugeGraph integration or live state.',
                      'RSI lab uses its own candidate-pool compiler; this S1 gate is not its promotion gate.',
                      'No production code or graph data changed.']}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({name:{'fake_calls':r['fake_client_calls'],'nonempty':r['nonempty_id_count'],'accepted':r['build_reset_true']['accepted'],'reset_calls':r['build_reset_true']['reset_calls']} for name,r in scenarios.items()},ensure_ascii=False))


if __name__=='__main__':main()
