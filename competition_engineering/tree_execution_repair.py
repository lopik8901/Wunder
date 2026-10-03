"""Audit execution cost of the frozen shallow-tree experiment, without refitting."""
from competition_engineering import root_cause_campaign as campaign
import argparse
import json
import time
import joblib
import numpy as np
from competition_engineering import compact_tree_onnx, campaign_onnx
from competition_engineering.manual_search_core import TRAIN_1024,load_combo,predict_combo,assess
from competition_engineering.manual_tiny_tree_t0 import design
from competition_engineering.autonomous_campaign import paired99
from competition_engineering.pipeline import cached,write_json
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_check_artifacts,_package,validate_callback,replay
from connectome.mlevolve_generated import validate_source

OUT=campaign.OUT/'tree_execution'
OLD=campaign.ROOT/'competition_engineering/runs/manual_tiny_tree_t0_20261001'

def prepare():
    OUT.mkdir(exist_ok=True)
    protocol=OUT/'protocol.json'
    if not protocol.exists():
        write_json(protocol,{'hypothesis':'Most of the measured single-row tree cost came from Python estimator overhead rather than the compact tree mechanism.',
            'fixed_candidate':'Historical32-tree7-leaf t0 residual, strength0.25, no refit or strength selection.',
            'old_artifact_sha256':campaign.sha(OLD/'tree.joblib'),
            'old_report_sha256':campaign.sha(OLD/'report.json'),
            'comparison':'Compile same leaf values and split routing to ONNX, fuse with identical incumbent, preserve all callback checks.',
            'rule':'Actual complete callback must pass90us infrastructure gate and76.92753780833335us practical threshold; compare frozen candidate score and paired99 uncertainty.',
            'parity':'Training sequence rowwise parity plus synthetic random and split-boundary regression tests.',
            'scientific_limit':'Removing execution overhead does not establish a score gain or justify treating historical adaptive strength as prospective.'})
        campaign.event('experiment_planned',id='tree_execution',protocol_sha256=campaign.sha(protocol))
    model=joblib.load(OLD/'tree.joblib')
    if not (OUT/'artifacts').exists():
        (OUT/'artifacts').mkdir()
        trees=[p[0].nodes for p in model._predictors]
        np.savez(OUT/'frozen_trees.npz',baseline=model._baseline_prediction,**{'tree'+str(i):n for i,n in enumerate(trees)})
        compact_tree_onnx.export(trees,float(model._baseline_prediction.ravel()[0]),OUT/'artifacts/fused.onnx',OUT,strength=.25,target=0)
        code=campaign_onnx.callback_source('current');validate_source(code,training=False)
        (OUT/'callback.py').write_text(code,encoding='utf-8')
        artifacts=_check_artifacts(OUT,{'callback.py':campaign.sha(OUT/'callback.py')})
        archive,deployment=_package(OUT)
        write_json(OUT/'export_identity.json',{'artifacts':artifacts,'deployment':deployment,'archive_sha256':campaign.sha(archive),
            'archive_bytes':archive.stat().st_size,'tree_sha256':campaign.sha(OUT/'frozen_trees.npz'),
            'source_sha256':campaign.sha(compact_tree_onnx.__file__)})
        campaign.event('execution_export_frozen',id='tree_execution',identity_sha256=campaign.sha(OUT/'export_identity.json'))
    return model

def check():
    model=prepare();combo=load_combo();destination=OUT/'report.json'
    if destination.exists():
        print(destination.read_text());return
    def predictor(z,base):
        p=base.copy();p[:,0]+=.25*model.predict(design(z,combo,base,np.arange(len(base)))).astype(np.float32)
        return p
    sandbox=GeneratedSandbox();sandbox.preflight(WORKER,OUT);deadline=time.monotonic()+1800
    validation=validate_callback(sandbox,OUT,TRAIN_1024,deadline)
    _,z=next(cached(TRAIN_1024));actual,_=replay(sandbox,OUT,[z],deadline)
    expected=predictor(z,predict_combo(z,combo))
    np.testing.assert_allclose(actual[0][z['need']],expected[z['need']],atol=3e-5,rtol=3e-5)
    validation.update(stream_batch_parity=True,label_mask_independence=True,finite_outputs=True,
        max_abs_parity=float(np.max(np.abs(actual[0][z['need']]-expected[z['need']]))))
    result=assess(predictor,diagnostics=True);ci=paired99(predictor)
    old=json.loads((OLD/'report.json').read_text())
    assert abs(result['candidate']['weighted_pearson']-old['search_result']['candidate']['weighted_pearson'])<1e-9
    report={'search':result,'paired99':ci,'callback_checks':validation,
        'practical_cpu_pass':validation['callback_us_per_row']<=76.92753780833335,
        'status':'historical_exploratory_candidate_execution_repaired',
        'identity':json.loads((OUT/'export_identity.json').read_text()),
        'decision':'Execution feasibility can reopen compact-tree research, but historical search uncertainty still decides whether this frozen candidate improves the incumbent.'}
    write_json(destination,report);campaign.event('experiment_completed',id='tree_execution',report_sha256=campaign.sha(destination),
        wp=result['candidate']['weighted_pearson'],paired99=ci,cpu_us=validation['callback_us_per_row'])
    print(json.dumps({'wp':result['candidate']['weighted_pearson'],'paired99':ci,'callback':validation,'status':report['status']}),flush=True)

def microbenchmark():
    """Same rows, one CPU and serialized runs isolate estimator overhead."""
    import os
    import onnxruntime as ort
    from threadpoolctl import threadpool_limits
    destination=OUT/'matched_microbenchmark.json'
    if destination.exists():
        print(destination.read_text());return
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    campaign.event('experiment_planned',id='tree_execution_matched_timing',
        scope='Frozen correction only, same2000 training feature rows, three alternating serialized timings on one CPU; no score selection.')
    model=joblib.load(OLD/'tree.joblib');combo=load_combo();_,z=next(cached(TRAIN_1024))
    idx=np.flatnonzero(z['need'])[:2000];base=predict_combo(z,combo)
    x=design(z,combo,base,idx);rows=[row.reshape(1,-1) for row in x]
    trees=[p[0].nodes for p in model._predictors]
    graph=compact_tree_onnx.ensemble_model(trees,float(model._baseline_prediction.ravel()[0]),114)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    functions={'sklearn':lambda row:model.predict(row)[0],
               'onnx':lambda row:session.run(None,{'x':row})[0][0,0]}
    timings={k:[] for k in functions};values={}
    with threadpool_limits(limits=1):
        for fn in functions.values():
            for row in rows[:100]:
                fn(row)
        for repeat in range(3):
            for name in (['sklearn','onnx'] if repeat%2==0 else ['onnx','sklearn']):
                started=time.perf_counter();values[name]=np.array([functions[name](row) for row in rows])
                timings[name].append((time.perf_counter()-started)/len(rows)*1e6)
    np.testing.assert_allclose(values['sklearn'],values['onnx'],atol=8e-7,rtol=8e-7)
    result={'microseconds_per_row':timings,'median_us':{k:float(np.median(v)) for k,v in timings.items()},
        'rows':len(rows),'repeats':3,'cpu_count':1,'max_abs_parity':float(np.max(np.abs(values['sklearn']-values['onnx']))),
        'scope':'Correction-only API timings; separate from complete isolated callback measurements.'}
    write_json(destination,result);campaign.event('experiment_completed',id='tree_execution_matched_timing',report_sha256=campaign.sha(destination))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','check','microbenchmark'],required=True)
    args=parser.parse_args()
    {'prepare':prepare,'check':check,'microbenchmark':microbenchmark}[args.phase]()
