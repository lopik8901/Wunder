"""Frozen graph export and replication-gated official search comparison."""
from competition_engineering import nonlinear_state_campaign as campaign
import json
import shutil
import time
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from competition_engineering import frozen_representation_onnx,campaign_onnx
from competition_engineering.pipeline import cached,write_json
from competition_engineering.manual_search_core import TRAIN_1024,SEARCH,SEARCH_ROOT_WP,load_combo,predict_combo
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.representation_evaluation import full_features,bootstrap_stats
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_check_artifacts,_package,validate_callback,replay
from competition_engineering.residual import sufficient
from competition_engineering.search_population_audit import pooled_correlations
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
from wnn_connectome_starterpack.utils import GlobalAccumulator

def export(destination,work,models,family,strengths):
    dummy=work/'artifacts/model.npz'
    np.savez(dummy,coef=np.zeros((371,2),np.float32),mean=models['mean'],scale=models['scale'],
        weights=models[family],strengths=np.array(strengths,np.float32),**{k:models[k] for k in ['pc','ph','pj','bc','bh','bj']})
    frozen_representation_onnx.export(dummy,destination,work,force_latent=True)
    model=onnx.load(destination);graph=model.graph;last=graph.node[-1];assert last.op_type=='Add'
    base=last.input[0];correction=next(n for n in graph.node if last.input[1] in n.output);assert correction.op_type=='MatMul'
    f=correction.input[0];counter=0
    def const(value,dtype=np.float32):
        nonlocal counter
        counter+=1;name='nonlinear_constant_'+str(counter);graph.initializer.append(nh.from_array(np.asarray(value,dtype=dtype),name));return name
    def op(kind,*inputs,**attrs):
        nonlocal counter
        counter+=1;name='nonlinear_value_'+str(counter);graph.node.append(h.make_node(kind,list(inputs),[name],**attrs));return name
    def gather(start,end): return op('Gather',f,const(np.arange(start,end),np.int64),axis=0)
    def tanh(x,w,b): return op('Tanh',op('Add',op('MatMul',x,const(w)),const(b)))
    if family=='linear': design=f
    elif family=='current': design=op('Concat',gather(0,115),tanh(gather(1,115),models['pc'],models['bc']),axis=0)
    elif family=='separable': design=op('Concat',f,tanh(gather(1,115),models['pc'][:,:64],models['bc'][:64]),tanh(gather(115,371),models['ph'],models['bh']),axis=0)
    elif family=='joint': design=op('Concat',f,tanh(gather(1,371),models['pj'],models['bj']),axis=0)
    else: raise ValueError('Unknown family')
    graph.output[0].name=op('Add',base,op('MatMul',design,const(models[family]*np.array(strengths,np.float32))))
    onnx.checker.check_model(model);onnx.save(model,destination)

def require_replication(report):
    if report.get('search_gate_passed') is not True: raise ValueError('Official search blocked: prospective replication failed')

def main():
    out=campaign.OUT;require_replication(json.loads((out/'tanh_replication.json').read_text()))
    if (out/'search_results.json').exists(): print('Frozen search comparison already exists');return
    models,training=campaign.load_models();names=training['families'];combo=load_combo();extractor=LatentExtractor();validations={}
    _,probe=next(cached(TRAIN_1024));base=predict_combo(probe,combo);f=full_features(probe,base,models,extractor,combo)
    for name in names:
        work=out/('check_'+name);work.mkdir(exist_ok=True);(work/'artifacts').mkdir(exist_ok=True)
        strengths=training['choices'][name]['strengths'];path=work/'identity.json'
        if not path.exists():
            export(work/'artifacts/fused.onnx',work,models,name,strengths)
            (work/'callback.py').write_text(campaign_onnx.callback_source('current'),encoding='utf-8')
            artifacts=_check_artifacts(work,{'callback.py':campaign.sha(work/'callback.py')});archive,deployment=_package(work)
            write_json(path,{'artifacts':artifacts,'deployment':deployment,'archive_sha256':campaign.sha(archive),'family':name,'strengths':strengths})
            campaign.event('deployment_frozen',family=name,identity_sha256=campaign.sha(path))
        identity=json.loads(path.read_text());assert all(campaign.sha(work/'deploy'/k)==v for k,v in identity['deployment'].items())
        if (work/'validation.json').exists(): validations[name]=json.loads((work/'validation.json').read_text());continue
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);deadline=time.monotonic()+1800
        validation=validate_callback(sandbox,work,TRAIN_1024,deadline);actual,_=replay(sandbox,work,[probe],deadline)
        expected=base+campaign.features(f,models,name)@models[name]*np.array(strengths,np.float32);need=probe['need']
        np.testing.assert_allclose(actual[0][need],expected[need],atol=3e-5,rtol=3e-5)
        validation.update(stream_batch_parity=True,finite_outputs=True,label_mask_independence=True,max_abs_parity=float(np.max(np.abs(actual[0][need]-expected[need]))))
        write_json(work/'validation.json',validation);validations[name]=validation
        campaign.event('callback_verified',family=name,cpu_us=validation['callback_us_per_row']);print(json.dumps({'family':name,'cpu_us':validation['callback_us_per_row']}),flush=True)
    campaign.event('official_search_started',replication_sha256=campaign.sha(out/'tanh_replication.json'))
    moments=[];root=GlobalAccumulator();accs={name:GlobalAccumulator() for name in names}
    atlases={name:SearchErrorDiagnostics(root_predictor=lambda z:predict_combo(z,combo)) for name in names}
    for _,z in cached(SEARCH):
        base=predict_combo(z,combo);f=full_features(z,base,models,extractor,combo);m=z['mask'];root.add(z['y'],base,m);row=[sufficient(z['y'][m],base[m])]
        for name in names:
            p=base+campaign.features(f,models,name)@models[name]*np.array(training['choices'][name]['strengths'],np.float32)
            accs[name].add(z['y'],p,m);atlases[name].add(z,p);row.append(sufficient(z['y'][m],p[m]))
        moments.append(row)
    moments=np.array(moments);np.savez(out/'search_moments.npz',moments=moments)
    reference=root.result();assert abs(reference['weighted_pearson']-SEARCH_ROOT_WP)<1e-12
    counts=bootstrap_counts(64,np.random.default_rng(20261033),draws=10000)
    boot=pooled_correlations(bootstrap_stats(moments,counts));delta=boot[:,1:]-boot[:,:1];reports={}
    for i,name in enumerate(names):
        score=accs[name].result();gain=score['weighted_pearson']-reference['weighted_pearson'];td=[score[t]-reference[t] for t in ['t0','t1']]
        ci=np.quantile(delta[:,i].mean(-1),[.005,.995]).tolist();cpu=validations[name]['callback_us_per_row']
        eligible=name=='joint' and gain>=.0002 and ci[0]>0 and min(td)>=-.0002 and cpu<=76.92753780833335
        result={'wp':score['weighted_pearson'],'delta':gain,'target_deltas':td,'paired99':ci,'cpu_us':cpu,
            'role':'primary' if name=='joint' else 'mechanism_control','status':'AWAITING_PROMOTION' if eligible else 'insufficient',
            'atlas':atlases[name].result(),'validation':validations[name]}
        write_json(out/('check_'+name)/'report.json',result);reports[name]=result
        if eligible:
            destination=out/'awaiting_promotion'/name;destination.parent.mkdir(exist_ok=True);shutil.copytree(out/('check_'+name)/'deploy',destination)
            write_json(destination/'status.json',{'status':'AWAITING_PROMOTION','report_sha256':campaign.sha(out/('check_'+name)/'report.json')})
        print(json.dumps({k:v for k,v in result.items() if k not in ['atlas','validation']}),flush=True)
    write_json(out/'search_results.json',reports)
    j=names.index('joint');contrast={}
    for name in names[:-1]:
        i=names.index(name);contrast[name]={'delta':reports['joint']['delta']-reports[name]['delta'],
            'paired99':np.quantile((delta[:,j]-delta[:,i]).mean(-1),[.005,.995]).tolist()}
    write_json(out/'search_mechanism.json',contrast)
    campaign.event('official_search_completed',report_sha256=campaign.sha(out/'search_results.json'),contrasts=contrast)

if __name__=='__main__': main()
