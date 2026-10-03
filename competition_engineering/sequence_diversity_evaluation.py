"""Deployment and official search remain gated on prospective replication."""
from competition_engineering import sequence_diversity as campaign
import json
import shutil
import time
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from competition_engineering import frozen_representation_onnx,campaign_onnx
from competition_engineering.manual_search_core import TRAIN_1024,SEARCH,load_combo,predict_combo,SEARCH_ROOT_WP
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.representation_evaluation import full_features,bootstrap_stats
from competition_engineering.pipeline import cached,write_json
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_check_artifacts,_package,validate_callback,replay
from competition_engineering.residual import sufficient
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.search_population_audit import pooled_correlations
from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
from wnn_connectome_starterpack.utils import GlobalAccumulator

def fused_parameters(model,name,linear=False):
    if linear: return {'linear':model[name+'_linear']*model[name+'_linear_strengths']}
    prefixes=[name+'_t'+str(t) for t in [0,1]];strengths=np.array([model[p+'_strength'] for p in prefixes],np.float32)
    linear_coef=np.column_stack([model[p+'_linear'][:,0] for p in prefixes])*strengths
    shared=all(np.array_equal(model[prefixes[0]+k],model[prefixes[1]+k]) for k in ['_w','_b'])
    if shared:
        w=model[prefixes[0]+'_w'];b=model[prefixes[0]+'_b'];o=np.column_stack([model[p+'_o'][:,0] for p in prefixes])*strengths
    else:
        w=np.column_stack([model[p+'_w'] for p in prefixes]);b=np.concatenate([model[p+'_b'] for p in prefixes]);o=np.zeros((64,2),np.float32)
        for t,p in enumerate(prefixes): o[t*32:(t+1)*32,t]=model[p+'_o'][:,0]*strengths[t]
    c=np.array([model[p+'_c'][0] for p in prefixes],np.float32)*strengths
    return {'linear':linear_coef,'w':w,'b':b,'o':o,'c':c}

def export(destination,work,model,name,normalization,linear=False):
    params=fused_parameters(model,name,linear)
    path=work/'artifacts/model.npz';np.savez(path,coef=params['linear'],**normalization,**params)
    frozen_representation_onnx.export(path,destination,work,force_latent=True)
    if linear: return
    graph_model=onnx.load(destination);g=graph_model.graph;last=g.node[-1];assert last.op_type=='Add'
    root=last.input[0];term=next(node for node in g.node if last.input[1] in node.output);assert term.op_type=='MatMul'
    features=term.input[0];counter=0
    def const(value,dtype=np.float32):
        nonlocal counter
        counter+=1;name='diversity_const_'+str(counter);g.initializer.append(nh.from_array(np.asarray(value,dtype=dtype),name));return name
    def op(kind,*inputs,**attrs):
        nonlocal counter
        counter+=1;name='diversity_value_'+str(counter);g.node.append(h.make_node(kind,list(inputs),[name],**attrs));return name
    x=op('Gather',features,const(np.arange(1,371),np.int64),axis=0)
    hidden=op('Tanh',op('Add',op('MatMul',x,const(params['w'])),const(params['b'])))
    correction=op('Add',last.input[1],op('Add',op('MatMul',hidden,const(params['o'])),const(params['c'])))
    g.output[0].name=op('Add',root,correction);onnx.checker.check_model(graph_model);onnx.save(graph_model,destination)

def main(owner=campaign):
    campaign=owner
    out=campaign.OUT;rep=json.loads((out/'replication.json').read_text())
    if rep.get('search_gate_passed') is not True: raise ValueError('Official search blocked by prospective replication')
    if (out/'search_results.json').exists(): print('Frozen comparison already exists');return
    model,_=campaign.load_models()
    with np.load(out/'normalization.npz') as q: norm={k:q[k] for k in q.files}
    variants={'broad_neural':('broad_r0',False),'narrow_neural':('narrow_r0',False),'broad_linear':('broad_r0',True)}
    combo=load_combo();extractor=LatentExtractor();validation={}
    _,probe=next(cached(TRAIN_1024));base=predict_combo(probe,combo);f=full_features(probe,base,norm,extractor,combo)
    for key,(name,linear) in variants.items():
        work=out/('check_'+key);work.mkdir(exist_ok=True);(work/'artifacts').mkdir(exist_ok=True);identity=work/'identity.json'
        if not identity.exists():
            export(work/'artifacts/fused.onnx',work,model,name,norm,linear)
            (work/'callback.py').write_text(campaign_onnx.callback_source('current'),encoding='utf-8')
            artifacts=_check_artifacts(work,{'callback.py':campaign.sha(work/'callback.py')});archive,deployment=_package(work)
            write_json(identity,{'artifacts':artifacts,'deployment':deployment,'archive_sha256':campaign.sha(archive)})
            campaign.event('deployment_frozen',candidate=key,identity_sha256=campaign.sha(identity))
        frozen=json.loads(identity.read_text());assert all(campaign.sha(work/'deploy'/k)==v for k,v in frozen['deployment'].items())
        if (work/'validation.json').exists(): validation[key]=json.loads((work/'validation.json').read_text());continue
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);deadline=time.monotonic()+1800
        checked=validate_callback(sandbox,work,TRAIN_1024,deadline);actual,_=replay(sandbox,work,[probe],deadline)
        expected=base+campaign.corrections(f,model,name,linear);need=probe['need']
        np.testing.assert_allclose(actual[0][need],expected[need],atol=3e-5,rtol=3e-5)
        checked.update(stream_batch_parity=True,label_mask_independence=True,finite_outputs=True,max_abs_parity=float(np.max(np.abs(actual[0][need]-expected[need]))))
        write_json(work/'validation.json',checked);validation[key]=checked
        campaign.event('callback_verified',candidate=key,cpu_us=checked['callback_us_per_row']);print(json.dumps({'candidate':key,'cpu_us':checked['callback_us_per_row']}),flush=True)
    campaign.event('official_search_started',replication_sha256=campaign.sha(out/'replication.json'))
    root=GlobalAccumulator();accs={key:GlobalAccumulator() for key in variants};atlases={key:SearchErrorDiagnostics(root_predictor=lambda z:predict_combo(z,combo)) for key in variants};moments=[]
    for _,z in cached(SEARCH):
        base=predict_combo(z,combo);f=full_features(z,base,norm,extractor,combo);m=z['mask'];root.add(z['y'],base,m);row=[sufficient(z['y'][m],base[m])]
        for key,(name,linear) in variants.items():
            p=base+campaign.corrections(f,model,name,linear);accs[key].add(z['y'],p,m);atlases[key].add(z,p);row.append(sufficient(z['y'][m],p[m]))
        moments.append(row)
    moments=np.array(moments);np.savez(out/'search_moments.npz',moments=moments);reference=root.result();assert abs(reference['weighted_pearson']-SEARCH_ROOT_WP)<1e-12
    counts=bootstrap_counts(64,np.random.default_rng(20261043),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));delta=boot[:,1:]-boot[:,:1];reports={}
    for i,key in enumerate(variants):
        score=accs[key].result();gain=score['weighted_pearson']-reference['weighted_pearson'];targets=[score[t]-reference[t] for t in ['t0','t1']];ci=np.quantile(delta[:,i].mean(-1),[.005,.995]).tolist();cpu=validation[key]['callback_us_per_row']
        eligible=key=='broad_neural' and gain>=.0002 and ci[0]>0 and min(targets)>=-.0002 and cpu<=76.92753780833335
        result={'wp':score['weighted_pearson'],'delta':gain,'target_deltas':targets,'paired99':ci,'cpu_us':cpu,
            'status':'AWAITING_PROMOTION' if eligible else 'insufficient','atlas':atlases[key].result(),'validation':validation[key]}
        write_json(out/('check_'+key)/'report.json',result);reports[key]=result
        if eligible:
            destination=out/'awaiting_promotion'/key;destination.parent.mkdir(exist_ok=True);shutil.copytree(out/('check_'+key)/'deploy',destination)
            write_json(destination/'status.json',{'status':'AWAITING_PROMOTION','report_sha256':campaign.sha(out/('check_'+key)/'report.json')})
        print(json.dumps({k:v for k,v in result.items() if k not in ['atlas','validation']}),flush=True)
    contrast={'delta':reports['broad_neural']['delta']-reports['narrow_neural']['delta'],
        'paired99':np.quantile((delta[:,0]-delta[:,1]).mean(-1),[.005,.995]).tolist()}
    write_json(out/'search_results.json',reports);write_json(out/'search_diversity_contrast.json',contrast)
    campaign.event('official_search_completed',report_sha256=campaign.sha(out/'search_results.json'),contrast=contrast)

if __name__=='__main__': main()
