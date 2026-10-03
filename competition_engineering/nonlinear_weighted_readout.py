"""Single fixed-penalty current-row fit justified by nonlinear mask transport."""
from competition_engineering import reassessment_campaign as campaign
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.special import expit
from competition_engineering.manual_search_core import COMBO,COMBO_SHA,TRAIN_1024,load_combo,predict_combo,assess
from competition_engineering.pipeline import cached,write_json
from competition_engineering.residual import sufficient,from_stats
from competition_engineering import campaign_onnx,autonomous_campaign as lab
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch,_check_artifacts,_package,validate_callback,replay
from connectome.mlevolve_generated import validate_source

OUT=campaign.OUT/'nonlinear_weighted_t0'

def tree_predict(x,nodes):
    if np.any(nodes['is_categorical']):
        raise ValueError('Categorical mask trees unsupported')
    index=np.zeros(len(x),dtype=np.int64)
    for depth in range(len(nodes)):
        leaf=nodes['is_leaf'][index].astype(bool)
        if leaf.all():
            return nodes['value'][index]
        active=np.flatnonzero(~leaf);node=index[active]
        left=x[active,nodes['feature_idx'][node]]<=nodes['num_threshold'][node]
        index[active]=np.where(left,nodes['left'][node],nodes['right'][node])
    raise ValueError('Invalid tree cycle')

def mask_focus(x,base,steps,combo,trees):
    p=np.clip(base,-2,2)
    f=np.column_stack((np.clip((x-combo['mean'])/combo['scale'],-8,8),p,np.abs(p),p*p,steps/20000.,steps>=13333)).astype(float)
    values=[]
    for fold in range(4):
        logits=np.full(len(x),float(trees['fold'+str(fold)+'_baseline'].ravel()[0]))
        for j in range(64):
            logits+=tree_predict(f,trees['fold'+str(fold)+'_tree'+str(j)])
        values.append(expit(logits)/trees['rates'][fold])
    return np.clip(np.mean(values,axis=0),.2,5)

def readout_scores(y,base,correction,focus):
    results=[];y=np.clip(y,-2,2);w=np.abs(y)*focus
    for strength in [0.,.05,.1,.25,.5,1.]:
        p=np.clip(base+strength*correction,-2,2)
        stats=np.array([w.sum(),np.dot(w,y),np.dot(w,p),np.dot(w,y*y),np.dot(w,p*p),np.dot(w,y*p)])
        wp=float(from_stats(np.repeat(stats[:,None],2,axis=1))[0])
        results.append({'strength':strength,'wp':wp})
    return results

def fit_readouts(train_files,output_dir):
    with np.load(Path(__file__).with_name('combo.npz')) as q:
        combo={k:q[k] for k in q.files}
    with np.load(Path(__file__).with_name('mask_trees.npz')) as q:
        trees={k:q[k] for k in q.files}
    rows=[[],[]]
    for count,path in enumerate(train_files):
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        idx=np.flatnonzero(z['need'])[::40];base=predict_combo(z,combo)[idx]
        x=np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)
        f=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),x)).astype(float)
        focus=mask_focus(z['x'][idx],base,z['step'][idx],combo,trees)
        y=np.clip(z['y'][idx,0],-2,2).astype(float)
        rows[count%5==0].append((f,base[:,0],y,focus))
        if (count+1)%128==0:
            print('TRAIN_PROGRESS='+str(count+1),flush=True)
    data=[tuple(np.concatenate([r[j] for r in split]) for j in range(4)) for split in rows]
    f,base,y,focus=data[0];coefficients={};report={};selections={}
    for name,multiplier in [('unweighted',np.ones(len(y))),('nonlinear_weighted',focus)]:
        w=np.abs(y)*multiplier
        coef=np.linalg.solve(f.T@(w[:,None]*f)+np.eye(f.shape[1])*len(y)*1.,f.T@(w*(y-base)))
        coefficients[name]=np.pad(coef.astype(np.float32),(0,224))
        sf,sbase,sy,sfocus=data[1]
        uniform=readout_scores(sy,sbase,sf@coef,np.ones(len(sy)))
        weighted=readout_scores(sy,sbase,sf@coef,sfocus)
        report[name]={'uniform_selection':uniform,'focused_selection':weighted}
        selections[name+'_focused']={'model':name,**max(weighted,key=lambda q:q['wp'])}
        if name=='unweighted':
            selections[name+'_uniform']={'model':name,**max(uniform,key=lambda q:q['wp'])}
    np.savez(output_dir+'/models.npz',**coefficients)
    Path(output_dir+'/training.json').write_text(json.dumps({'models':report,'selections':selections,
        'fit_rows':len(y),'selection_rows':len(data[1][2]),'penalty':1.,
        'focus_quantiles':np.quantile(focus,[0,.1,.5,.9,1]).tolist()}))
    return {'models':2,'fit_rows':len(y),'selection_rows':len(data[1][2])}

def source():
    combo_code=inspect.getsource(predict_combo).replace('f = features({**z, "p": base}, model["mean"], model["scale"])',
        'f = np.column_stack((np.ones(len(base), np.float32), np.clip((z["x"]-model["mean"])/model["scale"], -8, 8), base)).astype(np.float32)')
    code='import numpy as np\nimport json\nfrom pathlib import Path\nfrom scipy.special import expit\n'
    code+='\n'.join(inspect.getsource(fn) for fn in [from_stats,tree_predict,mask_focus,readout_scores])+'\n'+combo_code+'\n'
    code+=inspect.getsource(fit_readouts).replace('def fit_readouts(','def train(')
    validate_source(code,training=True)
    return code

def fit():
    OUT.mkdir(exist_ok=True)
    eligibility=json.loads((campaign.OUT/'nonlinear_selection_result.json').read_text())
    if 0 not in eligibility['eligible_targets']:
        raise ValueError('Prospective diagnostic launch rule not met')
    work=OUT/'isolated_training';config=OUT/'protocol.json'
    if not config.exists():
        protocol={'hypothesis':'Nonlinear causal mask weighting supplies a more faithful offline training population for a t0 correction.',
            'penalty':1.,'tier':1024,'target':0,'training_sampling':'stride40; every fifth sequence selects strength',
            'comparison':'Two matched coefficient fits, unweighted and nonlinear-weighted; uniform/focused training-side WP selection; no search strength grid.',
            'launch_evidence_sha256':campaign.sha(campaign.OUT/'nonlinear_selection_result.json'),
            'rule':{'minimum_delta':.0002,'paired99_lower':0.,'minimum_target_delta':-.0002,'callback_us_max':76.92753780833335},
            'mask_classifier':'Offline weights only, no additional inference state or cost.',
            'train_source_sha256':__import__('hashlib').sha256(source().encode()).hexdigest()}
        write_json(config,protocol);campaign.event('experiment_planned',id='nonlinear_weighted_t0',protocol_sha256=campaign.sha(config))
    if not (work/'artifacts/training.json').exists():
        work.mkdir(exist_ok=True);(work/'train.py').write_text(source(),encoding='utf-8');shutil.copyfile(COMBO,work/'combo.npz')
        with np.load(campaign.OUT/'nonlinear_mask_trees.npz') as q:
            trees={k:q[k] for k in q.files}
        with np.load(campaign.ROOT/'competition_engineering/runs/objective_alignment_20261002/causal_population_classifier.npz') as q:
            trees['rates']=q['rates']
        np.savez(work/'mask_trees.npz',**trees)
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,TRAIN_1024)
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            campaign.event('infrastructure_failure',id='nonlinear_weighted_t0',status=status,error=stderr.decode(errors='replace')[-1500:])
            raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={name:campaign.sha(work/name) for name in ['train.py','combo.npz','mask_trees.npz','artifacts/models.npz','artifacts/training.json']}
    if (work/'fit_identity.json').exists():
        if identity!=json.loads((work/'fit_identity.json').read_text()):
            raise ValueError('Frozen weighted fit changed')
    else:
        if identity['train.py']!=json.loads(config.read_text())['train_source_sha256'] or identity['combo.npz']!=COMBO_SHA:
            raise ValueError('Planned source identity changed')
        write_json(work/'fit_identity.json',identity);campaign.event('fit_frozen',id='nonlinear_weighted_t0',identity=identity)
    with np.load(work/'artifacts/models.npz') as q:
        coefficients={k:q[k] for k in q.files}
    with np.load(campaign.ROOT/'competition_engineering/runs/objective_alignment_20261002/isolated_training/artifacts/models.npz') as q:
        np.testing.assert_allclose(coefficients['unweighted'],q['raw_residual_1.0'],atol=1e-6,rtol=1e-5)
    print((work/'artifacts/training.json').read_text(),flush=True)

def check():
    work=OUT/'isolated_training';identity=json.loads((work/'fit_identity.json').read_text())
    assert all(campaign.sha(work/name)==sha for name,sha in identity.items())
    training=json.loads((work/'artifacts/training.json').read_text());combo=load_combo();results={}
    with np.load(work/'artifacts/models.npz') as q:
        coefficients={k:q[k] for k in q.files}
    for name,selected in training['selections'].items():
        directory=OUT/('checks_'+name)
        if (directory/'report.json').exists():
            results[name]=json.loads((directory/'report.json').read_text());continue
        coef=coefficients[selected['model']];strength=selected['strength']
        def predictor(z,base):
            f=np.column_stack((np.ones(len(base)),np.clip(base,-2,2),np.clip((z['x']-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
            p=base.copy();p[:,0]+=(strength*(f@coef[:115])).astype(np.float32)
            return p
        result=assess(predictor,diagnostics=True);ci=lab.paired99(predictor)
        directory.mkdir(exist_ok=True);(directory/'artifacts').mkdir(exist_ok=True)
        np.savez(directory/'artifacts/model.npz',coef=coef,mean=combo['mean'],scale=combo['scale'])
        campaign_onnx.export(directory/'artifacts/model.npz',directory/'artifacts/fused.onnx','current',strength=strength,target=0)
        (directory/'callback.py').write_text(campaign_onnx.callback_source('current'),encoding='utf-8')
        original={'callback.py':campaign.sha(directory/'callback.py')};artifacts=_check_artifacts(directory,original);archive,deployment=_package(directory)
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,directory);deadline=time.monotonic()+1800
        validation=validate_callback(sandbox,directory,TRAIN_1024,deadline)
        _,z=next(cached(TRAIN_1024));predictions,_=replay(sandbox,directory,[z],deadline);expected=predictor(z,predict_combo(z,combo))
        np.testing.assert_allclose(predictions[0][z['need']],expected[z['need']],atol=3e-5,rtol=3e-5)
        validation.update(stream_batch_parity=True,label_mask_independence=True,finite_outputs=True,max_abs_parity=float(np.max(np.abs(predictions[0][z['need']]-expected[z['need']]))))
        eligible=result['delta_combined']>=.0002 and ci[0]>0 and min(result['delta_per_target'])>=-.0002 and validation['callback_us_per_row']<=76.92753780833335
        report={'selection':selected,'search':result,'paired99':ci,'callback_checks':validation,
            'status':'AWAITING PROMOTION' if eligible else 'scientific_negative_or_insufficient_search_evidence',
            'identity':{'fit':identity,'artifacts':artifacts,'deployment':deployment,'package':campaign.sha(archive)}}
        write_json(directory/'report.json',report);results[name]=report
        if eligible:
            awaiting=OUT/('awaiting_promotion_'+name);shutil.copytree(directory/'deploy',awaiting)
            write_json(awaiting/'status.json',{'status':'AWAITING PROMOTION','report_sha256':campaign.sha(directory/'report.json'),'incumbent_sha256':COMBO_SHA})
        campaign.event('candidate_checked',id=name,wp=result['candidate']['weighted_pearson'],delta=result['delta_combined'],paired99=ci,status=report['status'],callback_us=validation['callback_us_per_row'])
        print(json.dumps({'id':name,'wp':result['candidate']['weighted_pearson'],'delta':result['delta_combined'],'paired99':ci,'cpu_us':validation['callback_us_per_row'],'status':report['status']}),flush=True)
    write_json(OUT/'candidate_checks.json',results)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','check'],required=True)
    args=parser.parse_args();fit() if args.phase=='train' else check()
