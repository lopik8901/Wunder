"""A distinct causal-information test: existing GRU state updates."""
from competition_engineering import nonlinear_state_campaign as campaign
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
from competition_engineering.pipeline import cached,write_json
from competition_engineering.replicated_readout import readout_design,grid_stats,choose_strengths
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from connectome.mlevolve_generated import validate_source
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations

OUT=campaign.OUT/'state_innovation'

def state_changes(hidden):
    return hidden-np.concatenate((np.zeros_like(hidden[:1]),hidden[:-1]),axis=0)

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    parent=campaign.initialize();old=json.loads((campaign.prior.OUT/'protocol.json').read_text());excluded=set(parent['excluded_groups'])
    for groups in parent['replication_groups'].values(): excluded.update(groups)
    excluded.update(json.loads((campaign.OUT/'state_trees/protocol.json').read_text())['replication_groups'])
    pool=json.loads((campaign.TRAIN_4096/'identity.json').read_text())['groups']
    fresh=np.random.default_rng(20261039).permutation(sorted(set(pool)-excluded))[:256].tolist()
    assert len(fresh)==256 and not set(fresh)&excluded
    result={'observation':'Three distinct nonlinear families offered little replicated increment over the causal-state linear signal. All consumed current updated states, but none explicitly consumed their one-step changes.',
        'hypothesis':'The direction and magnitude of the frozen GRU update carry residual information not accessible to a linear readout of current state and current observations.',
        'competing_explanation':'State updates are redundant with current observations or too noisy to generalize.',
        'fit':'Original1024 sequences, same deterministic128 rows/group as prior matched fits. Linear current+hidden371 control versus same plus256 standardized one-step hidden changes. Ridge0.1 per fit row; weighted residualMSE.',
        'normalization':'Fit-only hidden and update mean/std. Hidden floor0.05; update floor0.005. Clip standardized features to[-8,8]. No lag or regularization grid.',
        'causality':'delta_h[t]=h[t]-h[t-1], h[-1]=0 per sequence. Uses incoming and returned frozen GRU states already in callback; no new state evolution or memory.',
        'selection':'Original256 correction-disjoint selection sequences, pooled uniform/probability WP chooses per-target strength0,.05,.1,.25,.5,1.',
        'groups':{'fit_a':old['groups']['fit_a'],'fit_b':old['groups']['fit_b'],'selection':old['groups']['selection'],'replication':fresh},
        'gate':'Innovation combined positive both populations and both halves; probability paired95 lower>0; innovation-minus-control positive both populations and probability paired95 lower>0.',
        'search':'If gate passes, one frozen primary comparison; otherwise official search prohibited.',
        'interpretation':'This tests one causal feature, not a revival of old EMA/bilinear parameter tuning. Failure closes this one-step linear feature test.'}
    write_json(path,result)
    knowledge=json.loads((campaign.ROOT/'competition_engineering/search_knowledge.json').read_text());assert knowledge['version']==9
    knowledge['version']=10;knowledge['previous_snapshot']={'path':'runs/nonlinear_state_20261003/knowledge_v9.json','sha256':campaign.sha(campaign.OUT/'knowledge_v9.json')}
    knowledge['weakened'].append({'hypothesis':'Shallow frozen-state threshold regimes add a transferable correction beyond linear state features.',
        'reason':'160/192 splits used hidden state, but fresh state-minus-linear WP uniform-0.000047, proxy+0.000013, paired95 spans zero. No official search.'})
    knowledge['status']='Three nonlinear readout mechanisms exhausted locally; testing distinct causal state-update information.'
    write_json(campaign.OUT/'knowledge_v10.json',knowledge);write_json(campaign.ROOT/'competition_engineering/search_knowledge.json',knowledge)
    campaign.event('experiment_decision',id='state_trees',decision='No replicated incremental threshold signal. Branch to previously unused incoming-state information.',knowledge_sha256=campaign.sha(campaign.OUT/'knowledge_v10.json'))
    campaign.event('innovation_planned',protocol_sha256=campaign.sha(path));return result

def prepare():
    protocol=initialize();combo=campaign.load_combo();extractor=campaign.LatentExtractor();records=[]
    teacher=campaign.ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    for role_index,(role,groups) in enumerate(protocol['groups'].items()):
        directory=OUT/('replication' if role=='replication' else 'training');directory.mkdir(exist_ok=True)
        for count,group in enumerate(groups):
            path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json');origin=campaign.TRAIN_4096/f'{group:05d}.npz'
            if path.exists() and meta.exists():
                row=json.loads(meta.read_text());assert campaign.sha(path)==row['derived_sha256'];records.append(row);continue
            with np.load(origin) as q: z={k:q[k] for k in q.files}
            p,h=extractor.predict(z['x']);np.testing.assert_allclose(p[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
            change=state_changes(h)
            if role!='replication':
                with np.load(campaign.prior.TRAIN/f'{group:05d}.npz') as q:
                    indices=q['indices'].copy()
                if role_index<2:
                    pick=np.sort(np.random.default_rng(20261031+group).choice(len(indices),128,replace=False));indices=indices[pick]
            else: indices=np.sort(np.random.default_rng(20261039+group).choice(np.flatnonzero(z['need']),500,replace=False))
            base=campaign.predict_combo(z,combo)[indices]
            current=np.column_stack((np.ones(len(indices)),np.clip(base,-2,2),np.clip((z['x'][indices]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
            focus=campaign.probability_focus(z['x'][indices],base,z['step'][indices],combo,trees)[:,1]
            np.savez(path,current=current,hidden=h[indices],change=change[indices],base=base,y=z['y'][indices],focus=focus,group=group,indices=indices,role=role_index)
            row={'group':group,'role':role,'source_sha256':campaign.sha(origin),'derived_sha256':campaign.sha(path)};write_json(meta,row);records.append(row)
            if (count+1)%128==0: print('INNOVATION_PREPARED='+role+':'+str(count+1),flush=True)
    write_json(OUT/'training/identity.json',{'groups':sum([protocol['groups'][k] for k in ['fit_a','fit_b','selection']],[])})
    write_json(OUT/'replication/identity.json',{'groups':protocol['groups']['replication']})
    if not (OUT/'prepared_manifest.json').exists():
        write_json(OUT/'prepared_manifest.json',{'records':records,'teacher_sha256':campaign.sha(teacher)});campaign.event('innovation_prepared',sha256=campaign.sha(OUT/'prepared_manifest.json'))

def design(current,hidden,change,mean,scale,dm,ds):
    return np.column_stack((readout_design(current,hidden,mean,scale),np.clip((change.astype(float)-dm)/ds,-8,8))).astype(np.float32)

def fit(train_files,output_dir):
    sx=np.zeros(512);sxx=np.zeros(512);n=0
    for path in train_files:
        with np.load(path) as z:
            role=int(z['role'])
            if role not in [0,1,2]: raise ValueError('Replication mounted')
            if role<2:
                h=np.column_stack((z['hidden'],z['change'])).astype(float);sx+=h.sum(0);sxx+=(h*h).sum(0);n+=len(h)
    mean=sx/n;scale=np.sqrt(np.maximum(sxx/n-mean*mean,np.r_[np.full(256,.05**2),np.full(256,.005**2)]))
    grams=np.zeros((2,627,627));rhs=np.zeros((627,2));r2=np.zeros(2);mass=np.zeros(2)
    for count,path in enumerate(train_files):
        with np.load(path) as q: z={k:q[k] for k in q.files}
        if int(z['role'])==2: continue
        f=design(z['current'],z['hidden'],z['change'],mean[:256],scale[:256],mean[256:],scale[256:]).astype(float)
        y=np.clip(z['y'],-2,2).astype(float);w=np.abs(y);r=y-z['base'];r2+=(w*r*r).sum(0);mass+=w.sum(0)
        for t in [0,1]: grams[t]+=f.T@(w[:,t,None]*f);rhs[:,t]+=f.T@(w[:,t]*r[:,t])
        if (count+1)%256==0: print('GRAM_GROUPS='+str(count+1),flush=True)
    models={};metrics={}
    for name,width in [('linear',371),('innovation',627)]:
        c=np.column_stack([np.linalg.solve(grams[t,:width,:width]+np.eye(width)*n*.1,rhs[:width,t]) for t in [0,1]])
        full=np.zeros((627,2),np.float32);full[:width]=c;models[name]=full
        metrics[name]={'weighted_mse':[(r2[t]-2*c[:,t]@rhs[:width,t]+c[:,t]@grams[t,:width,:width]@c[:,t])/mass[t] for t in [0,1]]}
    moments=[]
    for path in train_files:
        with np.load(path) as q: z={k:q[k] for k in q.files}
        if int(z['role'])!=2: continue
        f=design(z['current'],z['hidden'],z['change'],mean[:256],scale[:256],mean[256:],scale[256:])
        moments.append([[grid_stats(z['y'],z['base'],f@models[k],focus) for focus in [np.ones(len(f)),z['focus']]] for k in ['linear','innovation']])
    choices={name:choose_strengths(np.array(moments).sum(0)[i]) for i,name in enumerate(['linear','innovation'])}
    np.savez(output_dir+'/models.npz',mean=mean[:256],scale=scale[:256],dm=mean[256:],ds=scale[256:],**models)
    np.savez(output_dir+'/selection_moments.npz',moments=moments)
    Path(output_dir+'/training.json').write_text(json.dumps({'metrics':metrics,'choices':choices,'fit_rows':n,
        'change_scale_quantiles':np.quantile(scale[256:],[0,.25,.5,.75,1]).tolist()}))
    return {'fits':2,'rows':n}

def source():
    code='import numpy as np\nimport json\nfrom pathlib import Path\n'
    code+='\n'.join(inspect.getsource(f) for f in [from_stats,focus_stats,readout_design,grid_stats,choose_strengths,design])+'\n'
    code+=inspect.getsource(fit).replace('def fit(','def train(');validate_source(code,training=True);return code

def train():
    protocol=initialize();manifest=json.loads((OUT/'prepared_manifest.json').read_text())
    expected=set(sum([protocol['groups'][k] for k in ['fit_a','fit_b','selection']],[]))
    assert set(json.loads((OUT/'training/identity.json').read_text())['groups'])==expected
    assert not expected&set(protocol['groups']['replication'])
    for row in manifest['records']:
        if row['group'] in expected: assert campaign.sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    work=OUT/'fit';work.mkdir(exist_ok=True)
    if not (work/'artifacts/training.json').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');campaign.event('innovation_fit_started',source_sha256=campaign.sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,OUT/'training')
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            campaign.event('implementation_failure',id='state_innovation',status=status,error=stderr.decode(errors='replace')[-1000:]);raise RuntimeError(stderr.decode(errors='replace')[-1000:])
    identity={p:campaign.sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/training.json','artifacts/selection_moments.npz']}
    if (work/'identity.json').exists(): assert identity==json.loads((work/'identity.json').read_text())
    else: write_json(work/'identity.json',identity);campaign.event('innovation_models_frozen',identity=identity)
    r=json.loads((work/'artifacts/training.json').read_text());print(json.dumps({'metrics':r['metrics'],'choices':{k:v['strengths'] for k,v in r['choices'].items()},'change_scale_quantiles':r['change_scale_quantiles']}),flush=True)

def load_models():
    work=OUT/'fit';identity=json.loads((work/'identity.json').read_text());assert all(campaign.sha(work/k)==v for k,v in identity.items())
    with np.load(work/'artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    return model,json.loads((work/'artifacts/training.json').read_text())

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    model,training=load_models();moments=[]
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_manifest.json').read_text())['records'] if r['role']=='replication'}
    for _,z in cached(OUT/'replication'):
        assert campaign.sha(OUT/'replication'/f"{int(z['group']):05d}.npz")==expected[int(z['group'])]
        f=design(z['current'],z['hidden'],z['change'],model['mean'],model['scale'],model['dm'],model['ds'])
        ps=[z['base']]+[z['base']+f@model[k]*np.array(training['choices'][k]['strengths'],np.float32) for k in ['linear','innovation']]
        moments.append([[focus_stats(z['y'],p,w) for w in [np.ones(len(f)),z['focus']]] for p in ps])
    moments=np.array(moments);np.savez(OUT/'replication_moments.npz',moments=moments)
    counts=bootstrap_counts(256,np.random.default_rng(20261040),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0));delta=boot[:,1:]-boot[:,:1];point=scores[1:]-scores[:1]
    variants={name:{'combined_uniform_probability':point[i].mean(-1).tolist(),'targets_uniform_probability':point[i].tolist(),
        'paired95':np.quantile(delta[:,i].mean(-1),[.025,.975],axis=0).tolist(),'paired99':np.quantile(delta[:,i].mean(-1),[.005,.995],axis=0).tolist()} for i,name in enumerate(['linear','innovation'])}
    contrast={'combined_uniform_probability':(point[1]-point[0]).mean(-1).tolist(),'paired95':np.quantile((delta[:,1]-delta[:,0]).mean(-1),[.025,.975],axis=0).tolist()}
    halves=[]
    for m in [moments[:128],moments[128:]]:
        s=pooled_correlations(m.sum(0));halves.append((s[2]-s[0]).mean(-1).tolist())
    gate=bool(np.all(point[1].mean(-1)>0) and np.all(np.array(halves)>0) and variants['innovation']['paired95'][0][1]>0
        and np.all(np.array(contrast['combined_uniform_probability'])>0) and contrast['paired95'][0][1]>0)
    result={'variants':variants,'innovation_minus_linear':contrast,'halves_uniform_probability':halves,'search_gate_passed':gate,
        'decision':'Eligible for frozen comparison' if gate else 'No official search; one-step state information did not establish incremental replicated value.'}
    write_json(path,result);campaign.event('innovation_replication_completed',gate=gate,report_sha256=campaign.sha(path));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','prepare','train','replicate'],required=True)
    args=p.parse_args();{'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
