"""Prospective frozen-state nonlinear readouts; designated training data only."""
from competition_engineering import representation_campaign as prior
import argparse
import hashlib
import inspect
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from competition_engineering.pipeline import cached,write_json
from competition_engineering.manual_search_core import ROOT,COMBO,COMBO_SHA,TRAIN_4096,load_combo,predict_combo
from competition_engineering.replicated_readout import readout_design,choose_strengths,grid_stats,verify_training_cache
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.probability_readout import probability_focus
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from connectome.mlevolve_generated import validate_source
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations

OUT=ROOT/'competition_engineering/runs/nonlinear_state_20261003'
sha=prior.sha

def event(kind,**payload):
    OUT.mkdir(exist_ok=True);path=OUT/'ledger.jsonl'
    row={'utc':datetime.now(timezone.utc).isoformat(),'boundary':'search-only','kind':kind,
        'previous_ledger_sha256':sha(path) if path.exists() else None,**payload}
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    assert sha(COMBO)==COMBO_SHA
    previous=json.loads((prior.OUT/'protocol.json').read_text())
    excluded=set(previous['excluded_recent_tree_selection_groups'])
    for groups in previous['groups'].values(): excluded.update(groups)
    excluded.update(json.loads((prior.OUT/'consensus_ensemble/protocol.json').read_text())['replication_groups'])
    pool=json.loads((TRAIN_4096/'identity.json').read_text())['groups']
    fresh=np.random.default_rng(20261030).permutation(sorted(set(pool)-excluded))[:512].tolist()
    assert len(fresh)==512 and not set(fresh)&excluded
    knowledge=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text());assert knowledge['version']==6
    write_json(OUT/'starting_knowledge_v6.json',knowledge)
    knowledge['version']=7
    knowledge['previous_snapshot']={'path':'runs/representation_20261002/knowledge_v6.json','sha256':sha(prior.OUT/'knowledge_v6.json')}
    knowledge['closed_branches']+=['Disagreement soft-threshold tuning without new evidence','Search-specific position gating without replicated evidence']
    knowledge['exploratory_only'].append({'wp':.6590721033812812,'paired99_delta':[-.0008065760838449942,.0011115846961819937],
        'reason':'Consensus failed matched-control replication and practical CPU constraint; not an incumbent.'})
    knowledge['status']='New prospective nonlinear-state campaign; frozen representation campaign complete.'
    write_json(OUT/'knowledge_v7.json',knowledge);write_json(ROOT/'competition_engineering/search_knowledge.json',knowledge)
    protocol={'started_unix':time.time(),'incumbent_sha256':COMBO_SHA,'incumbent_wp':.6588353223316641,
        'observation':'Frozen-state linear readouts repeatedly improve training-derived populations; conditional nonlinear use of hidden and current features remains untested.',
        'competing_explanations':['Readout function class misses conditional signal','Representation signal is population-specific','Finite-sequence variation dominates tiny gains'],
        'hypothesis':'Joint nonlinear features of frozen GRU state and present observations capture residual signal beyond linear and separable nonlinear controls.',
        'choice':'Random shallow tanh features with a convex ridge readout isolate function-class utility without neural optimization failures. This is a bounded function-class test, not a test of all nonlinear models.',
        'fit':'Existing1024 fitting sequences, deterministic128 of500 cached rows each; hidden normalization estimated on these1024 only. Same weighted residual MSE and ridge0.1 per row for all controls.',
        'families':['linear','current','separable','joint'],
        'features':'linear371; current115+128 tanh(current); separable371+64 tanh(current)+64 tanh(hidden); joint371+128 tanh(joint). Fixed seed20261031, Gaussian unit-variance projections and uniform[-1,1] offsets.',
        'selection':'Existing correction-disjoint256 selection groups; per-target strengths0,.05,.1,.25,.5,1 maximize minimum pooled WP gain across uniform and frozen probability populations. Primary family joint fixed before fitting.',
        'replication_groups':{'tanh':fresh[:256],'reserve':fresh[256:]},'excluded_groups':sorted(excluded),
        'replication_gate':'Joint combined point gain positive on both populations and both128-sequence halves; probability population paired95 lower>0; joint-minus-linear and joint-minus-current positive on both populations; joint-minus-separable probability point>0. Failure blocks official-search access for this hypothesis.',
        'search':'Only if replication gate passes: one frozen joint candidate compared with incumbent, plus frozen controls for mechanism; no official-search retuning.',
        'qualification':previous['qualification'],'uncertainty':'Sequence bootstrap; descriptive for reused search. Fresh groups independent of correction fitting/selection, not guaranteed independent of pretrained GRU.',
        'boundary':'No protected data, feedback, promotion or submission. Prior protected information encountered during archive handoff is excluded from experiment design.',
        'source_manifest_sha256':sha(prior.OUT/'prepared_manifest.json')}
    write_json(path,protocol);event('campaign_initialized',protocol_sha256=sha(path),knowledge_sha256=sha(OUT/'knowledge_v7.json'))
    return protocol

def prepare():
    protocol=initialize();directory=OUT/'replication_tanh';directory.mkdir(exist_ok=True)
    combo=load_combo();extractor=LatentExtractor();records=[]
    teacher=ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    for count,group in enumerate(protocol['replication_groups']['tanh']):
        path=directory/f'{group:05d}.npz';meta=directory/f'{group:05d}.json';origin=TRAIN_4096/f'{group:05d}.npz'
        if path.exists() and meta.exists():
            record=json.loads(meta.read_text());assert sha(path)==record['derived_sha256'];records.append(record);continue
        with np.load(origin) as q: z={k:q[k] for k in q.files}
        p,hidden=extractor.predict(z['x']);np.testing.assert_allclose(p[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
        idx=np.sort(np.random.default_rng(20261030+group).choice(np.flatnonzero(z['need']),500,replace=False))
        base=predict_combo(z,combo)[idx]
        current=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
        focus=probability_focus(z['x'][idx],base,z['step'][idx],combo,trees)[:,1]
        np.savez(path,current=current,hidden=hidden[idx],base=base,y=z['y'][idx],focus=focus,group=group,indices=idx)
        record={'group':group,'source_sha256':sha(origin),'derived_sha256':sha(path)};write_json(meta,record);records.append(record)
        if (count+1)%64==0: print('PREPARED='+str(count+1),flush=True)
    write_json(directory/'identity.json',{'groups':protocol['replication_groups']['tanh']})
    if not (OUT/'replication_manifest.json').exists():
        write_json(OUT/'replication_manifest.json',{'records':records,'teacher_sha256':sha(teacher)})
        event('replication_prepared',manifest_sha256=sha(OUT/'replication_manifest.json'))

def projections():
    rng=np.random.default_rng(20261031)
    current=rng.normal(size=(114,128)).astype(np.float32)/np.sqrt(114)
    hidden=rng.normal(size=(256,64)).astype(np.float32)/16
    joint=np.vstack((rng.normal(size=(114,128))/np.sqrt(228),rng.normal(size=(256,128))/np.sqrt(512))).astype(np.float32)
    return {'pc':current.astype(np.float32),'ph':hidden.astype(np.float32),'pj':joint,'bc':rng.uniform(-1,1,128).astype(np.float32),
        'bh':rng.uniform(-1,1,64).astype(np.float32),'bj':rng.uniform(-1,1,128).astype(np.float32)}

def features(f,params,family):
    if family=='linear': return f
    if family=='current': return np.column_stack((f[:,:115],np.tanh(f[:,1:115]@params['pc']+params['bc']))).astype(np.float32)
    if family=='separable': return np.column_stack((f,np.tanh(f[:,1:115]@params['pc'][:,:64]+params['bc'][:64]),np.tanh(f[:,115:]@params['ph']+params['bh']))).astype(np.float32)
    if family=='joint': return np.column_stack((f,np.tanh(f[:,1:]@params['pj']+params['bj']))).astype(np.float32)
    raise ValueError('Unknown feature family')

def fit(train_files,output_dir):
    params=projections();sx=np.zeros(256);sxx=np.zeros(256);count=0
    for path in train_files:
        with np.load(path) as z:
            role=int(z['role'])
            if role not in [0,1,2]: raise ValueError('Replication mounted in trainer')
            if role<2:
                h=z['hidden'].astype(float);sx+=h.sum(0);sxx+=(h*h).sum(0);count+=len(h)
    mean=sx/count;scale=np.sqrt(np.maximum(sxx/count-mean*mean,.05**2))
    dims={'linear':371,'current':243,'separable':499,'joint':499}
    grams={k:np.zeros((2,d,d)) for k,d in dims.items()};rhs={k:np.zeros((d,2)) for k,d in dims.items()};r2=np.zeros(2);mass=np.zeros(2);n=0
    for index,path in enumerate(train_files):
        with np.load(path) as q: z={k:q[k] for k in q.files}
        if int(z['role'])==2: continue
        idx=np.sort(np.random.default_rng(20261031+int(z['group'])).choice(len(z['y']),128,replace=False))
        f=readout_design(z['current'][idx],z['hidden'][idx],mean,scale)
        y=np.clip(z['y'][idx],-2,2).astype(float);w=np.abs(y);r=y-z['base'][idx];n+=len(f);mass+=w.sum(0);r2+=(w*r*r).sum(0)
        for name in dims:
            x=features(f,params,name).astype(float)
            for t in [0,1]: grams[name][t]+=x.T@(w[:,t,None]*x);rhs[name][:,t]+=x.T@(w[:,t]*r[:,t])
        if (index+1)%128==0: print('FIT_SEQUENCES='+str(index+1),flush=True)
    models={};metrics={}
    for name,dim in dims.items():
        c=np.column_stack([np.linalg.solve(grams[name][t]+np.eye(dim)*n*.1,rhs[name][:,t]) for t in [0,1]])
        models[name]=c.astype(np.float32)
        metrics[name]={'weighted_mse':[(r2[t]-2*c[:,t]@rhs[name][:,t]+c[:,t]@grams[name][t]@c[:,t])/mass[t] for t in [0,1]],
            'normal_equation_relative_residual':[float(np.linalg.norm((grams[name][t]+np.eye(dim)*n*.1)@c[:,t]-rhs[name][:,t])/max(np.linalg.norm(rhs[name][:,t]),1e-30)) for t in [0,1]]}
    moments=[]
    for path in train_files:
        with np.load(path) as q: z={k:q[k] for k in q.files}
        if int(z['role'])!=2: continue
        f=readout_design(z['current'],z['hidden'],mean,scale);row=[]
        for name in dims:
            correction=features(f,params,name)@models[name]
            row.append([grid_stats(z['y'],z['base'],correction,focus) for focus in [np.ones(len(f)),z['focus']]])
        moments.append(row)
    choices={name:choose_strengths(np.array(moments).sum(0)[i]) for i,name in enumerate(dims)}
    np.savez(output_dir+'/models.npz',mean=mean,scale=scale,**params,**models)
    np.savez(output_dir+'/selection_moments.npz',moments=moments)
    Path(output_dir+'/training.json').write_text(json.dumps({'metrics':metrics,'choices':choices,'fit_rows':n,'families':list(dims)}))
    return {'fits':4,'rows':n}

def source():
    code='import numpy as np\nimport json\nfrom pathlib import Path\n'
    code+='\n'.join(inspect.getsource(f) for f in [from_stats,focus_stats,readout_design,choose_strengths,grid_stats,projections,features])+'\n'
    code+=inspect.getsource(fit).replace('def fit(','def train(')
    validate_source(code,training=True);return code

def train():
    initialize();verify_training_cache();work=OUT/'tanh_fit';work.mkdir(exist_ok=True)
    if not (work/'artifacts/training.json').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');event('fit_started',source_sha256=sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,prior.TRAIN)
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            event('implementation_failure',status=status,error=stderr.decode(errors='replace')[-1000:]);raise RuntimeError(stderr.decode(errors='replace')[-1000:])
    identity={str(p.relative_to(work)).replace('\\','/'):sha(p) for p in [work/'train.py',work/'artifacts/models.npz',work/'artifacts/training.json',work/'artifacts/selection_moments.npz']}
    if (work/'identity.json').exists(): assert identity==json.loads((work/'identity.json').read_text())
    else: write_json(work/'identity.json',identity);event('models_frozen',identity=identity)
    report=json.loads((work/'artifacts/training.json').read_text());print(json.dumps({'metrics':report['metrics'],'choices':{k:v['strengths'] for k,v in report['choices'].items()}}),flush=True)

def load_models():
    work=OUT/'tanh_fit';identity=json.loads((work/'identity.json').read_text())
    assert all(sha(work/k)==v for k,v in identity.items())
    with np.load(work/'artifacts/models.npz') as q: models={k:q[k] for k in q.files}
    training=json.loads((work/'artifacts/training.json').read_text());return models,training

def replicate():
    path=OUT/'tanh_replication.json'
    if path.exists(): print(path.read_text());return
    models,training=load_models();moments=[];names=training['families']
    manifest=json.loads((OUT/'replication_manifest.json').read_text())
    expected={r['group']:r['derived_sha256'] for r in manifest['records']}
    for group,z in cached(OUT/'replication_tanh'):
        assert sha(OUT/'replication_tanh'/f"{int(z['group']):05d}.npz")==expected[int(z['group'])]
        f=readout_design(z['current'],z['hidden'],models['mean'],models['scale'])
        ps=[z['base']]+[z['base']+features(f,models,k)@models[k]*np.array(training['choices'][k]['strengths'],np.float32) for k in names]
        moments.append([[focus_stats(z['y'],p,w) for w in [np.ones(len(f)),z['focus']]] for p in ps])
    moments=np.array(moments);np.savez(OUT/'tanh_replication_moments.npz',moments=moments)
    counts=bootstrap_counts(len(moments),np.random.default_rng(20261032),draws=10000)
    scores=pooled_correlations(moments.sum(0));boot=pooled_correlations(bootstrap_stats(moments,counts));delta=boot[:,1:]-boot[:,:1];points=scores[1:]-scores[:1]
    variants={k:{'combined_uniform_probability':points[i].mean(-1).tolist(),'targets_uniform_probability':points[i].tolist(),
        'paired95':np.quantile(delta[:,i].mean(-1),[.025,.975],axis=0).tolist(),'paired99':np.quantile(delta[:,i].mean(-1),[.005,.995],axis=0).tolist()} for i,k in enumerate(names)}
    j=names.index('joint');contrasts={}
    for k in ['linear','current','separable']:
        i=names.index(k);contrasts[k]={'combined_uniform_probability':(points[j]-points[i]).mean(-1).tolist(),
            'paired95':np.quantile((delta[:,j]-delta[:,i]).mean(-1),[.025,.975],axis=0).tolist()}
    halves=[]
    for m in [moments[:128],moments[128:]]:
        score=pooled_correlations(m.sum(0));halves.append((score[j+1]-score[0]).mean(-1).tolist())
    gate=bool(np.all(points[j].mean(-1)>0) and np.all(np.array(halves)>0) and variants['joint']['paired95'][0][1]>0
        and all(np.all(np.array(contrasts[k]['combined_uniform_probability'])>0) for k in ['linear','current'])
        and contrasts['separable']['combined_uniform_probability'][1]>0)
    result={'variants':variants,'joint_minus_controls':contrasts,'joint_halves_uniform_probability':halves,'search_gate_passed':gate,
        'decision':'Freeze candidate for one search comparison' if gate else 'No official search: the preregistered nonlinear interaction hypothesis failed replication.',
        'moments_sha256':sha(OUT/'tanh_replication_moments.npz')}
    write_json(path,result);event('replication_completed',gate=gate,report_sha256=sha(path));print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['initialize','prepare','train','replicate'],required=True)
    args=parser.parse_args();{'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
