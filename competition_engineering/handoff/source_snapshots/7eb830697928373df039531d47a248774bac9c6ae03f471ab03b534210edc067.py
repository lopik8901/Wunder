"""Fixed-row-budget, paired prospective experiment in fitting-sequence diversity."""
from competition_engineering import nonlinear_state_campaign as previous
import argparse
import inspect
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from competition_engineering.pipeline import cached,write_json
from competition_engineering.replicated_readout import readout_design,grid_stats,choose_strengths
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from connectome.mlevolve_generated import validate_source
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations

ROOT=previous.ROOT
OUT=ROOT/'competition_engineering/runs/sequence_diversity_20261003'
sha=previous.sha

def event(kind,**payload):
    OUT.mkdir(exist_ok=True);path=OUT/'ledger.jsonl'
    row={'utc':datetime.now(timezone.utc).isoformat(),'boundary':'search-only','kind':kind,
        'previous_ledger_sha256':sha(path) if path.exists() else None,**payload}
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def sampling_indices(length,group,replica,arm,role):
    if role not in [0,1] or arm not in ['narrow','broad']: raise ValueError('Invalid fitting allocation')
    if role==1 and arm=='narrow': return np.array([],dtype=np.int64)
    permutation=np.random.default_rng(20261040+group+replica*100000).permutation(length)
    return np.sort(permutation[:128 if arm=='narrow' else 64])

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    original=json.loads((previous.prior.OUT/'protocol.json').read_text())
    recent=json.loads((previous.OUT/'protocol.json').read_text())
    narrow=original['groups']['fit_a']+original['groups']['fit_b'];selection=original['groups']['selection']
    extra=recent['replication_groups']['tanh']+recent['replication_groups']['reserve']
    extra+=json.loads((previous.OUT/'state_trees/protocol.json').read_text())['replication_groups']
    extra+=json.loads((previous.OUT/'state_innovation/protocol.json').read_text())['groups']['replication']
    excluded=set(recent['excluded_groups'])|set(extra)
    pool=json.loads((previous.TRAIN_4096/'identity.json').read_text())['groups']
    fresh=np.random.default_rng(20261041).permutation(sorted(set(pool)-excluded))[:512].tolist()
    assert len(narrow)==len(set(narrow))==1024 and len(extra)==len(set(extra))==1024
    assert not set(narrow)&set(extra) and not set(selection)&(set(narrow)|set(extra))
    assert len(fresh)==512 and not set(fresh)&excluded
    model,_=previous.load_models()
    np.savez(OUT/'normalization.npz',mean=model['mean'],scale=model['scale'])
    protocol={'started_unix':time.time(),'incumbent_wp':previous.SEARCH_ROOT_WP if hasattr(previous,'SEARCH_ROOT_WP') else .6588353223316641,
        'incumbent_sha256':previous.COMBO_SHA,
        'observation':'Joint learned heads reduce training residual MSE substantially while training-derived WP selects early checkpoints;1024 fitting sequences may be the limited statistical unit.',
        'hypothesis':'At fixed training rows and model/update budget, increasing fitting-sequence diversity improves fresh-sequence WP and reduces fitting-to-development deterioration.',
        'competing_explanations':['Model misses relevant information','Population composition changes rather than diversity alone explain results','Additional sequence diversity does not improve this readout'],
        'arms':{'narrow':{'sequences':1024,'rows_per_sequence':128},'broad':{'sequences':2048,'rows_per_sequence':64}},
        'fit_rows_per_model':131072,'groups':{'original_fit':narrow,'extra_fit':extra,'selection':selection,'replication':fresh},
        'design':'Both arms contain original1024 sequences. Broad adds all1024 recent development sequences without outcome-based ranking. Shared-group64 rows are a nested random subset of narrow128; two predeclared paired sampling/initialization seeds.',
        'normalization':'Identical frozen normalization learned only on original1024 fitting sequences; no extra or evaluation groups inform it.',
        'model':'371-feature ridge backbone0.1 per row, frozen after fitting within each arm;32-unit joint tanh residual head. GRU frozen. AdamW lr0.002 decay0.001 batch2048 epochs20; identical update count and paired initialization.',
        'selection':'Original256 correction-disjoint training-derived selection groups; targetwise checkpoint5/10/20 and strength0,.05,.1,.25,.5,1 maximize minimum pooled uniform/probability WP gain. Linear controls use same strength selector.',
        'primary':'Broad neural replica0, chosen prospectively. Replica1 tests repeatability; no best-seed choice or ensemble.',
        'gate':'Broad neural replica0 gains positive in both populations and both256-sequence halves, probability paired95 lower>0; both paired replicas improve over narrow in proxy WP; mean paired broad-minus-narrow gains positive both populations and probability paired95 lower>0.',
        'official_search':'Only after gate passes: one frozen comparison of primary broad neural replica0, narrow neural replica0 and broad linear replica0, all versus incumbent. No search retuning.',
        'qualification':recent['qualification'],
        'scope':'Fresh replication is disjoint from correction fitting/selection and earlier research development groups; independence from pretrained GRU is not claimed. Extra groups were observed in earlier campaigns and are now explicitly repurposed as fitting data.',
        'normalization_sha256':sha(OUT/'normalization.npz')}
    write_json(path,protocol)
    knowledge=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text());assert knowledge['version']==11
    write_json(OUT/'starting_knowledge_v11.json',knowledge);knowledge['version']=12
    knowledge['previous_snapshot']={'path':'runs/nonlinear_state_20261003/knowledge_v11.json','sha256':sha(previous.OUT/'knowledge_v11.json')}
    knowledge['status']='Prospective sequence-diversity experiment; frozen model and equal fitting-row/update budgets.'
    write_json(OUT/'knowledge_v12.json',knowledge);write_json(ROOT/'competition_engineering/search_knowledge.json',knowledge)
    event('experiment_preregistered',protocol_sha256=sha(path),knowledge_sha256=sha(OUT/'knowledge_v12.json'));return protocol

def prepare():
    protocol=initialize();training=OUT/'training';training.mkdir(exist_ok=True);replication=OUT/'replication';replication.mkdir(exist_ok=True)
    with np.load(OUT/'normalization.npz') as q: mean=q['mean'];scale=q['scale']
    recent=json.loads((previous.OUT/'protocol.json').read_text())
    sources={g:previous.prior.TRAIN/f'{g:05d}.npz' for g in protocol['groups']['original_fit']+protocol['groups']['selection']}
    for name,groups in [('replication_tanh',recent['replication_groups']['tanh']),('learned32/replication',recent['replication_groups']['reserve']),
        ('state_trees/replication',json.loads((previous.OUT/'state_trees/protocol.json').read_text())['replication_groups']),
        ('state_innovation/replication',json.loads((previous.OUT/'state_innovation/protocol.json').read_text())['groups']['replication'])]:
        sources.update({g:previous.OUT/name/f'{g:05d}.npz' for g in groups})
    records=[]
    for role,name in enumerate(['original_fit','extra_fit','selection']):
        for count,group in enumerate(protocol['groups'][name]):
            path=training/f'{group:05d}.npz';meta=path.with_suffix('.json');origin=sources[group]
            if path.exists() and meta.exists():
                row=json.loads(meta.read_text());assert sha(path)==row['derived_sha256'];records.append(row);continue
            with np.load(origin) as q: z={k:q[k] for k in q.files}
            f=readout_design(z['current'],z['hidden'],mean,scale)
            np.savez(path,f=f,base=z['base'],y=z['y'],focus=z['focus'],indices=z['indices'],group=group,role=role)
            row={'group':group,'role':name,'source_path':origin.relative_to(ROOT).as_posix(),'source_sha256':sha(origin),'derived_sha256':sha(path)}
            write_json(meta,row);records.append(row)
        print('PREPARED_TRAINING='+name,flush=True)
    extractor=previous.LatentExtractor();combo=previous.load_combo()
    teacher=ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    for count,group in enumerate(protocol['groups']['replication']):
        path=replication/f'{group:05d}.npz';meta=path.with_suffix('.json');origin=previous.TRAIN_4096/f'{group:05d}.npz'
        if path.exists() and meta.exists():
            row=json.loads(meta.read_text());assert sha(path)==row['derived_sha256'];records.append(row);continue
        with np.load(origin) as q: z={k:q[k] for k in q.files}
        p,hidden=extractor.predict(z['x']);np.testing.assert_allclose(p[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
        idx=np.sort(np.random.default_rng(20261041+group).choice(np.flatnonzero(z['need']),500,replace=False));base=previous.predict_combo(z,combo)[idx]
        current=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
        f=readout_design(current,hidden[idx],mean,scale);focus=previous.probability_focus(z['x'][idx],base,z['step'][idx],combo,trees)[:,1]
        np.savez(path,f=f,base=base,y=z['y'][idx],focus=focus,indices=idx,group=group,role=3)
        row={'group':group,'role':'replication','source_path':origin.relative_to(ROOT).as_posix(),'source_sha256':sha(origin),'derived_sha256':sha(path)}
        write_json(meta,row);records.append(row)
        if (count+1)%128==0: print('PREPARED_REPLICATION='+str(count+1),flush=True)
    write_json(training/'identity.json',{'groups':sum([protocol['groups'][k] for k in ['original_fit','extra_fit','selection']],[])})
    write_json(replication/'identity.json',{'groups':protocol['groups']['replication']})
    if not (OUT/'prepared_manifest.json').exists():
        write_json(OUT/'prepared_manifest.json',{'records':records,'teacher_sha256':sha(teacher)})
        event('data_prepared',manifest_sha256=sha(OUT/'prepared_manifest.json'))

def head_predict(f,models,prefix):
    return f@models[prefix+'_linear']+np.tanh(f[:,1:]@models[prefix+'_w']+models[prefix+'_b'])@models[prefix+'_o']+models[prefix+'_c']

def validate_allocation(rows,groups,arm):
    if rows!=131072 or groups!=(1024 if arm=='narrow' else 2048):
        raise ValueError('Unequal fitting budget')

def fit(train_files,output_dir):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    selected=[]
    for path in train_files:
        with np.load(path) as q:
            role=int(q['role'])
            if role not in [0,1,2]: raise ValueError('Replication mounted in fitting sandbox')
            if role==2: selected.append((q['f'].copy(),q['y'].copy(),q['base'].copy(),q['focus'].copy()))
    artifacts={};reports={};allmoments={}
    for replica in [0,1]:
        for arm in ['narrow','broad']:
            xs=[];ys=[];ws=[];bases=[];groups=[];gaps=[]
            for path in train_files:
                with np.load(path) as z:
                    role=int(z['role'])
                    if role==2: continue
                    idx=sampling_indices(len(z['f']),int(z['group']),replica,arm,role)
                    if not len(idx): continue
                    y=np.clip(z['y'][idx],-2,2).astype(np.float32)
                    xs.append(z['f'][idx]);ys.append(y-z['base'][idx]);ws.append(np.abs(y));bases.append(z['base'][idx]);groups.append(int(z['group']))
                    gaps.append(float(np.median(np.diff(z['indices'][idx]))))
            x=np.concatenate(xs);y=np.concatenate(ys);weight=np.concatenate(ws);fit_base=np.concatenate(bases);del xs,ys,ws,bases
            validate_allocation(len(x),len(groups),arm)
            linear=np.column_stack([np.linalg.solve(x.astype(float).T@(weight[:,t,None]*x)+np.eye(371)*len(x)*.1,
                x.astype(float).T@(weight[:,t]*y[:,t])) for t in [0,1]]).astype(np.float32)
            name=arm+'_r'+str(replica);artifacts[name+'_linear']=linear
            linear_moments=np.array([[grid_stats(sy,base,f@linear,pop) for pop in [np.ones(len(f)),focus]] for f,sy,base,focus in selected])
            linear_choice=choose_strengths(linear_moments.sum(0));artifacts[name+'_linear_strengths']=np.array(linear_choice['strengths'],np.float32)
            allmoments[name+'_linear']=linear_moments
            tx=torch.from_numpy(x[:,1:].copy());tw=torch.from_numpy(weight);residual=y-x@linear;ty=torch.from_numpy(residual)
            torch.manual_seed(20261034+replica)
            w=torch.nn.Parameter(torch.randn(370,32)*.06);b=torch.nn.Parameter(torch.zeros(32));o=torch.nn.Parameter(torch.zeros(32,2));c=torch.nn.Parameter(torch.zeros(2))
            optimizer=torch.optim.AdamW([w,b,o,c],lr=.002,weight_decay=.001)
            history=[{'epoch':0,'weighted_mse':((weight.astype(float)*residual.astype(float)**2).sum(0)/weight.sum(0,dtype=float)).tolist()}];states={};choices=[]
            for epoch in range(1,21):
                permutation=torch.randperm(len(tx))
                for start in range(0,len(tx),2048):
                    idx=permutation[start:start+2048];prediction=torch.tanh(tx[idx]@w+b)@o+c
                    loss=(tw[idx]*(prediction-ty[idx])**2).mean()
                    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                    optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_([w,b,o,c],10.,error_if_nonfinite=True);optimizer.step()
                if epoch in [5,10,20]:
                    state={'linear':linear,'w':w.detach().numpy().copy(),'b':b.detach().numpy().copy(),'o':o.detach().numpy().copy(),'c':c.detach().numpy().copy()}
                    prediction=x@linear+np.tanh(x[:,1:]@state['w']+state['b'])@state['o']+state['c']
                    mse=(weight.astype(float)*(prediction-y).astype(float)**2).sum(0)/weight.sum(0,dtype=float)
                    moments=[]
                    for f,sy,base,focus in selected:
                        corr=f@linear+np.tanh(f[:,1:]@state['w']+state['b'])@state['o']+state['c']
                        moments.append([grid_stats(sy,base,corr,pop) for pop in [np.ones(len(f)),focus]])
                    moments=np.array(moments);choice=choose_strengths(moments.sum(0));choices.append(choice);states[epoch]=state;allmoments[name+'_epoch'+str(epoch)]=moments
                    fit_wp=from_stats(focus_stats(y+fit_base,fit_base+prediction,np.ones(len(x))))
                    history.append({'epoch':epoch,'weighted_mse':mse.tolist(),'gradient_norm':float(norm),'uniform_training_wp':fit_wp.tolist()})
                    print(json.dumps({'arm':name,'epoch':epoch,'mse':mse.tolist(),'strengths':choice['strengths']}),flush=True)
            chosen=[]
            for t in [0,1]:
                index=int(np.argmax([max(np.array(choice['minimum_population_gains'])[:,t]) for choice in choices]));epoch=[5,10,20][index]
                prefix=name+'_t'+str(t);state=states[epoch]
                for k,v in state.items(): artifacts[prefix+'_'+k]=v if k in ['w','b'] else v[...,t:t+1]
                artifacts[prefix+'_strength']=np.array(choices[index]['strengths'][t],np.float32)
                chosen.append({'epoch':epoch,'strength':choices[index]['strengths'][t]})
            selected_correction=np.column_stack([head_predict(x,artifacts,name+'_t'+str(t))[:,0]*artifacts[name+'_t'+str(t)+'_strength'] for t in [0,1]])
            baseline_wp=from_stats(focus_stats(y+fit_base,fit_base,np.ones(len(x))))
            selected_wp=from_stats(focus_stats(y+fit_base,fit_base+selected_correction,np.ones(len(x))))
            reports[name]={'groups':groups,'fit_rows':len(x),'updates':20*64,'history':history,'chosen':chosen,'linear_choice':linear_choice,'choices':choices,
                'selected_uniform_fit_wp':selected_wp.tolist(),'baseline_uniform_fit_wp':baseline_wp.tolist(),'selected_uniform_fit_gain':(selected_wp-baseline_wp).tolist(),
                'population':{'feature_mean':x.mean(0,dtype=float).tolist(),'feature_std':x.std(0,dtype=float).tolist(),
                    'mean_abs_target':weight.mean(0,dtype=float).tolist(),'target_mean':(y+fit_base).mean(0,dtype=float).tolist(),
                    'residual_mean':y.mean(0,dtype=float).tolist(),'median_sampling_gap':float(np.median(gaps))}}
    np.savez(output_dir+'/models.npz',**artifacts);np.savez(output_dir+'/selection_moments.npz',**allmoments)
    Path(output_dir+'/training.json').write_text(json.dumps(reports));return {'fits':8,'rows_per_model':131072}

def source():
    code='import numpy as np\nimport torch\nimport json\nfrom pathlib import Path\n'
    code+='\n'.join(inspect.getsource(f) for f in [from_stats,focus_stats,grid_stats,choose_strengths,sampling_indices,head_predict,validate_allocation])+'\n'
    code+=inspect.getsource(fit).replace('def fit(','def train(');validate_source(code,training=True);return code

def train():
    protocol=initialize();manifest=json.loads((OUT/'prepared_manifest.json').read_text())
    expected=set(sum([protocol['groups'][k] for k in ['original_fit','extra_fit','selection']],[]))
    assert set(json.loads((OUT/'training/identity.json').read_text())['groups'])==expected and not expected&set(protocol['groups']['replication'])
    for row in manifest['records']:
        if row['role']!='replication': assert sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    work=OUT/'fit';work.mkdir(exist_ok=True)
    if not (work/'artifacts/training.json').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');event('fit_started',source_sha256=sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,OUT/'training')
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            event('implementation_failure',status=status,error=stderr.decode(errors='replace')[-1200:]);raise RuntimeError(stderr.decode(errors='replace')[-1200:])
    identity={p:sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/training.json','artifacts/selection_moments.npz']}
    if (work/'identity.json').exists(): assert identity==json.loads((work/'identity.json').read_text())
    else: write_json(work/'identity.json',identity);event('models_frozen',identity=identity)
    reports=json.loads((work/'artifacts/training.json').read_text());print(json.dumps({k:{'chosen':v['chosen'],'history':v['history']} for k,v in reports.items()}),flush=True)

def load_models():
    work=OUT/'fit';identity=json.loads((work/'identity.json').read_text());assert all(sha(work/k)==v for k,v in identity.items())
    with np.load(work/'artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    return model,json.loads((work/'artifacts/training.json').read_text())

def corrections(f,model,name,linear=False):
    if linear: return f@model[name+'_linear']*model[name+'_linear_strengths']
    return np.column_stack([head_predict(f,model,name+'_t'+str(t))[:,0]*model[name+'_t'+str(t)+'_strength'] for t in [0,1]])

def replication_gate(point,delta,halves):
    # Arrays: replica, allocation(narrow,broad), model(linear,neural), population, target.
    contrast=point[:,1,1]-point[:,0,1]
    difference=delta[:,:,1,1]-delta[:,:,0,1]
    average_ci=np.quantile(difference.mean(axis=(1,3)),[.025,.975],axis=0)
    primary_ci=np.quantile(delta[:,0,1,1].mean(-1),[.025,.975],axis=0)
    gate=bool(np.all(point[0,1,1].mean(-1)>0) and np.all(np.array(halves)>0) and primary_ci[0,1]>0
        and np.all(contrast.mean(-1)[:,1]>0) and np.all(contrast.mean(axis=(0,2))>0) and average_ci[0,1]>0)
    return gate,average_ci,primary_ci

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    model,training=load_models();moments=[]
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_manifest.json').read_text())['records'] if r['role']=='replication'}
    for _,z in cached(OUT/'replication'):
        assert sha(OUT/'replication'/f"{int(z['group']):05d}.npz")==expected[int(z['group'])]
        prediction=[z['base']]
        for r in [0,1]:
            for arm in ['narrow','broad']:
                for linear in [True,False]: prediction.append(z['base']+corrections(z['f'],model,arm+'_r'+str(r),linear))
        moments.append([[focus_stats(z['y'],p,pop) for pop in [np.ones(len(z['f'])),z['focus']]] for p in prediction])
    moments=np.array(moments);np.savez(OUT/'replication_moments.npz',moments=moments)
    counts=bootstrap_counts(512,np.random.default_rng(20261042),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0))
    point=(scores[1:]-scores[:1]).reshape(2,2,2,2,2);delta=(boot[:,1:]-boot[:,:1]).reshape(10000,2,2,2,2,2)
    variants={};contrasts={}
    for r in [0,1]:
        for a,arm in enumerate(['narrow','broad']):
            for m,name in enumerate(['linear','neural']):
                key=arm+'_r'+str(r)+'_'+name
                variants[key]={'combined_uniform_probability':point[r,a,m].mean(-1).tolist(),'targets_uniform_probability':point[r,a,m].tolist(),
                    'paired95':np.quantile(delta[:,r,a,m].mean(-1),[.025,.975],axis=0).tolist(),'paired99':np.quantile(delta[:,r,a,m].mean(-1),[.005,.995],axis=0).tolist()}
        for m,name in enumerate(['linear','neural']):
            contrasts['r'+str(r)+'_'+name]={'combined_uniform_probability':(point[r,1,m]-point[r,0,m]).mean(-1).tolist(),
                'paired95':np.quantile((delta[:,r,1,m]-delta[:,r,0,m]).mean(-1),[.025,.975],axis=0).tolist()}
    halves=[]
    for chunk in [moments[:256],moments[256:]]:
        score=pooled_correlations(chunk.sum(0));halves.append((score[4]-score[0]).mean(-1).tolist())
    gate,average_ci,primary_ci=replication_gate(point,delta,halves)
    result={'variants':variants,'broad_minus_narrow':contrasts,'mean_paired_neural_contrast':{
        'combined_uniform_probability':(point[:,1,1]-point[:,0,1]).mean(axis=(0,2)).tolist(),'paired95':average_ci.tolist()},
        'primary_halves_uniform_probability':halves,'search_gate_passed':gate,
        'decision':'Freeze primary for one search comparison' if gate else 'No official search: sequence-diversity hypothesis did not meet prospective replication rule.',
        'fresh_sequences':512,'fit_rows_each':131072,'bootstrap_draws':10000,'moments_sha256':sha(OUT/'replication_moments.npz')}
    write_json(path,result);event('replication_completed',gate=gate,report_sha256=sha(path));print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['initialize','prepare','train','replicate'],required=True)
    args=parser.parse_args();{'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
