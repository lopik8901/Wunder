"""Small learned nonlinear directions after the finite random basis test."""
from competition_engineering import nonlinear_state_campaign as campaign
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
import torch
from competition_engineering.replicated_readout import readout_design,grid_stats,choose_strengths,verify_training_cache
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.pipeline import cached,write_json
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from connectome.mlevolve_generated import validate_source
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations

OUT=campaign.OUT/'learned32'

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    parent=campaign.initialize();assert (campaign.OUT/'search_results.json').exists()
    protocol={'observation':'128 fixed joint nonlinear features replicated only a small increment over linear and failed official transfer; their full callback exceeded practical CPU.',
        'hypothesis':'A learned32-unit nonlinear basis extracts conditional directions missed by fixed random projections, with lower inference arithmetic.',
        'families':['current','separable','joint'],'fit':'Same1024 groups and128 rows/group as the convex experiment. Frozen linear weighted-ridge backbone; fit a32-unit tanh residual head by AdamW lr0.002 decay0.001 batch2048 epochs20. Fixed seed20261034. GRU frozen.',
        'controls':'Current-only head,16 current+16 hidden separable head,32 joint head. Each has a matched fixed linear backbone and zero-initialized nonlinear output.',
        'selection':'Training-only pooled WP selects each target from epochs5,10,20 and strengths0,.05,.1,.25,.5,1 using minimum gain across uniform/probability populations. Save selected per-target heads if epochs differ.',
        'replication_groups':parent['replication_groups']['reserve'],'gate':'Joint positive combined in both populations and both128-group halves; probability paired95 lower>0; joint beats its fixed linear control and current head on both populations, separable on probability population.',
        'search':'Only primary joint if gate passes, one frozen comparison; controls are mechanism diagnostics. Failed replication forbids official search.',
        'failure_diagnosis':'Track training losses and finite gradients; falling MSE alone does not establish WP benefit. Poor fit remains an optimization/function-class result, not proof of absent representation signal.',
        'source_fit_identity_sha256':campaign.sha(campaign.OUT/'tanh_fit/identity.json')}
    write_json(path,protocol)
    knowledge=json.loads((campaign.ROOT/'competition_engineering/search_knowledge.json').read_text());assert knowledge['version']==7
    knowledge['version']=8;knowledge['previous_snapshot']={'path':'runs/nonlinear_state_20261003/knowledge_v7.json','sha256':campaign.sha(campaign.OUT/'knowledge_v7.json')}
    knowledge['weakened'].append({'hypothesis':'A fixed128-feature joint tanh basis resolves linear readout transfer limitations.',
        'reason':'Fresh joint-minus-linear probability WP+0.000020 with95 interval crossing zero; official joint WP0.658388773, delta-0.000447, CPU80.00us.'})
    knowledge['status']='Fixed nonlinear basis complete; test a materially different smaller learned basis on reserved replication.'
    write_json(campaign.OUT/'knowledge_v8.json',knowledge);write_json(campaign.ROOT/'competition_engineering/search_knowledge.json',knowledge)
    campaign.event('experiment_decision',id='random_tanh',decision='Abandon fixed random basis tuning; test learned compact nonlinear directions.',knowledge_sha256=campaign.sha(campaign.OUT/'knowledge_v8.json'))
    campaign.event('learned32_planned',protocol_sha256=campaign.sha(path));return protocol

def head_mask(family):
    mask=np.ones((370,32),np.float32)
    if family=='current': mask[114:]=0
    elif family=='separable': mask[:114,16:]=0;mask[114:,:16]=0
    elif family!='joint': raise ValueError('Unknown family')
    return mask

def predict_head(f,model,prefix):
    linear=f@model[prefix+'_linear'];a=f[:,1:]@model[prefix+'_w']+model[prefix+'_b']
    return linear+np.tanh(a)@model[prefix+'_o']+model[prefix+'_c']

def fit(train_files,output_dir):
    torch.set_num_threads(1);torch.manual_seed(20261034);torch.use_deterministic_algorithms(True)
    with np.load('seed.npz') as q: seed={k:q[k] for k in q.files}
    mean=seed['mean'];scale=seed['scale'];train=[];targets=[];weights=[];selection=[]
    for path in train_files:
        with np.load(path) as q: z={k:q[k] for k in q.files}
        role=int(z['role'])
        if role not in [0,1,2]: raise ValueError('Replication data in fit')
        if role==2:
            f=readout_design(z['current'],z['hidden'],mean,scale)
            selection.append((f,z['y'],z['base'],z['focus']));continue
        idx=np.sort(np.random.default_rng(20261031+int(z['group'])).choice(len(z['y']),128,replace=False))
        f=readout_design(z['current'][idx],z['hidden'][idx],mean,scale)
        y=np.clip(z['y'][idx],-2,2).astype(np.float32)
        train.append(f);targets.append(y-z['base'][idx]);weights.append(np.abs(y))
    x=np.concatenate(train);y=np.concatenate(targets);weight=np.concatenate(weights);del train,targets,weights
    current=x[:,:115].astype(float);current_coef=np.zeros((371,2),np.float32)
    for t in [0,1]:
        gram=current.T@(weight[:,t,None]*current);rhs=current.T@(weight[:,t]*y[:,t])
        current_coef[:115,t]=np.linalg.solve(gram+np.eye(115)*len(x)*.1,rhs)
    del current
    tx=torch.from_numpy(x[:,1:].copy());tw=torch.from_numpy(weight)
    artifacts={'mean':mean,'scale':scale};reports={};selection_moments={}
    for family in ['current','separable','joint']:
        torch.manual_seed(20261034)
        linear=current_coef if family=='current' else seed['linear']
        residual=y-x@linear;ty=torch.from_numpy(residual)
        mask=torch.from_numpy(head_mask(family))
        w=torch.nn.Parameter(torch.randn(370,32)*.06);b=torch.nn.Parameter(torch.zeros(32))
        o=torch.nn.Parameter(torch.zeros(32,2));c=torch.nn.Parameter(torch.zeros(2))
        optimizer=torch.optim.AdamW([w,b,o,c],lr=.002,weight_decay=.001)
        history=[];checkpoints={};scores=[]
        history.append({'epoch':0,'weighted_mse':((weight*residual**2).sum(0)/weight.sum(0)).tolist()})
        for epoch in range(1,21):
            permutation=torch.randperm(len(tx));loss_sum=np.zeros(2)
            for start in range(0,len(tx),2048):
                idx=permutation[start:start+2048]
                output=torch.tanh(tx[idx]@(w*mask)+b)@o+c
                losses=tw[idx]*(output-ty[idx])**2;loss=losses.mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite optimizer loss')
                optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_([w,b,o,c],10.,error_if_nonfinite=True);optimizer.step()
                loss_sum+=losses.detach().sum(0).numpy()
            if epoch in [5,10,20]:
                state={'linear':linear,'w':(w*mask).detach().numpy().copy(),'b':b.detach().numpy().copy(),'o':o.detach().numpy().copy(),'c':c.detach().numpy().copy()}
                pred=x@linear+np.tanh(x[:,1:]@state['w']+state['b'])@state['o']+state['c']
                mse=(weight*(pred-y)**2).sum(0)/weight.sum(0);history.append({'epoch':epoch,'weighted_mse':mse.tolist(),'last_gradient_norm':float(norm)})
                moments=[]
                for f,sy,base,focus in selection:
                    corr=f@linear+np.tanh(f[:,1:]@state['w']+state['b'])@state['o']+state['c']
                    moments.append([grid_stats(sy,base,corr,pop) for pop in [np.ones(len(f)),focus]])
                moments=np.array(moments);choice=choose_strengths(moments.sum(0));scores.append(choice);checkpoints[epoch]=state
                selection_moments[family+'_'+str(epoch)]=moments
                print(json.dumps({'family':family,'epoch':epoch,'mse':mse.tolist(),'strengths':choice['strengths']}),flush=True)
        # Choose checkpoint per target using selection WP only. Store two separate small heads.
        chosen=[]
        for t in [0,1]:
            objective=[max(np.array(item['minimum_population_gains'])[:,t]) for item in scores]
            index=int(np.argmax(objective));epoch=[5,10,20][index];state=checkpoints[epoch];strength=scores[index]['strengths'][t]
            prefix=family+'_'+str(t)
            for k,v in state.items(): artifacts[prefix+'_'+k]=v if k in ['w','b'] else v[...,t:t+1]
            artifacts[prefix+'_strength']=np.array(strength,np.float32);chosen.append({'epoch':epoch,'strength':strength,'minimum_gain':objective[index]})
        artifacts[family+'_baseline']=linear
        reports[family]={'history':history,'chosen':chosen,'selections':scores}
    np.savez(output_dir+'/models.npz',**artifacts);np.savez(output_dir+'/selection_moments.npz',**selection_moments)
    Path(output_dir+'/training.json').write_text(json.dumps({'families':reports,'fit_rows':len(x)}))
    return {'fits':3,'rows':len(x)}

def source():
    code='import numpy as np\nimport torch\nimport json\nfrom pathlib import Path\n'
    code+='\n'.join(inspect.getsource(f) for f in [from_stats,focus_stats,readout_design,choose_strengths,grid_stats,head_mask])+'\n'
    code+=inspect.getsource(fit).replace('def fit(','def train(');validate_source(code,training=True);return code

def train():
    initialize();verify_training_cache();work=OUT/'fit';work.mkdir(exist_ok=True)
    models,_=campaign.load_models()
    if not (work/'artifacts/training.json').exists():
        np.savez(work/'seed.npz',mean=models['mean'],scale=models['scale'],linear=models['linear'])
        (work/'train.py').write_text(source(),encoding='utf-8');campaign.event('learned_fit_started',source_sha256=campaign.sha(work/'train.py'),seed_sha256=campaign.sha(work/'seed.npz'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,campaign.prior.TRAIN)
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            campaign.event('implementation_failure',id='learned32',status=status,error=stderr.decode(errors='replace')[-1200:]);raise RuntimeError(stderr.decode(errors='replace')[-1200:])
    identity={p:campaign.sha(work/p) for p in ['seed.npz','train.py','artifacts/models.npz','artifacts/training.json','artifacts/selection_moments.npz']}
    if (work/'identity.json').exists(): assert identity==json.loads((work/'identity.json').read_text())
    else: write_json(work/'identity.json',identity);campaign.event('learned_models_frozen',identity=identity)
    report=json.loads((work/'artifacts/training.json').read_text());print(json.dumps({k:{'chosen':v['chosen'],'history':v['history']} for k,v in report['families'].items()}),flush=True)

def prepare():
    protocol=initialize();directory=OUT/'replication';directory.mkdir(exist_ok=True);records=[]
    combo=campaign.load_combo();extractor=campaign.LatentExtractor()
    teacher=campaign.ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    for count,group in enumerate(protocol['replication_groups']):
        path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json');origin=campaign.TRAIN_4096/f'{group:05d}.npz'
        if path.exists() and meta.exists():
            row=json.loads(meta.read_text());assert campaign.sha(path)==row['derived_sha256'];records.append(row);continue
        with np.load(origin) as q: z={k:q[k] for k in q.files}
        p,h=extractor.predict(z['x']);np.testing.assert_allclose(p[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
        idx=np.sort(np.random.default_rng(20261035+group).choice(np.flatnonzero(z['need']),500,replace=False));base=campaign.predict_combo(z,combo)[idx]
        current=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
        focus=campaign.probability_focus(z['x'][idx],base,z['step'][idx],combo,trees)[:,1]
        np.savez(path,current=current,hidden=h[idx],base=base,y=z['y'][idx],focus=focus,group=group,indices=idx)
        row={'group':group,'source_sha256':campaign.sha(origin),'derived_sha256':campaign.sha(path)};write_json(meta,row);records.append(row)
        if (count+1)%64==0: print('PREPARED='+str(count+1),flush=True)
    write_json(directory/'identity.json',{'groups':protocol['replication_groups']})
    if not (OUT/'replication_manifest.json').exists():
        write_json(OUT/'replication_manifest.json',{'records':records,'teacher_sha256':campaign.sha(teacher)});campaign.event('learned_replication_prepared',sha256=campaign.sha(OUT/'replication_manifest.json'))

def load_models():
    work=OUT/'fit';identity=json.loads((work/'identity.json').read_text());assert all(campaign.sha(work/k)==v for k,v in identity.items())
    with np.load(work/'artifacts/models.npz') as q: models={k:q[k] for k in q.files}
    return models

def corrections(f,models,family):
    return np.column_stack([predict_head(f,models,family+'_'+str(t))[:,0]*models[family+'_'+str(t)+'_strength'] for t in [0,1]])

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    models=load_models();names=['linear','current','separable','joint'];moments=[]
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'replication_manifest.json').read_text())['records']}
    for _,z in cached(OUT/'replication'):
        assert campaign.sha(OUT/'replication'/f"{int(z['group']):05d}.npz")==expected[int(z['group'])]
        f=readout_design(z['current'],z['hidden'],models['mean'],models['scale'])
        # Fixed linear control retains the original predeclared WP-selected strengths.
        linear=f@models['joint_baseline']*np.array([.5,.25],np.float32)
        ps=[z['base'],z['base']+linear]+[z['base']+corrections(f,models,name) for name in names[1:]]
        moments.append([[focus_stats(z['y'],p,w) for w in [np.ones(len(f)),z['focus']]] for p in ps])
    moments=np.array(moments);np.savez(OUT/'replication_moments.npz',moments=moments)
    counts=bootstrap_counts(256,np.random.default_rng(20261036),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0))
    delta=boot[:,1:]-boot[:,:1];point=scores[1:]-scores[:1]
    variants={name:{'combined_uniform_probability':point[i].mean(-1).tolist(),'targets_uniform_probability':point[i].tolist(),
        'paired95':np.quantile(delta[:,i].mean(-1),[.025,.975],axis=0).tolist(),'paired99':np.quantile(delta[:,i].mean(-1),[.005,.995],axis=0).tolist()} for i,name in enumerate(names)}
    contrasts={name:{'combined_uniform_probability':(point[3]-point[i]).mean(-1).tolist(),'paired95':np.quantile((delta[:,3]-delta[:,i]).mean(-1),[.025,.975],axis=0).tolist()} for i,name in enumerate(names[:3])}
    halves=[]
    for m in [moments[:128],moments[128:]]:
        s=pooled_correlations(m.sum(0));halves.append((s[4]-s[0]).mean(-1).tolist())
    gate=bool(np.all(point[3].mean(-1)>0) and np.all(np.array(halves)>0) and variants['joint']['paired95'][0][1]>0
        and all(np.all(np.array(contrasts[k]['combined_uniform_probability'])>0) for k in ['linear','current']) and contrasts['separable']['combined_uniform_probability'][1]>0)
    result={'variants':variants,'joint_minus_controls':contrasts,'joint_halves_uniform_probability':halves,'search_gate_passed':gate,
        'decision':'Eligible for frozen comparison' if gate else 'No official-search evaluation; learned joint mechanism did not replicate.'}
    write_json(path,result);campaign.event('learned_replication_completed',gate=gate,report_sha256=campaign.sha(path));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','train','prepare','replicate'],required=True)
    args=p.parse_args();{'initialize':initialize,'train':train,'prepare':prepare,'replicate':replicate}[args.phase]()
