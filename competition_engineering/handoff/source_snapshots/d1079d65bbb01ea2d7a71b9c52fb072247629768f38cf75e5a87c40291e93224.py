"""Matched current and frozen-state readouts, with training-only selection."""
from competition_engineering import representation_campaign as campaign
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.pipeline import write_json
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from connectome.mlevolve_generated import validate_source

OUT=campaign.OUT/'replicated_readout'

def readout_design(current,hidden,mean,scale):
    return np.column_stack((current,np.clip((hidden.astype(float)-mean)/scale,-8,8))).astype(np.float32)

def grid_stats(y,base,correction,focus):
    return np.array([focus_stats(y,base+s*correction,focus) for s in [0.,.05,.1,.25,.5,1.]])

def choose_strengths(moments):
    scores=np.array([[from_stats(item) for item in population] for population in moments])
    gains=scores-scores[:,:1]
    worst=gains.min(axis=0)
    choice=worst.argmax(axis=0)
    strengths=np.array([0.,.05,.1,.25,.5,1.])[choice]
    return {'strengths':strengths.tolist(),'scores_uniform_probability':scores.tolist(),
        'minimum_population_gains':worst.tolist(),'selected_indices':choice.tolist()}

def fit_models(train_files,output_dir):
    sx=np.zeros(256);sxx=np.zeros(256);count=0
    for path in train_files:
        with np.load(path) as z:
            role=int(z['role'])
            if role not in (0,1,2):
                raise ValueError('Unexpected development role in training sandbox')
            if role<2:
                h=z['hidden'].astype(float);sx+=h.sum(0);sxx+=(h*h).sum(0);count+=len(h)
    mean=sx/count;scale=np.sqrt(np.maximum(sxx/count-mean*mean,.05**2))
    grams=np.zeros((2,2,371,371));rhs=np.zeros((2,371,2));r2=np.zeros((2,2));mass=np.zeros((2,2));rows=np.zeros(2,dtype=int)
    for index,path in enumerate(train_files):
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        role=int(z['role'])
        if role==2:
            continue
        f=readout_design(z['current'],z['hidden'],mean,scale).astype(float)
        y=np.clip(z['y'],-2,2).astype(float);residual=y-z['base'];weight=np.abs(y)
        for target in (0,1):
            w=weight[:,target];grams[role,target]+=f.T@(w[:,None]*f)
            rhs[role,:,target]+=f.T@(w*residual[:,target])
            r2[role,target]+=np.dot(w,residual[:,target]**2);mass[role,target]+=w.sum()
        rows[role]+=len(f)
        if (index+1)%128==0:
            print('FIT_GRAM_SEQUENCES='+str(index+1),flush=True)
    coefficients={};fit_metrics={}
    for name,dim in [('current',115),('latent',371)]:
        fit_metrics[name]={}
        for suffix,groups in [('a',[0]),('b',[1]),('pooled',[0,1])]:
            gram=grams[groups,:,:dim,:dim].sum(0);right=rhs[groups,:dim,:].sum(0);n=int(rows[groups].sum())
            coef=np.column_stack([np.linalg.solve(gram[t]+np.eye(dim)*n*.1,right[:,t]) for t in (0,1)])
            full=np.zeros((371,2),np.float32);full[:dim]=coef
            coefficients[name+'_'+suffix]=full
            mse=[]
            for target in (0,1):
                c=full[:dim,target].astype(float)
                mse.append(float((r2[groups,target].sum()-2*np.dot(c,right[:,target])+c@gram[target]@c)/mass[groups,target].sum()))
            fit_metrics[name][suffix]={'weighted_residual_mse':mse,'rows':n}
    sequence_stats=[];selection_groups=[]
    for path in train_files:
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        if int(z['role'])!=2:
            continue
        f=readout_design(z['current'],z['hidden'],mean,scale)
        block=[]
        for name in ['current','latent']:
            correction=f@coefficients[name+'_pooled']
            block.append([grid_stats(z['y'],z['base'],correction,focus) for focus in [np.ones(len(f)),z['focus']]])
        sequence_stats.append(block);selection_groups.append(int(z['group']))
    totals=np.array(sequence_stats).sum(0)
    choices={name:choose_strengths(totals[i]) for i,name in enumerate(['current','latent'])}
    np.savez(output_dir+'/models.npz',mean=mean,scale=scale,**coefficients)
    np.savez(output_dir+'/selection_moments.npz',moments=sequence_stats,groups=selection_groups)
    np.savez(output_dir+'/fit_grams.npz',grams=grams,rhs=rhs,rows=rows,residual_squared=r2,mass=mass)
    report={'fit_metrics':fit_metrics,'selections':choices,'replicas':'A512,B512,pooled1024 sequences',
        'selection_sequences':len(selection_groups),'hidden_scale_quantiles':np.quantile(scale,[0,.1,.5,.9,1]).tolist(),
        'fixed_penalty':.1,'fit_rows':int(rows.sum()),'selection_rows':len(selection_groups)*500}
    Path(output_dir+'/training.json').write_text(json.dumps(report))
    return {'fits':6,'bivariate_readouts':6,'selections':{k:v['strengths'] for k,v in choices.items()}}

def source():
    code='import numpy as np\nimport json\nfrom pathlib import Path\n'
    code+='\n'.join(inspect.getsource(fn) for fn in [from_stats,focus_stats,readout_design,grid_stats,choose_strengths])+'\n'
    code+=inspect.getsource(fit_models).replace('def fit_models(','def train(')
    validate_source(code,training=True);return code

def verify_training_cache():
    protocol=campaign.initialize();manifest=json.loads((campaign.OUT/'prepared_manifest.json').read_text())
    expected={g for role in ['fit_a','fit_b','selection'] for g in protocol['groups'][role]}
    actual=set(json.loads((campaign.TRAIN/'identity.json').read_text())['groups'])
    if actual!=expected or actual&set(protocol['groups']['replication']):
        raise ValueError('Derived training boundary mismatch')
    for row in manifest['sequences']:
        if row['group'] in expected:
            assert campaign.sha(campaign.TRAIN/f"{row['group']:05d}.npz")==row['derived_sha256']

def train():
    OUT.mkdir(exist_ok=True);work=OUT/'isolated_training';protocol=OUT/'fit_protocol.json'
    verify_training_cache()
    if not protocol.exists():
        write_json(protocol,{'campaign_protocol_sha256':campaign.sha(campaign.OUT/'protocol.json'),
            'prepared_manifest_sha256':campaign.sha(campaign.OUT/'prepared_manifest.json'),
            'source_sha256':__import__('hashlib').sha256(source().encode()).hexdigest(),
            'scope':'Only derived fitting and selection groups are mounted; replication and search caches absent.'})
        campaign.event('model_experiment_planned',id='frozen_state_readout',protocol_sha256=campaign.sha(protocol))
    if not (work/'artifacts/training.json').exists():
        work.mkdir(exist_ok=True);(work/'train.py').write_text(source(),encoding='utf-8')
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,campaign.TRAIN)
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            campaign.event('implementation_failure',id='frozen_state_readout',status=status,error=stderr.decode(errors='replace')[-1500:])
            raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={name:campaign.sha(work/name) for name in ['train.py','artifacts/models.npz','artifacts/training.json','artifacts/selection_moments.npz','artifacts/fit_grams.npz']}
    if (work/'fit_identity.json').exists():
        assert identity==json.loads((work/'fit_identity.json').read_text())
    else:
        assert identity['train.py']==json.loads(protocol.read_text())['source_sha256']
        write_json(work/'fit_identity.json',identity);campaign.event('models_frozen_before_replication_and_search',identity=identity)
    print((work/'artifacts/training.json').read_text(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train'],required=True)
    train()
