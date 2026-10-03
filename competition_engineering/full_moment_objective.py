"""One evidence-gated exact-row residual/tangent replication, search only."""
import inspect
import json
import shutil
import time
import argparse
from pathlib import Path
from competition_engineering import objective_alignment as oa
import numpy as np
from competition_engineering.manual_search_core import ROOT, COMBO, COMBO_SHA, TRAIN_1024, TRAIN_4096
from competition_engineering.generated_runner import GeneratedSandbox, WORKER, _launch, digest
from competition_engineering.pipeline import write_json
from connectome.mlevolve_generated import validate_source


def full_train(train_files, output_dir):
    with np.load(Path(__file__).with_name('combo.npz')) as q:
        combo={k:q[k].copy() for k in q.files}
    moments=np.zeros(6); rows=0
    for count,path in enumerate(train_files):
        if count%256==0:
            Path(output_dir+'/progress.json').write_text(json.dumps({'stage':'objective_moments','sequences_seen':count,'sequences_total':len(train_files)}))
        if count%5==0:
            continue
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        idx=np.flatnonzero(z['need']); base=predict_combo(z,combo)[idx,TARGET]
        y=np.clip(z['y'][idx,TARGET],-2,2).astype(np.float64);p=np.clip(base,-2,2).astype(np.float64);w=np.abs(y)
        moments+=np.array([w.sum(),w@y,w@p,w@(y*y),w@(p*p),w@(y*p)])
        rows+=len(idx)
    mass,sy,sp,syy,spp,syp=moments
    my=sy/mass;mp=sp/mass;slope=(syp-sy*mp)/(spp-sp*mp)
    gram=np.zeros((115,115)); rhs={name:np.zeros(115) for name in ['raw_residual','pearson_tangent']}
    for count,path in enumerate(train_files):
        if count%256==0:
            Path(output_dir+'/progress.json').write_text(json.dumps({'stage':'exact_readout_statistics','sequences_seen':count,'sequences_total':len(train_files)}))
        if count%5==0:
            continue
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        idx=np.flatnonzero(z['need']);both=predict_combo(z,combo)[idx]
        base=both[:,TARGET].astype(np.float64)
        f=np.column_stack((np.ones(len(idx)),np.clip(both,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8))).astype(np.float64)
        y=np.clip(z['y'][idx,TARGET],-2,2).astype(np.float64);w=np.abs(y)
        target=y-my-slope*(np.clip(base,-2,2)-mp);target[np.abs(base)>=2]=0
        gram+=f.T@(w[:,None]*f)
        rhs['raw_residual']+=f.T@(w*(y-base));rhs['pearson_tangent']+=f.T@(w*target)
    coefficients={}
    for objective in rhs:
        for ridge in [.01,.1,1.]:
            name=objective+'_'+str(ridge)
            coefficients[name]=np.linalg.solve(gram+np.eye(115)*rows*ridge,rhs[objective]).astype(np.float32)
    strengths=[0.,.05,.1,.25,.5,1.]
    sums={name:np.zeros((2,6,6,2)) for name in coefficients}
    losses={name:np.zeros((2,6,3)) for name in coefficients}
    counts=[0,0]
    for count,path in enumerate(train_files):
        if count%256==0:
            Path(output_dir+'/progress.json').write_text(json.dumps({'stage':'training_and_selection_metrics','sequences_seen':count,'sequences_total':len(train_files)}))
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        idx=np.flatnonzero(z['need']);both=predict_combo(z,combo)[idx]
        f=np.column_stack((np.ones(len(idx)),np.clip(both,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
        y=np.clip(z['y'][idx,TARGET],-2,2).astype(np.float64);w=np.abs(y)
        target=np.column_stack((y,y));split=int(count%5==0);counts[split]+=len(idx)
        for name,coef in coefficients.items():
            correction=f@coef
            for j,strength in enumerate(strengths):
                raw=(both[:,TARGET]+strength*correction).astype(np.float32)
                prediction=np.column_stack((raw,raw))
                sums[name][split,j]+=sufficient(target,prediction)
                losses[name][split,j]+=np.array([w.sum(),w@((y-raw)**2),w@((y-np.clip(raw,-2,2))**2)])
    report={'fit_rows':counts[0],'selection_rows':counts[1],'stride':1,'tangent_slope':float(slope),'models':{}}
    for name in coefficients:
        report['models'][name]={}
        for split,key in enumerate(['fit','internal_selection']):
            report['models'][name][key]=[{'strength':strength,'wp':float(from_stats(sums[name][split,j])[0]),
                'weighted_raw_mse':float(losses[name][split,j,1]/losses[name][split,j,0]),
                'weighted_clipped_mse':float(losses[name][split,j,2]/losses[name][split,j,0])}
                for j,strength in enumerate(strengths)]
    np.savez(output_dir+'/models.npz',**{name:np.pad(coef,(0,224)) for name,coef in coefficients.items()},
             mean=combo['mean'],scale=combo['scale'])
    Path(output_dir+'/training.json').write_text(json.dumps(report))
    return {'fit_rows':counts[0],'selection_rows':counts[1],'models':len(coefficients)}


def source():
    code=oa.source().split('def train(')[0]+inspect.getsource(full_train).replace('def full_train(','def train(')
    validate_source(code,training=True)
    return code


def output_directory(tier=1024):
    return ROOT/('competition_engineering/runs/full_moment_objective_'+('_4096_' if tier==4096 else '')+'t0_20261002')


def main(tier=1024):
    diagnostic=json.loads((oa.output_directory()/'training_thinning_diagnosis.json').read_text())
    if 0 not in diagnostic['eligible_targets_for_full_moment_replication']:
        raise ValueError('No scientific support for exact-moment replication')
    oa.TARGET=0;oa.OUT=output_directory(tier)
    training_cache=TRAIN_1024 if tier==1024 else TRAIN_4096
    oa.OUT.mkdir(exist_ok=True)
    start=json.loads((oa.output_directory()/'campaign.json').read_text())['started_unix']
    deadline=time.monotonic()+start+7200-time.time()
    if deadline<=time.monotonic():
        raise TimeoutError('Shared campaign resources exhausted')
    config_path=oa.OUT/'campaign.json'
    if not config_path.exists():
        write_json(config_path,{'started_unix':start,'budget_seconds':7200,'target':0,'tier':tier,'stride':1,
            'hypothesis':'Deterministic row thinning introduces objective-moment error sufficient to obscure a weak current-row correction.',
            'support':'Exact moments beat stride40 control and frozen incumbent under unchanged search rule.',
            'abandon':'No practically reliable WP gain; no further row-phase or penalty tuning.',
            'reference_sha256':COMBO_SHA,'boundary':'Fixed training'+str(tier)+' and search64 only; training worker sees training data only'})
        shutil.copyfile(__file__,oa.OUT/'runner_snapshot.py')
        oa.event('hypothesis_planned',config=json.loads(config_path.read_text()))
    work=oa.OUT/'isolated_training'
    if not (work/'artifacts/training.json').exists():
        work.mkdir();(work/'train.py').write_text(source(),encoding='utf-8');shutil.copyfile(COMBO,work/'combo.npz')
        oa.event('experiment_planned',id='exact_full_moments',train_source_sha256=digest(work/'train.py'),target=0,stride=1)
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',min(deadline,time.monotonic()+1800),training_cache)
        stdout,stderr=process.communicate(timeout=min(1800,max(1,deadline-time.monotonic())))
        watcher.join(timeout=1);(work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            oa.event('infrastructure_failure',status=status,error=stderr.decode(errors='replace')[-1500:])
            raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    oa.verify_fit_identity(work)
    training=json.loads((work/'artifacts/training.json').read_text())
    with np.load(work/'artifacts/models.npz') as q:
        coefficients={k:q[k].copy() for k in training['models']}
    models={'incumbent':(None,0.)};selections={}
    for family in ['raw_residual','pearson_tangent']:
        choices=[(name,row) for name,info in training['models'].items() if name.startswith(family)
                 for row in info['internal_selection']]
        name,row=max(choices,key=lambda x:x[1]['wp']);key=family+'_select_wp'
        selections[key]={'model':name,'strength':row['strength'],'internal_selection':row}
        models[key]=(coefficients[name],row['strength'])
    for name,coef in coefficients.items():
        for strength in [.05,.1,.25,.5,1.]:
            models[name+'@'+str(strength)]=(coef,strength)
    scores=oa.search_metrics(models)
    write_json(oa.OUT/'diagnosis.json',{'training':training,'selections':selections,'search':scores,
               'caveat':'Adaptive reused search evidence only'})
    for key in selections:
        oa.event('experiment_completed',id=key,selection=selections[key],search=scores[key],delta=scores[key]['wp']-scores['incumbent']['wp'])
    print(json.dumps({'exact_rows':training['fit_rows']+training['selection_rows'],'selected_search':{k:scores[k] for k in selections}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tier',type=int,choices=[1024,4096],default=1024)
    main(parser.parse_args().tier)
