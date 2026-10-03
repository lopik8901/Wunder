"""Train-only diagnosis and bounded risk-variance objective experiment.

Groups are quartiles of training sequence mean frozen predictions. Groups are
used offline only; the learned correction is the same current-row readout.
"""
import inspect
import json
import shutil
import time
import argparse
from pathlib import Path
from competition_engineering import objective_alignment as oa
import numpy as np
from scipy.optimize import minimize
from competition_engineering.manual_search_core import ROOT, TRAIN_1024, COMBO, COMBO_SHA
from competition_engineering.generated_runner import GeneratedSandbox, WORKER, _launch, digest
from competition_engineering.pipeline import write_json
from connectome.mlevolve_generated import validate_source


def group_value_gradient(coef, groups, reference, variance_penalty, ridge):
    values=[]; gradients=[]
    for (f,base,y),zero in zip(groups,reference):
        value,gradient=oa.objective_value_gradient(coef,f,base,y,0.)
        values.append(value-zero); gradients.append(gradient)
    values=np.array(values); gradients=np.array(gradients)
    centered=values-values.mean()
    value=values.mean()+variance_penalty*np.mean(centered**2)+ridge*np.dot(coef,coef)/2
    gradient=gradients.mean(0)+2*variance_penalty*np.mean(centered[:,None]*gradients,axis=0)+ridge*coef
    return value,gradient


def train_groups(train_files, output_dir):
    with np.load(Path(__file__).with_name('combo.npz')) as q:
        combo={k:q[k].copy() for k in q.files}
    fit=[]; selection=[]
    for count,path in enumerate(train_files):
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        idx=np.flatnonzero(z['need'])[::40]
        both=predict_combo(z,combo)[idx]
        x=np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)
        f=np.column_stack((np.ones(len(idx)),np.clip(both,-2,2),x)).astype(np.float64)
        row=(f,both[:,TARGET],np.clip(z['y'][idx,TARGET],-2,2),float(np.clip(both[:,TARGET],-2,2).mean()))
        (selection if count%5==0 else fit).append(row)
    edges=np.quantile([row[3] for row in fit],[.25,.5,.75])
    groups=[]
    for group in range(4):
        rows=[row for row in fit if np.searchsorted(edges,row[3],side='right')==group]
        groups.append(tuple(np.concatenate([row[k] for row in rows]) for k in range(3)))
    held=tuple(np.concatenate([row[k] for row in selection]) for k in range(3))
    pooled=tuple(np.concatenate([row[k] for row in groups]) for k in range(3))
    dimension=groups[0][0].shape[1]
    zero=np.zeros(dimension)
    reference=[]; gradients=[]
    for f,base,y in groups:
        value,gradient=objective_value_gradient(zero,f,base,y,0.)
        reference.append(value); gradients.append(gradient)
    gradients=np.array(gradients)
    norms=np.linalg.norm(gradients,axis=1)
    cosines=gradients@gradients.T/np.maximum(norms[:,None]*norms[None,:],1e-20)
    minimum=float(np.min(cosines[np.triu_indices(4,1)]))
    report={'group_mean_edges':edges.tolist(),'gradient_cosines':cosines.tolist(),
            'minimum_gradient_cosine':minimum,'baseline_group_wp':(-np.array(reference)).tolist(),
            'groups':'quartiles of mean frozen training-sequence prediction; internal correction selection excludes every fifth sequence',
            'launch_rule':'Fit risk-variance variants only if minimum pairwise training-group gradient cosine is below0.75',
            'models':{}}
    coefficients={}
    if minimum<.75:
        for beta in [0.,100.,1000.]:
            for ridge in [.1,1.]:
                name='group_beta'+str(beta)+'_ridge'+str(ridge)
                opt=minimize(group_value_gradient,zero,args=(groups,reference,beta,ridge),jac=True,
                             method='L-BFGS-B',options={'maxiter':80,'ftol':1e-10})
                coefficients[name]=np.pad(opt.x.astype(np.float32),(0,224))
                strengths=[0.,.05,.1,.25,.5,1.]
                report['models'][name]={'fit':metrics(pooled[2],pooled[1],pooled[0]@opt.x,strengths),
                    'internal_selection':metrics(held[2],held[1],held[0]@opt.x,strengths),
                    'group_wp_gains':[-objective_value_gradient(opt.x,*row,0.)[0]+reference[i] for i,row in enumerate(groups)],
                    'optimizer':{'success':bool(opt.success),'iterations':int(opt.nit),'message':str(opt.message)}}
        report['decision']='Heterogeneous training gradients justify the predeclared risk-variance comparison.'
    else:
        report['decision']='Training gradients already agree; no risk-variance fit scientifically justified.'
    np.savez(output_dir+'/models.npz',**coefficients,mean=combo['mean'],scale=combo['scale'])
    Path(output_dir+'/training.json').write_text(json.dumps(report))
    return {'models':len(coefficients),'minimum_gradient_cosine':minimum}


def main(target):
    oa.TARGET=target
    oa.OUT=ROOT/('competition_engineering/runs/group_objective_t'+str(target)+'_20261002')
    oa.OUT.mkdir(exist_ok=True)
    config_path=oa.OUT/'campaign.json'
    parent=oa.output_directory()/'campaign.json'
    start=json.loads(parent.read_text())['started_unix']
    deadline=time.monotonic()+start+7200-time.time()
    if deadline<=time.monotonic():
        raise TimeoutError('Shared campaign resource window exhausted')
    if not config_path.exists():
        write_json(config_path,{'started_unix':start,'budget_seconds':7200,'target':target,'tier':1024,
            'hypothesis':'Pooled objective gradients vary across sequence populations; penalizing variation in group WP improvements may select a transferable correction.',
            'support':'Robust variant beats matched group-mean control and frozen incumbent under unchanged paired search rule.',
            'abandon':'No gradient heterogeneity or no practically reliable WP gain; do not tune arbitrary group boundaries.',
            'reference_sha256':COMBO_SHA,'boundary':'Fixed training1024 and search64 only; current-row label-free inference',
            'library':'v5; group-risk variance paper is motivation, not a task guarantee'})
        shutil.copyfile(__file__,oa.OUT/'runner_snapshot.py')
        oa.event('hypothesis_planned',config=json.loads(config_path.read_text()))
    work=oa.OUT/'isolated_training'
    if not (work/'artifacts/training.json').exists():
        work.mkdir()
        code=oa.source().split('def train(')[0]
        code+=inspect.getsource(group_value_gradient).replace('oa.objective_value_gradient','objective_value_gradient')+'\n'
        code+=inspect.getsource(train_groups).replace('def train_groups(','def train(')
        validate_source(code,training=True)
        (work/'train.py').write_text(code,encoding='utf-8'); shutil.copyfile(COMBO,work/'combo.npz')
        oa.event('experiment_planned',id='sequence_group_risk_variance',train_source_sha256=digest(work/'train.py'),
                 gradient_launch_threshold=.75,variance_penalties=[0,100,1000],ridge=[.1,1.],target=target)
        sandbox=GeneratedSandbox(); sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',min(deadline,time.monotonic()+1800),TRAIN_1024)
        stdout,stderr=process.communicate(timeout=min(1800,max(1,deadline-time.monotonic())))
        watcher.join(timeout=1); (work/'stdout.log').write_bytes(stdout); (work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            oa.event('infrastructure_failure',status=status,error=stderr.decode(errors='replace')[-1500:])
            raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    oa.verify_fit_identity(work)
    training=json.loads((work/'artifacts/training.json').read_text())
    if not training['models']:
        write_json(oa.OUT/'report.json',{'training':training,'status':'abandoned_before_fit'})
        oa.event('scientific_interpretation',decision=training['decision']); print(training,flush=True); return
    with np.load(work/'artifacts/models.npz') as q:
        coefficients={k:q[k].copy() for k in training['models']}
    models={'incumbent':(None,0.)}; selections={}
    for beta in [0.,100.,1000.]:
        choices=[(name,row) for name,info in training['models'].items() if name.startswith('group_beta'+str(beta)+'_')
                 for row in info['internal_selection']]
        name,row=max(choices,key=lambda item:item[1]['wp'])
        key='group_beta'+str(beta)+'_select_wp'
        selections[key]={'model':name,'strength':row['strength'],'internal_selection':row}
        models[key]=(coefficients[name],row['strength'])
    for name,coef in coefficients.items():
        for strength in [.05,.1,.25,.5,1.]:
            models[name+'@'+str(strength)]=(coef,strength)
    scores=oa.search_metrics(models)
    write_json(oa.OUT/'diagnosis.json',{'training':training,'selections':selections,'search':scores,
               'caveat':'Adaptive search evidence only; no protected evaluation'})
    for name in selections:
        oa.event('experiment_completed',id=name,selection=selections[name],search=scores[name],
                 delta=scores[name]['wp']-scores['incumbent']['wp'])
    print(json.dumps({'training_gradient_cosine_min':training['minimum_gradient_cosine'],
                     'selected_search':{k:scores[k] for k in selections}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--target',type=int,choices=[0,1],default=0)
    main(p.parse_args().target)
