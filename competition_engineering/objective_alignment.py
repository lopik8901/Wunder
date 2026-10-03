"""Search-only objective diagnosis and train-isolated, current-row experiments.

No temporal feature is fitted. Fixed training sequences supply both fitting and
an internal sequence holdout; search labels never enter the training worker.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import inspect
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from competition_engineering import autonomous_campaign as lab
from competition_engineering import campaign_onnx
from competition_engineering.generated_runner import GeneratedSandbox, WORKER, _launch, _check_artifacts, _package, validate_callback, digest
from competition_engineering.manual_search_core import ROOT, COMBO, COMBO_SHA, SEARCH, TRAIN_1024, TRAIN_4096, load_combo, predict_combo, assess
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import sufficient, from_stats
from connectome.mlevolve_generated import validate_source

OUT = ROOT / 'competition_engineering/runs/objective_alignment_20261002'
TARGET = 0


def output_directory(tier=1024, target=0):
    suffix = ('_4096' if tier==4096 else '') + ('_t1' if target==1 else '')
    return ROOT / ('competition_engineering/runs/objective_alignment'+suffix+'_20261002')


def event(kind, **payload):
    path = OUT / 'ledger.jsonl'
    record = {'utc': lab.datetime.now(lab.timezone.utc).isoformat(), 'kind': kind,
              'evidence': 'reused_search_evidence_not_independent_validation',
              'previous_ledger_sha256': digest(path) if path.exists() else None, **payload}
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')


def verify_fit_identity(work):
    identity_path=work/'fit_identity.json'
    files=['train.py','combo.npz','artifacts/models.npz','artifacts/training.json']
    actual={name:digest(work/name) for name in files}
    if identity_path.exists():
        if actual != json.loads(identity_path.read_text()):
            raise ValueError('Frozen objective fit identity changed')
    else:
        planned=[json.loads(line) for line in (OUT/'ledger.jsonl').read_text().splitlines()
                 if json.loads(line).get('kind')=='experiment_planned']
        if not planned or planned[-1]['train_source_sha256']!=actual['train.py'] or actual['combo.npz']!=COMBO_SHA:
            raise ValueError('Objective fit source or incumbent differs from planned identity')
        write_json(identity_path,actual)
        event('fit_identity_frozen',identity=actual)
    return actual


def objective_value_gradient(coef, f, base, y, penalty):
    """Negative exact clipped weighted correlation, plus coefficient ridge."""
    raw = base + f @ coef
    p = np.clip(raw, -2, 2)
    w = np.abs(y)
    mass = w.sum()
    yc = y - np.dot(w, y) / mass
    pc = p - np.dot(w, p) / mass
    vy = np.dot(w, yc * yc)
    vp = max(np.dot(w, pc * pc), 1e-12)
    cov = np.dot(w, yc * pc)
    denom = np.sqrt(vy * vp)
    direction = w * (yc - cov / vp * pc) / denom
    direction[np.abs(raw) >= 2] = 0
    return -cov / denom + penalty * np.dot(coef, coef) / 2, -f.T @ direction + penalty * coef


def metrics(y, base, correction, strengths):
    w = np.abs(y)
    result = []
    for strength in strengths:
        raw = base + strength * correction
        target = np.column_stack((y, y))
        pred = np.column_stack((raw, raw))
        result.append({'strength': strength,
                       'weighted_raw_mse': float(np.dot(w, (y-raw)**2) / w.sum()),
                       'weighted_clipped_mse': float(np.dot(w, (y-np.clip(raw,-2,2))**2) / w.sum()),
                       'wp': float(from_stats(sufficient(target, pred))[0]),
                       'saturated_weight_fraction': float(np.dot(w, np.abs(raw)>=2)/w.sum())})
    return result


def fit_models(train_files, output_dir):
    """This function is copied into the isolated training program."""
    with np.load(Path(__file__).with_name('combo.npz')) as q:
        combo = {k:q[k].copy() for k in q.files}
    rows = [[], []]
    for count, path in enumerate(train_files):
        with np.load(path) as q:
            z = {k:q[k] for k in q.files}
        idx = np.flatnonzero(z['need'])[::40]
        base = predict_combo(z, combo)[idx, TARGET]
        x = np.clip((z['x'][idx]-combo['mean'])/combo['scale'], -8, 8)
        f = np.column_stack((np.ones(len(idx)), np.clip(predict_combo(z,combo)[idx],-2,2), x)).astype(np.float64)
        rows[count % 5 == 0].append((f, base, np.clip(z['y'][idx,TARGET],-2,2)))
        if (count+1)%256 == 0:
            print('TRAIN_PROGRESS='+str(count+1), flush=True)
    data = [tuple(np.concatenate([r[j] for r in split]) for j in range(3)) for split in rows]
    f, base, y = data[0]
    w = np.abs(y)
    gram = f.T @ (w[:,None]*f)
    p = np.clip(base,-2,2)
    my = np.dot(w,y)/w.sum(); mp = np.dot(w,p)/w.sum()
    slope = np.dot(w,(y-my)*(p-mp))/np.dot(w,(p-mp)**2)
    tangent = (y-my)-slope*(p-mp)
    tangent[np.abs(base)>=2] = 0
    coefficients = {}; report = {}
    strengths = [0., .05, .1, .25, .5, 1.]
    for penalty in [.01, .1, 1.]:
        for objective, target in [('raw_residual', y-base), ('pearson_tangent', tangent)]:
            name = objective+'_'+str(penalty)
            coef = np.linalg.solve(gram + np.eye(f.shape[1])*len(y)*penalty, f.T@(w*target))
            coefficients[name] = np.pad(coef.astype(np.float32), (0,224))
            report[name] = {'fit': metrics(y,base,f@coef,strengths),
                            'internal_holdout': metrics(data[1][2],data[1][1],data[1][0]@coef,strengths)}
    for penalty in [.01, .1, 1.]:
        name = 'direct_wp_'+str(penalty)
        opt = minimize(objective_value_gradient, np.zeros(f.shape[1]), args=(f,base,y,penalty),
                       jac=True, method='L-BFGS-B', options={'maxiter':80, 'ftol':1e-10})
        coefficients[name] = np.pad(opt.x.astype(np.float32), (0,224))
        report[name] = {'fit': metrics(y,base,f@opt.x,strengths),
                        'internal_holdout': metrics(data[1][2],data[1][1],data[1][0]@opt.x,strengths),
                        'optimizer': {'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit)}}
    np.savez(output_dir+'/models.npz', **coefficients, mean=combo['mean'], scale=combo['scale'])
    Path(output_dir+'/training.json').write_text(json.dumps({'models':report,'fit_rows':len(y),'holdout_rows':len(data[1][2]),
                   'tangent_slope':float(slope),'split':'every fifth training sequence reserved; stride40'}))
    return {'models':len(coefficients),'fit_rows':len(y),'holdout_rows':len(data[1][2])}


def source():
    combo_code = inspect.getsource(predict_combo).replace(
        'f = features({**z, "p": base}, model["mean"], model["scale"])',
        'f = np.column_stack((np.ones(len(base), np.float32), np.clip((z["x"]-model["mean"])/model["scale"], -8, 8), base)).astype(np.float32)')
    code = 'import numpy as np\nimport json\nfrom pathlib import Path\nfrom scipy.optimize import minimize\n'
    code += 'TARGET = '+str(TARGET)+'\n'
    code += inspect.getsource(sufficient)+'\n'+inspect.getsource(from_stats)+'\n'+combo_code+'\n'
    code += inspect.getsource(objective_value_gradient)+'\n'+inspect.getsource(metrics)+'\n'
    code += inspect.getsource(fit_models).replace('def fit_models(', 'def train(').replace('holdout','selection')
    validate_source(code, training=True)
    return code


def initial_diagnosis():
    """Before fitting: loss/score geometry of an already frozen raw ridge."""
    combo=load_combo(); sums={}; losses={}
    for _,z in cached(SEARCH):
        idx=np.flatnonzero(z['mask']); incumbent=predict_combo(z,combo)
        base=(z['p']*combo['base_scale']+combo['base_bias']).astype(np.float32)
        f=np.column_stack((np.ones(len(idx)),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8),base[idx])).astype(np.float32)
        correction=f@combo['root_coef'][:,TARGET]
        y=np.clip(z['y'][idx],-2,2).astype(np.float64); w=np.abs(y)
        for strength in [0.,.25,.5,.75,1.,1.25,1.5,2.]:
            p=incumbent[idx].copy(); p[:,TARGET]=base[idx,TARGET]+strength*correction
            sums.setdefault(strength,np.zeros((6,2))); losses.setdefault(strength,np.zeros((3,2)))
            sums[strength]+=sufficient(y,p)
            losses[strength]+=np.stack((w.sum(0),(w*(y-p)**2).sum(0),(w*(y-np.clip(p,-2,2))**2).sum(0)))
    result=[{'strength':s,'wp':float(from_stats(m).mean()),'target_wp':float(from_stats(m)[TARGET]),
             'raw_mse':float(losses[s][1,TARGET]/losses[s][0,TARGET]),'clipped_mse':float(losses[s][2,TARGET]/losses[s][0,TARGET])} for s,m in sums.items()]
    inversions=[{'lower_mse':a,'higher_wp':b} for a in result for b in result if a['raw_mse']<b['raw_mse'] and a['wp']<b['wp']]
    report={'fixed_raw_ridge_strength_path':result,'loss_wp_order_inversions':inversions,
            'interpretation':'Within an existing frozen residual direction, lower search raw MSE can accompany lower WP.' if inversions else 'No ordering inversion on this fixed direction; geometry alone does not establish harmful mismatch.',
            'limitation':'Search losses are diagnostic; this does not prove lower training loss causes search regression.'}
    write_json(OUT/'initial_diagnosis.json',report)
    event('pre_experiment_diagnosis',**report)
    print(json.dumps({'phase':'pre_experiment_diagnosis','path':result,'inversions':len(inversions)}),flush=True)


def search_metrics(models):
    combo = load_combo()
    sums = {name:np.zeros((6,2)) for name in models}
    loss = {name:np.zeros((3,2)) for name in models}
    for _, z in cached(SEARCH):
        idx = np.flatnonzero(z['mask'])
        base = predict_combo(z,combo)
        f = lab.design(z['x'],base,combo['mean'],combo['scale'],'linear',idx)[:,:115]
        y = np.clip(z['y'][idx],-2,2).astype(np.float64); w=np.abs(y)
        for name,(coef,strength) in models.items():
            prediction=base[idx].copy()
            if coef is not None:
                prediction[:,TARGET] += (strength*(f@coef[:115])).astype(np.float32)
            sums[name] += sufficient(y,prediction)
            loss[name] += np.stack((w.sum(0),(w*(y-prediction)**2).sum(0),
                                    (w*(y-np.clip(prediction,-2,2))**2).sum(0)))
    return {name:{'wp_per_target':from_stats(s).tolist(),'wp':float(from_stats(s).mean()),
                  'weighted_raw_mse':(loss[name][1]/loss[name][0]).tolist(),
                  'weighted_clipped_mse':(loss[name][2]/loss[name][0]).tolist()}
            for name,s in sums.items()}


def main():
    global OUT, TARGET
    parser=argparse.ArgumentParser()
    parser.add_argument('--tier',type=int,choices=[1024,4096],default=1024)
    parser.add_argument('--target',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    TARGET=args.target
    OUT=output_directory(args.tier,args.target)
    train_cache=TRAIN_1024 if args.tier==1024 else TRAIN_4096
    OUT.mkdir(exist_ok=True)
    config_path = OUT/'campaign.json'
    if not config_path.exists():
        parent=ROOT/'competition_engineering/runs/objective_alignment_20261002/campaign.json'
        started=json.loads(parent.read_text())['started_unix'] if parent.exists() else time.time()
        write_json(config_path, {'started_unix':started,'budget_seconds':7200,'tier':args.tier,'target':TARGET,
                   'incumbent':0.6588353223316641,'incumbent_sha256':COMBO_SHA,
                   'boundary':'only fixed TRAIN_'+str(args.tier)+' and fixed SEARCH; no protected evaluation',
                   'temporal_branch':'closed; current-row coefficients only',
                   'plan':'matched residual/tangent/direct-WP objectives; training-sequence holdout chooses penalty and strength; fixed search is diagnostic',
                   'source_sha256':digest(Path(__file__))})
        event('campaign_started', config=json.loads(config_path.read_text()))
        shutil.copyfile(__file__,OUT/'runner_snapshot.py')
    config=json.loads(config_path.read_text())
    deadline=time.monotonic()+max(0,config['started_unix']+config['budget_seconds']-time.time())
    if deadline<=time.monotonic():
        raise TimeoutError('Resource window exhausted')
    if not (OUT/'initial_diagnosis.json').exists():
        initial_diagnosis()
    work=OUT/'isolated_training'
    if not (work/'artifacts/training.json').exists():
        work.mkdir()
        (work/'train.py').write_text(source(),encoding='utf-8')
        shutil.copyfile(COMBO,work/'combo.npz')
        event('experiment_planned', id='matched_objectives', objectives=['raw_residual','pearson_tangent','direct_wp'],
              selection='internal training sequence holdout only', train_source_sha256=digest(work/'train.py'))
        sandbox=GeneratedSandbox(); sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',min(deadline,time.monotonic()+1800),train_cache)
        stdout,stderr=process.communicate(timeout=min(1800,max(1,deadline-time.monotonic())))
        watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout); (work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            event('infrastructure_failure',status=status,error=stderr.decode(errors='replace')[-1500:])
            raise RuntimeError('Isolated training failed: '+stderr.decode(errors='replace')[-1500:])
    verify_fit_identity(work)
    training=json.loads((work/'artifacts/training.json').read_text().replace('selection','holdout'))
    with np.load(work/'artifacts/models.npz') as q:
        coeff={name:q[name].copy() for name in training['models']}
    models={'incumbent':(None,0.)}; selections={}
    for family in ['raw_residual','pearson_tangent','direct_wp']:
        choices=[(name,row) for name,info in training['models'].items() if name.startswith(family)
                 for row in info['internal_holdout']]
        for criterion in ['wp','weighted_raw_mse']:
            name,row=max(choices,key=lambda item:item[1][criterion]) if criterion=='wp' else min(choices,key=lambda item:item[1][criterion])
            key=family+'_select_'+criterion
            selections[key]={'model':name,'strength':row['strength'],'internal_holdout':row}
            models[key]=(coeff[name],row['strength'])
    # Diagnostic grid: measures mismatch; never used as an undisclosed independent test.
    for name,c in coeff.items():
        for strength in [.05,.1,.25,.5,1.]:
            models[name+'@'+str(strength)]=(c,strength)
    scores=search_metrics(models)
    write_json(OUT/'diagnosis.json',{'training':training,'selections':selections,'search':scores,
               'caveat':'adaptive reused search evidence; neither independent validation nor promotion'})
    for family in ['raw_residual','pearson_tangent','direct_wp']:
        key=family+'_select_wp'; candidate=scores[key]
        event('experiment_completed',id=family,selection=selections[key],search=candidate,
              delta=candidate['wp']-scores['incumbent']['wp'],
              decision='refine only if directional evidence or holdout/search agreement; otherwise abandon')
    print(json.dumps({'selections':selections,'selected_search':{k:scores[k] for k in selections}}),flush=True)


if __name__=='__main__':
    main()
