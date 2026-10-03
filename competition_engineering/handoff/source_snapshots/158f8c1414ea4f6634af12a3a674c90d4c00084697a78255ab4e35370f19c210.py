"""Bounded matched comparison of population weighting with stable penalty scale."""
from competition_engineering import root_cause_campaign as campaign
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.special import expit
from competition_engineering.manual_search_core import COMBO, COMBO_SHA, TRAIN_1024, predict_combo
from competition_engineering.nonlinear_weighted_readout import tree_predict, mask_focus, readout_scores
from competition_engineering.residual import from_stats
from competition_engineering.pipeline import write_json
from competition_engineering.generated_runner import GeneratedSandbox, WORKER, _launch
from connectome.mlevolve_generated import validate_source

OUT=campaign.OUT/'probability_readout'

def normalize_focus(focus):
    focus=np.asarray(focus,dtype=float)
    if focus.ndim!=1 or not np.isfinite(focus).all() or np.any(focus<0) or focus.mean()<=0:
        raise ValueError('Invalid population weights')
    return focus/focus.mean()

def probability_focus(x,base,steps,combo,trees):
    p=np.clip(base,-2,2)
    f=np.column_stack((np.clip((x-combo['mean'])/combo['scale'],-8,8),p,np.abs(p),p*p,steps/20000.,steps>=13333)).astype(float)
    probability=[]
    for fold in range(4):
        logits=np.full(len(x),float(trees['fold'+str(fold)+'_baseline'].ravel()[0]))
        for j in range(64):
            logits+=tree_predict(f,trees['fold'+str(fold)+'_tree'+str(j)])
        probability.append(expit(logits))
    values=np.array(probability)
    return np.column_stack((np.clip(np.mean(values/trees['rates'][:,None],axis=0),.2,5),values.mean(0)))

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
        focus=probability_focus(z['x'][idx],base,z['step'][idx],combo,trees)
        y=np.clip(z['y'][idx,0],-2,2).astype(float)
        rows[count%5==0].append((f,base[:,0],y,focus))
        if (count+1)%128==0:
            print('TRAIN_PROGRESS='+str(count+1),flush=True)
    data=[tuple(np.concatenate([r[j] for r in split]) for j in range(4)) for split in rows]
    f,base,y,focus=data[0];coefficients={};report={};selections={}
    for column,name in enumerate(['capped_normalized','probability_normalized']):
        multiplier=normalize_focus(focus[:,column]);w=np.abs(y)*multiplier
        coef=np.linalg.solve(f.T@(w[:,None]*f)+np.eye(f.shape[1])*len(y),f.T@(w*(y-base)))
        coefficients[name]=coef.astype(np.float32)
        sf,sbase,sy,sfocus=data[1]
        uniform=readout_scores(sy,sbase,sf@coef,np.ones(len(sy)))
        weighted=readout_scores(sy,sbase,sf@coef,sfocus[:,column])
        report[name]={'uniform_selection':uniform,'focused_selection':weighted,
                      'focus_training_mean':float(focus[:,column].mean()),
                      'normalized_focus_quantiles':np.quantile(multiplier,[0,.1,.5,.9,1]).tolist()}
        selections[name]={'model':name,**max(weighted,key=lambda q:q['wp'])}
    np.savez(output_dir+'/models.npz',**coefficients)
    Path(output_dir+'/training.json').write_text(json.dumps({'models':report,'selections':selections,
        'fit_rows':len(y),'selection_rows':len(data[1][2]),'penalty':1.,'normalization':'Each fit population has mean-one focus; selection WP scale invariant.'}))
    return {'models':2,'fit_rows':len(y),'selection_rows':len(data[1][2])}

def source():
    combo_code=inspect.getsource(predict_combo).replace('f = features({**z, "p": base}, model["mean"], model["scale"])',
        'f = np.column_stack((np.ones(len(base), np.float32), np.clip((z["x"]-model["mean"])/model["scale"], -8, 8), base)).astype(np.float32)')
    code='import numpy as np\nimport json\nfrom pathlib import Path\nfrom scipy.special import expit\n'
    code+='\n'.join(inspect.getsource(fn) for fn in [from_stats,tree_predict,normalize_focus,probability_focus,readout_scores])+'\n'+combo_code+'\n'
    code+=inspect.getsource(fit_readouts).replace('def fit_readouts(','def train(')
    validate_source(code,training=True)
    return code

def train():
    OUT.mkdir(exist_ok=True)
    evidence=json.loads((campaign.OUT/'weight_diagnosis_matched.json').read_text())
    if 0 not in evidence['eligible_targets']:
        raise ValueError('Prospective population-diagnostic launch rule failed')
    protocol=OUT/'protocol.json';work=OUT/'isolated_training'
    if not protocol.exists():
        write_json(protocol,{'hypothesis':'Probability-weighted t0 training and selection transfer better after removing arbitrary population truncation.',
            'matched_control':'Capped and uncapped populations both normalized to mean-one before the same fixed ridge penalty1; otherwise identical.',
            'selection':'Every fifth training sequence, focused official-form WP strengths [0,.05,.1,.25,.5,1]; no search selection.',
            'training':'1024 designated training sequences, stride40, current115-feature readout, target0 only.',
            'qualification':{'minimum_delta':.0002,'paired99_lower':0.,'minimum_target_delta':-.0002,'cpu_us_max':76.92753780833335},
            'launch_evidence_sha256':campaign.sha(campaign.OUT/'weight_diagnosis_matched.json'),
            'source_sha256':__import__('hashlib').sha256(source().encode()).hexdigest()})
        campaign.event('experiment_planned',id='probability_readout',protocol_sha256=campaign.sha(protocol))
    if not (work/'artifacts/training.json').exists():
        work.mkdir(exist_ok=True);(work/'train.py').write_text(source(),encoding='utf-8')
        shutil.copyfile(COMBO,work/'combo.npz')
        shutil.copyfile(campaign.PREVIOUS/'nonlinear_weighted_t0/isolated_training/mask_trees.npz',work/'mask_trees.npz')
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,TRAIN_1024)
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            campaign.event('implementation_failure',id='probability_readout',status=status,error=stderr.decode(errors='replace')[-1500:])
            raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={name:campaign.sha(work/name) for name in ['train.py','combo.npz','mask_trees.npz','artifacts/models.npz','artifacts/training.json']}
    if (work/'fit_identity.json').exists():
        assert identity==json.loads((work/'fit_identity.json').read_text())
    else:
        assert identity['combo.npz']==COMBO_SHA
        assert identity['train.py']==json.loads(protocol.read_text())['source_sha256']
        write_json(work/'fit_identity.json',identity);campaign.event('fit_frozen',id='probability_readout',identity=identity)
    print((work/'artifacts/training.json').read_text(),flush=True)

def check():
    from competition_engineering import nonlinear_weighted_readout as runner
    runner.OUT=OUT;runner.campaign=campaign
    runner.check()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','check'],required=True)
    args=parser.parse_args();train() if args.phase=='train' else check()
