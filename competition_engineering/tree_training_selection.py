"""Correction-disjoint training selection for the now-deployable frozen tree."""
from competition_engineering import root_cause_campaign as campaign
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.special import expit
from competition_engineering.manual_search_core import COMBO,COMBO_SHA,TRAIN_1024,TRAIN_4096,predict_combo,load_combo,assess
from competition_engineering.nonlinear_weighted_readout import tree_predict
from competition_engineering.probability_readout import probability_focus
from competition_engineering.residual import from_stats
from competition_engineering.pipeline import cached,write_json
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch,_check_artifacts,_package,validate_callback,replay
from competition_engineering import compact_tree_onnx,campaign_onnx
from competition_engineering.autonomous_campaign import paired99
from connectome.mlevolve_generated import validate_source

OUT=campaign.OUT/'tree_training_selection'

def selection_moments(y,base,correction,focus):
    y=np.clip(y,-2,2).astype(float);w=np.abs(y)*focus
    return np.array([[w.sum(),np.dot(w,y),np.dot(w,p),np.dot(w,y*y),np.dot(w,p*p),np.dot(w,y*p)]
        for strength in [0.,.05,.1,.25,.5,1.] for p in [np.clip(base+strength*correction,-2,2)]])

def select(train_files,output_dir):
    inputs={}
    for name in ['combo','mask_trees','frozen_trees','excluded_groups']:
        with np.load(Path(__file__).with_name(name+'.npz')) as q:
            inputs[name]={k:q[k] for k in q.files}
    combo=inputs['combo'];model=inputs['frozen_trees'];excluded=set(inputs['excluded_groups']['groups'].tolist())
    groups=[];blocks=[]
    for path in train_files:
        group=int(Path(path).stem)
        if group in excluded:
            continue
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        eligible=np.flatnonzero(z['need'])
        idx=np.sort(np.random.default_rng(20261012+group).choice(eligible,500,replace=False))
        base=predict_combo(z,combo)[idx]
        x=np.column_stack((np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8),np.clip(base,-2,2))).astype(np.float32)
        correction=np.full(len(x),float(model['baseline'].ravel()[0]))
        for j in range(32):
            correction+=tree_predict(x,model['tree'+str(j)])
        correction=correction.astype(np.float32)
        focus=probability_focus(z['x'][idx],base,z['step'][idx],combo,inputs['mask_trees'])[:,1]
        blocks.append([selection_moments(z['y'][idx,0],base[:,0],correction,v) for v in [np.ones(len(x)),focus]])
        groups.append(group)
        if len(groups)==256:
            break
    if len(groups)!=256 or set(groups)&excluded:
        raise ValueError('Correction-disjoint selection partition failed')
    totals=np.array(blocks).sum(0);scores={}
    for j,name in enumerate(['uniform','probability']):
        scores[name]=[{'strength':strength,'wp':float(from_stats(np.repeat(s[:,None],2,axis=1))[0])}
            for strength,s in zip([0.,.05,.1,.25,.5,1.],totals[j])]
    selected=max(scores['probability'],key=lambda q:q['wp'])
    np.savez(output_dir+'/selection_moments.npz',moments=blocks,groups=groups)
    Path(output_dir+'/selection.json').write_text(json.dumps({'groups':groups,'rows':256*500,'scores':scores,'selected':selected,
        'scope':'Excluded all1024 original tree-fitting sequences; GRU and incumbent provenance still prevent a whole-system independence claim.'}))
    return {'selection_sequences':256,'selected':selected}

def source():
    combo_code=inspect.getsource(predict_combo).replace('f = features({**z, "p": base}, model["mean"], model["scale"])',
        'f = np.column_stack((np.ones(len(base), np.float32), np.clip((z["x"]-model["mean"])/model["scale"], -8, 8), base)).astype(np.float32)')
    code='import numpy as np\nimport json\nfrom pathlib import Path\nfrom scipy.special import expit\n'
    code+='\n'.join(inspect.getsource(fn) for fn in [from_stats,tree_predict,probability_focus,selection_moments])+'\n'+combo_code+'\n'
    code+=inspect.getsource(select).replace('def select(','def train(')
    validate_source(code,training=True);return code

def train():
    OUT.mkdir(exist_ok=True);protocol=OUT/'protocol.json';work=OUT/'isolated_selection'
    if not protocol.exists():
        write_json(protocol,{'hypothesis':'Nonlinear frozen-tree corrections may retain useful scoring-population signal even though current-row linear readouts do not.',
            'mechanism':'Same frozen32 shallow trees; select scalar strength only on training sequences disjoint from all1024 original tree-fitting sequences.',
            'selection':'First256 eligible sequences from pre-existing4096 training-cache order,500 random required rows/sequence,seed20261012+group. Pooled probability-weighted clipped WP.',
            'strengths':[0.,.05,.1,.25,.5,1.],'search_use':'One frozen training-selected strength, no search choice.',
            'qualification':{'minimum_delta':.0002,'paired99_lower':0.,'minimum_target_delta':-.0002,'cpu_us_max':76.92753780833335},
            'training_cache_sha256':campaign.sha(TRAIN_4096/'identity.json'),
            'tree_fit_cache_sha256':campaign.sha(TRAIN_1024/'identity.json'),
            'source_sha256':__import__('hashlib').sha256(source().encode()).hexdigest(),
            'tree_sha256':campaign.sha(campaign.OUT/'tree_execution/frozen_trees.npz'),
            'limitation':'Mask teacher used search mask indicators, never target labels; training selection is correction-disjoint, not independent of the supplied base model.'})
        campaign.event('experiment_planned',id='tree_training_selection',protocol_sha256=campaign.sha(protocol))
    if not (work/'artifacts/selection.json').exists():
        work.mkdir(exist_ok=True);(work/'train.py').write_text(source(),encoding='utf-8')
        shutil.copyfile(COMBO,work/'combo.npz')
        shutil.copyfile(campaign.PREVIOUS/'nonlinear_weighted_t0/isolated_training/mask_trees.npz',work/'mask_trees.npz')
        shutil.copyfile(campaign.OUT/'tree_execution/frozen_trees.npz',work/'frozen_trees.npz')
        groups=json.loads((TRAIN_1024/'identity.json').read_text())['groups'];np.savez(work/'excluded_groups.npz',groups=groups)
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,TRAIN_4096)
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            campaign.event('implementation_failure',id='tree_training_selection',status=status,error=stderr.decode(errors='replace')[-1500:])
            raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={name:campaign.sha(work/name) for name in ['train.py','combo.npz','mask_trees.npz','frozen_trees.npz','excluded_groups.npz','artifacts/selection.json','artifacts/selection_moments.npz']}
    if (work/'selection_identity.json').exists():
        assert identity==json.loads((work/'selection_identity.json').read_text())
    else:
        assert identity['combo.npz']==COMBO_SHA and identity['train.py']==json.loads(protocol.read_text())['source_sha256']
        write_json(work/'selection_identity.json',identity);campaign.event('selection_frozen',id='tree_training_selection',identity=identity)
    report=json.loads((work/'artifacts/selection.json').read_text())
    print(json.dumps({k:report[k] for k in ['scores','selected','rows','scope']}),flush=True)

def check():
    work=OUT/'isolated_selection';identity=json.loads((work/'selection_identity.json').read_text())
    assert all(campaign.sha(work/name)==value for name,value in identity.items())
    selection=json.loads((work/'artifacts/selection.json').read_text())['selected'];strength=selection['strength']
    destination=OUT/'report.json'
    if destination.exists():
        print(destination.read_text());return
    with np.load(work/'frozen_trees.npz') as q:
        trees=[q['tree'+str(i)] for i in range(32)];baseline=float(q['baseline'].ravel()[0])
    combo=load_combo()
    def predictor(z,base):
        # Frozen tree routing; target values and scoring flags are not consumed.
        x=np.column_stack((np.clip((z['x']-combo['mean'])/combo['scale'],-8,8),np.clip(base,-2,2))).astype(np.float32)
        correction=np.full(len(x),baseline)
        for nodes in trees:
            correction+=tree_predict(x,nodes)
        p=base.copy();p[:,0]+=strength*correction.astype(np.float32);return p
    directory=OUT/'callback_check';directory.mkdir(exist_ok=True);(directory/'artifacts').mkdir(exist_ok=True)
    compact_tree_onnx.export(trees,baseline,directory/'artifacts/fused.onnx',directory,strength=strength)
    (directory/'callback.py').write_text(campaign_onnx.callback_source('current'),encoding='utf-8')
    artifacts=_check_artifacts(directory,{'callback.py':campaign.sha(directory/'callback.py')});archive,deployment=_package(directory)
    sandbox=GeneratedSandbox();sandbox.preflight(WORKER,directory);deadline=time.monotonic()+1800
    validation=validate_callback(sandbox,directory,TRAIN_1024,deadline)
    _,z=next(cached(TRAIN_1024));actual,_=replay(sandbox,directory,[z],deadline);expected=predictor(z,predict_combo(z,combo))
    np.testing.assert_allclose(actual[0][z['need']],expected[z['need']],atol=3e-5,rtol=3e-5)
    validation.update(stream_batch_parity=True,label_mask_independence=True,finite_outputs=True,
        max_abs_parity=float(np.max(np.abs(actual[0][z['need']]-expected[z['need']]))))
    result=assess(predictor,diagnostics=True);ci=paired99(predictor)
    eligible=result['delta_combined']>=.0002 and ci[0]>0 and min(result['delta_per_target'])>=-.0002 and validation['callback_us_per_row']<=76.92753780833335
    report={'selection':selection,'search':result,'paired99':ci,'callback_checks':validation,
        'status':'AWAITING PROMOTION' if eligible else 'scientific_negative_or_insufficient_search_evidence',
        'identity':{'selection':identity,'artifacts':artifacts,'deployment':deployment,'archive_sha256':campaign.sha(archive)}}
    write_json(destination,report)
    if eligible:
        awaiting=OUT/'awaiting_promotion';shutil.copytree(directory/'deploy',awaiting)
        write_json(awaiting/'status.json',{'status':'AWAITING PROMOTION','report_sha256':campaign.sha(destination)})
    campaign.event('experiment_completed',id='tree_training_selection',report_sha256=campaign.sha(destination),
        wp=result['candidate']['weighted_pearson'],paired99=ci,cpu_us=validation['callback_us_per_row'],status=report['status'])
    print(json.dumps({'selection':selection,'wp':result['candidate']['weighted_pearson'],'paired99':ci,'callback':validation,'status':report['status']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','check'],required=True)
    args=parser.parse_args();train() if args.phase=='train' else check()
