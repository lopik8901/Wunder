"""Prospective modeling campaign on frozen causal representations."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from competition_engineering.manual_search_core import ROOT,TRAIN_1024,TRAIN_4096,COMBO,COMBO_SHA,load_combo,predict_combo
from competition_engineering.pipeline import write_json
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.probability_readout import probability_focus

OUT=ROOT/'competition_engineering/runs/representation_20261002'
PREVIOUS=ROOT/'competition_engineering/runs/root_cause_20261002'
TRAIN=OUT/'prepared_training'
REPLICATION=OUT/'prepared_replication'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def event(kind,**payload):
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'ledger.jsonl'
    row={'utc':datetime.now(timezone.utc).isoformat(),'kind':kind,'boundary':'search-only',
         'previous_ledger_sha256':sha(path) if path.exists() else None,**payload}
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def group_plan(fit_groups,pool_groups,excluded,seed=20261013):
    if len(fit_groups)!=1024 or len(set(fit_groups))!=1024 or len(set(pool_groups))!=len(pool_groups):
        raise ValueError('Expected unique fixed training groups')
    rng=np.random.default_rng(seed);fit=rng.permutation(fit_groups).tolist()
    choices=rng.permutation(sorted(set(pool_groups)-set(fit_groups)-set(excluded))).tolist()
    if len(choices)<512:
        raise ValueError('Insufficient training-derived development groups')
    return {'fit_a':fit[:512],'fit_b':fit[512:],'selection':choices[:256],'replication':choices[256:512]}

def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        return json.loads((OUT/'protocol.json').read_text())
    assert sha(COMBO)==COMBO_SHA
    knowledge=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text())
    write_json(OUT/'starting_knowledge.json',knowledge)
    fit=json.loads((TRAIN_1024/'identity.json').read_text())['groups']
    pool=json.loads((TRAIN_4096/'identity.json').read_text())['groups']
    excluded=json.loads((PREVIOUS/'tree_training_selection/isolated_selection/artifacts/selection.json').read_text())['groups']
    plan=group_plan(fit,pool,excluded)
    protocol={'started_unix':time.time(),'incumbent_sha256':COMBO_SHA,'incumbent_wp':knowledge['incumbent']['wp'],
        'reassessment':{
            'supported':['Population-dependent correction utility','WP selection over residual MSE','Cheap compact nonlinear callbacks'],
            'weakened':['Objective or weighting tweaks alone resolve remaining performance','Point-gradient disagreement reliably rejects a representation'],
            'inadequately_tested':['Learned frozen GRU hidden information beyond current112 inputs and two output values','Representation gains replicated across disjoint training sequence groups'],
            'competing_explanations':['Missing causal information in current-row correction basis','Representation already exhausted by the pretrained output','Population-specific utility and finite-sequence noise'],
            'decision':'Selection/transport is real but not established as the highest-leverage modeling bottleneck. Test a previously unfit learned representation using a minimal frozen selection/replication mechanism.'},
        'hypothesis':'Frozen forward GRU activations contain useful correction information missing from current-row readouts; a fixed linear probe can recover it without new dynamics.',
        'fit':'Current115-dimensional control versus current-plus256 frozen GRU states, same weighted residual MSE and fixed ridge penalty0.1 per fit row. Fit A,B and pooled coefficients.',
        'sampling':'500 uniformly sampled required rows per training sequence; seed20261013+group. All prior rows used by frozen causal state extraction.',
        'normalization':'Hidden mean/std learned on fitting groups only; std floor0.05, clip standardized features to[-8,8].',
        'selection':'On256 disjoint training sequences, select each target strength from[0,.05,.1,.25,.5,1] maximizing the minimum WP gain across uniform and frozen-probability-weighted populations. Zero allowed.',
        'replication':'A separate256 training sequences are not mounted in the fitting sandbox; evaluate frozen strengths, A/B replica consistency and paired sequence intervals before search.',
        'search_comparisons':['current_selected','latent_selected','current_fixed025','latent_fixed025'],
        'search_rule':'One predeclared fixed-strength pair prevents a noisy proxy selector from hiding representation information. No search tuning; report targetwise effects and matched representation contrast.',
        'qualification':{'minimum_delta':.0002,'paired99_lower':0.,'minimum_target_delta':-.0002,'cpu_us_max':76.92753780833335},
        'groups':plan,'excluded_recent_tree_selection_groups':excluded,
        'source_identity_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [TRAIN_1024/'identity.json',TRAIN_4096/'identity.json']},
        'scope':'Training-derived development, not independent base-model validation. The fixed search remains adaptively reused. Old temporal dynamics and broad loss grids remain closed.'}
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'research_state_map.json',protocol['reassessment'])
    event('research_reassessed',protocol_sha256=sha(OUT/'protocol.json'),incumbent_sha256=COMBO_SHA)
    return protocol

def prepare():
    protocol=initialize();combo=load_combo();extractor=LatentExtractor()
    tree_path=ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(tree_path) as q:
        trees={k:q[k] for k in q.files}
    from competition_engineering.frozen_latent_readout import latent_graph
    graph_bytes=latent_graph().SerializeToString();graph_path=OUT/'frozen_extractor.onnx'
    if graph_path.exists():
        assert graph_path.read_bytes()==graph_bytes
    else:
        graph_path.write_bytes(graph_bytes)
    roles=['fit_a','fit_b','selection','replication'];records=[];done=0
    for role_index,role in enumerate(roles):
        source=TRAIN_1024 if role_index<2 else TRAIN_4096
        destination=TRAIN if role_index<3 else REPLICATION;destination.mkdir(exist_ok=True)
        for group in protocol['groups'][role]:
            path=destination/f'{group:05d}.npz';meta=destination/f'{group:05d}.json';original=source/f'{group:05d}.npz'
            if path.exists() and meta.exists():
                record=json.loads(meta.read_text())
                assert record['derived_sha256']==sha(path)
                records.append(record);done+=1;continue
            with np.load(original) as q:
                z={k:q[k] for k in q.files}
            prediction,hidden=extractor.predict(z['x'])
            np.testing.assert_allclose(prediction[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
            if done==0:
                changed=z['x'].copy();changed[10000:]+=.7
                changed_p,changed_h=extractor.predict(changed)
                np.testing.assert_array_equal(prediction[:10000],changed_p[:10000])
                np.testing.assert_array_equal(hidden[:10000],changed_h[:10000])
                p2,h2=extractor.predict(z['x']);np.testing.assert_array_equal(prediction,p2);np.testing.assert_array_equal(hidden,h2)
                write_json(OUT/'extractor_checks.json',{'baseline_parity':True,'causal_prefix':True,'reset_determinism':True,'graph_sha256':sha(graph_path)})
            eligible=np.flatnonzero(z['need'])
            idx=np.sort(np.random.default_rng(20261013+group).choice(eligible,500,replace=False))
            base=predict_combo(z,combo)[idx]
            norm=np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)
            current=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),norm)).astype(np.float32)
            focus=probability_focus(z['x'][idx],base,z['step'][idx],combo,trees)[:,1]
            np.savez(path,current=current,hidden=hidden[idx],base=base,y=z['y'][idx],focus=focus,role=role_index,group=group,indices=idx)
            record={'group':group,'role':role,'source_npz_sha256':sha(original),'source_cache':source.relative_to(ROOT).as_posix(),
                'derived_sha256':sha(path),'extractor_sha256':sha(graph_path),'rows':len(idx)}
            write_json(meta,record);records.append(record);done+=1
            if done%64==0:
                write_json(OUT/'progress.json',{'phase':'frozen_representation_preparation','sequences':done,'total':1536})
                print('PREPARED_SEQUENCES='+str(done),flush=True)
    for destination,names in [(TRAIN,roles[:3]),(REPLICATION,roles[3:])]:
        identity={'groups':[g for role in names for g in protocol['groups'][role]],'roles':names,
            'scope':'Derived only from designated training caches; no official search or protected samples.',
            'protocol_sha256':sha(OUT/'protocol.json'),'extractor_sha256':sha(graph_path),'teacher_sha256':sha(tree_path)}
        path=destination/'identity.json'
        if path.exists():
            assert json.loads(path.read_text())==identity
        else:
            write_json(path,identity)
    if not (OUT/'prepared_manifest.json').exists():
        write_json(OUT/'prepared_manifest.json',{'sequences':records,'protocol_sha256':sha(OUT/'protocol.json')})
        event('training_features_prepared',sequences=done,manifest_sha256=sha(OUT/'prepared_manifest.json'))
    print(json.dumps({'prepared':done,'training':1280,'replication':256}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['initialize','prepare'],required=True)
    args=parser.parse_args();initialize() if args.phase=='initialize' else prepare()
