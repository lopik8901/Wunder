"""Training-only prospective audit of a frozen historical candidate shortlist.

No coefficients are fit and no protected validation sequences are read. Fresh
here means disjoint from local correction-fitting caches, not a guarantee about
the undisclosed provenance of the supplied pretrained GRU.
"""
from __future__ import annotations
from competition_engineering import reassessment_campaign as campaign
import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from competition_engineering.manual_search_core import ROOT,TRAIN_1024,TRAIN_4096,SEARCH,load_combo,predict_combo
from competition_engineering.pipeline import read_sequence,cached,write_json
from competition_engineering.gru_sequence import SequenceGRU
from competition_engineering.objective_alignment import output_directory
from competition_engineering.full_moment_objective import output_directory as full_directory
from competition_engineering.residual import sufficient,from_stats
from tools.diagnose_causal_scoring_population import classifier_features

def new_groups(total,excluded,seed=20261003,size=512):
    eligible=np.array(sorted(set(range(total))-set(excluded)))
    if len(eligible)<size:
        raise ValueError('Insufficient correction-disjoint training sequences')
    return np.random.default_rng(seed).permutation(eligible)[:size].tolist()

def shortlist():
    models=[('incumbent',np.zeros(115),0,None)]
    for target in (0,1):
        for tier in (1024,4096):
            directory=output_directory(tier,target)
            diagnosis=json.loads((directory/'diagnosis.json').read_text())
            chosen=diagnosis['selections']['raw_residual_select_wp']
            fit=directory/'isolated_training'
            identity=json.loads((fit/'fit_identity.json').read_text())
            assert all(campaign.sha(fit/name)==sha for name,sha in identity.items())
            with np.load(fit/'artifacts/models.npz') as q:
                coef=q[chosen['model']].copy()
            if np.any(coef[115:]!=0):
                raise ValueError('Shortlist unexpectedly changes temporal coefficients')
            models.append((f'raw_t{target}_{tier}',coef[:115]*chosen['strength'],target,diagnosis['search']['raw_residual_select_wp']['wp']))
    # One separately labelled exploratory point; never treated as an incumbent.
    directory=full_directory(4096);fit=directory/'isolated_training'
    identity=json.loads((fit/'fit_identity.json').read_text())
    assert all(campaign.sha(fit/name)==sha for name,sha in identity.items())
    with np.load(fit/'artifacts/models.npz') as q:
        coef=q['raw_residual_1.0'].copy()
    if np.any(coef[115:]!=0):
        raise ValueError('Exploratory point changes temporal coefficients')
    models.append(('historical_exploratory_point',coef[:115],0,.6590796311188727))
    return models

def focus_stats(y,p,focus):
    y=np.clip(y,-2,2).astype(np.float64);p=np.clip(p,-2,2).astype(np.float64)
    w=np.abs(y)*np.asarray(focus)[:,None]
    return np.stack((w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)))

def freeze_protocol():
    path=campaign.OUT/'prospective_selection_protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    train=ROOT/'competition_engineering/assets/wnn_connectome_starterpack/datasets/train.parquet'
    parquet=pq.ParquetFile(train);excluded=set();identities={}
    for name in ['gru_train_pilot','gru_train_scale_phase2','gru_train_medium_4096','gru_train_holdout_phase2']:
        identity=ROOT/'competition_engineering/cache'/name/'identity.json'
        q=json.loads(identity.read_text())
        if not q['split'].startswith('train') or not Path(q['data']['path']).name=='train.parquet':
            raise ValueError('Only training cache identities may define exclusion')
        excluded.update(q['groups']);identities[name]=campaign.sha(identity)
    groups=new_groups(parquet.num_row_groups,excluded)
    models=shortlist();np.savez(campaign.OUT/'prospective_shortlist.npz',**{name:coef for name,coef,_,_ in models})
    protocol={'observation':'64-sequence gradient uncertainty is large; correction internal-selection WP may not describe official scoring populations.',
        'hypothesis':'Training-side prospective selection can separate fitting overfit from scoring/distribution mismatch.',
        'competing_explanations':['Correction training overfit','Mask population differences','Search-sequence sampling noise','Model near-stationarity'],
        'models':[{'name':name,'target':target,'previous_search_wp':score} for name,coef,target,score in models],
        'coefficients_sha256':campaign.sha(campaign.OUT/'prospective_shortlist.npz'),
        'source_sha256':campaign.sha(Path(__file__)),
        'training_file':str(train.relative_to(ROOT)),'training_file_bytes':train.stat().st_size,
        'excluded_correction_groups':len(excluded),'exclusion_identity_sha256':identities,
        'selection_groups':groups[:256],'audit_groups':groups[256:],
        'selection_rule':'Choose one candidate per mask by pooled official-form WP on256 selection sequences, including incumbent; freeze choice before audit.',
        'masks':['All required rows','Bernoulli causal selection probability from frozen cross-fitted classifier ensemble','Uniform Bernoulli mask with same expected density'],
        'mask_seed':20261004,
        'causal_mask':'Uses only present input/frozen prediction/current step and predeclared independent random coins; never targets.',
        'audit_rule':'A prospective selection procedure is informative only if its selected candidate beats incumbent with paired99 lower>0 on disjoint256 audit groups.',
        'scope_limits':['Training data only; no true scoring mask exists there.','Base pretrained GRU provenance unknown, so independence applies only to local corrections.',
                        'Previously search-selected models remain adaptively chosen; this is a diagnostic, not protected evaluation.',
                        'Proxy-masked WP does not establish official-test improvement.'],
        'decision':'If training-side winners do not predict official search behavior, do not use this pool as a replacement promotion gate.'}
    write_json(path,protocol);campaign.event('experiment_planned',id='prospective_selection',protocol_sha256=campaign.sha(path))
    return protocol

def run():
    result_path=campaign.OUT/'prospective_selection_result.json'
    if result_path.exists():
        return
    protocol=freeze_protocol();combo=load_combo();models=shortlist()
    baseline=ROOT/'competition_engineering/assets/wnn_connectome_starterpack/baseline/baseline.onnx'
    if campaign.sha(baseline)!='321ec67b93f3c5dde2c0901ca4532072dad5f65753c2e6519e93ea3555f6a6fd':
        raise ValueError('Official GRU identity changed')
    extractor=SequenceGRU(baseline)
    _,preflight=next(cached(TRAIN_1024));started=time.perf_counter()
    prediction=extractor.predict(preflight['x'][None])[0]
    np.testing.assert_allclose(prediction[preflight['need']],preflight['p'][preflight['need']],atol=3e-6,rtol=3e-6)
    preflight_seconds=time.perf_counter()-started
    classifier_path=ROOT/'competition_engineering/runs/objective_alignment_20261002/causal_population_classifier.npz'
    with np.load(classifier_path) as q:
        classifier={k:q[k].copy() for k in q.files}
    parquet=pq.ParquetFile(ROOT/protocol['training_file']);stats=[];counts=[];completed=[]
    partial=campaign.OUT/'prospective_sequence_stats.npz'
    if partial.exists():
        with np.load(partial) as q:
            stats=list(q['stats']);counts=list(q['counts']);completed=q['groups'].tolist()
    order=protocol['selection_groups']+protocol['audit_groups']
    if completed!=order[:len(completed)]:
        raise ValueError('Prospective checkpoint group order changed')
    # Freeze selector choices after first256 sequences, before auditing the rest.
    choices_path=campaign.OUT/'prospective_selected.json'
    for index,group in enumerate(order[len(completed):],start=len(completed)):
        seq,need,x,y,_=read_sequence(parquet,group)
        raw=extractor.predict(x[None])[0]
        z={'x':x,'y':y,'p':raw,'need':need,'mask':need,'step':np.arange(len(x))}
        base=predict_combo(z,combo);idx=np.flatnonzero(need)
        f=np.column_stack((np.ones(len(idx)),np.clip(base[idx],-2,2),np.clip((x[idx]-combo['mean'])/combo['scale'],-8,8)))
        probability=expit(classifier_features(x[idx],base[idx],idx,classifier['mean'],classifier['scale'])@classifier['coef'].mean(0))
        rng=np.random.default_rng(protocol['mask_seed']+group);coin=rng.random(len(idx))
        masks=[np.ones(len(idx),bool),coin<probability,coin<float(probability.mean())]
        moments=[]
        for name,coef,target,score in models:
            candidate=base[idx].copy();candidate[:,target]+=f@coef
            moments.append([sufficient(y[idx][mask],candidate[mask]) for mask in masks])
        stats.append(moments);counts.append([int(m.sum()) for m in masks]);completed.append(group)
        if (index+1)%16==0 or index==255 or index==len(order)-1:
            np.savez(partial,stats=np.array(stats),counts=np.array(counts),groups=np.array(completed))
            write_json(campaign.OUT/'prospective_progress.json',{'completed':index+1,'phase':'selection' if index<256 else 'audit','preflight_seconds':preflight_seconds})
            print('prospective',index+1,flush=True)
        if index==255 and not choices_path.exists():
            selection=np.array(stats).sum(0)
            scores=np.array([[from_stats(s).mean() for s in model] for model in selection])
            choices={name:int(scores[:,j].argmax()) for j,name in enumerate(['uniform','causal_proxy','random_proxy'])}
            write_json(choices_path,{'choices':choices,'selection_wp':scores.tolist(),
                                     'shortlist_sha256':protocol['coefficients_sha256'],'checkpoint_sha256':campaign.sha(partial)})
            campaign.event('prospective_selection_frozen_before_audit',choices=choices,sha256=campaign.sha(choices_path))
    stats=np.array(stats);choices=json.loads(choices_path.read_text())['choices'];rng=np.random.default_rng(20261005)
    outcomes={};bootstrap_indices=rng.integers(256,size=(10000,256))
    for phase,block in [('selection',stats[:256]),('audit',stats[256:])]:
        sums=block.sum(0);scores=np.array([[from_stats(s).mean() for s in model] for model in sums])
        rows={}
        for j,mask in enumerate(['uniform','causal_proxy','random_proxy']):
            rows[mask]={}
            boot=block[bootstrap_indices,:,j].sum(1) if phase=='audit' else None
            # Vectorize pooled moments to avoid 60k Python scoring calls.
            if boot is not None:
                w,sy,sp,syy,spp,syp=np.moveaxis(boot,-2,0)
                corr=(syp-sy*sp/w)/np.sqrt(np.maximum(syy-sy*sy/w,1e-30)*np.maximum(spp-sp*sp/w,1e-30))
                delta=corr.mean(-1)-corr[:,0].mean(-1)[:,None]
            for k,model in enumerate(models):
                rows[mask][model[0]]={'wp':float(scores[k,j]),'delta':float(scores[k,j]-scores[0,j]),
                    'paired99':np.quantile(delta[:,k],[.005,.995]).tolist() if boot is not None else None,
                    'selected':k==choices[mask]}
        outcomes[phase]=rows
    result={'outcomes':outcomes,'choices':choices,'sequence_count':len(stats),'mask_row_counts':np.array(counts).sum(0).tolist(),
        'official_gru_preflight_seconds':preflight_seconds,'source_sha256':campaign.sha(Path(__file__)),
        'stats_sha256':campaign.sha(partial),'protocol_sha256':campaign.sha(campaign.OUT/'prospective_selection_protocol.json'),
        'boundary':'Correction-disjoint training sequences only; no official-search or protected labels used for prospective selection/audit.'}
    write_json(result_path,result);campaign.event('experiment_completed',id='prospective_selection',result_sha256=campaign.sha(result_path))
    print(json.dumps({'choices':choices,'audit_selected':{mask:outcomes['audit'][mask][models[k][0]] for mask,k in choices.items()}}),flush=True)

if __name__=='__main__':
    run()
