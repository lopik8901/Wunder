"""Explain feature/selector confounding without fitting or selecting candidates."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from competition_engineering import book_side_campaign as b
from competition_engineering import prediction_shape_campaign as c
from competition_engineering.pipeline import cached,write_json

def run():
    path=b.OUT/'selection_counterfactual.json'
    if path.exists():print(path.read_text());return
    assert not json.loads((b.OUT/'development.json').read_text())['development_gate_passed']
    protocol=b.OUT/'selection_counterfactual_protocol.json'
    write_json(protocol,{'observation':'Primary minus failed; repeat minus and controls chose different t0 strengths. Positive averaged contrasts may mix representation and selector effects.','counterfactual':'Apply the ORIGINAL selection-frozen minus_r1 strength vector to all six frozen maps on already consumed development rows. No new model, strength search, selected candidate or replication access. Also quantify selection uncertainty for the SAME fixed vector against zero on the already consumed selection population.','rule':'Report matched-strength minus-control contrasts with paired95 intervals for both fits and their mean. Diagnostic cannot change the failed primary gate.'});c.event('side_selection_counterfactual_preregistered',protocol_sha256=b.sha(protocol))
    work=b.OUT/'fit_work';assert all(b.sha(work/p)==h for p,h in json.loads((work/'identity.json').read_text()).items())
    with np.load(work/'artifacts/models.npz') as q:model={k:q[k] for k in q.files}
    strength=model['minus_r1_strength'];keys=[f+'_r'+str(r) for r in [0,1] for f in ['core','minus','plus']];result={}
    for role,directory,n,seed in [('development',b.OUT/'development',256,20261088),('selection',b.OUT/'training',128,20261089)]:
        moments=[]
        for group,z in cached(directory):
            if role=='selection' and int(z['role'])!=1:continue
            predictions=[z['base']]
            for key in keys:predictions.append(z['base']+b.design(z,key.split('_')[0],model)@model[key+'_coef']*strength)
            moments.append([[c.focus_stats(z['y'],p,w) for w in [np.ones(len(z['base'])),z['focus']]] for p in predictions])
        moments=np.array(moments);assert len(moments)==n;counts=c.bootstrap_counts(n,np.random.default_rng(seed),draws=10000);scores=c.pooled_correlations(moments.sum(0));boot=c.pooled_correlations(c.bootstrap_stats(moments,counts));delta=boot-boot[:,:1];point=scores-scores[:1]
        contrasts={}
        for control,js in [('core',[1,4]),('plus',[3,6])]:
            target_point=np.stack([point[2]-point[js[0]],point[5]-point[js[1]]]);draw=np.stack([delta[:,2]-delta[:,js[0]],delta[:,5]-delta[:,js[1]]],axis=1)
            contrasts[control]={'paired_fit_target':target_point.tolist(),'paired_fit_combined':target_point.mean(-1).tolist(),'mean_paired95':np.quantile(draw.mean((1,3)),[.025,.975],axis=0).tolist()}
        result[role]={'target_gains':{key:point[j+1].tolist() for j,key in enumerate(keys)},'target_gain_paired95':{key:np.quantile(delta[:,j+1],[.025,.975],axis=0).tolist() for j,key in enumerate(keys)},'matched_strength_contrasts':contrasts}
        np.savez(b.OUT/('counterfactual_'+role+'_moments.npz'),moments=moments)
    r={'fixed_strength':strength.tolist(),'populations':result,'interpretation':'Matched-strength diagnostic on consumed data; no new selected candidate. Original development gate remains failed.'}
    write_json(path,r);c.event('side_selection_counterfactual_completed',report_sha256=b.sha(path));print(json.dumps({role:row['matched_strength_contrasts'] for role,row in result.items()}))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
