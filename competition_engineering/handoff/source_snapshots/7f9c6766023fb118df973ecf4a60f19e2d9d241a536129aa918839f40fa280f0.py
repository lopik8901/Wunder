"""Analyze frozen fitting/development behavior; no candidate tuning or search."""
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from competition_engineering import alternative_representation as c
from competition_engineering import learned_causal_campaign as l
from competition_engineering.pipeline import cached,write_json
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.search_population_audit import pooled_correlations

def main():
    path=l.OUT/'generalization_diagnostic.json'
    if path.exists(): print(path.read_text());return
    with np.load(l.OUT/'fit/artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    totals={};errors={};energy={};rows={}
    for group,z in cached(l.OUT/'training'):
        role='fit' if int(z['role'])==0 else 'development'
        for replica in [0,1]:
            if role=='fit':
                idx=np.random.default_rng(20261053+group+replica*100000).permutation(len(z['base']))[:128]
                probe={k:v[idx] for k,v in z.items() if k in ['core','windows','valid','y','base','focus']}
            else: probe=z
            base=probe['base'];y=np.clip(probe['y'],-2,2);weight=np.abs(y)
            for family in ['current','temporal']:
                name=family+'_r'+str(replica);key=role+'_'+name
                correction=[];unshrunk=[]
                for target in [0,1]:
                    suffix=name+'_t'+str(target)
                    with torch.no_grad():
                        f=l.tensor_features(torch.from_numpy(probe['windows']),torch.from_numpy(probe['valid']),*[torch.from_numpy(model[suffix+'_'+v]) for v in ['projection','weights','bias']],family=='current').numpy()
                    raw=(probe['core']@model[suffix+'_linear']+f@model[suffix+'_head'])[:,0];unshrunk.append(raw);correction.append(raw*model[suffix+'_strength'])
                correction=np.column_stack(correction);unshrunk=np.column_stack(unshrunk);stand=np.column_stack([l.numpy_standalone(probe,model,name,t) for t in [0,1]])
                candidates=[base,base+correction,base+unshrunk,stand]
                moment=np.array([[focus_stats(probe['y'],p,pop) for pop in [np.ones(len(base)),probe['focus']]] for p in candidates])
                totals[key]=totals.get(key,np.zeros_like(moment))+moment
                values=np.stack([weight.sum(0),(weight*(y-base)**2).sum(0),(weight*(y-base-unshrunk)**2).sum(0),(weight*(y-base-correction)**2).sum(0)])
                errors[key]=errors.get(key,np.zeros_like(values))+values;energy[key]=energy.get(key,np.zeros(2))+(weight*unshrunk**2).sum(0);rows[key]=rows.get(key,0)+len(base)
    report={}
    for key,m in totals.items():
        score=pooled_correlations(m);error=errors[key]
        report[key]={'rows':rows[key],'baseline_selected_unshrunk_standalone_wp_uniform_probability':score.tolist(),'selected_gain':(score[1]-score[0]).mean(-1).tolist(),'unshrunk_gain':(score[2]-score[0]).mean(-1).tolist(),'weighted_baseline_unshrunk_selected_mse':(error[1:]/error[:1]).tolist(),'correction_rms':np.sqrt(energy[key]/error[0]).tolist()}
    write_json(path,{'frozen_models':report,'interpretation':'Same frozen selected weights and exact fitting samplers; no new tuning. Residual-training feature learning can improve fitting error while degrading development/replication WP and raw-target standalone decoding.'})
    c.event('generalization_gap_diagnosed',report_sha256=c.sha(path));print(json.dumps(report),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main()
