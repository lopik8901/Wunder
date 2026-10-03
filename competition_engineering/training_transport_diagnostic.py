"""Test source-to-search transport after nonlinear weighting; no fitting."""
from competition_engineering import reassessment_campaign as campaign
import json
import numpy as np
from competition_engineering.manual_search_core import TRAIN_1024,load_combo,predict_combo
from competition_engineering.pipeline import cached,write_json
from competition_engineering.gradient_uncertainty import primitives,gradient,bootstrap_counts,resample,cosine
from competition_engineering.nonlinear_selection_diagnostic import weighted_primitives
from competition_engineering.nonlinear_weighted_readout import mask_focus,OUT as FIT_OUT
from competition_engineering.residual import from_stats

def run():
    destination=campaign.OUT/'training_transport.json'
    if destination.exists():
        return
    protocol={'observation':'Nonlinear mask weighting improves within-search t0 gradient representation but its weighted correction did not improve official WP.',
        'hypothesis':'Within-search mask transport does not imply source-to-search target transport.',
        'competing_explanations':['Insufficient effective training sequences after weighting','Stable source/selected-search conditional conflict','High-dimensional gradient uncertainty','Raw loss geometry still misses a transferable weighted direction'],
        'tests':'Frozen mask ensemble on1024 training sequences; exact fixed stride40 phase; paired source/weighted gradients and official search gradients; effective sequence mass.',
        'next_fit_rule':'A new directional fit requires positive99 lower for weighted training-to-official alignment and improvement over uniform training. Scaling requires a clearly identified effective-sample limitation and stable source-side gradient direction.',
        'decision':'If transport is uncertain, gather a genuinely new causal-information lead rather than tune weights, penalties or loss.'}
    config=campaign.OUT/'training_transport_protocol.json'
    write_json(config,protocol);campaign.event('experiment_planned',id='training_transport',protocol_sha256=campaign.sha(config))
    with np.load(FIT_OUT/'isolated_training/mask_trees.npz') as q:
        trees={k:q[k] for k in q.files}
    combo=load_combo();stats={k:[] for k in ['uniform','weighted']};cross={k:[] for k in stats};focus_values=[]
    for index,(_,z) in enumerate(cached(TRAIN_1024)):
        idx=np.flatnonzero(z['need'])[::40];base=predict_combo(z,combo)[idx]
        f=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)))
        focus=mask_focus(z['x'][idx],base,z['step'][idx],combo,trees);focus_values.append(focus)
        for name,multiplier in [('uniform',np.ones(len(idx))),('weighted',focus)]:
            s,c=weighted_primitives(f,z['y'][idx],base,multiplier);stats[name].append(s);cross[name].append(c)
        if (index+1)%128==0:
            print('transport',index+1,flush=True)
    stats={k:np.array(v) for k,v in stats.items()};cross={k:np.array(v) for k,v in cross.items()}
    np.savez(campaign.OUT/'training_transport_moments.npz',**{**{k+'_stats':v for k,v in stats.items()},**{k+'_cross':v for k,v in cross.items()}})
    rng=np.random.default_rng(20261008);counts=bootstrap_counts(1024,rng,draws=5000)
    boots={k:resample(stats[k],cross[k],rng,counts=counts) for k in stats}
    with np.load(campaign.OUT/'gradient_primitives.npz') as q:
        ss=q['search_scored_stats'];sc=q['search_scored_cross']
    search=resample(ss,sc,rng,draws=5000);point_search=gradient(ss.sum(0),sc.sum(0))
    cosines={k:cosine(b,search) for k,b in boots.items()};improvement=cosines['weighted']-cosines['uniform']
    summaries={}
    for key in stats:
        g=gradient(stats[key].sum(0),cross[key].sum(0));mass=stats[key][:,0]
        split=[]
        for _ in range(500):
            order=rng.permutation(1024);a=order[:512];b=order[512:]
            split.append(cosine(gradient(stats[key][a].sum(0),cross[key][a].sum(0)),gradient(stats[key][b].sum(0),cross[key][b].sum(0))))
        summaries[key]={'wp':from_stats(stats[key].sum(0)).tolist(),'point_search_cosine':cosine(g,point_search).tolist(),
            'search_cosine99':np.quantile(cosines[key],[.005,.995],axis=0).tolist(),
            'effective_sequence_mass':(mass.sum(0)**2/(mass*mass).sum(0)).tolist(),
            'gradient_norm':np.linalg.norm(g,axis=0).tolist(),'bootstrap_noise_norm':np.sqrt(((boots[key]-g)**2).sum(1).mean(0)).tolist(),
            'split_half_cosine_01_50_99':np.quantile(split,[.01,.5,.99],axis=0).tolist()}
    lower=np.quantile(cosines['weighted'],.005,axis=0);gain_lower=np.quantile(improvement,.005,axis=0)
    eligible=[t for t in range(2) if lower[t]>0 and gain_lower[t]>0]
    values=np.concatenate(focus_values)
    result={'populations':summaries,'alignment_improvement99':np.quantile(improvement,[.005,.995],axis=0).tolist(),
        'focus_quantiles':np.quantile(values,[0,.1,.5,.9,1]).tolist(),'focus_lower_cap_fraction':float((values==.2).mean()),'focus_upper_cap_fraction':float((values==5).mean()),
        'eligible_targets_for_directional_fit':eligible,'protocol_sha256':campaign.sha(config),
        'moments_sha256':campaign.sha(campaign.OUT/'training_transport_moments.npz'),
        'limitation':'Descriptive gradient uncertainty conditional on frozen mask models and incumbent; no adaptive error guarantee.'}
    write_json(destination,result);campaign.event('experiment_completed',id='training_transport',result_sha256=campaign.sha(destination),eligible_targets=eligible)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    run()
