"""Paired population comparison of the prospectively frozen shortlist."""
from competition_engineering import reassessment_campaign as campaign
import json
import numpy as np
from scipy.special import expit
from competition_engineering.prospective_selection_diagnostic import shortlist
from competition_engineering.manual_search_core import SEARCH,load_combo,predict_combo,SEARCH_ROOT_WP
from competition_engineering.pipeline import cached,write_json
from competition_engineering.residual import sufficient,from_stats
from tools.diagnose_causal_scoring_population import classifier_features

def pooled_correlations(stats):
    w,sy,sp,syy,spp,syp=np.moveaxis(stats,-2,0)
    return (syp-sy*sp/w)/np.sqrt(np.maximum(syy-sy*sy/w,1e-30)*np.maximum(spp-sp*sp/w,1e-30))

def run():
    destination=campaign.OUT/'search_population_audit.json'
    if destination.exists():
        return
    models=shortlist();combo=load_combo();mask_names=['uniform','official','causal_proxy','random_proxy']
    protocol={'hypothesis':'Frozen corrections may improve uniform populations while worsening official selected populations, or gains may be dominated by sequence noise.',
        'models':[m[0] for m in models],'masks':mask_names,'draws':10000,'seed':20261006,
        'coefficient_sha256':campaign.sha(campaign.OUT/'prospective_shortlist.npz'),
        'rule':'Population-specific conflict requires paired99 interval for official-minus-uniform candidate delta to exclude zero. Otherwise do not claim conflict.',
        'decision':'No fit or selection from search labels; inspect only prospectively frozen choices and descriptive contrasts.'}
    path=campaign.OUT/'search_population_protocol.json'
    if not path.exists():
        write_json(path,protocol);campaign.event('experiment_planned',id='search_population_audit',protocol_sha256=campaign.sha(path))
    with np.load(campaign.ROOT/'competition_engineering/runs/objective_alignment_20261002/causal_population_classifier.npz') as q:
        classifier={k:q[k] for k in q.files}
    stats=[];counts=[]
    for index,(group,z) in enumerate(cached(SEARCH)):
        idx=np.flatnonzero(z['need']);base=predict_combo(z,combo)[idx];y=z['y'][idx]
        f=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)))
        probability=expit(classifier_features(z['x'][idx],base,z['step'][idx],classifier['mean'],classifier['scale'])@classifier['coef'][index%4])
        coin=np.random.default_rng(20261006+group).random(len(idx))
        masks=[np.ones(len(idx),bool),z['mask'][idx],coin<probability,coin<probability.mean()]
        row=[]
        for name,coef,target,previous in models:
            prediction=base.copy();prediction[:,target]+=f@coef
            row.append([sufficient(y[m],prediction[m]) for m in masks])
        stats.append(row);counts.append([int(m.sum()) for m in masks])
    stats=np.array(stats);np.savez(campaign.OUT/'search_population_moments.npz',stats=stats,counts=counts)
    wp=pooled_correlations(stats.sum(0));scores=wp.mean(-1)
    if abs(scores[0,1]-SEARCH_ROOT_WP)>2e-6:
        raise ValueError('Frozen official search incumbent did not reproduce')
    for k,model in enumerate(models):
        if model[3] is not None and abs(scores[k,1]-model[3])>2e-6:
            raise ValueError('Historical frozen correction did not reproduce')
    rng=np.random.default_rng(20261006);indices=rng.integers(64,size=(10000,64))
    boot=pooled_correlations(stats[indices].sum(1)).mean(-1)
    delta=boot-boot[:,0,None,:];point=scores-scores[0,None,:]
    results={}
    for k,model in enumerate(models):
        results[model[0]]={'populations':{mask:{'wp':float(scores[k,j]),'delta':float(point[k,j]),
            'paired99':np.quantile(delta[:,k,j],[.005,.995]).tolist()} for j,mask in enumerate(mask_names)},
            'official_minus_uniform_delta':float(point[k,1]-point[k,0]),
            'official_minus_uniform_delta_paired99':np.quantile(delta[:,k,1]-delta[:,k,0],[.005,.995]).tolist()}
    result={'models':results,'rows_per_mask':np.array(counts).sum(0).tolist(),
        'scope':'Reused fixed64-sequence search; descriptive uncertainty, no independence or adaptive correction.',
        'protocol_sha256':campaign.sha(path),'moments_sha256':campaign.sha(campaign.OUT/'search_population_moments.npz')}
    write_json(destination,result);campaign.event('experiment_completed',id='search_population_audit',result_sha256=campaign.sha(destination))
    print(json.dumps(results),flush=True)

if __name__=='__main__':
    run()
