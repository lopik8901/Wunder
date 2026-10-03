"""Aggregate objective-gradient and within/between-sequence transfer diagnosis."""
import numpy as np
from competition_engineering import objective_alignment as oa
from competition_engineering.manual_search_core import TRAIN_1024, SEARCH, load_combo, predict_combo
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import sufficient, from_stats
from threadpoolctl import threadpool_limits


def summarize(directory, training, propensity=None, stride=40):
    combo=load_combo(); blocks=[]; gradients=np.zeros((3,115,2)); nx=0
    sx=np.zeros(112); sxx=np.zeros(112)
    for _,z in cached(directory):
        idx=np.flatnonzero(z['need'])[::stride] if training else np.flatnonzero(z['mask'])
        base=predict_combo(z,combo)[idx]
        x=np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)
        f=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),x)).astype(np.float64)
        y=np.clip(z['y'][idx],-2,2).astype(np.float64); p=np.clip(base,-2,2).astype(np.float64)
        focus=np.ones(len(idx))
        if propensity is not None:
            position=np.searchsorted(propensity['step_edges'][1:-1],z['step'][idx],side='right')
            magnitude=np.searchsorted(propensity['magnitude_edges'][1:],np.max(np.abs(base),axis=1),side='right')
            focus=np.asarray(propensity['focus'])[position,magnitude]
        w=np.abs(y)*focus[:,None]; active=np.abs(base)<2
        gradients += np.stack((f.T@(w*active),f.T@(w*active*y),f.T@(w*active*p)))
        blocks.append(np.stack((w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0))))
        nx+=len(x); sx+=x.sum(0); sxx+=(x*x).sum(0)
    blocks=np.array(blocks); sums=blocks.sum(0); mass,sy,sp,syy,spp,syp=sums
    my=sy/mass; mp=sp/mass; vy=syy-sy*my; vp=spp-sp*mp; cov=syp-sy*mp
    slope=cov/vp
    gradient=(gradients[1]-my*gradients[0]-slope*(gradients[2]-mp*gradients[0]))/np.sqrt(vy*vp)
    within=(blocks[:,5]-blocks[:,1]*blocks[:,2]/blocks[:,0]).sum(0)
    return {'rows':nx,'sequences':len(blocks),'wp':from_stats(sums).tolist(),
            'between_sequence_covariance_fraction':((cov-within)/cov).tolist(),
            'weighted_target_mean':my.tolist(),'weighted_prediction_mean':mp.tolist(),
            'slope':slope.tolist(),'mean':sx/nx,'variance':np.maximum(sxx/nx-(sx/nx)**2,1e-12),'gradient':gradient}


def scoring_propensity():
    spec={'step_edges':[99,500,2000,5000,10000,20000],
          'magnitude_edges':[0,.25,.5,.75,1,1.5,2]}
    required=np.zeros((5,7)); selected=required.copy(); combo=load_combo()
    for _,z in cached(SEARCH):
        base=predict_combo(z,combo)
        position=np.searchsorted(spec['step_edges'][1:-1],z['step'],side='right')
        amplitude=np.searchsorted(spec['magnitude_edges'][1:],np.max(np.abs(base),axis=1),side='right')
        np.add.at(required,(position[z['need']],amplitude[z['need']]),1)
        mask=z['mask'] & z['need']
        np.add.at(selected,(position[mask],amplitude[mask]),1)
    spec.update(required_counts=required.tolist(),selected_counts=selected.tolist(),
                focus=np.clip((selected+.5)/(required+1)/(selected.sum()/required.sum()),.2,5).tolist(),
                incumbent_sha256=oa.COMBO_SHA,
                boundary='Aggregated search scoring indicators and frozen causal predictions only; no target values')
    return spec


def main():
    path=oa.OUT/'transfer_diagnosis.json'
    if path.exists():
        return
    with threadpool_limits(limits=1):
        train=summarize(TRAIN_1024,True); search=summarize(SEARCH,False)
        search_uniform=summarize(SEARCH,True)
    delta=(search['mean']-train['mean'])/np.sqrt(train['variance'])
    top=np.argsort(-np.abs(delta))[:10]
    cosine=(train['gradient']*search['gradient']).sum(0)/np.maximum(
        np.linalg.norm(train['gradient'],axis=0)*np.linalg.norm(search['gradient'],axis=0),1e-20)
    def cos(a,b):
        return ((a*b).sum(0)/np.maximum(np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0),1e-20)).tolist()
    result={'training':{k:v for k,v in train.items() if k not in ('mean','variance','gradient')},
            'search':{k:v for k,v in search.items() if k not in ('mean','variance','gradient')},
            'gradient_cosine_per_target':cosine.tolist(),
            'training_vs_uniform_search_gradient_cosine':cos(train['gradient'],search_uniform['gradient']),
            'uniform_vs_scored_search_gradient_cosine':cos(search_uniform['gradient'],search['gradient']),
            'uniform_search':{k:v for k,v in search_uniform.items() if k not in ('mean','variance','gradient')},
            'top_input_standardized_mean_shifts':[{'feature':int(i),'shift':float(delta[i])} for i in top],
            'interpretation':'Checks whether pooled objective directions and feature populations transfer across training/search; covariate shift is not established by mean differences alone.',
            'next_question':'If aligned fitting fails at both tiers, test whether correction gradients are stable across training sequence groups before considering shift adaptation.',
            'boundary':'aggregate statistics from fixed training and search caches only'}
    write_json(path,result); oa.event('objective_transfer_diagnosis',**result)
    print(result,flush=True)


def diagnose_population_alignment():
    path=oa.OUT/'population_gradient_diagnosis_v2.json'
    if path.exists():
        return
    with threadpool_limits(limits=1):
        propensity=scoring_propensity()
        scored=summarize(SEARCH,False)
        uniform=summarize(SEARCH,True,stride=1)
        weighted=summarize(SEARCH,True,propensity,stride=1)
        train=summarize(TRAIN_1024,True)
        train_weighted=summarize(TRAIN_1024,True,propensity)
    def cos(a,b):
        return ((a*b).sum(0)/np.maximum(np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0),1e-20)).tolist()
    result={'uniform_search_to_scored_gradient_cosine':cos(uniform['gradient'],scored['gradient']),
            'weighted_search_to_scored_gradient_cosine':cos(weighted['gradient'],scored['gradient']),
            'uniform_training_to_scored_gradient_cosine':cos(train['gradient'],scored['gradient']),
            'weighted_training_to_scored_gradient_cosine':cos(train_weighted['gradient'],scored['gradient']),
            'propensity':propensity,
            'search_gradient_scope':'All1273664 required search rows for uniform/weighted directions; all108139 officially scored rows for reference; replaces noisy stride40 diagnostic',
            'limitation':'Coarse weighting requires within-bin selection ignorability, which is unproven. Gradient alignment is a diagnostic and not a search score improvement.',
            'prior_negative':'Raw residual coarse propensity fitting failed in the previous campaign; a new objective-aware fit requires materially improved gradient alignment.'}
    write_json(path,result); oa.event('scoring_population_gradient_diagnosis',**result)
    print(result,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--population',action='store_true')
    if parser.parse_args().population:
        diagnose_population_alignment()
    else:
        main()
