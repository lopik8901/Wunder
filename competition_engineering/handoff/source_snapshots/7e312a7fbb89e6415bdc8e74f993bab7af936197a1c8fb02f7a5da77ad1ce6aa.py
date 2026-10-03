"""Sequence-bootstrap diagnostic of correction gradients and selection contrast.

This is supervisor-side aggregate diagnosis, not model fitting. No diagnostic
search gradient is ever exported as a deployable coefficient or training target.
"""
from __future__ import annotations
from competition_engineering import reassessment_campaign as campaign
import numpy as np
from scipy.special import expit
from competition_engineering.manual_search_core import TRAIN_1024, SEARCH, load_combo, predict_combo, SEARCH_ROOT_WP
from competition_engineering.pipeline import cached,write_json
from competition_engineering.residual import sufficient,from_stats
from tools.diagnose_causal_scoring_population import classifier_features

def primitives(f,y,base):
    y=np.clip(y,-2,2).astype(np.float64);p=np.clip(base,-2,2).astype(np.float64)
    w=np.abs(y); active=np.abs(base)<2
    stats=np.stack((w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)))
    cross=np.stack((f.T@(w*active),f.T@(w*active*y),f.T@(w*active*p)))
    return stats,cross

def gradient(stats,cross):
    mass,sy,sp,syy,spp,syp=np.moveaxis(stats,-2,0)
    my=sy/mass;mp=sp/mass;vy=syy-sy*my;vp=spp-sp*mp;cov=syp-sy*mp
    if np.any(mass<=0) or np.any(vy<=1e-12) or np.any(vp<=1e-12):
        raise ValueError('Degenerate objective population')
    f0,fy,fp=np.moveaxis(cross,-3,0)
    return (fy-my[...,None,:]*f0-(cov/vp)[...,None,:]*(fp-mp[...,None,:]*f0))/np.sqrt(vy*vp)[...,None,:]

def cosine(a,b):
    return (a*b).sum(-2)/np.maximum(np.linalg.norm(a,axis=-2)*np.linalg.norm(b,axis=-2),1e-30)

def bootstrap_counts(n,rng,draws=2000,size=None):
    size=n if size is None else size
    return np.stack([np.bincount(rng.integers(n,size=size),minlength=n) for _ in range(draws)])

def resample(stats,cross,rng,draws=2000,size=None,counts=None):
    n=len(stats);size=n if size is None else size
    counts=bootstrap_counts(n,rng,draws,size) if counts is None else counts
    draws=len(counts)
    s=(counts@stats.reshape(n,-1)).reshape(draws,*stats.shape[1:])
    f=(counts@cross.reshape(n,-1)).reshape(draws,*cross.shape[1:])
    return gradient(s,f)

def collect():
    destination=campaign.OUT/'gradient_primitives.npz'
    if destination.exists():
        return
    combo=load_combo();output={}
    classifier_path=campaign.ROOT/'competition_engineering/runs/objective_alignment_20261002/causal_population_classifier.npz'
    with np.load(classifier_path) as q:
        classifier={k:q[k] for k in q.files}
    for name,directory in [('training',TRAIN_1024),('search',SEARCH)]:
        records={k:[] for k in (['uniform'] if name=='training' else ['uniform','scored','unscored','weighted'])}
        crosses={k:[] for k in records};strata=[];row_counts=[]
        for index,(group,z) in enumerate(cached(directory)):
            idx=np.flatnonzero(z['need'])
            if name=='training':
                # Random fixed phase prevents repeatedly selecting one aliasing phase.
                phase=int(np.random.default_rng(20261002+group).integers(5));idx=idx[phase::5]
            base=predict_combo(z,combo)[idx]
            p=np.clip(base,-2,2);x=np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)
            f=np.column_stack((np.ones(len(idx)),p,x)).astype(np.float64)
            y=z['y'][idx]
            scored=z['mask'][idx]
            if name=='search':
                probability=expit(classifier_features(z['x'][idx],base,z['step'][idx],classifier['mean'],classifier['scale'])@classifier['coef'][index%4])
                weight=np.clip(probability/classifier['rates'][index%4],.2,5)
                focuses={'uniform':np.ones(len(idx)),'scored':scored.astype(float),'unscored':(~scored).astype(float),'weighted':weight}
                buckets=np.searchsorted([.01,.02,.04,.08,.16,.32,.64],probability)
                stepbin=np.searchsorted([6667,13333],z['step'][idx])
                cell=np.zeros((2,96,7,2))
                for target in range(2):
                    pred_bin=np.searchsorted([-1,0,1],p[:,target])
                    codes=buckets*12+stepbin*4+pred_bin
                    yy=np.clip(y[:,target],-2,2).astype(float);pp=p[:,target].astype(float);ww=np.abs(yy)
                    v=np.column_stack((ww,ww*yy,ww*pp,ww*yy*yy,ww*pp*pp,ww*yy*pp,np.ones(len(idx))))
                    for flag in (0,1):
                        chosen=scored==bool(flag)
                        for j in range(7):
                            np.add.at(cell[flag,:,j,target],codes[chosen],v[chosen,j])
                strata.append(cell)
            else:
                focuses={'uniform':np.ones(len(idx))}
            for key,focus in focuses.items():
                selected=focus>0
                s,c=primitives(f[selected],y[selected],base[selected])
                if not np.all(focus[selected]==1):
                    yy=np.clip(y[selected],-2,2).astype(float);pp=np.clip(base[selected],-2,2).astype(float)
                    w=np.abs(yy)*focus[selected,None];active=np.abs(base[selected])<2
                    s=np.stack((w.sum(0),(w*yy).sum(0),(w*pp).sum(0),(w*yy*yy).sum(0),(w*pp*pp).sum(0),(w*yy*pp).sum(0)))
                    c=np.stack((f[selected].T@(w*active),f[selected].T@(w*active*yy),f[selected].T@(w*active*pp)))
                records[key].append(s);crosses[key].append(c)
            row_counts.append(len(idx))
            if (index+1)%128==0:
                write_json(campaign.OUT/'progress.json',{'phase':name,'sequences':index+1})
                print(name,index+1,flush=True)
        for key in records:
            output[name+'_'+key+'_stats']=np.array(records[key]);output[name+'_'+key+'_cross']=np.array(crosses[key])
        output[name+'_row_counts']=np.array(row_counts)
        if strata:
            output['search_strata']=np.array(strata)
    np.savez(destination,**output)
    campaign.event('diagnostic_aggregates_frozen',sha256=campaign.sha(destination),classifier_sha256=campaign.sha(classifier_path))

def matched_contrast(cells,valid):
    totals=cells.sum(0) if cells.ndim==5 else cells
    n=totals[:,:,6,:]
    common=np.minimum(n[0],n[1])*valid
    moments=[]
    for flag in (0,1):
        factor=common/np.maximum(n[flag],1)
        moments.append((totals[flag,:,:6,:]*factor[:,None,:]).sum(0))
    stats=np.array(moments)
    mse=(stats[:,3]+stats[:,4]-2*stats[:,5])/stats[:,0]
    return {'wp':np.stack([from_stats(s) for s in stats]),'mse':mse,'mse_ratio':mse[1]/mse[0],
            'matched_rows_per_population':common.sum(0)}

def diagnose():
    if not (campaign.OUT/'research_state_map.json').exists():
        raise ValueError('Complete synthesis before experiments')
    path=campaign.OUT/'gradient_uncertainty_v2.json'
    if path.exists():
        return
    if not (campaign.OUT/'gradient_protocol.json').exists():
        protocol={'observation':'Low training/search gradient cosine used as a fit-launch rule without sequence uncertainty.',
            'competing_explanations':['Stable distribution/scoring conflict','High-dimensional gradient noise from only64 sequences','Residual model near-stationarity'],
            'hypothesis':'Point-gradient conflict overstates evidence if within-search gradients are unstable or conflict intervals include both signs.',
            'support':'Wide cosine intervals or negative split-half agreement; contrast exceeds same-size training subset variation.',
            'weaken':'Tight directional confidence and stable split-half conflict.',
            'stratification':'Fixed eight propensity bins, four target-specific prediction bins, three position bins; cross-fitted frozen mask classifier.',
            'selection_contrast_rule':'Residual MSE selected/unselected ratio99CI upper<0.85 after observable matching motivates deeper selection diagnosis, not automatic weighting.',
            'draws':2000,'seed':20261002,'training_sampling':'Fixed randomized stride5 phase per sequence; no score-driven phase choice',
            'decision':'No model is fit from these diagnostics; no new incumbent can be declared.'}
        write_json(campaign.OUT/'gradient_protocol.json',protocol);campaign.event('experiment_planned',id='gradient_uncertainty',protocol_sha256=campaign.sha(campaign.OUT/'gradient_protocol.json'))
    collect()
    with np.load(campaign.OUT/'gradient_primitives.npz') as q:
        data={k:q[k] for k in q.files}
    rng=np.random.default_rng(20261002);grads={};boots={};report={}
    search_counts=bootstrap_counts(len(data['search_scored_stats']),rng)
    for key in ('training_uniform','search_uniform','search_scored','search_unscored','search_weighted'):
        s=data[key+'_stats'];c=data[key+'_cross'];g=gradient(s.sum(0),c.sum(0));b=resample(s,c,rng,counts=search_counts if key.startswith('search_') else None)
        grads[key]=g;boots[key]=b
        noise=np.sqrt(((b-g)**2).sum(1).mean(0));norm=np.linalg.norm(g,axis=0)
        split=[]
        for _ in range(500):
            order=rng.permutation(len(s));a=order[:len(s)//2];z=order[len(s)//2:]
            split.append(cosine(gradient(s[a].sum(0),c[a].sum(0)),gradient(s[z].sum(0),c[z].sum(0))))
        report[key]={'wp':from_stats(s.sum(0)).tolist(),'sequences':len(s),'gradient_norm':norm.tolist(),
            'bootstrap_noise_norm':noise.tolist(),'norm_to_noise_ratio':(norm/np.maximum(noise,1e-30)).tolist(),
            'split_half_cosine_01_50_99':np.quantile(split,[.01,.5,.99],axis=0).tolist()}
    contrasts={}
    for a,b in [('training_uniform','search_scored'),('search_uniform','search_scored'),('search_weighted','search_scored')]:
        contrasts[a+'_vs_'+b]={'point_cosine':cosine(grads[a],grads[b]).tolist(),
            'paired_bootstrap_cosine99':np.quantile(cosine(boots[a],boots[b]),[.005,.995],axis=0).tolist(),
            'positive_direction_fraction':((boots[b]*grads[a]).sum(1)>0).mean(0).tolist()}
    s=data['training_uniform_stats'];c=data['training_uniform_cross'];small=resample(s,c,rng,size=64)
    reference=cosine(small,grads['training_uniform'])
    contrast=cosine(grads['search_scored'],grads['training_uniform'])
    cells=data['search_strata'];counts=cells.sum(0)[:,:,6,:];valid=(counts[0]>=100)&(counts[1]>=100)
    matched=matched_contrast(cells,valid);ratios=[];wps=[]
    for _ in range(2000):
        out=matched_contrast(cells[rng.integers(len(cells),size=len(cells))],valid)
        ratios.append(out['mse_ratio']);wps.append(out['wp'][1]-out['wp'][0])
    result={'population_gradients':report,'contrasts':contrasts,
        'same_size_training_subset_cosine_01_50_99':np.quantile(reference,[.01,.5,.99],axis=0).tolist(),
        'search_cosine_lower_tail_in_training_subset_distribution':(reference<contrast).mean(0).tolist(),
        'matched_selection_contrast':{k:v.tolist() for k,v in matched.items()},
        'matched_selection_mse_ratio99':np.quantile(ratios,[.005,.995],axis=0).tolist(),
        'matched_selection_wp_delta99':np.quantile(wps,[.005,.995],axis=0).tolist(),
        'training_rows':int(data['training_row_counts'].sum()),'search_required_rows':int(data['search_row_counts'].sum()),
        'interpretation_limits':['Sequence bootstrap is descriptive on adaptively reused search.','Observable matching tests only this representation; it does not prove full conditional target shift.',
                                 'Same-size training subset reference assumes sequences exchangeable; incumbent was already trained on training data.',
                                 'Gradient directions depend on feature scaling; these are not statistical guarantees or new candidate coefficients.'],
        'bootstrap_pairing':'Identical sequence counts across search populations; independent training/search resampling.',
        'supersedes':'gradient_uncertainty.json: search population cosine intervals used unpaired draws; preserved as implementation failure, not scientific evidence.',
        'protocol_sha256':campaign.sha(campaign.OUT/'gradient_protocol.json'),'primitives_sha256':campaign.sha(campaign.OUT/'gradient_primitives.npz')}
    write_json(path,result);campaign.event('experiment_completed',id='gradient_uncertainty',result_sha256=campaign.sha(path))
    print(__import__('json').dumps({k:v for k,v in result.items() if k not in ('population_gradients','interpretation_limits')}),flush=True)

if __name__=='__main__':
    diagnose()
