"""Cross-fitted causal scoring-indicator diagnostic; no search-target fitting.

Search targets are used only after the scoring-indicator classifier is frozen,
to diagnose whether its population weights align objective gradients.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from competition_engineering import objective_alignment as oa
from competition_engineering.manual_search_core import SEARCH, load_combo, predict_combo
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import from_stats


def classifier_features(x, base, steps, mean, scale):
    normalized=np.clip((x-mean)/scale,-8,8)
    p=np.clip(base,-2,2)
    return np.column_stack((np.ones(len(x)),normalized,p,np.abs(p),p*p,steps/20000.,steps>=13333)).astype(np.float64)


def classifier_loss(coef, x, scored, ridge):
    logits=x@coef
    regular=coef.copy(); regular[0]=0
    value=np.mean(np.logaddexp(0,logits)-scored*logits)+ridge*np.dot(regular,regular)/2
    gradient=x.T@(expit(logits)-scored)/len(x)+ridge*regular
    return value,gradient


def stats_and_gradient(stats, features):
    mass,sy,sp,syy,spp,syp=stats
    my=sy/mass;mp=sp/mass;vy=syy-sy*my;vp=spp-sp*mp;cov=syp-sy*mp
    gradient=(features[1]-my*features[0]-(cov/vp)*(features[2]-mp*features[0]))/np.sqrt(vy*vp)
    return gradient


def binary_log_loss(indicator, probability):
    indicator=np.asarray(indicator,dtype=np.float64)
    probability=np.clip(np.asarray(probability,dtype=np.float64),1e-12,1-1e-12)
    return float(np.mean(-indicator*np.log(probability)-(1-indicator)*np.log1p(-probability)))


def main():
    destination=oa.OUT/'causal_population_diagnosis.json'
    if destination.exists():
        return
    combo=load_combo()
    oa.event('hypothesis_planned',id='causal_scoring_population_diagnostic',
         question='Can current inputs predict scoring selection across held-out search sequences, and improve population-gradient alignment beyond failed coarse bins?',
         classifier_target='Scoring indicator only; never target0/target1',
         features='Current112 inputs, frozen predictions, prediction magnitudes/squares, current step',
         rule='Allow a weighted fit only with cross-fitted log-loss improvement, AUC>=0.75 and target-gradient cosine>=0.5 improving by>=0.25',
         caveat='Diagnosing selection ignorability on observed features; no target conditional-distribution guarantee')
    feature_rows=[];labels=[];folds=[]
    with threadpool_limits(limits=1):
        for index,(group,z) in enumerate(cached(SEARCH)):
            eligible=np.flatnonzero(z['need'])
            idx=np.random.default_rng(20261002+group).choice(eligible,min(1000,len(eligible)),replace=False)
            base=predict_combo(z,combo)[idx]
            feature_rows.append(classifier_features(z['x'][idx],base,z['step'][idx],combo['mean'],combo['scale']))
            labels.append(z['mask'][idx].astype(np.float64));folds.append(np.full(len(idx),index%4))
        x=np.concatenate(feature_rows);scored=np.concatenate(labels);fold=np.concatenate(folds)
        coefficients=[];rates=[];optimizer=[]
        for k in range(4):
            active=fold!=k; rate=float(scored[active].mean()); initial=np.zeros(x.shape[1]);initial[0]=np.log(rate/(1-rate))
            result=minimize(classifier_loss,initial,args=(x[active],scored[active],.1),jac=True,
                            method='L-BFGS-B',options={'maxiter':150,'ftol':1e-10})
            coefficients.append(result.x);rates.append(rate)
            optimizer.append({'success':bool(result.success),'iterations':int(result.nit),'message':str(result.message)})
        # Classifier fitting above does not use any scoring-target values.
        classifier_path=oa.OUT/'causal_population_classifier.npz'
        if classifier_path.exists():
            with np.load(classifier_path) as frozen:
                np.testing.assert_array_equal(frozen['coef'],np.array(coefficients))
                np.testing.assert_array_equal(frozen['rates'],rates)
        else:
            np.savez(classifier_path,coef=np.array(coefficients),rates=rates,
                     mean=combo['mean'],scale=combo['scale'])
        totals={k:np.zeros((6,2)) for k in ['uniform','weighted','scored']}
        feature_moments={k:np.zeros((3,115,2)) for k in totals}
        probabilities=[];indicators=[];prior_probabilities=[]
        for index,(_,z) in enumerate(cached(SEARCH)):
            idx=np.flatnonzero(z['need']);base=predict_combo(z,combo)[idx]
            clf=classifier_features(z['x'][idx],base,z['step'][idx],combo['mean'],combo['scale'])
            probability=expit(clf@coefficients[index%4]);prior=rates[index%4]
            probabilities.append(probability);indicators.append(z['mask'][idx]);prior_probabilities.append(np.full(len(idx),prior))
            p=np.clip(base,-2,2).astype(np.float64);y=np.clip(z['y'][idx],-2,2).astype(np.float64)
            f=np.column_stack((np.ones(len(idx)),p,np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)))
            for name,focus in [('uniform',np.ones(len(idx))),('weighted',np.clip(probability/prior,.2,5)),('scored',z['mask'][idx].astype(np.float64))]:
                w=np.abs(y)*focus[:,None];active=np.abs(base)<2
                totals[name]+=np.stack((w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)))
                feature_moments[name]+=np.stack((f.T@(w*active),f.T@(w*active*y),f.T@(w*active*p)))
        grad={k:stats_and_gradient(totals[k],feature_moments[k]) for k in totals}
        def cosine(a,b):
            return ((a*b).sum(0)/np.maximum(np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0),1e-20))
        uniform=cosine(grad['uniform'],grad['scored']);weighted=cosine(grad['weighted'],grad['scored'])
        probability=np.concatenate(probabilities);indicator=np.concatenate(indicators);prior=np.concatenate(prior_probabilities)
        auc=float(roc_auc_score(indicator,probability)); ll=binary_log_loss(indicator,probability); prior_ll=binary_log_loss(indicator,prior)
        eligible=[int(t) for t in range(2) if weighted[t]>=.5 and weighted[t]-uniform[t]>=.25 and auc>=.75 and ll<prior_ll]
    report={'cross_fitted_auc':auc,'cross_fitted_log_loss':ll,'prior_log_loss':prior_ll,
            'uniform_to_scored_gradient_cosine':uniform.tolist(),'weighted_to_scored_gradient_cosine':weighted.tolist(),
            'optimizer':optimizer,'eligible_targets_for_weighted_fit':eligible,'classifier_fit_rows':len(x),
            'diagnostic_rows':len(indicator),'source_sha256':oa.digest(Path(__file__)),
            'classifier_sha256':oa.digest(oa.OUT/'causal_population_classifier.npz'),
            'decision':'Proceed only for eligible targets; otherwise abandon this weighting hypothesis before fit.',
            'caveat':'Classifiers are fitted only to scoring indicators, with sequence cross-fitting. Target gradients are adaptive search diagnostics, not independent validation.'}
    write_json(destination,report);oa.event('causal_scoring_population_diagnostic_completed',**report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
