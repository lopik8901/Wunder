"""Fixed nonlinear mask-predictor diagnostic, trained on mask indicators only."""
from competition_engineering import reassessment_campaign as campaign
import json
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from competition_engineering.manual_search_core import SEARCH,load_combo,predict_combo
from competition_engineering.pipeline import cached,write_json
from competition_engineering.gradient_uncertainty import primitives,gradient,bootstrap_counts,resample,cosine
from tools.diagnose_causal_scoring_population import classifier_features,binary_log_loss

def weighted_primitives(f,y,base,focus):
    yy=np.clip(y,-2,2).astype(float);pp=np.clip(base,-2,2).astype(float)
    w=np.abs(yy)*focus[:,None];active=np.abs(base)<2
    s=np.stack((w.sum(0),(w*yy).sum(0),(w*pp).sum(0),(w*yy*yy).sum(0),(w*pp*pp).sum(0),(w*yy*pp).sum(0)))
    c=np.stack((f.T@(w*active),f.T@(w*active*yy),f.T@(w*active*pp)))
    return s,c

def run():
    path=campaign.OUT/'nonlinear_selection_result.json'
    if path.exists():
        return
    config=campaign.OUT/'nonlinear_selection_protocol.json'
    if not config.exists():
        write_json(config,{'observation':'t1 correction efficacy differs between official and uniform populations, surviving clipping/centering/influence checks; linear causal proxy does not reproduce it.',
            'hypothesis':'Nonlinear current-input mask structure improves population transport beyond the frozen linear scoring predictor.',
            'model':{'family':'HistGradientBoostingClassifier','max_iter':64,'max_leaf_nodes':15,'learning_rate':.1,'min_samples_leaf':40,'l2_regularization':1.,'early_stopping':False},
            'training_target':'Scoring indicator only; t0/t1 never classifier targets or features.',
            'cross_fit':'Four fixed sequence folds, exactly1000 random required rows per sequence, same sample as prior classifier.',
            'fit_launch_rule':'Log loss improvement>=0.02 over frozen linear predictor; for at least one target, paired99 lower of cosine improvement>0 and weighted-vs-official cosine99 lower>0. No launch from higher AUC alone.',
            'scope':'Diagnostic mask learning on fixed search only; no new practical incumbent or adaptive uncertainty guarantee.'})
        campaign.event('experiment_planned',id='nonlinear_selection',protocol_sha256=campaign.sha(config))
    combo=load_combo();features=[];labels=[];folds=[]
    for index,(group,z) in enumerate(cached(SEARCH)):
        required=np.flatnonzero(z['need']);idx=np.random.default_rng(20261002+group).choice(required,min(1000,len(required)),replace=False)
        base=predict_combo(z,combo)[idx]
        features.append(classifier_features(z['x'][idx],base,z['step'][idx],combo['mean'],combo['scale'])[:,1:])
        labels.append(z['mask'][idx]);folds.append(np.full(len(idx),index%4))
    features=np.concatenate(features);labels=np.concatenate(labels);folds=np.concatenate(folds)
    models=[];training=[];frozen={};started=time.perf_counter()
    for k in range(4):
        m=HistGradientBoostingClassifier(max_iter=64,max_leaf_nodes=15,learning_rate=.1,min_samples_leaf=40,l2_regularization=1.,early_stopping=False,random_state=20261007)
        m.fit(features[folds!=k],labels[folds!=k]);models.append(m)
        training.append({'fold':k,'sequences':48,'rows':int((folds!=k).sum()),'iterations':int(m.n_iter_)})
        # Freeze native node arrays for forensic reproduction, not callback use.
        for j,tree in enumerate(m._predictors):
            frozen[f'fold{k}_tree{j}']=tree[0].nodes
        frozen[f'fold{k}_baseline']=m._baseline_prediction
    np.savez(campaign.OUT/'nonlinear_mask_trees.npz',**frozen)
    campaign.event('mask_predictor_frozen_before_target_diagnostics',sha256=campaign.sha(campaign.OUT/'nonlinear_mask_trees.npz'),training_seconds=time.perf_counter()-started)
    with np.load(campaign.ROOT/'competition_engineering/runs/objective_alignment_20261002/causal_population_classifier.npz') as q:
        linear={k:q[k] for k in q.files}
    stats={k:[] for k in ['linear','nonlinear','official']};cross={k:[] for k in stats}
    probs=[];linear_probs=[];indicators=[]
    for index,(_,z) in enumerate(cached(SEARCH)):
        idx=np.flatnonzero(z['need']);base=predict_combo(z,combo)[idx]
        x=classifier_features(z['x'][idx],base,z['step'][idx],combo['mean'],combo['scale'])
        probability=models[index%4].predict_proba(x[:,1:])[:,1]
        lp=expit(x@linear['coef'][index%4]);prior=linear['rates'][index%4]
        probs.append(probability);linear_probs.append(lp);indicators.append(z['mask'][idx])
        f=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),x[:,1:113]))
        for name,focus in [('linear',np.clip(lp/prior,.2,5)),('nonlinear',np.clip(probability/prior,.2,5)),('official',z['mask'][idx].astype(float))]:
            s,c=weighted_primitives(f,z['y'][idx],base,focus)
            stats[name].append(s);cross[name].append(c)
    stats={k:np.array(v) for k,v in stats.items()};cross={k:np.array(v) for k,v in cross.items()}
    np.savez(campaign.OUT/'nonlinear_selection_moments.npz',**{**{k+'_stats':v for k,v in stats.items()},**{k+'_cross':v for k,v in cross.items()}})
    counts=bootstrap_counts(64,np.random.default_rng(20261007),draws=10000)
    boots={k:resample(stats[k],cross[k],None,counts=counts) for k in stats}
    points={k:gradient(stats[k].sum(0),cross[k].sum(0)) for k in stats}
    linear_cos=cosine(boots['linear'],boots['official']);nonlinear_cos=cosine(boots['nonlinear'],boots['official'])
    improvement=nonlinear_cos-linear_cos
    probability=np.concatenate(probs);lp=np.concatenate(linear_probs);indicator=np.concatenate(indicators)
    ll=binary_log_loss(indicator,probability);linear_ll=binary_log_loss(indicator,lp)
    lower=np.quantile(nonlinear_cos,.005,axis=0);improved_lower=np.quantile(improvement,.005,axis=0)
    eligible=[t for t in range(2) if linear_ll-ll>=.02 and lower[t]>0 and improved_lower[t]>0]
    result={'cross_fitted_auc':float(roc_auc_score(indicator,probability)),'cross_fitted_log_loss':ll,'linear_log_loss':linear_ll,
        'point_cosine_linear':cosine(points['linear'],points['official']).tolist(),
        'point_cosine_nonlinear':cosine(points['nonlinear'],points['official']).tolist(),
        'nonlinear_cosine99':np.quantile(nonlinear_cos,[.005,.995],axis=0).tolist(),
        'cosine_improvement99':np.quantile(improvement,[.005,.995],axis=0).tolist(),
        'eligible_targets':eligible,'training':training,'classifier_sha256':campaign.sha(campaign.OUT/'nonlinear_mask_trees.npz'),
        'protocol_sha256':campaign.sha(config),'decision':'Fit only if prospective transport rule passes; otherwise abandon nonlinear propensity before correction fitting.',
        'limitation':'Search is adaptively reused; passing diagnostic rule would motivate an experiment, not establish improvement.'}
    write_json(path,result);campaign.event('experiment_completed',id='nonlinear_selection',result_sha256=campaign.sha(path),eligible_targets=eligible)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    run()
