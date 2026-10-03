"""Localize the frozen t1 population contrast without fitting a new model."""
from competition_engineering import reassessment_campaign as campaign
import json
import numpy as np
from competition_engineering.manual_search_core import SEARCH,load_combo,predict_combo
from competition_engineering.prospective_selection_diagnostic import shortlist,focus_stats
from competition_engineering.pipeline import cached,write_json
from competition_engineering.search_population_audit import pooled_correlations

def within_correlation(stats):
    w,sy,sp,syy,spp,syp=np.moveaxis(stats,-2,0)
    within_cov=(syp-sy*sp/w).sum(0)
    within_y=(syy-sy*sy/w).sum(0);within_p=(spp-sp*sp/w).sum(0)
    return within_cov/np.sqrt(np.maximum(within_y*within_p,1e-30))

def run():
    path=campaign.OUT/'population_mechanism.json'
    if path.exists():
        return
    protocol={'observation':'Frozen t1 corrections have negative official-minus-uniform WP delta with paired99 interval excluding zero.',
        'hypotheses':['Prediction clipping changes correction efficacy','Between-sequence shifts dominate pooled result','A few influential sequences cause conflict','Conflict localized to position'],
        'models':['incumbent','raw_t1_1024','raw_t1_4096'],
        'tests':'Fixed clipped/unclipped-prediction contrast, pooled/within moments, leave-one-out, delete five largest absolute contrast influences, three fixed position thirds.',
        'decision':'Diagnostics may explain the conflict but cannot choose a correction strength, train a gating function or replace official metric.'}
    config=campaign.OUT/'population_mechanism_protocol.json'
    write_json(config,protocol);campaign.event('experiment_planned',id='population_mechanism',protocol_sha256=campaign.sha(config))
    combo=load_combo();models=[m for m in shortlist() if m[0] in protocol['models']];blocks=[];positions=[]
    for _,z in cached(SEARCH):
        idx=np.flatnonzero(z['need']);base=predict_combo(z,combo)[idx];y=np.clip(z['y'][idx],-2,2).astype(float)
        f=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)))
        masks=[np.ones(len(idx),bool),z['mask'][idx]];step=z['step'][idx];row=[];posrow=[]
        for name,coef,target,score in models:
            pred=base.copy();pred[:,target]+=f@coef;types=[];pos=[]
            for clipped in (True,False):
                p=np.clip(pred,-2,2).astype(float) if clipped else pred.astype(float)
                moments=[]
                for mask in masks:
                    yy=y[mask];pp=p[mask];w=np.abs(yy)
                    moments.append(np.stack((w.sum(0),(w*yy).sum(0),(w*pp).sum(0),(w*yy*yy).sum(0),(w*pp*pp).sum(0),(w*yy*pp).sum(0))))
                types.append(moments)
            for mask in masks:
                pos.append([focus_stats(y,pred,mask & (step>=lo)&(step<hi)) for lo,hi in [(99,6667),(6667,13333),(13333,20000)]])
            row.append(types);posrow.append(pos)
        blocks.append(row);positions.append(posrow)
    blocks=np.array(blocks);positions=np.array(positions)
    np.savez(campaign.OUT/'population_mechanism_moments.npz',blocks=blocks,positions=positions)
    sums=blocks.sum(0);scores=pooled_correlations(sums).mean(-1)
    results={}
    for k,model in enumerate(models):
        clipped_delta=scores[k,0]-scores[0,0];unclipped_delta=scores[k,1]-scores[0,1]
        within=within_correlation(blocks[:,k,0]).mean(-1)-within_correlation(blocks[:,0,0]).mean(-1)
        leave=[]
        for i in range(len(blocks)):
            reduced=pooled_correlations(sums-blocks[i]).mean(-1)
            d=reduced[k,0]-reduced[0,0];leave.append(d[1]-d[0])
        contrast=clipped_delta[1]-clipped_delta[0];influence=contrast-np.array(leave)
        worst=np.argsort(-np.abs(influence))[:5]
        reduced=pooled_correlations(sums-blocks[worst].sum(0)).mean(-1)
        d=reduced[k,0]-reduced[0,0]
        pos_scores=pooled_correlations(positions.sum(0)).mean(-1)
        results[model[0]]={'clipped_combined_delta_uniform_official':clipped_delta.tolist(),
            'unclipped_prediction_combined_delta_uniform_official':unclipped_delta.tolist(),
            'within_sequence_correlation_combined_delta_uniform_official':within.tolist(),
            'leave_one_out_population_contrast_range':[float(min(leave)),float(max(leave))],
            'delete_five_largest_absolute_influences_population_contrast':float(d[1]-d[0]),
            'fixed_position_combined_delta_uniform_official':(pos_scores[k]-pos_scores[0]).tolist()}
    result={'models':results,'interpretation':'Unclipped and within-sequence values are diagnostic counterfactual metrics, never candidate selection scores.',
        'protocol_sha256':campaign.sha(config),'moments_sha256':campaign.sha(campaign.OUT/'population_mechanism_moments.npz')}
    write_json(path,result);campaign.event('experiment_completed',id='population_mechanism',result_sha256=campaign.sha(path))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    run()
