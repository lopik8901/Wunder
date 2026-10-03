"""Test whether the observed position pattern transports before fitting a gate."""
from competition_engineering import representation_campaign as campaign
import json
import numpy as np
from competition_engineering.representation_evaluation import load_frozen,full_features,bootstrap_stats
from competition_engineering.replicated_readout import readout_design
from competition_engineering.manual_search_core import SEARCH,load_combo,predict_combo
from competition_engineering.pipeline import cached,write_json
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.search_population_audit import pooled_correlations

def position_contrast(moments):
    """Middle correction delta minus outer-row pooled correction delta."""
    middle=pooled_correlations(moments[...,1,:,:])
    outer=pooled_correlations(moments[...,[0,2],:,:].sum(-3))
    # The candidate axis immediately precedes population and target axes.
    return (middle[...,1,:,:]-middle[...,0,:,:])-(outer[...,1,:,:]-outer[...,0,:,:])

def run():
    destination=campaign.OUT/'position_diagnosis.json'
    if destination.exists():
        print(destination.read_text());return
    protocol={'observation':'Frozen-state fixed025 t1 search gains are positive early/late but negative in the middle; earlier current-row corrections showed a similar pattern.',
        'competing_explanations':['Position-dependent model utility','Official-population effect absent from training','Sequence noise in a post-inspection slice'],
        'experiment':'No fitting. Fixed thirds and strength0.25. Cross-fitted training A evaluated with B coefficients and vice versa; frozen pooled coefficients on replication and official search.',
        'contrast':'Middle t1 correction WP delta minus pooled outer-rows correction WP delta, with paired sequence99 intervals.',
        'position_fit_rule':'Proceed only if training-crossfit proxy contrast99 upper<0, replication proxy point<0, and official-search contrast99 upper<0. Otherwise position gating lacks replicated mechanistic support.',
        'limitations':'This is an adaptive diagnostic, not fresh official evidence. Correlation contrasts condition on the frozen models.'}
    path=campaign.OUT/'position_protocol.json';write_json(path,protocol)
    campaign.event('diagnostic_planned',id='position_transport',protocol_sha256=campaign.sha(path))
    models,_,_=load_frozen();records={};combo=load_combo();extractor=LatentExtractor()
    for split in ['training_crossfit','replication','search']:
        blocks=[]
        if split=='search':
            iterator=cached(SEARCH)
        else:
            directory=campaign.TRAIN if split=='training_crossfit' else campaign.REPLICATION
            iterator=cached(directory)
        for _,z in iterator:
            if split=='training_crossfit' and int(z['role'])==2:
                continue
            if split=='search':
                base=predict_combo(z,combo);f=full_features(z,base,models,extractor,combo)
                idx=np.flatnonzero(z['need']);base=base[idx];f=f[idx];y=z['y'][idx];steps=z['step'][idx]
                focuses=[np.ones(len(idx)),z['mask'][idx].astype(float)];coef=models['latent_pooled']
            else:
                base=z['base'];f=readout_design(z['current'],z['hidden'],models['mean'],models['scale'])
                y=z['y'];steps=z['indices'];focuses=[np.ones(len(y)),z['focus']]
                coef=models['latent_'+('b' if int(z['role'])==0 else 'a')] if split=='training_crossfit' else models['latent_pooled']
            p=base+.25*(f@coef);code=np.searchsorted([6667,13333],steps)
            block=[]
            for prediction in [base,p]:
                block.append([[focus_stats(y,prediction,focus*(code==position)) for position in range(3)] for focus in focuses])
            blocks.append(block)
        records[split]=np.array(blocks)
        print('POSITION_SPLIT='+split,flush=True)
    np.savez(campaign.OUT/'position_moments.npz',**records)
    report={}
    for split,moments in records.items():
        point=position_contrast(moments.sum(0))
        counts=bootstrap_counts(len(moments),np.random.default_rng(20261016),draws=10000)
        draws=position_contrast(bootstrap_stats(moments,counts))
        wp=pooled_correlations(moments.sum(0));d=wp[1]-wp[0]
        report[split]={'position_delta_per_target_by_population':d.tolist(),
            'middle_minus_outer_delta_per_target_by_population':point.tolist(),
            'paired99_middle_minus_outer':np.quantile(draws,[.005,.995],axis=0).tolist(),
            'population_order':['uniform','official' if split=='search' else 'probability'],'sequences':len(moments)}
    eligible=(report['training_crossfit']['paired99_middle_minus_outer'][1][1][1]<0
        and report['replication']['middle_minus_outer_delta_per_target_by_population'][1][1]<0
        and report['search']['paired99_middle_minus_outer'][1][1][1]<0)
    result={'splits':report,'position_fit_justified':eligible,
        'decision':'Fit the predeclared position mechanism only if the replicated rule passes; otherwise do not choose a gate from the official-search slice.',
        'protocol_sha256':campaign.sha(path),'moments_sha256':campaign.sha(campaign.OUT/'position_moments.npz')}
    write_json(destination,result);campaign.event('diagnostic_completed',id='position_transport',eligible=eligible,result_sha256=campaign.sha(destination))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    run()
