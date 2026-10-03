"""Versioned primary-source context for population-weighting assumptions."""
import shutil
import json
from competition_engineering import root_cause_campaign as campaign
from competition_engineering.pipeline import write_json
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library

def publish_revision(parent,destination,card):
    """Validate a complete staged version before making it the final directory."""
    if destination.exists():
        raise ValueError('Immutable library version exists')
    card=validate_card(card)
    parents,_=load_library(parent)
    for previous in parents:
        if previous['card_id']==card['card_id']:
            if card['version']<=previous['version']:
                raise ValueError('Card revision must increase its version')
        elif previous['source_url']==card['source_url']:
            raise ValueError('Duplicate source: revise existing card')
    staging=destination.with_name(destination.name+'.staging')
    shutil.copytree(parent,staging)
    write_json(staging/(card['card_id']+'.json'),card)
    cards,manifest=load_library(staging)
    staging.rename(destination)
    return cards,manifest

def main():
    parent=campaign.ROOT/'competition_engineering/research_cards/v8';destination=parent.with_name('v9')
    if destination.exists():
        raise ValueError('Immutable library version exists')
    card={'schema_version':1,'card_id':'importance_weighted_model_selection','version':2,'card_type':'paper',
        'family':'distribution_shift','tags':['selection','weights','covariate-shift','assumptions'],
        'title':'Covariate Shift Adaptation by Importance Weighted Cross Validation',
        'authors':['Masashi Sugiyama','Matthias Krauledat','Klaus-Robert Mueller'],
        'year':2007,'source_url':'https://www.jmlr.org/papers/v8/sugiyama07a.html',
        'source_locator':'Primary publisher abstract, JMLR8(35):985-1005',
        'curator':'Codex, primary-source review','verified_on':'2026-10-02','claim_confidence':'moderate',
        'mechanism':'Importance weighting adjusts model-selection risk when input distributions differ and the conditional output law remains unchanged.',
        'useful_when':'Curator inference: training-only selection uses a population proxy whose correction rankings disagree with the actual selected population.',
        'diagnostic_signatures':'Curator inference: check exact population moments and fixed-candidate efficacy, not classifier AUC alone.',
        'assumptions':'Covariate shift preserves the conditional distribution of outputs given inputs; usable importance ratios are required.',
        'limitations':'The paper does not establish unbiasedness for a finite-sample ratio statistic such as clipped weighted Pearson, learned proxy probabilities, or adaptively reused data.',
        'causality':'Curator inference: a scoring proxy can supply offline weights; deployed predictions still use only present and past observations.',
        'incremental_inference':'No extra online computation if weights are used only during training.',
        'training_compute':'Curator adaptation: matched fixed-penalty readouts with training-side WP selection.',
        'cpu_deployment':'Existing current-row readout; benchmark actual callback.',
        'dependencies':'NumPy and SciPy for the proposed adaptation; no author code imported.',
        'licensing':'Original curator paraphrase and bibliographic metadata.',
        'concise_relevance':'Supports diagnosing the weighting assumption and selection population before further fitting.'}
    cards,manifest=publish_revision(parent,destination,card)
    write_json(campaign.OUT/'library_v9_manifest.json',{'parent_manifest_sha256':load_library(parent)[1],
        'manifest_sha256':manifest,'count':len(cards),'added':[],'revised':[card['card_id']],
        'files':{p.name:campaign.sha(p) for p in sorted(destination.glob('*.json'))}})
    campaign.event('library_expanded',version='v9',manifest_sha256=manifest,count=len(cards))
    print(manifest)

if __name__=='__main__':
    main()
