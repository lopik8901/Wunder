"""Record primary domain-adaptation assumptions; no task impossibility claim."""
import shutil
from competition_engineering import reassessment_campaign as campaign
from competition_engineering.pipeline import write_json
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library

def main():
    parent=campaign.ROOT/'competition_engineering/research_cards/v7';destination=parent.with_name('v8')
    if destination.exists():
        raise ValueError('Immutable library version exists')
    shutil.copytree(parent,destination)
    card={'schema_version':1,'card_id':'domain_adaptation_identifiability','version':1,'card_type':'paper',
        'family':'distribution_shift','tags':['selection','transport','assumptions','generalization'],
        'title':'Impossibility Theorems for Domain Adaptation','authors':['Shai Ben-David','Tyler Lu','Teresa Luu','David Pal'],
        'year':2010,'source_url':'https://proceedings.mlr.press/v9/david10a.html',
        'source_locator':'Primary abstract and introduction; weighting discussion in PDF',
        'curator':'Codex, primary-source review','verified_on':'2026-10-02','claim_confidence':'moderate',
        'mechanism':'Classification counterexamples show that input-distribution reweighting alone need not identify successful adaptation.',
        'useful_when':'Curator inference: mask prediction improves without reliable target-objective transport.',
        'diagnostic_signatures':'Curator inference: corrected-population model ranking disagrees with actual selected-population ranking.',
        'assumptions':'Results concern agnostic classification with labeled source and unlabeled target data.',
        'limitations':'No theorem establishes impossibility for this labeled-search regression task or clipped correlation.',
        'causality':'Curator inference: target transport requires a stable relation usable from causal observable inputs.',
        'incremental_inference':'No proposed inference architecture.','training_compute':'Theory; no implementation imported.',
        'cpu_deployment':'No incremental cost.','dependencies':'None.','licensing':'Metadata and original curator paraphrase.',
        'concise_relevance':'Separates high mask-classifier AUC from justified learning weights.'}
    write_json(destination/(card['card_id']+'.json'),validate_card(card))
    cards,manifest=load_library(destination)
    write_json(campaign.OUT/'library_v8_manifest.json',{'parent_manifest_sha256':load_library(parent)[1],
        'manifest_sha256':manifest,'count':len(cards),'added':[card['card_id']],
        'files':{p.name:campaign.sha(p) for p in sorted(destination.glob('*.json'))}})
    campaign.event('library_expanded',version='v8',manifest_sha256=manifest,count=len(cards))
    print(manifest)

if __name__=='__main__':
    main()
