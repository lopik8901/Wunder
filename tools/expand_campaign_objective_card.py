"""Append a source-grounded functional-gradient card in a new library version."""
import shutil

from competition_engineering.autonomous_campaign import CAMPAIGN, ROOT, event, sha
from competition_engineering.pipeline import write_json
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library


def main():
    parent=ROOT/'competition_engineering/research_cards/v2'
    destination=ROOT/'competition_engineering/research_cards/v3'
    if destination.exists():
        raise ValueError('Version already exists')
    shutil.copytree(parent,destination)
    card={
        'schema_version':1,'card_id':'functional_gradient_fitting','version':1,
        'card_type':'paper','family':'objective','tags':['objective','gradient','correlation','clipping','residual'],
        'title':'Greedy function approximation: A gradient boosting machine.',
        'authors':['Jerome H. Friedman'],'year':2001,
        'source_url':'https://doi.org/10.1214/aos/1013203451','source_locator':'Publisher abstract, Annals of Statistics 29(5), 1189-1232',
        'curator':'Codex, primary publisher abstract review','verified_on':'2026-10-01','claim_confidence':'limited',
        'mechanism':'Stagewise additive function estimation is connected to steepest descent in function space and generalized beyond least-squares fitting.',
        'useful_when':'Curator inference: a residual fit can be misaligned with the actual evaluation objective, even when it reduces squared error.',
        'diagnostic_signatures':'Curator inference: clipped predictions and non-unit covariance-to-prediction-variance slopes motivate checking the objective gradient.',
        'assumptions':'Curator adaptation: estimate all objective moments from training rows, and test the fitted direction on fixed search evidence.',
        'limitations':'The paper does not validate this task, clipped weighted Pearson, or the proposed single ridge gradient step. The pooled objective is nonseparable and its derivative must be independently checked.',
        'causality':'Curator adaptation: objective gradients may use training targets, but the deployed correction receives only causal input state.',
        'incremental_inference':'Curator inference: one fitted linear readout adds no target-dependent online state.',
        'training_compute':'Proposed adaptation requires one training pass for moments and another for a bounded ridge fit.',
        'cpu_deployment':'Curator inference: changing the fitting target need not increase callback cost; benchmark the exported model.',
        'dependencies':'Proposed adaptation uses NumPy; no paper implementation is copied.',
        'licensing':'Original curator paraphrase and bibliographic metadata only.',
        'concise_relevance':'Tests an objective mismatch diagnosed before retrieval, rather than assuming a larger architecture is needed.',
    }
    write_json(destination/(card['card_id']+'.json'),validate_card(card))
    cards,manifest=load_library(destination)
    write_json(CAMPAIGN/'library_manifest_v3.json',{
        'version':'v3','parent_version':'v2','parent_manifest_sha256':load_library(parent)[1],
        'manifest_sha256':manifest,'count':len(cards),'added':[card['card_id']],
        'files':{q.name:sha(q) for q in sorted(destination.glob('*.json'))},
        'diagnosis_source':'objective_diagnostic.json; training/search clipping geometry recorded before literature retrieval',
    })
    event('library_expanded',version='v3',manifest_sha256=manifest,cards=len(cards),added=card['card_id'])
    print(manifest)


if __name__=='__main__':
    main()
