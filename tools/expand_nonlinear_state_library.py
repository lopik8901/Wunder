"""Record primary random-feature inspiration and bounded task adaptation."""
from competition_engineering import nonlinear_state_campaign as campaign
from competition_engineering.pipeline import write_json
from tools.expand_root_cause_library import publish_revision
from connectome.research_retrieval import load_library

def main():
    parent=campaign.ROOT/'competition_engineering/research_cards/v11';destination=parent.with_name('v12')
    card={'schema_version':1,'card_id':'random_shallow_features','version':1,'card_type':'paper','family':'nonlinear_readout',
        'tags':['random','nonlinear','readout','representation'],
        'title':'Weighted Sums of Random Kitchen Sinks: Replacing minimization with randomization in learning',
        'authors':['Ali Rahimi','Benjamin Recht'],'year':2008,
        'source_url':'https://papers.nips.cc/paper_files/paper/2008/hash/0efe32849d230d7f53049ddc4a4b0c60-Abstract.html',
        'source_locator':'Primary proceedings page and author-hosted paper abstract',
        'curator':'Codex, primary-source review','verified_on':'2026-10-03','claim_confidence':'limited',
        'mechanism':'Fit a weighted sum of randomized nonlinear features; the paper analyzes approximation and learning in shallow random networks.',
        'useful_when':'Curator inference: test nonlinear readout utility while avoiding nonconvex feature-learning optimization as a confound.',
        'diagnostic_signatures':'Curator inference: joint frozen-state features should beat matched linear, current-only, and separable feature controls on disjoint sequence groups.',
        'assumptions':'The randomized feature distribution and finite basis must represent useful functions for the task.',
        'limitations':'No guarantee for128 tanh features, clipped weighted correlation, population transfer or a shared pretrained representation. A failed finite basis does not rule out learned nonlinear readouts.',
        'causality':'Curator adaptation: current observations and existing forward GRU states only; frozen recurrent parameters.',
        'incremental_inference':'One fixed matrix product, tanh and a linear readout.',
        'training_compute':'Fixed-penalty weighted least squares; select correction strength by pooled training-derived WP.',
        'cpu_deployment':'Fuse with the existing ONNX callback; measure full callback before eligibility.',
        'dependencies':'NumPy for fitting and ONNX Runtime for deployment; no author code imported.',
        'licensing':'Original curator paraphrase and bibliographic metadata.',
        'concise_relevance':'A controlled finite nonlinear-function-class test over frozen causal state.'}
    cards,manifest=publish_revision(parent,destination,card)
    write_json(campaign.OUT/'library_v12_manifest.json',{'manifest_sha256':manifest,'parent_manifest_sha256':load_library(parent)[1],
        'count':len(cards),'added':[card['card_id']],'files':{p.name:campaign.sha(p) for p in destination.glob('*.json')}})
    campaign.event('library_expanded',version=12,manifest_sha256=manifest,count=len(cards));print(manifest)

if __name__=='__main__': main()
