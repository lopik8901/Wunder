"""Add primary linear-probe motivation without importing a task guarantee."""
from competition_engineering import representation_campaign as campaign
from competition_engineering.pipeline import write_json
from tools.expand_root_cause_library import publish_revision
from connectome.research_retrieval import load_library

def main():
    campaign.initialize()
    parent=campaign.ROOT/'competition_engineering/research_cards/v9';destination=parent.with_name('v10')
    card={'schema_version':1,'card_id':'linear_representation_probes','version':1,'card_type':'paper',
        'family':'frozen_representation','tags':['representation','probe','hidden','linear','diagnostic'],
        'title':'Understanding intermediate layers using linear classifier probes',
        'authors':['Guillaume Alain','Yoshua Bengio'],'year':2016,
        'source_url':'https://arxiv.org/abs/1610.01644','source_locator':'Primary abstract, version4,22November2018',
        'curator':'Codex, primary-source review','verified_on':'2026-10-02','claim_confidence':'limited',
        'mechanism':'Independently trained linear classifiers probe information available in intermediate representations without changing the underlying network.',
        'useful_when':'Curator inference: a frozen predictor compresses learned states into few outputs and current-input corrections may miss retained information.',
        'diagnostic_signatures':'Curator inference: compare a fixed-state readout with a matched current-feature control on separate sequence groups.',
        'assumptions':'The paper studies classification representations in neural networks; probe quality is conditional on the representation and task.',
        'limitations':'It does not prove that a regression probe improves clipped weighted correlation or that training gains transport across selected populations.',
        'causality':'Curator adaptation: only current forward GRU states are reused; no future rows, recurrent parameter updates or new memory states.',
        'incremental_inference':'Curator adaptation: one matrix-vector readout on already computed hidden activations.',
        'training_compute':'Curator adaptation: frozen forward pass followed by fixed-penalty weighted regression and training-derived replication.',
        'cpu_deployment':'Fuse readout into the existing graph and benchmark the complete callback.',
        'dependencies':'NumPy for fitting; ONNX Runtime for frozen feature extraction and inference.',
        'licensing':'Original curator paraphrase and bibliographic metadata; no paper code imported.',
        'concise_relevance':'Tests retained causal information directly instead of treating a noisy point-gradient rejection as conclusive.'}
    cards,manifest=publish_revision(parent,destination,card)
    write_json(campaign.OUT/'library_v10_manifest.json',{'manifest_sha256':manifest,'parent_manifest_sha256':load_library(parent)[1],
        'count':len(cards),'added':[card['card_id']],'files':{p.name:campaign.sha(p) for p in sorted(destination.glob('*.json'))}})
    campaign.event('library_expanded',version='v10',manifest_sha256=manifest,count=len(cards))
    print(manifest)

if __name__=='__main__':
    main()
