"""Add bounded primary-source context for a frozen disagreement experiment."""
from competition_engineering import representation_campaign as campaign
from competition_engineering.pipeline import write_json
from tools.expand_root_cause_library import publish_revision
from connectome.research_retrieval import load_library

def main():
    campaign.initialize()
    parent=campaign.ROOT/'competition_engineering/research_cards/v10';destination=parent.with_name('v11')
    card={'schema_version':1,'card_id':'ensemble_disagreement','version':1,'card_type':'paper',
        'family':'ensemble','tags':['ensemble','disagreement','uncertainty','regression','diagnostic'],
        'title':'Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles',
        'authors':['Balaji Lakshminarayanan','Alexander Pritzel','Charles Blundell'],'year':2017,
        'source_url':'https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html',
        'source_locator':'Primary NeurIPS abstract', 'curator':'Codex, primary-source review','verified_on':'2026-10-02','claim_confidence':'limited',
        'mechanism':'The authors use ensembles to estimate predictive uncertainty and report benchmark results for regression and classification.',
        'useful_when':'Curator inference: independently fitted predictors disagree on corrections and a conservative combination can be tested against an amplitude-matched control.',
        'diagnostic_signatures':'Curator inference: the disagreement-specific rule should beat a control with the same overall correction scale on a fresh sequence group.',
        'assumptions':'Ensemble members must contain meaningfully different predictive errors for disagreement to add information.',
        'limitations':'The paper does not establish that two linear readouts of a shared frozen representation have calibrated uncertainty, nor a benefit for clipped weighted Pearson.',
        'causality':'Curator adaptation: both readouts consume the existing causal current features and frozen forward states only.',
        'incremental_inference':'Curator adaptation: one extra fixed matrix product and elementwise operations.',
        'training_compute':'Curator adaptation: reuse disjoint fixed readouts; evaluate the rule on an unused training-derived group.',
        'cpu_deployment':'Fuse the rule into ONNX and benchmark the full callback.',
        'dependencies':'NumPy and ONNX Runtime; no author code imported.','licensing':'Original curator paraphrase and bibliographic metadata; no paper code imported.',
        'concise_relevance':'Motivates testing disagreement as a falsifiable conditional correction mechanism, with an explicit matched control.'}
    cards,manifest=publish_revision(parent,destination,card)
    write_json(campaign.OUT/'library_v11_manifest.json',{'manifest_sha256':manifest,'parent_manifest_sha256':load_library(parent)[1],
        'count':len(cards),'added':[card['card_id']],'files':{p.name:campaign.sha(p) for p in sorted(destination.glob('*.json'))}})
    campaign.event('library_expanded',version='v11',manifest_sha256=manifest,count=len(cards))
    print(manifest)

if __name__=='__main__':
    main()
