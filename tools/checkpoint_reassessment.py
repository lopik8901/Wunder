"""Persist experimental decisions and bounded planner-visible lessons."""
import json
import time
from competition_engineering import reassessment_campaign as campaign
from competition_engineering.pipeline import write_json

def main():
    knowledge=json.loads((campaign.OUT/'starting_knowledge.json').read_text());knowledge['version']=2
    knowledge['established'] +=[
        {'lesson':'Point-gradient cosine is unreliable at the current64-sequence search size: median split-half agreement was0.16 for t0 and0.07 for t1. Include sequence uncertainty before rejecting a representation.',
         'source':'runs/reassessment_20261002/gradient_uncertainty_v2.json'},
        {'lesson':'Frozen t1 corrections improve all-required-row WP but regress officially scored WP. Their paired population contrasts exclude zero, survive unclipped prediction and within-sequence checks, and remain negative after influential-sequence deletion.',
         'source':'runs/reassessment_20261002/population_mechanism.json'},
        {'lesson':'A nonlinear causal mask predictor improved target0 population-gradient transport over the linear predictor under a prospective uncertainty rule. This supports a bounded weighted-readout experiment, not a new incumbent.',
         'source':'runs/reassessment_20261002/nonlinear_selection_result.json'}]
    knowledge['weakened'] +=[
        {'hypothesis':'Low point-gradient alignment establishes a stable causal transfer conflict.','reason':'Bootstrap intervals span both signs and split-half gradients are unstable.'},
        {'hypothesis':'Clipping or a few outlier sequences explain the t1 population contrast.','reason':'Contrast survives diagnostic counterfactuals.'},
        {'hypothesis':'The existing linear causal proxy is a sufficient selection substitute.','reason':'Prospective training masks selected an officially inferior t1 correction; audit intervals included zero.'}]
    snapshot=campaign.OUT/'knowledge_v2.json'
    if not snapshot.exists():
        write_json(snapshot,knowledge);write_json(campaign.KNOWLEDGE,knowledge)
        campaign.event('knowledge_updated',version=2,sha256=campaign.sha(snapshot))
    decisions=[
        {'experiment':'gradient_uncertainty_v2','decision':'Replace point-only directional rejection with uncertainty-aware diagnosis. Do not reopen all models from ambiguous cosines.'},
        {'experiment':'prospective_selection','decision':'Keep correction-disjoint training audit as a diagnostic; proxy masks do not replace official search.'},
        {'experiment':'search_population_audit','decision':'Follow robust t1 population contrast; t0 contrasts remain inconclusive.'},
        {'experiment':'population_mechanism','decision':'Rule down clipping, between-sequence effects alone, and influential-sequence explanations; examine selector representation.'},
        {'experiment':'nonlinear_selection','decision':'Launch fixed-penalty t0 readout only; t1 transport-improvement rule failed.'}]
    path=campaign.OUT/'decisions_checkpoint.json'
    if not path.exists():
        write_json(path,decisions);campaign.event('decisions_checkpointed',sha256=campaign.sha(path))
    write_json(campaign.OUT/'CHECKPOINT.json',{'elapsed_seconds':time.time()-json.loads((campaign.OUT/'campaign.json').read_text())['started_unix'],
        'incumbent_unchanged':True,'completed_diagnostics':5,'next':'Complete isolated nonlinear-weighted t0 fit and unchanged callback/search checks',
        'knowledge_sha256':campaign.sha(campaign.KNOWLEDGE),'library_manifest_sha256':json.loads((campaign.OUT/'library_v8_manifest.json').read_text())['manifest_sha256']})

if __name__=='__main__':
    main()
