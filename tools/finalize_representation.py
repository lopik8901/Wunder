"""Write the immutable representation-campaign checkpoint."""
import json
from pathlib import Path
from competition_engineering import representation_campaign as campaign
from competition_engineering.pipeline import write_json
from connectome.research_retrieval import load_library

def evidence_paths(out):
    return [out/'protocol.json',out/'knowledge_v5.json',out/'knowledge_v6.json',
        out/'replicated_readout/replication.json',out/'replicated_readout/candidate_checks.json',
        out/'consensus_ensemble/protocol.json',out/'consensus_ensemble/replication.json',
        out/'consensus_ensemble/candidate_checks.json',out/'consensus_ensemble/mechanism_comparison.json',
        out/'library_v10_manifest.json',out/'library_v11_manifest.json']

def main():
    out=campaign.OUT
    if (out/'FINAL_REPORT.json').exists(): return
    latest=campaign.ROOT/'competition_engineering/search_knowledge.json'
    knowledge=json.loads(latest.read_text())
    consensus=json.loads((out/'consensus_ensemble/candidate_checks.json').read_text())
    rep=json.loads((out/'consensus_ensemble/replication.json').read_text())
    if knowledge['version']==5:
        knowledge['version']=6
        knowledge['previous_snapshot']={'path':'runs/representation_20261002/knowledge_v5.json','sha256':campaign.sha(out/'knowledge_v5.json')}
        knowledge['established'].append({'lesson':'The average of independently fitted frozen-state readouts replicated positive training-derived gains. A disagreement soft-threshold did not beat a matched-amplitude control on fresh replication, so disagreement is not supported as the source of utility.', 'source':'runs/representation_20261002/consensus_ensemble/replication.json'})
        knowledge['weakened'].append({'hypothesis':'Readout disagreement is a transferable per-row indicator for shrinking causal corrections.', 'reason':'Consensus trailed the amplitude-matched control on both frozen replication populations; its higher reused-search point is uncertain and above the CPU limit.'})
        knowledge['status']='Representation readouts repeatedly improve training-derived populations but have no credible official-search improvement. This campaign checkpoint is complete.'
        write_json(out/'knowledge_v6.json',knowledge);write_json(latest,knowledge)
    else:
        assert knowledge['version']==6 and (out/'knowledge_v6.json').exists()
    manifest=load_library(campaign.ROOT/'competition_engineering/research_cards/v11')[1]
    checks={name:{'wp':r['search']['candidate']['weighted_pearson'],'delta':r['search']['delta_combined'],'paired99':r['paired99'],'cpu_us':r['callback_checks']['callback_us_per_row'],'status':r['status']} for name,r in consensus.items()}
    report={'schema_version':1,'boundary':'search-only','incumbent':knowledge['incumbent'],'library_manifest_sha256':manifest,
        'best_observed_search_candidate':{'name':'consensus','wp':checks['consensus']['wp'],'reason':'Point estimate only; does not qualify.'},
        'best_credible_candidate':None,'awaiting_promotion':[],'candidates':checks,
        'replication':rep,'decisions':['Frozen-state representation has replicated development signal but no official transfer.', 'Position transport was not stable enough to justify a fit.', 'Disagreement soft-threshold is abandoned: it lost to matched amplitude on fresh replication.', 'Do not tune consensus for its reused-search point estimate.'],
        'infrastructure':{'tests_added':'Consensus algebra and ONNX head parity regression tests.', 'implementation_failure':None},
        'reproducibility_sha256':{str(p.relative_to(campaign.ROOT)).replace('\\','/'):campaign.sha(p) for p in evidence_paths(out)}}
    write_json(out/'FINAL_REPORT.json',report)
    md='# Connectome representation research checkpoint — 2 October 2026\n\n**Practical incumbent retained: 0.658835322. No candidate awaits promotion.**\n\n| Candidate | Search WP | Δ WP | Paired 99% interval | Callback µs/row | Decision |\n|---|---:|---:|---|---:|---|\n'
    for name,r in checks.items(): md+=f"| {name} | {r['wp']:.9f} | {r['delta']:+.9f} | [{r['paired99'][0]:+.7f}, {r['paired99'][1]:+.7f}] | {r['cpu_us']:.2f} | insufficient |\n"
    md+='\nThe consensus point is the best observed in this continuation, but it is not credible: its interval crosses zero and its 77.49 µs callback exceeds the 76.93 µs limit. It was not preserved for promotion.\n\nFrozen GRU-state readouts recovered a positive effect on two training-derived replications, showing that learned causal states carry correction information beyond the current features. That information did not reliably transfer to the reused 64-sequence official search. Position effects also failed replication.\n\nThe final disagreement experiment used two disjoint readouts, fixed strength 0.25, and a predeclared soft-threshold rule. On a fresh 256-sequence replication it lost to the amplitude-matched control by −0.000063 uniform and −0.000134 probability-weighted combined WP. The rule is abandoned; the higher search point does not overturn its mechanism failure.\n\nThe campaign remains strictly search-only: no promotion, full validation, leaderboard, or submission was run. The incumbent archive is unchanged. New regression tests cover consensus algebra and fused ONNX parity. The research library now has v11 with the primary [deep ensembles paper](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html); its limits for shared-readout clipped-WP use are explicit.\n\nThe next credible direction is a materially distinct causal architecture or feature family, evaluated first through the existing disjoint sequence replication protocol. Further selection, position, loss, or agreement shrinkage tuning is not justified by this evidence.\n'
    (out/'FINAL_REPORT.md').write_text(md,encoding='utf-8')
    campaign.event('campaign_checkpointed',report_sha256=campaign.sha(out/'FINAL_REPORT.json'),review_sha256=campaign.sha(out/'FINAL_REPORT.md'),knowledge_sha256=campaign.sha(out/'knowledge_v6.json'))
    write_json(out/'CHECKPOINT.json',{'status':'complete_checkpoint','completed':'Frozen representation, position transport, and disagreement ensemble experiments','incumbent':knowledge['incumbent'],'knowledge_version':6,'awaiting_promotion':[]})
    print(json.dumps({'report_sha256':campaign.sha(out/'FINAL_REPORT.json'),'knowledge_sha256':campaign.sha(out/'knowledge_v6.json')}))

if __name__=='__main__': main()
