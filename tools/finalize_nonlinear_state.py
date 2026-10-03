"""Freeze modeling evidence, source snapshots and a complete checkpoint report."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
from competition_engineering import nonlinear_state_campaign as campaign
from competition_engineering.pipeline import write_json
from competition_engineering.search_population_audit import pooled_correlations
from connectome.research_retrieval import load_library

def sequence_stability(path,primary):
    with np.load(path) as q: moments=q['moments']
    scores=pooled_correlations(moments.sum(0));point=(scores[primary]-scores[0]).mean(-1)
    single=pooled_correlations(moments);gains=(single[:,primary]-single[:,0]).mean(-1)
    leave=pooled_correlations(moments.sum(0)[None]-moments);leave_gain=(leave[:,primary]-leave[:,0]).mean(-1)
    return {'pooled_gain':point.tolist(),'sequence_positive_fraction':np.mean(gains>0,axis=0).tolist(),
        'leave_one_sequence_out_gain_range':np.stack([leave_gain.min(0),leave_gain.max(0)]).tolist(),
        'interpretation':'Sequence fractions are descriptive; pooled WP is recomputed after deletion, never averaged from per-sequence correlations.'}

def main():
    out=campaign.OUT
    if (out/'FINAL_REPORT.json').exists(): raise ValueError('Completed checkpoint is immutable')
    def read(relative): return json.loads((out/relative).read_text())
    rep=read('tanh_replication.json');learned=read('learned32/replication.json');trees=read('state_trees/replication.json');innovation=read('state_innovation/replication.json')
    if innovation['search_gate_passed'] and not (out/'state_innovation/search_results.json').exists():
        raise ValueError('Qualified innovation experiment must finish its predeclared search comparison before closure')
    tests={k:read('tests_'+k+'.json') for k in ['windows','linux']}
    assert tests['windows']['passed']>=85 and tests['linux']['passed']>=86
    checks=read('search_results.json');extra=read('state_innovation/search_results.json') if (out/'state_innovation/search_results.json').exists() else {}
    allchecks={**{'tanh_'+k:v for k,v in checks.items()},**{'innovation_'+k:v for k,v in extra.items()}}
    awaiting=[name for name,row in allchecks.items() if row['status']=='AWAITING_PROMOTION']
    stability={}
    for name,file,index in [('tanh','tanh_replication_moments.npz',4),('learned32','learned32/replication_moments.npz',4),('trees','state_trees/replication_moments.npz',3),('innovation','state_innovation/replication_moments.npz',2)]:
        stability[name]=sequence_stability(out/file,index)
    write_json(out/'sequence_stability.json',stability)
    knowledge_path=campaign.ROOT/'competition_engineering/search_knowledge.json';knowledge=json.loads(knowledge_path.read_text());assert knowledge['version']==10
    knowledge['version']=11;knowledge['previous_snapshot']={'path':'runs/nonlinear_state_20261003/knowledge_v10.json','sha256':campaign.sha(out/'knowledge_v10.json')}
    knowledge['established'].append({'lesson':'Multiple disjoint training-derived groups recover frozen-state linear correction gains. Fixed joint tanh, learned tanh and shallow threshold trees do not establish a robust incremental mechanism beyond that linear signal.',
        'source':'runs/nonlinear_state_20261003/FINAL_REPORT.json'})
    knowledge['weakened'].append({'hypothesis':'One-step frozen GRU state updates add robust incremental causal information to the current-state linear readout.',
        'reason':innovation['decision']+' Contrast: '+json.dumps(innovation['innovation_minus_linear'])})
    knowledge['status']='Four prospective model mechanisms completed at a reproducible session checkpoint; incumbent retained pending any separately qualified candidate.'
    knowledge['next_research_question']='Does increasing the number of fitting sequences at a fixed sampled-row budget reduce the learned-readout generalization gap? Compare sequence diversity with within-sequence density, retaining an unused prospective replication group.'
    write_json(out/'knowledge_v11.json',knowledge);write_json(knowledge_path,knowledge)
    state_map={'supported':['WP-based selection can reject later checkpoints despite substantially lower training MSE','Frozen-state linear readouts retain repeatable development signal','Prospective replication prevents testing failed mechanisms on official search'],
        'weakly_supported':['Random joint tanh features improve uniform development WP over a linear control; proxy increment and official transfer remain unestablished'],
        'weakened':['Fixed smooth interactions resolve transfer','Learned smooth interactions resolve transfer','Shallow hidden-state threshold regimes add meaningful incremental signal'],
        'abandoned_locally':['Random-basis width/seed tuning','Learned-head epoch or strength tuning after failed replication','Tree depth/strength tuning after failed replication','Consensus and position gates from earlier campaigns'],
        'inadequately_tested':['Sequence diversity at fixed fitting-row budget','Changing the frozen representation itself','Larger or longer-history models with new causal evidence'],
        'next_hypothesis':knowledge['next_research_question'],
        'next_design':'Hold architecture, optimizer, total sampled rows, selection population and gate fixed; compare1024 sequences×128 rows with2048×64. Previously observed replication groups may enter the larger fit only if a newly reserved disjoint group is used for evaluation. Freeze the assignment before fitting.',
        'scientific_limit':'Strong training-error reductions with early WP-selected checkpoints motivate the diversity test; they do not prove sequence overfitting is the unique root cause.'}
    write_json(out/'research_state_map.json',state_map)
    campaign.event('research_synthesis',state_map_sha256=campaign.sha(out/'research_state_map.json'),knowledge_sha256=campaign.sha(out/'knowledge_v11.json'),awaiting=awaiting)
    sources=['competition_engineering/'+name for name in ['nonlinear_state_campaign.py','nonlinear_state_evaluation.py','learned_state_readout.py','state_tree_readout.py','state_innovation_readout.py',
        'representation_campaign.py','replicated_readout.py','representation_evaluation.py','frozen_latent_readout.py','frozen_representation_onnx.py','compact_tree_onnx.py','campaign_onnx.py',
        'manual_search_core.py','probability_readout.py','prospective_selection_diagnostic.py','gradient_uncertainty.py','search_population_audit.py','search_error_diagnostics.py','residual.py','pipeline.py','generated_runner.py','generated_sandbox.py','generated_worker.py']]
    sources+=['connectome/mlevolve_generated.py','tests/test_nonlinear_state_campaign.py','tools/expand_nonlinear_state_library.py','tools/nonlinear_campaign_checkpoint.py','tools/finalize_nonlinear_state.py']
    snapshot=out/'source_snapshots';snapshot.mkdir(exist_ok=True)
    for source in sources: shutil.copyfile(campaign.ROOT/source,snapshot/source.replace('/','__'))
    cards,manifest=load_library(campaign.ROOT/'competition_engineering/research_cards/v12')
    evidence=[p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['ledger.jsonl','CHECKPOINT.json','FINAL_REPORT.json','FINAL_REPORT.md']]
    evidence += list((campaign.ROOT/'competition_engineering/research_cards/v12').glob('*.json'))
    evidence += [campaign.COMBO,campaign.COMBO.parent/'baseline.onnx',campaign.prior.OUT/'protocol.json',campaign.prior.OUT/'prepared_manifest.json',
        campaign.prior.TRAIN/'identity.json',campaign.TRAIN_4096/'identity.json']
    evidence += list((campaign.prior.OUT/'replicated_readout/isolated_training/artifacts').glob('*.npz'))
    best=max(allchecks,key=lambda k:allchecks[k]['wp'])
    report={'status':'complete_session_checkpoint','boundary':'search-only','starting_incumbent':.6588353223316641,'ending_incumbent':.6588353223316641,
        'incumbent_sha256':campaign.COMBO_SHA,'experiments':4,'fitted_bivariate_model_variants':11,'fresh_replication_sequences':1024,
        'official_search_passes':1+bool(extra),'official_search_candidates':len(allchecks),'search_results':allchecks,
        'replication':{'random_tanh':rep,'learned32':learned,'state_trees':trees,'state_innovation':innovation},
        'sequence_stability':stability,'best_new_observed':{'name':best,'wp':allchecks[best]['wp'],'paired99':allchecks[best]['paired99'],'status':allchecks[best]['status']},
        'best_credible':{'name':'manual_targetwise_combo_v1','wp':.6588353223316641},'awaiting_promotion':awaiting,
        'best_historical_exploratory':{'wp':.6590796311188727,'status':'Historical adaptive point only; not replaced or optimized around.'},
        'tests':tests,'library_manifest_sha256':manifest,'library_cards':len(cards),'new_papers':['Rahimi and Recht2008, Weighted Sums of Random Kitchen Sinks'],
        'infrastructure':'New guarded fit/export paths and tests. No production fit failure. A float32 unit-test tolerance was replaced by checks against a float64 reference. Isolation and evaluation boundaries unchanged.',
        'research_state':state_map,'elapsed_seconds':time.time()-read('protocol.json')['started_unix'],
        'limits':['Fresh groups are disjoint from correction fit/selection, not necessarily independent of the supplied pretrained GRU','Official64-sequence search is adaptively reused; intervals are descriptive','Failed gates prohibit official scores; unmeasured scores and callback costs are not inferred'],
        'ledger_prefix_sha256':campaign.sha(out/'ledger.jsonl'),
        'reproducibility_sha256':{p.relative_to(campaign.ROOT).as_posix():campaign.sha(p) for p in evidence}}
    write_json(out/'FINAL_REPORT.json',report)
    lines=['# Connectome nonlinear-state research checkpoint — 3 October 2026','',
        '**Practical incumbent retained: 0.658835322.** '+('Qualified candidates are separately frozen: '+', '.join(awaiting) if awaiting else 'No candidate awaits promotion.'),'',
        'Four prospective mechanisms were tested, with11 fitted bivariate model variants and1024 fresh training-derived replication sequences. The frozen GRU was unchanged. Each experiment updated the hash-chained ledger and persistent knowledge before branching.','',
        '## Official-search results','',
        '| Candidate | WP | Delta | Paired99% delta interval | Callback µs/row |','|---|---:|---:|---|---:|']
    for name,row in allchecks.items(): lines.append(f"| {name} | {row['wp']:.9f} | {row['delta']:+.9f} | [{row['paired99'][0]:+.7f}, {row['paired99'][1]:+.7f}] | {row['cpu_us']:.2f} |")
    lines += ['',f"Best newly measured point: {best}, WP{allchecks[best]['wp']:.9f}. The practical incumbent remains stronger. Historical0.659079631 and consensus0.659072103 remain exploratory points from earlier campaigns.",'',
        '## Experimental trajectory','',
        '1. **Fixed nonlinear features.** Tested a371-feature linear readout, a current-only nonlinear control, separable current/hidden nonlinearities and128 joint tanh features. Closed-form weighted-ridge solves had normal-equation residuals near1e-15. WP-selected strengths were frozen before replication. Joint gained0.001838 uniform/0.001088 proxy WP; joint-minus-linear was0.000223/0.000020. The permissive prospective gate passed. The one official-search pass found no benefit over linear and no incumbent improvement; joint CPU80.00µs also exceeded76.93µs. Fixed-basis tuning was abandoned.','',
        '2. **Learned32-unit nonlinear directions.** Tested current-only, separable and joint heads over fixed linear backbones, keeping the GRU frozen. Later epochs cut joint training MSE from approximately1.331/1.310 to1.062/1.066, yet WP chose epoch5 for both targets. Fresh replication joint-minus-linear was−0.000223 uniform/+0.000079 proxy; joint also lost to separable on proxy rows and one uniform half was negative. Gate failed; official search was not consulted. Optimization succeeded at its loss, while incremental model utility failed to replicate.','',
        '3. **Threshold regimes.** Fit32 depth2 histogram trees per target atop fixed linear backbones, with a matched current-only tree control. State models used hidden features in160 of192 splits. Fresh incremental WP over linear was−0.000047 uniform/+0.000013 proxy, with95% intervals spanning zero. Gate failed; official search was not consulted. Tree use of hidden state did not demonstrate useful additional signal.','',
        '4. **One-step state updates.** Tested the distinct causal feature h[t]−h[t−1], computed on full sequences before sampling, with zero initial state. A627-feature readout was compared with a matched371-feature current-state control; each selected strength by training-derived WP. Result: '+innovation['decision'],'',
        '## Fresh replication and uncertainty','',
        '| Primary model | Uniform WP delta | Proxy WP delta | Proxy paired95% interval | Search gate |','|---|---:|---:|---|---|']
    for name,r,key in [('Random joint',rep,'joint'),('Learned joint',learned,'joint'),('State trees',trees,'state'),('State update',innovation,'innovation')]:
        v=r['variants'][key];ci=[v['paired95'][0][1],v['paired95'][1][1]]
        lines.append(f"| {name} | {v['combined_uniform_probability'][0]:+.7f} | {v['combined_uniform_probability'][1]:+.7f} | [{ci[0]:+.7f}, {ci[1]:+.7f}] | {'pass' if r['search_gate_passed'] else 'fail'} |")
    lines += ['', 'These deltas compare against the frozen incumbent on the same development population. They are not official-search scores. Incremental contrasts against matched controls, targetwise effects, both sequence halves, paired95/99 intervals and leave-one-sequence-out ranges are recorded in FINAL_REPORT.json. A positive baseline comparison does not establish the proposed mechanism when its matched control performs similarly.','',
        '## Lessons and remaining limitation','',
        'Frozen states repeatedly support a useful linear development correction. Three nonlinear function classes did not reliably add enough incremental value to explain or resolve official-population transfer. Lower training error remains a poor selection signal, especially for the learned head. The evidence does not establish an exhausted GRU representation, an irreducible score ceiling, or a unique population-shift root cause.','',
        'The strongest next hypothesis is insufficient fitting-sequence diversity: learned heads can substantially reduce training error while WP favors early checkpoints. Test sequence count at fixed row budget, architecture and optimizer (1024×128 versus2048×64), with a prospectively reserved unused replication group. This is a falsifiable generalization hypothesis, not a recommendation to enlarge the network or tune this search score.','',
        '## Implementation, sources and boundaries','',
        f"Regression checks: Windows{tests['windows']['passed']} passed/{tests['windows']['skipped']} platform skip; WSL{tests['linux']['passed']} passed. Tests cover interaction controls, ONNX parity, exact tree routing, state-update causality/reset, source guards and the failed-replication search block. All actual fits used the existing network-disabled sandbox; replication data was never mounted there.",'',
        'Library v12 adds the primary [Rahimi–Recht2008 random nonlinear features paper](https://papers.nips.cc/paper_files/paper/2008/hash/0efe32849d230d7f53049ddc4a4b0c60-Abstract.html). Existing linear-probe, neural-additive and gradient-boosting cards informed controls and alternatives. All task adaptations and lack of a clipped-WP transfer guarantee are explicit.','',
        'No protected promotion, holdout, full-validation, leaderboard or submission results were used or run in this campaign. Prior archive-handoff protected information was excluded from research decisions. The incumbent artifact hash is unchanged. Failed replication candidates have no inferred official score or full-callback latency.','',
        'This is a reproducible session checkpoint, not a claim that all scientifically possible research or machine resources are exhausted. Exact artifacts, source snapshots, cache identities, test records and knowledge snapshots are hashed in FINAL_REPORT.json. Verify with `python -m tools.nonlinear_campaign_checkpoint --phase verify`.','']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    campaign.event('campaign_checkpointed',json_sha256=campaign.sha(out/'FINAL_REPORT.json'),markdown_sha256=campaign.sha(out/'FINAL_REPORT.md'))
    write_json(out/'CHECKPOINT.json',{'status':'complete_session_checkpoint','experiments':4,'knowledge_version':11,'incumbent_wp':.6588353223316641,'awaiting_promotion':awaiting,
        'next_hypothesis':knowledge['next_research_question']})
    print(json.dumps({'hashes':len(report['reproducibility_sha256']),'best_new_point':report['best_new_observed'],'awaiting':awaiting}))

if __name__=='__main__': main()
