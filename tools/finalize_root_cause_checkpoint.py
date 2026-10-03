"""Freeze the authorized research checkpoint without changing the incumbent."""
import json
import re
import shutil
from pathlib import Path
from competition_engineering import root_cause_campaign as campaign
from competition_engineering.manual_search_core import COMBO,COMBO_SHA,SEARCH_ROOT_WP,TRAIN_1024,TRAIN_4096,SEARCH
from competition_engineering.pipeline import write_json
from connectome.research_retrieval import load_library

def main():
    out=campaign.OUT
    if (out/'FINAL_REPORT.json').exists():
        raise ValueError('Checkpoint report is immutable')
    assert campaign.sha(COMBO)==COMBO_SHA
    def read(relative):
        return json.loads((out/relative).read_text())
    weights=read('weight_diagnosis_matched.json')
    checks=read('probability_readout/candidate_checks.json')
    checks['historical_tree_strength025']=read('tree_execution/report.json')
    checks['training_selected_tree_strength1']=read('tree_training_selection/report.json')
    timing=read('tree_execution/matched_microbenchmark.json')
    tree_selection=read('tree_training_selection/isolated_selection/artifacts/selection.json')
    test_text=(out/'regression_tests.log').read_text()
    passed=re.search(r'(\d+) passed in ([\d.]+)s',test_text)
    assert passed and int(passed[1])==67 and 'failed' not in test_text
    awaiting=[name for name,row in checks.items() if row['status']=='AWAITING PROMOTION']
    assert not awaiting,'Handle a qualifying candidate before checkpoint closure'
    knowledge_path=campaign.ROOT/'competition_engineering/search_knowledge.json'
    knowledge=json.loads(knowledge_path.read_text());assert knowledge['version']==3
    knowledge['version']=4
    knowledge['previous_snapshot']={'path':'runs/reassessment_20261002/knowledge_v3.json',
        'sha256':campaign.sha(campaign.PREVIOUS/'knowledge_v3.json')}
    knowledge['established'] += [
        {'lesson':'Use matched weight normalization when comparing regularized weighted fits. Mean-one focus makes the fixed penalty invariant to arbitrary population-weight units.',
         'source':'runs/root_cause_20261002/probability_readout/protocol.json'},
        {'lesson':'Frozen-tree execution was limited by estimator API overhead. Same-row one-CPU timing: sklearn192.53us versus ONNX3.46us for the correction; the full causal callback passes at68.50us.',
         'source':'runs/root_cause_20261002/tree_execution/matched_microbenchmark.json'},
        {'lesson':'A probability-weighted proxy rejected the new linear t0 correction on training-side WP, retaining the incumbent. The same proxy selected a frozen-tree strength on256 correction-disjoint training sequences that did not yield a credible official-search gain.',
         'source':'runs/root_cause_20261002/tree_training_selection/report.json'}]
    knowledge['weakened'] += [
        {'hypothesis':'Population-weight truncation is the dominant remaining cause of failed correction transfer.',
         'reason':'Removing truncation and fold-prior scaling improved t0 search-gradient alignment, but individual-component99 intervals included zero; the matched probability-weighted readout selected zero correction.'},
        {'hypothesis':'Compact shallow trees are intrinsically too costly for the callback budget.',
         'reason':'Identical frozen trees pass the complete callback budget after ONNX compilation.'},
        {'hypothesis':'Current causal mask weighting reliably ranks corrections across representations.',
         'reason':'On correction-disjoint training, tree strength1 improved proxy t0 WP by0.000846; official search combined delta was-0.000141 with99 interval spanning both signs.'}]
    knowledge['exploratory_only'].append({'wp':checks['historical_tree_strength025']['search']['candidate']['weighted_pearson'],
        'paired99_delta':checks['historical_tree_strength025']['paired99'],
        'reason':'Historical search-selected tree strength0.25; execution repaired, not a newly discovered score improvement.'})
    knowledge['next_research_question']='Which causal feature information predicts correction utility in the officially scored population across sequences? Compact tree execution is now feasible, but further fits need a transferable signal beyond training/proxy WP.'
    knowledge['status']='Good checkpoint reached: weighting hypothesis tested, tree execution repaired, training-only tree selection completed; incumbent retained.'
    write_json(out/'knowledge_v4.json',knowledge);write_json(knowledge_path,knowledge)
    campaign.event('knowledge_updated',version=4,snapshot_sha256=campaign.sha(out/'knowledge_v4.json'))
    campaign.event('infrastructure_repair_recorded',id='atomic_library_revision',
        failure='Duplicate source URL correctly rejected, but the initial builder had already created an invalid final directory. This was a publication failure, not model evidence.',
        repair='Preserved rejected attempt separately; revise existing card, validate staging, then atomically rename. Duplicate rejection/publication regression passes.',
        rejected_evidence='rejected_library_v9')
    campaign.event('infrastructure_repair_recorded',id='frozen_knowledge_verifier',
        failure='Closed-campaign verifier compared its snapshot with mutable current knowledge, which would reject a legitimate later version.',
        repair='Verify immutable snapshot and its existing report hash; regression covers an advanced live pointer and tampered snapshot.')
    campaign.event('research_decision',id='checkpoint',
        decision='Stop at requested good checkpoint. No weight, loss, temporal or tree-strength sweep; preserve CPU repair and new evidence. Exact remaining modeling limitation is unresolved.')
    sources=[campaign.ROOT/path for path in [
        'competition_engineering/root_cause_campaign.py','competition_engineering/probability_readout.py',
        'competition_engineering/compact_tree_onnx.py','competition_engineering/tree_execution_repair.py',
        'competition_engineering/tree_training_selection.py','competition_engineering/campaign_onnx.py',
        'competition_engineering/nonlinear_weighted_readout.py','competition_engineering/gradient_uncertainty.py',
        'competition_engineering/nonlinear_selection_diagnostic.py','competition_engineering/manual_search_core.py',
        'tests/test_root_cause_campaign.py','tools/expand_root_cause_library.py',
        'tools/verify_reassessment.py','tools/verify_root_cause_checkpoint.py']]
    sources.append(Path(__file__).resolve())
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for path in sources:
        shutil.copyfile(path,snapshots/path.relative_to(campaign.ROOT).as_posix().replace('/','__'))
    library=campaign.ROOT/'competition_engineering/research_cards/v9';cards,library_hash=load_library(library)
    evidence=[p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('ledger.jsonl','CHECKPOINT.json')]
    evidence += [directory/'identity.json' for directory in [TRAIN_1024,TRAIN_4096,SEARCH]]
    evidence += list(library.glob('*.json'))
    evidence += [COMBO,COMBO.parent/'baseline.onnx',campaign.PREVIOUS/'FINAL_REPORT.json',
        campaign.PREVIOUS/'knowledge_v3.json',campaign.PREVIOUS/'nonlinear_mask_trees.npz',
        campaign.ROOT/'competition_engineering/runs/manual_tiny_tree_t0_20261001/tree.joblib',
        campaign.ROOT/'competition_engineering/runs/manual_tiny_tree_t0_20261001/report.json']
    result={'status':'checkpoint_complete','starting_incumbent':SEARCH_ROOT_WP,'ending_incumbent':SEARCH_ROOT_WP,
        'incumbent_sha256':COMBO_SHA,'new_coefficient_fits':2,'new_mask_classifier_fits':0,
        'population_diagnostics':2,'historical_tree_execution_repair':True,'correction_disjoint_training_selection_sequences':256,
        'candidate_checks':checks,'weight_diagnostic':weights,'matched_tree_timing':timing,
        'tree_training_selection':tree_selection,'awaiting_promotion':awaiting,
        'best_credible':{'wp':SEARCH_ROOT_WP,'name':'frozen practical incumbent'},
        'best_alternative_checked':{'wp':checks['historical_tree_strength025']['search']['candidate']['weighted_pearson'],
            'name':'historical tree strength0.25, now deployable','paired99':checks['historical_tree_strength025']['paired99'],
            'status':'exploratory historical score, not established improvement'},
        'best_historical_point':{'wp':0.6590796311188727,'paired99':[-0.0005840183,0.0010744564],'status':'exploratory only'},
        'supported':['Official WP model selection','Weight-scale control in penalized comparisons',
            'Compact-tree execution feasibility after removing estimator overhead',
            'Population proxy improves some diagnostics but does not ensure correction-utility transport'],
        'weakened':['Weight clipping alone explains failed transfer','Compact trees necessarily violate CPU budget',
            'Mask-proxy WP reliably selects official-search improvements across representations'],
        'root_cause_assessment':'Concrete engineering causes repaired: tree API overhead and weight-scale confounding. The remaining score bottleneck is unresolved population-dependent correction utility, amplified by reused-search sequence uncertainty. A unique modeling root cause or conditional label shift is not established.',
        'new_papers':[],'revised_research_cards':['importance_weighted_model_selection,version2'],
        'library_version':'v9','library_card_count':len(cards),'library_manifest_sha256':library_hash,
        'research_sources':{'paper':'https://www.jmlr.org/papers/v8/sugiyama07a.html',
            'operator':'https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html'},
        'infrastructure_failures':['Duplicate-card version publication repaired; invalid attempt preserved.'],
        'repairs':['Compiled numeric tree ensemble with float32 split-boundary routing parity',
            'Mean-one population weights in matched regularized fits','Atomic validated research-library publication',
            'Closed-campaign verification independent of mutable current-knowledge pointer'],
        'tests':{'passed':int(passed[1]),'seconds':float(passed[2]),'log_sha256':campaign.sha(out/'regression_tests.log')},
        'boundary':'This continuation used only designated training/search caches, frozen search-only artifacts, and primary literature. No protected evaluation/results, full validation, leaderboard or submission used.',
        'limitations':['All search comparisons reuse64 sequences; intervals are descriptive and do not correct adaptive selection.',
            'Historical tree strength0.25 was search-selected earlier; recompilation does not create new performance evidence.',
            'Tree selection excludes original correction-training sequences, but does not establish independent base-model evaluation.',
            'Mask teacher was cross-fitted on search indicators; its training transfer and learning uncertainty remain limitations.',
            'The zero-correction interval[0,0] reflects identical predictions, not certainty about future performance.'],
        'strongest_next_question':knowledge['next_research_question'],
        'stopping_reason':'User-authorized good checkpoint after completed controlled experiments, repaired execution, and clean regression checks; not a claim of resource exhaustion or all hypotheses exhausted.',
        'reproducibility_sha256':{p.relative_to(campaign.ROOT).as_posix():campaign.sha(p) for p in sorted(set(evidence))},
        'ledger_prefix_sha256':campaign.sha(out/'ledger.jsonl')}
    write_json(out/'FINAL_REPORT.json',result)
    lines=['# Connectome research checkpoint — 2 October 2026',
        '', '**Practical incumbent retained: 0.658835322. No candidate awaits promotion.**',
        '', 'This continuation completed two population diagnostics, two matched ridge fits, an exact tree execution repair, and training-only strength selection on256 sequences excluded from the tree fit. Prior completed campaigns and their reports were preserved.',
        '', '**Results**','',
        '| Candidate | Search WP | Delta vs incumbent | Paired99% interval | Full callback µs/row |',
        '|---|---:|---:|---|---:|']
    for name,row in checks.items():
        lines.append(f"| {name} | {row['search']['candidate']['weighted_pearson']:.9f} | {row['search']['delta_combined']:+.9f} | [{row['paired99'][0]:+.7f}, {row['paired99'][1]:+.7f}] | {row['callback_checks']['callback_us_per_row']:.2f} |")
    lines += ['', 'The0.658882537 tree point is historical, now made deployable; its uncertainty still spans zero. The best historical point overall remains0.659079631, also exploratory with paired99% interval[-0.0005840,+0.0010745]. Neither replaces the incumbent.',
        '', '**Trajectory and decisions**','',
        '1. Reviewed the frozen research map and completed reassessment. WP selection remains established; objective-only and old temporal grids stayed closed.',
        '2. Diagnosed capped scoring weights. Removing clipping and fold-prior scaling raised t0 gradient cosine from0.703 to0.786; paired99% improvement interval[0.0030,0.1805]. A matched follow-up isolated the components: each component interval crossed zero, so clipping alone is not established as the cause.',
        '3. Fit two matched t0 readouts with mean-one weights and the same penalty. The probability-weighted training selector chose strength0; the capped control chose0.05 and scored a tiny negative delta. Decision: do not continue weight or penalty tuning.',
        '4. Compiled the historical32-tree correction into one ONNX ensemble operator. Same-row, one-CPU correction timings were192.53µs for sklearn and3.46µs for ONNX. Complete causal callback cost68.50µs; all required checks passed. Decision: remove execution cost as grounds for rejecting this compact tree family.',
        '5. Selected tree strength using probability-weighted pooled WP on256 correction-disjoint training sequences,128000 sampled rows. It chose1.0, with proxy t0 WP gain0.00084645. Official search combined delta was-0.00014067 and its99% interval crossed zero. Decision: population transfer remains unresolved across both linear and nonlinear corrections; stop at this checkpoint.',
        '', '**Root causes and remaining limitation**','',
        'Two concrete problems were addressed: estimator API overhead blocked a feasible tree representation, and arbitrary weight scale could change effective ridge regularization. Neither repair established a stronger scorer.',
        'The strongest unresolved limitation is population-dependent correction utility. Better mask prediction and closer population moments do not reliably predict official-scored gains. Reused64-sequence uncertainty makes small adaptive improvements difficult to establish. This does not prove conditional label shift, an irreducible ceiling, or a unique architectural root cause.',
        '', '**Repairs and verification**','',
        '- Numeric-tree export preserves float32 split routing, including threshold-adjacent values. Full replay maximum absolute differences were below2.4e-7.',
        '- Mean-one focus normalization makes regularized fits invariant to arbitrary global weight units.',
        '- A duplicate research-card source was rejected. Its incomplete directory is preserved under rejected_library_v9; publishing now validates a staged revision before atomic rename. This is an infrastructure failure, not a scientific negative.',
        '- Closed-campaign verification now checks its hashed knowledge snapshot instead of requiring the mutable latest-knowledge file to remain unchanged.',
        '- 67 Linux regression tests passed, including the new regressions and existing search-boundary, metric, library, history and callback checks.',
        '', '**Research provenance and boundary**','',
        'Library v9 revises the existing [importance-weighted model-selection card](https://www.jmlr.org/papers/v8/sugiyama07a.html), retaining33 unique papers. No new paper was needed. Its covariate-shift assumption and lack of a clipped-Pearson guarantee remain explicit. Export follows the [official ONNX ensemble operator specification](https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html).',
        '', 'No protected evaluation, protected feedback, full validation, leaderboard evaluation or submission was used in this continuation. Training-only selection is correction-disjoint, not an independent assessment of the supplied GRU. Search intervals remain descriptive under adaptive reuse.',
        '', '**Next question**','',knowledge['next_research_question'],
        '', 'This is the requested good checkpoint, not a claim that resources or all scientifically justified research directions are exhausted.',
        '', '**Reproducibility**','',
        f'Incumbent SHA256: `{COMBO_SHA}`.',
        f'Library manifest SHA256: `{library_hash}`.',
        f'Knowledge snapshot SHA256: `{campaign.sha(out/"knowledge_v4.json")}`.',
        '', 'FINAL_REPORT.json contains exact input/config/model/package/source-snapshot hashes, candidate atlases and paired intervals. ledger.jsonl contains the hash-chained decisions. Verify with `python -m tools.verify_root_cause_checkpoint`.', '']
    # Keep numeric prose readable without altering identifiers or exact hashes.
    prose='\n'.join(lines)
    for before,after in [('on256','on 256'),('The0.','The 0.'),('remains0.','remains 0.'),('from0.','from 0.'),('to0.','to 0.'),
        ('strength0','strength 0'),('chose0','chose 0'),('historical32','historical 32'),('were192','were 192'),('and3.46','and 3.46'),
        ('cost68','cost 68'),('sequences,128000','sequences, 128000'),('chose1.0','chose 1.0'),('gain0.','gain 0.'),('was-0.','was -0.'),
        ('its99%','its 99%'),('Reused64','Reused 64'),('below2.4','below 2.4'),('retaining33','retaining 33')]:
        prose=prose.replace(before,after)
    (out/'FINAL_REPORT.md').write_text(prose,encoding='utf-8')
    write_json(out/'CHECKPOINT.json',{'status':'complete','incumbent':SEARCH_ROOT_WP,'awaiting_promotion':[],
        'report_sha256':campaign.sha(out/'FINAL_REPORT.json'),'knowledge_version':4,'next_question':knowledge['next_research_question']})
    campaign.event('campaign_checkpointed',review_sha256=campaign.sha(out/'FINAL_REPORT.json'),report_sha256=campaign.sha(out/'FINAL_REPORT.md'))
    print(json.dumps({'report':str(out/'FINAL_REPORT.md'),'incumbent':SEARCH_ROOT_WP,'tests':int(passed[1]),'awaiting':awaiting}))

if __name__=='__main__':
    main()
