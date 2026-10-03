"""Freeze the reassessment evidence, final decisions, and reproducibility map."""
import json
import shutil
import time
from pathlib import Path
from competition_engineering import reassessment_campaign as campaign
from competition_engineering.manual_search_core import COMBO,COMBO_SHA,SEARCH_ROOT_WP,TRAIN_1024,TRAIN_4096,SEARCH
from competition_engineering.pipeline import write_json
from connectome.research_retrieval import load_library

def main():
    out=campaign.OUT
    if (out/'FINAL_REPORT.json').exists():
        raise ValueError('Final research review is immutable')
    if campaign.sha(COMBO)!=COMBO_SHA:
        raise ValueError('Practical incumbent identity changed')
    names=['gradient_uncertainty_v2','prospective_selection_result','search_population_audit',
           'population_mechanism','nonlinear_selection_result','training_transport']
    diagnostics={name:json.loads((out/(name+'.json')).read_text()) for name in names}
    checks=json.loads((out/'nonlinear_weighted_t0/candidate_checks.json').read_text())
    awaiting=[name for name,record in checks.items() if record['status']=='AWAITING PROMOTION']
    best_name=max(checks,key=lambda k:checks[k]['search']['candidate']['weighted_pearson'])
    best=checks[best_name]
    if diagnostics['training_transport']['eligible_targets_for_directional_fit']:
        raise ValueError('Directional fit prerequisites passed; finish justified work before closing')
    knowledge=json.loads((out/'knowledge_v2.json').read_text());knowledge['version']=3
    for row in knowledge['established']:
        row['lesson']=row['lesson'].replace('current64','current 64').replace('was0.16','was 0.16').replace('and0.07','and 0.07').replace('target0','target 0')
    knowledge['established'] +=[{'lesson':'Nonlinear weighting improved training-side t0 gradient stability, but source-to-official-search alignment remained uncertain. Better mask prediction alone did not establish a stronger correction.',
                                'source':'runs/reassessment_20261002/training_transport.json'}]
    knowledge['weakened'] +=[{'hypothesis':'A nonlinear causal scoring predictor alone makes a fixed current-row residual readout transferable.',
                            'reason':'Weighted fit scored0.658830952; paired99 interval crosses zero. Mask-informed selection point gain was only0.0000115.'}]
    knowledge['exploratory_only'] +=[{'wp':best['search']['candidate']['weighted_pearson'],'paired99_delta':best['paired99'],
                                     'reason':'Training-proxy-selected unweighted readout; tiny adaptive search gain, not an established improvement.'}]
    knowledge['next_research_question']='Which training-derived causal representation or evaluation population reproduces the stable t1 selected-versus-uniform correction contrast without fitting search-target residuals?'
    knowledge['status']='Checkpointed; no further fit met current evidence prerequisites.'
    write_json(out/'knowledge_v3.json',knowledge);write_json(campaign.KNOWLEDGE,knowledge)
    campaign.event('knowledge_updated',version=3,sha256=campaign.sha(out/'knowledge_v3.json'))
    campaign.event('research_decision',id='training_transport',decision='No new directional fit; source-to-official improvement interval includes zero. Weighting improved source stability, so insufficient training count is not independently established as the dominant cause.')
    sources=[campaign.ROOT/'competition_engineering'/name for name in [
        'reassessment_campaign.py','gradient_uncertainty.py','prospective_selection_diagnostic.py','search_population_audit.py',
        'population_mechanism_diagnostic.py','nonlinear_selection_diagnostic.py','nonlinear_weighted_readout.py','training_transport_diagnostic.py']]
    sources +=[campaign.ROOT/'connectome/mlevolve_history.py',campaign.ROOT/'tests/test_reassessment_campaign.py',
               campaign.ROOT/'tools/expand_reassessment_library.py',campaign.ROOT/'tools/expand_transport_card.py',
               campaign.ROOT/'tools/checkpoint_reassessment.py',Path(__file__).resolve()]
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for path in sources:
        target=snapshots/(str(path.relative_to(campaign.ROOT)).replace('\\','__').replace('/','__'))
        shutil.copyfile(path,target)
    evidence=[p for p in out.rglob('*') if p.is_file() and p.name not in ('ledger.jsonl','CHECKPOINT.json','prospective_progress.json','progress.json')]
    # Source inputs include only fixed training/search identities, not mixed ledgers.
    evidence +=[directory/'identity.json' for directory in [TRAIN_1024,TRAIN_4096,SEARCH]]
    library=campaign.ROOT/'competition_engineering/research_cards/v8';cards,library_hash=load_library(library)
    evidence +=list(library.glob('*.json'))
    elapsed=time.time()-json.loads((out/'campaign.json').read_text())['started_unix']
    result={'status':'closed_no_further_evidence_gated_fit','starting_incumbent':SEARCH_ROOT_WP,'ending_incumbent':SEARCH_ROOT_WP,
        'incumbent_sha256':COMBO_SHA,'incumbent_unchanged':True,
        'history_review':{'mlevolve_attempts':82,'routine_metric_records':74,'original_cards':29,
            'scope':'Dedicated search-only histories, manual reports/atlases, temporal and objective-alignment campaigns.'},
        'diagnostic_experiments':6,'new_mask_classifier_fits':4,'current_row_readout_fits':2,
        'readout_controls':'One matched unweighted fit reproduced original coefficients; three prospectively selected variants checked.',
        'diagnostics':diagnostics,'candidate_checks':checks,
        'best_new_observed':{'name':best_name,'wp':best['search']['candidate']['weighted_pearson'],'delta':best['search']['delta_combined'],
                             'paired99':best['paired99'],'status':'exploratory_only'},
        'best_historical_observed':{'wp':0.6590796311188727,'status':'exploratory_only; original adaptive99 interval crossed zero'},
        'best_credible_candidate':{'name':'frozen practical incumbent','wp':SEARCH_ROOT_WP},
        'awaiting_promotion':awaiting,
        'mechanisms_supported':['WP selection over residual MSE','Sequence-level uncertainty for directional diagnosis',
            'Population-dependent t1 correction efficacy in reused search','Nonlinear current-input mask prediction improves within-search t0 population representation'],
        'mechanisms_weakened':['Point-gradient disagreement proves stable transfer conflict','Clipping/outlier sequences alone explain t1 contrast',
            'Linear causal proxy is an adequate official selection substitute','Nonlinear mask weighting alone establishes a better fixed current-row correction',
            'Objective misalignment alone is the main remaining bottleneck'],
        'remaining_modeling_uncertainty':['Conditional target shift is not proved by the matched contrast; its intervals include zero.',
            'Broader causal representation and architecture families remain inadequately tested, not ruled out.',
            'The exact causal information limiting predictions is still unidentified.'],
        'limiting_performance_assessment':'Most directly evidenced limitation is reliable selection and transport across populations, compounded by low effective statistical resolution of a repeatedly reused64-sequence search. This identifies a research bottleneck, not a unique irreducible model ceiling.',
        'strongest_next_question':knowledge['next_research_question'],
        'ending_reason':'The justified nonlinear-weighting experiment completed; no candidate qualified and the follow-up source-to-official transport rule failed. Current evidence does not justify more weight, penalty, objective or training-size tuning. Untested architectures require a new measurable causal-information hypothesis rather than a blind sweep.',
        'credits_exhausted_claim':False,'elapsed_seconds':elapsed,
        'new_papers':[{'id':c['card_id'],'url':c['source_url']} for c in cards if c['card_id'] in ['model_selection_variance','cross_validation_uncertainty','adaptive_analysis_validity','domain_adaptation_identifiability']],
        'library_version':'v8','library_manifest_sha256':library_hash,
        'infrastructure_repairs':['Legacy null/missing search-record fields handled with exact metric-domain/finite guards.',
            'Search-population bootstrap now uses shared sequence draws; original invalid intervals preserved separately.',
            'Persistent manual-search lessons now reach MLEvolve planner context without rescoring its historical tree root.',
            'Native frozen mask trees reproduced in isolated training without new dependencies or source-guard relaxation.'],
        'regression_tests':{'linux_passed':59,'failures':0,'cache_provider_disabled':True},
        'deployment_cost':{'callback_us_range':[min(r['callback_checks']['callback_us_per_row'] for r in checks.values()),max(r['callback_checks']['callback_us_per_row'] for r in checks.values())],
            'mask_predictor_callback_cost':0,'causality_reset_determinism_parity_finiteness_checks_passed':True},
        'boundary_audit':{'protected_evaluation_run':False,'protected_cache_read':False,'submission_performed':False,
            'incident':'Initial inventory read competition_engineering/ledger.json, which mixed historical search and protected entries. This file was excluded thereafter from synthesis/evidence/experiment selection. No protected feedback was used to choose experiments.',
            'training_audit':'512 sequences disjoint from local correction caches; no claim of independence from undisclosed pretrained-GRU data. No true scoring mask exists on these training rows.'},
        'reproducibility_sha256':{str(p.relative_to(campaign.ROOT)):campaign.sha(p) for p in evidence},
        'ledger_prefix_sha256':campaign.sha(out/'ledger.jsonl')}
    # The actual filtered history count, rather than status=success, is authoritative.
    history=json.loads((out/'mlevolve_search_history.json').read_text())
    result['history_review']['routine_metric_records']=sum('combined_WP' in r['search_scores'] for r in history)
    write_json(out/'FINAL_REPORT.json',result)
    lines=['# Connectome search-only reassessment — 2 October 2026','',
        f'Practical incumbent retained: **{SEARCH_ROOT_WP:.9f}**, SHA256 `{COMBO_SHA}`. No candidate awaits promotion.',
        '', 'The campaign reviewed82 MLEvolve attempts, the dedicated manual search history and candidate atlases, temporal negatives, training-size/calibration/nonlinear findings, the60-fit objective-alignment campaign and29 existing research cards before experimentation.',
        '', '## Experiment trajectory','',
        '| Experiment | Observation and reasoning | Result and decision |','|---|---|---|',
        '| Gradient uncertainty | Point cosines had been used to reject transfer without sequence uncertainty. | Search split-half median cosine0.157 for t0,0.073 for t1;99% intervals include both signs. Point-only rejection weakened. |',
        '| Prospective training selection | Separate256 selection and256 audit sequences excluded local correction caches. Freeze six historical candidates; use uniform, causal proxy and random proxy masks. | All chose raw t1/4096. Audit gains+0.000425,+0.000242,+0.000356 respectively; all99% intervals cross zero. Proxy pools remain diagnostics. |',
        '| Search population audit | Compare identical frozen candidates across uniform and official masks with shared sequence bootstrap. | t1/1024 official-minus-uniform gain−0.001074,99%CI[−0.001806,−0.000433]; t1/4096−0.001521,CI[−0.002662,−0.000488]. t0 contrasts inconclusive. |',
        '| Population mechanism | Test clipping, within/between moments, influential sequences and fixed position thirds. | t1 conflict persists without clipping, within sequences, after any single deletion and after five largest-influence deletions; greatest deterioration in middle scored rows. |',
        '| Nonlinear scoring predictor | Test a single64-tree/15-leaf classifier with four sequence folds and scoring indicators only. | AUC0.8864 versus0.8155; log loss0.1977 versus0.2473. t0 alignment0.703,99%CI[0.239,0.908]; improvement CI[0.033,0.862]. t0 fit justified; t1 improvement rule failed. |',
        '| Weighted t0 readout | Fixed penalty1.0,1,024 training sequences, two matched coefficients, training-only uniform/focused WP strength selection. Mask trees used offline only. | No qualifying candidate. Matched control reproduced. Details below. |',
        '| Source transport follow-up | Does improved within-search mask representation carry from training to official search? | Weighted t0 source alignment0.477,99%CI[−0.287,0.699]; improvement CI[−0.218,0.409]. No directional-fit launch. Training stability improved, so further scaling is not independently justified. |',
        '', '## Checked candidates','',
        '| Candidate | Official search WP | Delta | Paired99% interval | CPU µs/row |','|---|---:|---:|---|---:|']
    for name,r in checks.items():
        lines.append(f"| {name} | {r['search']['candidate']['weighted_pearson']:.9f} | {r['search']['delta_combined']:+.9f} | [{r['paired99'][0]:+.7f}, {r['paired99'][1]:+.7f}] | {r['callback_checks']['callback_us_per_row']:.2f} |")
    lines +=['', 'Best new observed point: **0.658846790**, an unweighted readout whose strength was selected using nonlinear-proxy WP. Its gain+0.000011468 is exploratory only. The historical0.659079631 point also remains exploratory. The best credible practical candidate remains the frozen0.658835322 combo.',
        '', '## What now appears limiting','',result['limiting_performance_assessment'],
        '', 'Training and all-required search WP differ materially from official selected-population WP. A mask classifier can predict selection well yet fail to reproduce correction behavior. The t1 contrast is stable under several fixed diagnostic comparisons; it does not prove full conditional label shift, mask causality, or a universally useful inverse-weighting solution. The matched residual contrast remains uncertain.',
        '', 'The new weighted model did not materially change residual/position errors; t1 was unchanged, while small middle t0 gains were offset by late deterioration. Objective/loss and old temporal grids stayed closed. Architecture families not adequately tested remain open questions, not scientific negatives.',
        '', '## Research process and repairs','',
        'The persistent WP-selection lesson and weakened objective-misalignment hypothesis were stored before synthesis. Manual research knowledge now appears in MLEvolve planning context; its historical scored root is unchanged. This corrects context anchoring without fabricating a tree score.',
        '', 'Legacy optional fields caused two synthesis failures and were repaired. An initial diagnostic used independent bootstrap draws for correlated search populations; its intervals were preserved as invalid implementation evidence and replaced by a paired version. All59 Linux regression checks passed, including metric-domain guards, paired identical-population resampling, analytic derivatives, common-population matching, within-sequence moments, fresh group partitioning, mask-tree parity/causal-prefix checks, source safety and ONNX callback checks.',
        '', 'No protected evaluation or submission ran. Boundary incident: the initial inventory read a legacy ledger containing mixed historical search and protected entries. It was excluded thereafter from research synthesis and decision evidence. Dedicated search-only files supplied the experiment history.',
        '', 'All intervals are descriptive on reused search and do not correct adaptive experimentation. Fresh training audit sequences are correction-disjoint, but the supplied GRU training provenance is undisclosed. Proxy masks are not the official score and do not create an independent official validation set.',
        '', '## Primary papers and remaining question','']
    lines +=['- ['+c['card_id']+']('+c['source_url']+')' for c in cards if c['card_id'] in [p['id'] for p in result['new_papers']]]
    lines +=['', 'Library v8 preserves previous cards and adds four primary-source cards. Publication assumptions, task limitations and curator inference are distinguished. Domain-adaptation impossibility results are not claimed to prove this task impossible.',
        '', '**Strongest next research question:** '+result['strongest_next_question'],
        '', result['ending_reason'],
        '', 'Stopped because no further fit met current scientific prerequisites, not because credits were proved depleted. Six diagnostic experiments, four sequence-fold mask-classifier fits and two readout coefficient fits completed.',
        '', '## Reproducibility','',f'Library manifest SHA256: `{library_hash}`. Immutable sources are in `source_snapshots/`; exact config/model/archive/input-identity/card hashes and complete candidate error atlases are in `FINAL_REPORT.json`. The hash-chained ledger records plans, freezes, results, repairs and decisions.',
        '', '| Key artifact | SHA256 |','|---|---|']
    for relative in ['campaign.json','research_state_map.json','nonlinear_weighted_t0/protocol.json','nonlinear_weighted_t0/isolated_training/fit_identity.json','nonlinear_mask_trees.npz','knowledge_v3.json','library_v8_manifest.json']:
        lines.append('| `'+relative+'` | `'+campaign.sha(out/relative)+'` |')
    lines.append('')
    # Improve spacing in narrative numerals without altering artifact identifiers.
    text='\n'.join(lines)
    for a,b in [('reviewed82','reviewed 82'),('the60-fit','the 60-fit'),('and29','and 29'),('Separate256','Separate 256'),('and256','and 256'),('cosine0.','cosine 0.'),('t0,0.','t0, 0.'),(';99%','; 99%'),('all99%','all 99%'),('gain−','gain −'),('alignment0.','alignment 0.'),('log loss0.','log loss 0.'),('AUC0.','AUC 0.'),('versus0.','versus 0.'),('penalty1.0','penalty 1.0'),('1.0,1,024','1.0, 1,024'),('single64-tree','single 64-tree'),('All59','All 59'),('historical0.','historical 0.'),('frozen0.','frozen 0.'),('gain+0.','gain +0.'),('Audit gains+','Audit gains +')]:
        text=text.replace(a,b)
    (out/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    campaign.event('campaign_ended',reason=result['ending_reason'],incumbent_sha256=COMBO_SHA,awaiting_candidates=awaiting,
        report_sha256=campaign.sha(out/'FINAL_REPORT.md'),review_sha256=campaign.sha(out/'FINAL_REPORT.json'),regression_tests_passed=59)
    write_json(out/'CHECKPOINT.json',{'status':result['status'],'incumbent':SEARCH_ROOT_WP,'awaiting':awaiting,
        'report_sha256':campaign.sha(out/'FINAL_REPORT.md'),'review_sha256':campaign.sha(out/'FINAL_REPORT.json'),
        'ledger_sha256':campaign.sha(out/'ledger.jsonl')})
    print(json.dumps({'status':result['status'],'best_new':result['best_new_observed'],'awaiting':awaiting,'library':library_hash}),flush=True)

if __name__=='__main__':
    main()
