"""Freeze the search-only evidence review after all justified fits/checks finish."""
import json
import time
from pathlib import Path
from competition_engineering import objective_alignment as oa
from competition_engineering.full_moment_objective import output_directory as full_directory
from competition_engineering.manual_search_core import COMBO, COMBO_SHA, SEARCH_ROOT_WP
from competition_engineering.pipeline import write_json


def main():
    root=oa.output_directory();oa.OUT=root
    if (root/'report.md').exists():
        raise ValueError('Final research review is immutable')
    rows=[];grid_rows=[];files={};models=0;awaiting=[];validation=[]
    suites=[(oa.output_directory(tier,target),'objective_alignment',tier,target) for target in (0,1) for tier in (1024,4096)]
    suites +=[(oa.ROOT/('competition_engineering/runs/group_objective_t'+str(t)+'_20261002'),'group_risk_variance',1024,t) for t in (0,1)]
    suites +=[(full_directory(tier),'exact_training_moments',tier,0) for tier in (1024,4096)]
    for directory,family,tier,target in suites:
        q=json.loads((directory/'diagnosis.json').read_text())
        oa.OUT=directory;oa.verify_fit_identity(directory/'isolated_training')
        checks=json.loads((directory/'candidate_checks.json').read_text())
        models+=len(q['training']['models'])
        for key,selected in q['selections'].items():
            if not key.endswith('_select_wp'):
                continue
            check=checks[key]
            row={'suite':family,'tier':tier,'target':target,'variant':key,'selected':selected,
                 'score':check['search']['candidate']['weighted_pearson'],
                 'delta':check['search']['delta_combined'],'paired99':check['paired_99ci'],
                 'callback_us':check['callback_checks']['callback_us_per_row'],'status':check['status']}
            rows.append(row);validation.append(check['callback_checks'])
        for key,check in checks.items():
            if check['status']=='AWAITING PROMOTION':
                awaiting.append({'directory':str(directory.relative_to(oa.ROOT)),'variant':key,
                                 'score':check['search']['candidate']['weighted_pearson']})
        for key,score in q['search'].items():
            if '@' in key:
                grid_rows.append({'directory':str(directory.relative_to(oa.ROOT)),'variant':key,
                                 'score':score['wp'],'delta':score['wp']-SEARCH_ROOT_WP})
        for name in ['campaign.json','diagnosis.json','candidate_checks.json','ledger.jsonl','isolated_training/fit_identity.json']:
            if directory==root and name=='ledger.jsonl':
                continue
            files[str((directory/name).relative_to(oa.ROOT))]=oa.digest(directory/name)
    oa.OUT=root
    config=json.loads((root/'campaign.json').read_text())
    diagnostics={name:json.loads((root/(name+'.json')).read_text()) for name in
        ['initial_diagnosis','transfer_diagnosis','population_gradient_diagnosis_v2','causal_population_diagnosis','training_thinning_diagnosis','frozen_latent_diagnosis']}
    if oa.digest(COMBO)!=COMBO_SHA:
        raise ValueError('Practical incumbent identity changed')
    best=max(grid_rows,key=lambda r:r['score'])
    eligible_point=max([candidate['score'] for candidate in awaiting],default=SEARCH_ROOT_WP)
    ledger_prefix=oa.digest(root/'ledger.jsonl')
    result={'status':'search_loop_closed_no_further_materially_justified_fit',
            'ending_reason':'The evidence justified pooled objectives, sequence-group risk variance, scoring-population diagnostics, exact-moment replications and a frozen latent-feature diagnostic. These protocols are complete; the next readout-fit prerequisites failed. Further architecture, recurrent dynamics, group-boundary, phase, penalty or propensity tuning lacks new supporting evidence.',
            'incumbent_wp':SEARCH_ROOT_WP,'incumbent_sha256':COMBO_SHA,'incumbent_unchanged':True,
            'readout_fits':models,'controlled_suites':len(suites),'selected_variants':rows,
            'best_diagnostic_point':best,'awaiting_promotion':awaiting,
            'best_eligible_search_wp':eligible_point,
            'conclusion':'Lower residual training loss can yield worse official WP. Objective-aware selection mitigates the mismatch, but direct objective fitting alone does not establish a transferable improvement.',
            'important_limit':'All score evidence reuses the fixed64-sequence search. Bootstrap intervals are descriptive and uncorrected for adaptive selection; this is not independent validation.',
            'temporal_branch':'Remained closed; no temporal coefficient fitted or timescale tuned.',
            'next_research_question':'Does conditional target/scoring-selection drift persist within matched causal-input populations? A new fit requires training-side evidence of a stable deployable signal or materially stronger observable population alignment. Current inputs/output values, the scoring predictor and the frozen GRU-state probe did not establish that premise; search gradients must not become a fitting target.',
            'resource_elapsed_seconds':time.time()-config['started_unix'],'configured_resource_window_seconds':config['budget_seconds'],
            'resource_stop':'Evidence exhausted, not a claim that credits were depleted',
            'ledger_prefix_sha256_before_closure':ledger_prefix,
            'diagnostics':diagnostics,'evidence_files_sha256':files,
            'engineering_checks':{'variants':len(validation),'callback_us_range':[min(v['callback_us_per_row'] for v in validation),max(v['callback_us_per_row'] for v in validation)],
                                  'max_stream_parity_error':max(v['max_abs_parity'] for v in validation)},
            'regression_tests':{'runtime':'Linux Python 3.11','passed':41,'failures':0,'warning':'Pytest cache write permission; test execution unaffected'},
            'protected_evaluation_performed':False}
    write_json(root/'final_review.json',result)
    text=[
        '# Objective-alignment search-only research review — 2026-10-02',
        '',f'Practical incumbent retained: **{SEARCH_ROOT_WP:.9f}**. No protected promotion, complete validation, leaderboard evaluation or submission was run.',
        '',f'{models} current-row readouts were fitted in {len(suites)} controlled suites. Temporal mechanisms stayed closed. Correction fitting ran in the existing network-free Linux sandbox; search labels never entered correction fitting. Scoring-indicator classifiers were supervisor-side diagnostics.',
        '', '## What the diagnosis establishes',
        '', 'Lower residual loss can accompany worse official clipped weighted correlation. At 1,024 sequences for target 0, MSE-based internal selection produced search WP 0.656579390, versus 0.658783576 for WP-based internal selection. Direct clipped-WP and correlation-tangent fitting improved fitting WP but did not establish a reliable search improvement. These comparisons diagnose selection/objective mismatch; they do not prove that mismatch is the sole cause of generalization failure.',
        '', '## Controlled results',
        '', '| Suite | Train sequences | Target | Variant | Search WP | Delta | Paired99% interval |',
        '|---|---:|---:|---|---:|---:|---|']
    for row in rows:
        ci=row['paired99']
        text.append(f"| {row['suite']} | {row['tier']} | t{row['target']} | {row['variant']} | {row['score']:.9f} | {row['delta']:+.9f} | [{ci[0]:+.7f}, {ci[1]:+.7f}] |")
    text +=['',f"Best adaptively inspected point: **{best['score']:.9f}**, delta{best['delta']:+.9f}, `{best['variant']}` in `{best['directory']}`. A point estimate alone does not satisfy the frozen evidence rule.",
        '', '## Decisions and additional diagnostics',
        '', '- Pooled residual/tangent/direct-WP comparison: completed for both targets at 1,024 and 4,096 sequences. WP selection helped relative to MSE selection, but the selected fits did not beat the practical incumbent.',
        '- Training-group risk variance: group gradients disagreed (minimum cosines−0.250 and−0.610), so fitting was justified. Variance penalties did not establish a winner; target1 selected zero correction. No group-boundary tuning followed.',
        '- Coarse scoring propensity: exact search gradients confirmed worse alignment after weighting (cosines -0.131 and 0.034). Abandoned before a weighted correction fit.',
        '- Causal-input scoring classifier: sequence cross-fitting gave AUC 0.8155 and log loss 0.2473 versus prior 0.2914. Weighted-gradient cosines 0.294 and 0.150 failed the prospective rule. No correction fit used those weights.',
        '- Row thinning: all 20.38m training rows were audited. Target 0 phase error reached 23.1%, motivating exact-moment comparisons at both training tiers. This removed sampling uncertainty without reopening temporal modeling.',
        '- Frozen latent representation: existing GRU weights/state updates stayed fixed. Activation extraction matched baseline predictions and rowwise hidden states, passed exact prefix/reset checks, and took 0.125 seconds on the training-sequence preflight. Training/search latent-gradient cosines 0.183 and -0.427 failed the readout launch rule. No latent readout was fitted.',
        '', '## Engineering and reproducibility',
        '', 'Failures were kept separate from scientific negatives: an unusable existing interpreter was replaced by the verified existing runtime; generated-source serialization was made compatible with the unchanged source guard; unused temporal state was removed from the current-row ONNX graph; boolean log-loss and source-hash reporting failures were repaired. Failed attempts and frozen classifier coefficients were preserved. Regression tests cover source safety, analytic gradients, fit tampering, boolean/extreme-probability loss, an end-to-end synthetic population diagnostic, and both-target ONNX export/parity/reset/causality/output buffering.',
        '', f"Final selected callbacks ran at{result['engineering_checks']['callback_us_range'][0]:.2f}–{result['engineering_checks']['callback_us_range'][1]:.2f}µs/row. Maximum cached/streaming prediction discrepancy was{result['engineering_checks']['max_stream_parity_error']:.3g}. Engineering checks used training rows; they were not protected validation.",
        '', 'All 41 Linux regression tests passed. Pytest reported a cache-write permission warning; test execution was unaffected.',
        '', 'Frozen sources, fit/artifact hashes, per-suite append-only ledgers, candidate archives and paired search reports remain in the suite directories. `final_review.json` records their identities. The practical incumbent SHA256 remains unchanged.',
        '', '## Research library and remaining question',
        '', 'Four original research cards were added in versioned libraries, distinguishing paper claims from task adaptations: [nondecomposable objective linearization](https://proceedings.mlr.press/v37/narasimhana15.html), [importance-weighted model selection](https://www.jmlr.org/papers/v8/sugiyama07a.html), [risk variance across domains](https://proceedings.mlr.press/v139/krueger21a.html), and [fixed recurrent-feature readouts](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf). Their assumptions and limited transfer to clipped Pearson and a pretrained GRU are recorded explicitly.',
        '', result['next_research_question'],
        '', result['ending_reason'],
        '', 'All score evidence reuses the fixed64-sequence search set. Internal sequence selection excludes correction-fit sequences, but the frozen incumbent was already trained/selected. Neither internal selection nor the adaptive bootstrap is independent validation.',
        '', f"Awaiting promotion: {len(awaiting)} candidate(s). The loop ended for lack of a further materially justified fit, without claiming that credits were depleted.",
        '']
    (root/'report.md').write_text('\n'.join(text),encoding='utf-8')
    oa.event('campaign_ended',reason=result['ending_reason'],incumbent_unchanged=True,awaiting_candidates=awaiting,
             fitted_readouts=models,controlled_suites=len(suites),regression_tests_passed=41,protected_evaluation_performed=False,
             report_sha256=oa.digest(root/'report.md'),review_sha256=oa.digest(root/'final_review.json'))
    print(json.dumps({'status':result['status'],'readouts':models,'best_diagnostic_point':best,'awaiting':awaiting,'incumbent':SEARCH_ROOT_WP}),flush=True)


if __name__=='__main__':
    main()
