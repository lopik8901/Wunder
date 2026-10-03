# Connectome search-only reassessment — 2 October 2026

Practical incumbent retained: **0.658835322**, SHA256 `7aa3204522ef118f664e6677f5e9a6ed5ad296b0e9466d9884906abc5b8ddb0a`. No candidate awaits promotion.

The campaign reviewed 82 MLEvolve attempts, the dedicated manual search history and candidate atlases, temporal negatives, training-size/calibration/nonlinear findings, the 60-fit objective-alignment campaign and 29 existing research cards before experimentation.

## Experiment trajectory

| Experiment | Observation and reasoning | Result and decision |
|---|---|---|
| Gradient uncertainty | Point cosines had been used to reject transfer without sequence uncertainty. | Search split-half median cosine 0.157 for t0, 0.073 for t1; 99% intervals include both signs. Point-only rejection weakened. |
| Prospective training selection | Separate 256 selection and 256 audit sequences excluded local correction caches. Freeze six historical candidates; use uniform, causal proxy and random proxy masks. | All chose raw t1/4096. Audit gains +0.000425,+0.000242,+0.000356 respectively; all 99% intervals cross zero. Proxy pools remain diagnostics. |
| Search population audit | Compare identical frozen candidates across uniform and official masks with shared sequence bootstrap. | t1/1024 official-minus-uniform gain −0.001074,99%CI[−0.001806,−0.000433]; t1/4096−0.001521,CI[−0.002662,−0.000488]. t0 contrasts inconclusive. |
| Population mechanism | Test clipping, within/between moments, influential sequences and fixed position thirds. | t1 conflict persists without clipping, within sequences, after any single deletion and after five largest-influence deletions; greatest deterioration in middle scored rows. |
| Nonlinear scoring predictor | Test a single 64-tree/15-leaf classifier with four sequence folds and scoring indicators only. | AUC 0.8864 versus 0.8155; log loss 0.1977 versus 0.2473. t0 alignment 0.703,99%CI[0.239,0.908]; improvement CI[0.033,0.862]. t0 fit justified; t1 improvement rule failed. |
| Weighted t0 readout | Fixed penalty 1.0, 1,024 training sequences, two matched coefficients, training-only uniform/focused WP strength selection. Mask trees used offline only. | No qualifying candidate. Matched control reproduced. Details below. |
| Source transport follow-up | Does improved within-search mask representation carry from training to official search? | Weighted t0 source alignment 0.477,99%CI[−0.287,0.699]; improvement CI[−0.218,0.409]. No directional-fit launch. Training stability improved, so further scaling is not independently justified. |

## Checked candidates

| Candidate | Official search WP | Delta | Paired99% interval | CPU µs/row |
|---|---:|---:|---|---:|
| unweighted_focused | 0.658846790 | +0.000011468 | [-0.0001230, +0.0001389] | 68.89 |
| unweighted_uniform | 0.658783576 | -0.000051746 | [-0.0007175, +0.0005809] | 65.74 |
| nonlinear_weighted_focused | 0.658830952 | -0.000004370 | [-0.0001301, +0.0001074] | 68.27 |

Best new observed point: **0.658846790**, an unweighted readout whose strength was selected using nonlinear-proxy WP. Its gain +0.000011468 is exploratory only. The historical 0.659079631 point also remains exploratory. The best credible practical candidate remains the frozen 0.658835322 combo.

## What now appears limiting

Most directly evidenced limitation is reliable selection and transport across populations, compounded by low effective statistical resolution of a repeatedly reused64-sequence search. This identifies a research bottleneck, not a unique irreducible model ceiling.

Training and all-required search WP differ materially from official selected-population WP. A mask classifier can predict selection well yet fail to reproduce correction behavior. The t1 contrast is stable under several fixed diagnostic comparisons; it does not prove full conditional label shift, mask causality, or a universally useful inverse-weighting solution. The matched residual contrast remains uncertain.

The new weighted model did not materially change residual/position errors; t1 was unchanged, while small middle t0 gains were offset by late deterioration. Objective/loss and old temporal grids stayed closed. Architecture families not adequately tested remain open questions, not scientific negatives.

## Research process and repairs

The persistent WP-selection lesson and weakened objective-misalignment hypothesis were stored before synthesis. Manual research knowledge now appears in MLEvolve planning context; its historical scored root is unchanged. This corrects context anchoring without fabricating a tree score.

Legacy optional fields caused two synthesis failures and were repaired. An initial diagnostic used independent bootstrap draws for correlated search populations; its intervals were preserved as invalid implementation evidence and replaced by a paired version. All 59 Linux regression checks passed, including metric-domain guards, paired identical-population resampling, analytic derivatives, common-population matching, within-sequence moments, fresh group partitioning, mask-tree parity/causal-prefix checks, source safety and ONNX callback checks.

No protected evaluation or submission ran. Boundary incident: the initial inventory read a legacy ledger containing mixed historical search and protected entries. It was excluded thereafter from research synthesis and decision evidence. Dedicated search-only files supplied the experiment history.

All intervals are descriptive on reused search and do not correct adaptive experimentation. Fresh training audit sequences are correction-disjoint, but the supplied GRU training provenance is undisclosed. Proxy masks are not the official score and do not create an independent official validation set.

## Primary papers and remaining question

- [adaptive_analysis_validity](http<LOCAL_PATH>
- [cross_validation_uncertainty](http<LOCAL_PATH>
- [domain_adaptation_identifiability](http<LOCAL_PATH>
- [model_selection_variance](http<LOCAL_PATH>

Library v8 preserves previous cards and adds four primary-source cards. Publication assumptions, task limitations and curator inference are distinguished. Domain-adaptation impossibility results are not claimed to prove this task impossible.

**Strongest next research question:** Which training-derived causal representation or evaluation population reproduces the stable t1 selected-versus-uniform correction contrast without fitting search-target residuals?

The justified nonlinear-weighting experiment completed; no candidate qualified and the follow-up source-to-official transport rule failed. Current evidence does not justify more weight, penalty, objective or training-size tuning. Untested architectures require a new measurable causal-information hypothesis rather than a blind sweep.

Stopped because no further fit met current scientific prerequisites, not because credits were proved depleted. Six diagnostic experiments, four sequence-fold mask-classifier fits and two readout coefficient fits completed.

## Reproducibility

Library manifest SHA256: `fb42fbd84ca992afccaebb6feec01d725e2bc2d87a235d87dc4d398ace276224`. Immutable sources are in `source_snapshots/`; exact config/model/archive/input-identity/card hashes and complete candidate error atlases are in `FINAL_REPORT.json`. The hash-chained ledger records plans, freezes, results, repairs and decisions.

| Key artifact | SHA256 |
|---|---|
| `campaign.json` | `2d89c30b99b500e59a71f6b32b80f17317bd4e2a7e10f0f9e0e4f41e741f3070` |
| `research_state_map.json` | `e4faf80f96ee78e7d5798ae472466014de22eb06ada4e3d09dc61575022b35cb` |
| `nonlinear_weighted_t0/protocol.json` | `596081c9327dc01d8215b8754f5ef8b26fa284727158cf5c7cf9c26fbf1457b6` |
| `nonlinear_weighted_t0/isolated_training/fit_identity.json` | `935e3b2a95024bfc4a932f64056064d48925fcaefb2bc7eff3117dc56fca875d` |
| `nonlinear_mask_trees.npz` | `0306488a899d5b1460413e96817cd5b616fc2d8ccf11d04aeefc7f19789f637e` |
| `knowledge_v3.json` | `92c3708cca4075b2ea942b371ba0a16932693d3652bca48216fd107d4ee73dfd` |
| `library_v8_manifest.json` | `397047c4d28b3eea9ffa38dd627dc9fb98e2406ab1ef98b718d87c6fef8448cf` |
