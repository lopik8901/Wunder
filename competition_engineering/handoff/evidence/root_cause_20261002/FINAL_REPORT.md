# Connectome research checkpoint — 2 October 2026

**Practical incumbent retained: 0.658835322. No candidate awaits promotion.**

This continuation completed two population diagnostics, two matched ridge fits, an exact tree execution repair, and training-only strength selection on 256 sequences excluded from the tree fit. Prior completed campaigns and their reports were preserved.

**Results**

| Candidate | Search WP | Delta vs incumbent | Paired99% interval | Full callback µs/row |
|---|---:|---:|---|---:|
| capped_normalized | 0.658832680 | -0.000002642 | [-0.0001256, +0.0001069] | 66.46 |
| probability_normalized | 0.658835322 | +0.000000000 | [+0.0000000, +0.0000000] | 64.41 |
| historical_tree_strength 025 | 0.658882537 | +0.000047215 | [-0.0002873, +0.0003637] | 68.50 |
| training_selected_tree_strength1 | 0.658694657 | -0.000140666 | [-0.0013889, +0.0009151] | 71.30 |

The 0.658882537 tree point is historical, now made deployable; its uncertainty still spans zero. The best historical point overall remains 0.659079631, also exploratory with paired99% interval[-0.0005840,+0.0010745]. Neither replaces the incumbent.

**Trajectory and decisions**

1. Reviewed the frozen research map and completed reassessment. WP selection remains established; objective-only and old temporal grids stayed closed.
2. Diagnosed capped scoring weights. Removing clipping and fold-prior scaling raised t0 gradient cosine from 0.703 to 0.786; paired99% improvement interval[0.0030,0.1805]. A matched follow-up isolated the components: each component interval crossed zero, so clipping alone is not established as the cause.
3. Fit two matched t0 readouts with mean-one weights and the same penalty. The probability-weighted training selector chose strength 0; the capped control chose 0.05 and scored a tiny negative delta. Decision: do not continue weight or penalty tuning.
4. Compiled the historical 32-tree correction into one ONNX ensemble operator. Same-row, one-CPU correction timings were 192.53µs for sklearn and 3.46µs for ONNX. Complete causal callback cost 68.50µs; all required checks passed. Decision: remove execution cost as grounds for rejecting this compact tree family.
5. Selected tree strength using probability-weighted pooled WP on 256 correction-disjoint training sequences, 128000 sampled rows. It chose 1.0, with proxy t0 WP gain 0.00084645. Official search combined delta was -0.00014067 and its 99% interval crossed zero. Decision: population transfer remains unresolved across both linear and nonlinear corrections; stop at this checkpoint.

**Root causes and remaining limitation**

Two concrete problems were addressed: estimator API overhead blocked a feasible tree representation, and arbitrary weight scale could change effective ridge regularization. Neither repair established a stronger scorer.
The strongest unresolved limitation is population-dependent correction utility. Better mask prediction and closer population moments do not reliably predict official-scored gains. Reused 64-sequence uncertainty makes small adaptive improvements difficult to establish. This does not prove conditional label shift, an irreducible ceiling, or a unique architectural root cause.

**Repairs and verification**

- Numeric-tree export preserves float32 split routing, including threshold-adjacent values. Full replay maximum absolute differences were below 2.4e-7.
- Mean-one focus normalization makes regularized fits invariant to arbitrary global weight units.
- A duplicate research-card source was rejected. Its incomplete directory is preserved under rejected_library_v9; publishing now validates a staged revision before atomic rename. This is an infrastructure failure, not a scientific negative.
- Closed-campaign verification now checks its hashed knowledge snapshot instead of requiring the mutable latest-knowledge file to remain unchanged.
- 67 Linux regression tests passed, including the new regressions and existing search-boundary, metric, library, history and callback checks.

**Research provenance and boundary**

Library v9 revises the existing [importance-weighted model-selection card](http<LOCAL_PATH> retaining 33 unique papers. No new paper was needed. Its covariate-shift assumption and lack of a clipped-Pearson guarantee remain explicit. Export follows the [official ONNX ensemble operator specification](http<LOCAL_PATH>

No protected evaluation, protected feedback, full validation, leaderboard evaluation or submission was used in this continuation. Training-only selection is correction-disjoint, not an independent assessment of the supplied GRU. Search intervals remain descriptive under adaptive reuse.

**Next question**

Which causal feature information predicts correction utility in the officially scored population across sequences? Compact tree execution is now feasible, but further fits need a transferable signal beyond training/proxy WP.

This is the requested good checkpoint, not a claim that resources or all scientifically justified research directions are exhausted.

**Reproducibility**

Incumbent SHA256: `7aa3204522ef118f664e6677f5e9a6ed5ad296b0e9466d9884906abc5b8ddb0a`.
Library manifest SHA256: `01d0518710aeec60d796ad47de986c30893d66e3b8b461bfe7b60a79b1ad26a9`.
Knowledge snapshot SHA256: `f0c2a6a8ee5d96d48cab12d0bfa9627cf5a38d77c84c9f9b032b99188c7fae07`.

FINAL_REPORT.json contains exact input/config/model/package/source-snapshot hashes, candidate atlases and paired intervals. ledger.jsonl contains the hash-chained decisions. Verify with `python -m tools.verify_root_cause_checkpoint`.
