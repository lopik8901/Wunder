# Connectome nonlinear-state research checkpoint — 3 October 2026

**Practical incumbent retained: 0.658835322.** No candidate awaits promotion.

Four prospective mechanisms were tested, with11 fitted bivariate model variants and1024 fresh training-derived replication sequences. The frozen GRU was unchanged. Each experiment updated the hash-chained ledger and persistent knowledge before branching.

## Official-search results

| Candidate | WP | Delta | Paired99% delta interval | Callback µs/row |
|---|---:|---:|---|---:|
| tanh_linear | 0.658444206 | -0.000391116 | [-0.0029412, +0.0017267] | 74.81 |
| tanh_current | 0.657527517 | -0.001307805 | [-0.0036150, +0.0003360] | 78.86 |
| tanh_separable | 0.658394429 | -0.000440893 | [-0.0030598, +0.0017655] | 83.07 |
| tanh_joint | 0.658388773 | -0.000446549 | [-0.0031249, +0.0016918] | 80.00 |

Best newly measured point: tanh_linear, WP0.658444206. The practical incumbent remains stronger. Historical0.659079631 and consensus0.659072103 remain exploratory points from earlier campaigns.

## Experimental trajectory

1. **Fixed nonlinear features.** Tested a371-feature linear readout, a current-only nonlinear control, separable current/hidden nonlinearities and128 joint tanh features. Closed-form weighted-ridge solves had normal-equation residuals near1e-15. WP-selected strengths were frozen before replication. Joint gained0.001838 uniform/0.001088 proxy WP; joint-minus-linear was0.000223/0.000020. The permissive prospective gate passed. The one official-search pass found no benefit over linear and no incumbent improvement; joint CPU80.00µs also exceeded76.93µs. Fixed-basis tuning was abandoned.

2. **Learned32-unit nonlinear directions.** Tested current-only, separable and joint heads over fixed linear backbones, keeping the GRU frozen. Later epochs cut joint training MSE from approximately1.331/1.310 to1.062/1.066, yet WP chose epoch5 for both targets. Fresh replication joint-minus-linear was−0.000223 uniform/+0.000079 proxy; joint also lost to separable on proxy rows and one uniform half was negative. Gate failed; official search was not consulted. Optimization succeeded at its loss, while incremental model utility failed to replicate.

3. **Threshold regimes.** Fit32 depth2 histogram trees per target atop fixed linear backbones, with a matched current-only tree control. State models used hidden features in160 of192 splits. Fresh incremental WP over linear was−0.000047 uniform/+0.000013 proxy, with95% intervals spanning zero. Gate failed; official search was not consulted. Tree use of hidden state did not demonstrate useful additional signal.

4. **One-step state updates.** Tested the distinct causal feature h[t]−h[t−1], computed on full sequences before sampling, with zero initial state. A627-feature readout was compared with a matched371-feature current-state control; each selected strength by training-derived WP. Result: No official search; one-step state information did not establish incremental replicated value.

## Fresh replication and uncertainty

| Primary model | Uniform WP delta | Proxy WP delta | Proxy paired95% interval | Search gate |
|---|---:|---:|---|---|
| Random joint | +0.0018378 | +0.0010883 | [+0.0001693, +0.0020170] | pass |
| Learned joint | +0.0009036 | +0.0010479 | [+0.0002839, +0.0017540] | fail |
| State trees | +0.0018285 | +0.0017995 | [+0.0009178, +0.0027035] | fail |
| State update | +0.0024063 | +0.0021089 | [+0.0012306, +0.0030264] | fail |

These deltas compare against the frozen incumbent on the same development population. They are not official-search scores. Incremental contrasts against matched controls, targetwise effects, both sequence halves, paired95/99 intervals and leave-one-sequence-out ranges are recorded in FINAL_REPORT.json. A positive baseline comparison does not establish the proposed mechanism when its matched control performs similarly.

## Lessons and remaining limitation

Frozen states repeatedly support a useful linear development correction. Three nonlinear function classes did not reliably add enough incremental value to explain or resolve official-population transfer. Lower training error remains a poor selection signal, especially for the learned head. The evidence does not establish an exhausted GRU representation, an irreducible score ceiling, or a unique population-shift root cause.

The strongest next hypothesis is insufficient fitting-sequence diversity: learned heads can substantially reduce training error while WP favors early checkpoints. Test sequence count at fixed row budget, architecture and optimizer (1024×128 versus2048×64), with a prospectively reserved unused replication group. This is a falsifiable generalization hypothesis, not a recommendation to enlarge the network or tune this search score.

## Implementation, sources and boundaries

Regression checks: Windows85 passed/1 platform skip; WSL86 passed. Tests cover interaction controls, ONNX parity, exact tree routing, state-update causality/reset, source guards and the failed-replication search block. All actual fits used the existing network-disabled sandbox; replication data was never mounted there.

Library v12 adds the primary [Rahimi–Recht2008 random nonlinear features paper](http<LOCAL_PATH> Existing linear-probe, neural-additive and gradient-boosting cards informed controls and alternatives. All task adaptations and lack of a clipped-WP transfer guarantee are explicit.

No protected promotion, holdout, full-validation, leaderboard or submission results were used or run in this campaign. Prior archive-handoff protected information was excluded from research decisions. The incumbent artifact hash is unchanged. Failed replication candidates have no inferred official score or full-callback latency.

This is a reproducible session checkpoint, not a claim that all scientifically possible research or machine resources are exhausted. Exact artifacts, source snapshots, cache identities, test records and knowledge snapshots are hashed in FINAL_REPORT.json. Verify with `python -m tools.nonlinear_campaign_checkpoint --phase verify`.
