# Causal context diagnostic checkpoint — 3 October 2026

**Best credible candidate remains the frozen incumbent: 0.658835322. No candidate awaits promotion.**

Reserved prospectively: 512 fitting sequences, 128 development sequences and two 256-sequence replication blocks. Only fitting and development were read. Both replication blocks remain untouched, as does the earlier reserved block. Independence concerns new correction fitting and selection; incumbent pretraining exposure is not claimed independent.

The diagnostic tested whether feature-only first-512-row context (32 projected-input means and 32 log standard deviations) predicts sequence-level derivatives of clipped weighted correlation for a fixed current-input correction basis. All assessed rows follow the context prefix. Reference scoring moments and context normalization were estimated on fitting data only. A mean-gradient predictor served as the control. The probability-weighted population is a proxy, not the official scoring mask.

| Population | Target 0 error reduction | Target 1 error reduction |
|---|---:|---:|
| Uniform | -6.91% | -12.98% |
| Probability proxy | -16.65% | -10.41% |

Every point estimate and both development halves worsened. Three of four paired 95% intervals exclude improvement; the uniform target-0 interval crosses zero. The predeclared gate failed, so no replication, conditional model, or official search was run. Complete intervals are in development_diagnostic.json.

Within-sequence interleaved gradient reliability was about 0.88 for uniform weighting and 0.65–0.68 for the proxy. Temporal dependence can inflate this descriptive measure; it is not an estimate of irreducible noise. Stable sequence differences alone do not establish that the tested observable context predicts them.

The exact clipped correlation gradient passed finite-difference regression, including clipping saturation and probability weights. Prefix-context regression covers causality, future independence and reset behavior. Windows and Linux suite output is preserved alongside this report. No implementation failure occurred in this diagnostic.

Checkpoint decision: abandon this specific predictor. Prior temporal, readout and supervision counterfactuals also failed; another architecture or context-window sweep lacks a new mechanism justification. Resources are not claimed exhausted. Preserve independent sequences until evidence identifies a materially different, observable and transferable signal.

No protected evaluation, promotion, full validation, leaderboard evaluation or submission occurred. The incumbent and its archive are unchanged.
