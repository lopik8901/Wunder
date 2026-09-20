# WunderNN Connectome Task Contract

Build a deterministic, causal, sequential `PredictionModel` that predicts `t0` and `t1`, two undisclosed indicators of future price movement for instrument `i0`. Instrument `i1` is input context only and has no targets. The exact target semantics are not disclosed.

## Data and sequence contract

- Each independent sequence has exactly 20,000 ordered observations (`step_in_seq` 0 through 19,999).
- Steps 0-98 are a 99-observation warm-up. Process them and update state, but return `None` because `need_prediction` is false.
- Steps 99-19,999 require two finite predictions on every row, including rows that will not be scored.
- A prediction may use only the current observation and earlier observations from the same sequence. Future rows, future-derived preprocessing, and full-sequence fitting at inference are leakage.
- Reset all recurrent, rolling, normalization, and cache state whenever `seq_ix` changes. Never carry state across sequences.
- `seq_ix` values are shuffled identifiers, not chronological values. Do not use them as features, sort keys, or evidence of temporal proximity.

## Features

`DataPoint.state` contains 112 ordered `float32` features. Do not alphabetically sort their names.

| Positions | Documented meaning |
|---:|---|
| 0-10 | `i0_p0`...`i0_p10`, bid price-like |
| 11-21 | `i0_p11`...`i0_p21`, ask price-like |
| 22-32 | `i0_v0`...`i0_v10`, bid volume-like |
| 33-43 | `i0_v11`...`i0_v21`, ask volume-like |
| 44-47 | `i0_dp0`...`i0_dp3`, trade price |
| 48-51 | `i0_dv0`...`i0_dv3`, trade volume |
| 52-62 | `i1_p0`...`i1_p10`, bid price-like |
| 63-73 | `i1_p11`...`i1_p21`, ask price-like |
| 74-84 | `i1_v0`...`i1_v10`, bid volume-like |
| 85-95 | `i1_v11`...`i1_v21`, ask volume-like |
| 96-99 | `i1_dp0`...`i1_dp3`, trade price |
| 100-103 | `i1_dv0`...`i1_dv3`, trade volume |
| 104-111 | `a0`...`a7`, additional anonymized features |

The group labels above are known; finer semantics are not. An index is an identifier, not an order-book depth level: for example, `p0` is not guaranteed to be the best quote and `p10` is not guaranteed to be deepest. Do not invent price ordering, distance-from-best, or depth semantics from feature indices.

## Scoring contract

Validation exposes `is_scored`; test scoring uses a hidden mask. The platform scores only `need_prediction AND is_scored`, but `PredictionModel.predict` never receives `is_scored` or targets. Do not condition inference, feature construction, or model state on the visible validation mask. Predict every required row as though it were scored.

Optimize the exact official Global Weighted Pearson objective. For each target `k` independently, pool every selected row across all sequences, then define:

```text
y = clip(target_k, -2, 2)
p = clip(prediction_k, -2, 2)
w = abs(y)
WP_k = sum(w*(y-weighted_mean(y))*(p-weighted_mean(p))) /
       sqrt(sum(w*(y-weighted_mean(y))^2) * sum(w*(p-weighted_mean(p))^2))
score = (WP_t0 + WP_t1) / 2
```

Clipping applies to both targets and predictions before weighting. Weights are the absolute clipped target values, so zero targets have zero weight. Use global pooled moments, not a mean of per-sequence correlations and not a correlation of concatenated targets. Use float64 metric arithmetic; the official zero-weight/zero-variance threshold is `1e-8`, constant predictions score zero for that target, and numerical results are bounded to `[-1, 1]`. Do not replace, rescale, round, or otherwise modify the official metric for candidate ranking.

## Submission interface and feasibility

The ZIP root must contain `solution.py` defining a no-argument `PredictionModel` with `predict(self, data_point)`. `data_point` provides `seq_ix`, `step_in_seq`, `need_prediction`, and the 112-feature `state`. Return `None` when `need_prediction` is false; otherwise return exactly two finite, `float32`-compatible values of shape `(2,)` in `[t0, t1]` order. Process every callback row and produce identical predictions on repeated runs.

The final solution runs offline in isolated Linux on Python 3.11 with 1 vCPU, 16 GB RAM, no GPU, a 60-minute limit for the entire hidden test set, and a 20 MB maximum submission ZIP. It may use only supplied competition data; all code, weights, and dependencies needed at inference must be packaged and available offline.

## Experiment boundary

- Task context supplies competition knowledge.
- `connectome.evaluate_model(...)` enforces causal callback behavior and exact scoring. Call `connectome.emit_result(result)` so the authoritative `CONNECTOME_RESULT_JSON=` record returns `WP_t0`, `WP_t1`, and their official combined score to MLEvolve.
- Deployment checks must enforce the Python, CPU, RAM, runtime, offline, deterministic, and archive-size constraints before submission.

During MLEvolve experiments, authenticated subset files are available as `./input/train.parquet` and `./input/valid.parquet`. Candidate code must train only from the training file, construct `SequenceData` from validation columns in their stored order, call `connectome.evaluate_model(ModelClass, validation_data)`, and finish with `connectome.emit_result(result)`. The evaluator, rather than candidate code, is authoritative for callback validation and scoring.

Authoritative local references: `wnn_connectome_starterpack/docs/data_overview.md`, `wnn_connectome_starterpack/docs/submission_guide.md`, `wnn_connectome_starterpack/docs/rules.md`, `wnn_connectome_starterpack/docs/faq.md`, `wnn_connectome_starterpack/METRIC.md`, and `wnn_connectome_starterpack/utils.py`.
