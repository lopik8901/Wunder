# Connectome research handoff — 3 October 2026

This is the canonical cross-device research entrypoint. Read this document,
`handoff/research_state.json`, `handoff/data_roles.json`,
`handoff/experiment_ledger.jsonl`, `handoff/mlevolve_attempts.json`, and the
v16 research library before proposing work. Follow `../CONTINUE_RESEARCH.md`.
The research session is ended. Bootstrap must summarize and wait for the user's
approval before starting experiments. No candidate awaits promotion.

## Frozen practical incumbent

Search WP **0.6588353223316641** (rounded **0.658835322**), t0
**0.6479917119604004**, t1 **0.6696789327029276**. These are historical results
on the reused64-sequence official search population, not independent validation.

| Artifact | Repository-relative external path | SHA256 |
|---|---|---|
| Frozen coefficients | `competition_engineering/deployment/manual_targetwise_combo_v1/combo.npz` | `7aa3204522ef118f664e6677f5e9a6ed5ad296b0e9466d9884906abc5b8ddb0a` |
| Existing archive | `competition_engineering/submissions/manual_targetwise_combo_v1.zip` | `c8acc7153f60b7540c3b988877f30af8a1ec799d87026d09b46a9c3a37521076` |

The full external manifest records the ONNX, callbacks and every required
dependency hash. These binaries are **not in Git**. Transfer the small
`competition_engineering/handoff_external/connectome_required_artifacts.zip`
separately and verify it. This is a transfer bundle, not a new competition
submission or a replacement incumbent.

Pipeline:112 float32 current inputs feed the supplied forward two-layer GRU,
with128 hidden units per layer. All rows, including99 warm-up rows, update state;
reset on each sequence. Affine base calibration is scale `[0.85,0.5]`, bias
`[-0.25,0.25]`. A115-feature current readout uses intercept, normalized/clipped
112 inputs and two calibrated predictions. Target0 uses the4096-sequence
magnitude-weighted ridge expert. Target1 uses4096-sequence low/high ridge experts,
switching when `max(abs(root_prediction)) >= 1`. Frozen strengths are1.
The callback returns target0 magnitude output plus target1 gated output.
Preserve its exact float32/double arithmetic and graph/model identities.
Coefficient provenance is in `manual_targetwise_combo.py` and the safe
autonomous-campaign trajectory. Do not replace the historical MLEvolve root's
actual0.6548654996 model/score with this incumbent by relabeling it.

## Scientific trajectory

The portable evidence directory retains hypotheses, specifications, per-target
scores, intervals, training/selection/replication identities, frozen-model
hashes, latency records, repairs and decisions. The exported event ledger
preserves263 source events, with source provenance and its own hash chain.
The82-attempt MLEvolve history is an explicit search-only retrospective export,
not raw prompts or mixed supervisor feedback. The early pilot was contaminated
by a feedback-boundary incident and must not be called independent clean evidence.
Historical negatives are retained, not deleted. Exact frozen source versions
are deduplicated under `handoff/source_snapshots/` by SHA256.

| Family / campaign | Main finding | Interpretation |
|---|---|---|
| Historical MLEvolve ridge/calibration/TCN and manual lag, histogram, quadratic, random-feature, multiscale GPU, EMA, magnitude, gating, mask and spline fits | Magnitude and gated target-specific linear fits formed the incumbent. Other local families did not establish a stronger practical candidate. | Reused-search trajectory; some failures are infrastructure, not mechanism rejection. See82-attempt ledger and autonomous report. |
| Additive memory / diagonal bilinear memory | Full-callback WP0.658397455 /0.658244825, below incumbent; additive99% interval crosses zero, bilinear95% below zero. | Do not continue the tested temporal mechanisms without new evidence. Execution rehearsals were repaired separately. |
| Objective alignment |60 fits in8 suites: raw/clipped residual objectives, Pearson tangent/direct WP, group-risk and exact-moment counterfactuals. Objective-aware selection was useful but no convincing replacement. | Broad loss/penalty sweeping is closed. This does not reject objective-aligned selection. |
| Reassessment / population transport | Point-gradient uncertainty is large; stable t1 uniform-versus-official contrast survives clipping, within-sequence and influential-sequence controls. Better scoring-mask prediction does not ensure correction transport. | No established unique label-shift mechanism or universal weighting fix. Proxy masks are diagnostics. |
| Root-cause campaign | Mean-one weights remove arbitrary regularization-scale confounding. Exact ONNX tree export cuts correction cost192.53→3.46µs, full callback68.50µs. Training-selected tree strength regressed search. | Infrastructure/execution feasibility improved; score did not. Compact trees are not intrinsically too slow. |
| Frozen GRU state readouts | Positive correction signal replicated on training-derived populations (example proxy gain0.000859,99%[0.000234,0.001471]); official-search transfer remained essentially zero/uncertain. | States contain correction information beyond current inputs; useful development information is not an official-score win. |
| Mean/consensus ensembles and position | Consensus search0.659072103,99% interval crosses zero, callback77.49µs exceeds practical limit. Disagreement threshold loses to matched-amplitude control on fresh replication; position fails replication. | Ensemble point does not validate disagreement mechanism; position/consensus tuning closed. |
| Nonlinear frozen state | Fixed joint tanh, learned32-unit head, shallow state trees and one-step innovations did not establish robust incremental utility beyond matched linear controls. One qualified random probe scored0.658388773 and80µs on search. | Lower training MSE and hidden-feature usage are not a mechanism result; not proof that every nonlinear family is exhausted. |
| Fitting-sequence diversity |1024×128 versus2048×64, two seeds, same131072-row budget/model. Initial mean proxy contrast−0.000016898,95%[−0.000814,+0.000754]; composition-balanced mean+0.000308696,95%[−0.000174,+0.000810]. | **Inconclusive**, including the balanced follow-up. Both gates failed; not proof that diversity never helps.1024 fresh replication sequences consumed. |
| Alternative representations | Random86-row TCN, exact delays/current nonlinear controls,512-row order4 Legendre memory, learned matched temporal/static encoders, independent decoder, raw-target supervision. All five stage gates failed on separate fresh256-sequence blocks. | Random probes do not reject trained architectures generally. Trained temporal models overfit and failed matched transfer; decoder/supervision rescues did not fix complementarity.1280 fresh replication sequences consumed. |
| Prefix context | Feature-only first512-row mean/log-std descriptor64→later WP gradients. Development error reductions−6.91%,−12.98%,−16.65%,−10.41%; both halves negative. | Close this predictor. Interleaved reliability~0.88 uniform/~0.65–0.68 proxy is descriptive and temporally dependent; not an irreducible-noise estimate.512 replication sequences preserved. |
| Prediction calibration | Historical seven-knot experiment found in source audit, so no claim the family was new. Fresh two-way192-sequence crossfit with fixed bounded17-knot curve: uniform−0.001590/−0.000777; proxy−0.000139/+0.000032 versus affine. Own-fitting gains disappear across sequences. | Failed signal gate before selection/development access; no curve rescue sweep. Small parameter count does not ensure transfer. |
| Bid/ask semantic asymmetry | Eight bounded side summaries versus state/current and common-mode controls. Primary uniform/proxy+0.000157/+0.000104,95% intervals cross zero. Repeat+0.000417/+0.000241 also uncertain. | Primary gate failed. Matching the selection-frozen strength vector removes the repeat's apparent feature advantage; both fits trail both controls. No replication consumed. |

Learned temporal fitting gains~0.08–0.09 unshrunk proxy WP contrasted with
development losses~0.054–0.056. Better residual MSE, poor-row MSE, fitting WP,
standalone decoding and apparent ensemble diversity can coexist with worse
incremental WP. Original-target and independent-decoder probes help identify
representation/readout dependence, but did not establish an incumbent complement.
Standalone figures from different populations are not paired causal estimates.

## Exploratory maxima and uncertainty

| Historical official-search point | Why it does not replace the incumbent |
|---|---|
|0.659079631119|Adaptive exploratory maximum; paired99% delta[−0.000584018,+0.001074456].|
|0.659072103381 consensus|99% delta[−0.0008066,+0.0011116], mechanism failure and77.49µs callback.|
|0.658882537421 compiled historical tree|99% delta[−0.0002873,+0.0003637]; repaired execution is not a newly demonstrated gain.|
|0.658846790151 proxy-selected linear readout|Gain0.000011468,99% delta[−0.00012299,+0.00013893].|

Recent development-only maxima are **not competition scores**. Fixed TCN's
primary proxy gain+0.000432463 failed repeatability/GRU controls. The side-feature
repeat's+0.000241135 proxy gain did not pass its interval or primary gate and
was explained by strength-selection differences. All intervals on reused search
are descriptive; they do not correct adaptive experimentation.

## Data state and boundaries

`handoff/reservations/untouched_512.json` is the **exact byte-for-byte** frozen
causal-context protocol, with row-group/sequence-ID assignments. Its SHA256 is
in `handoff/data_roles.json` and the artifact manifest. It reserves
`replication_1`256 and `replication_2`256. Neither was feature/target prepared,
fit, selected on or evaluated. Identity-only metadata checks and opaque dataset
checksums do not expose outcomes or consume these pools.

An additional older256-sequence `future_reserved` block also remains reserved;
it is separate from the512 and must not be silently reallocated.
There are1391 train row groups outside all documented assignments/exclusions;
reserve any future roles prospectively, including disjoint sequence IDs, before
looking at their outcomes. The4096 earlier research pool has no never-observed
replication groups left. Recent shape fitting384 and side-selection128 /
development256 have been consumed. Same fitting groups were deliberately reused
for the two mechanisms; side selection/development were unread at preregistration.

The dataset registry contains exact legacy `tune`64 / `confirm`64 validation
IDs and protected512 training-role IDs. All other validation groups are
protected. The official tune64 is heavily adaptively reused and cannot be
relabelled independent validation. Training-derived proxy WP uses a frozen
selection teacher, not the official scoring mask. Correction-disjoint roles do
not imply independence from supplied GRU pretraining, whose provenance is unknown.

No protected outcomes are included here. The raw root ledger and old supervisor
records contain mixed historical protected information; **do not use them as
research memory**. Historical protected exposure status was not audited for this
handoff: never assume a protected pool is untouched or reset counters. Protected
evaluation/promotion/full validation/submission requires separate user approval
and supervisor-owned evidence. Recent campaigns and this handoff ran none of
those operations. No candidate currently awaits promotion.

## Constraints and infrastructure

Global WP clips y and predictions to±2 and weights by `abs(clipped y)`, pooling
moments over selected rows before taking each target correlation and averaging.
Do not average sequence scores. Candidate inference is causal and CPU-only,
one vCPU,16GB RAM,20MB archive,60-minute whole-test runtime. The practical local
callback guard is**76.9275378083µs/row**, tighter than the infrastructure90µs
failure cap. Measure the entire isolated callback, not only the correction,
and serialize benchmarks. Timing is machine-dependent, not transferable proof.

Production generated fits require Linux bubblewrap/user namespaces, network
disabled, only the designated fitting/selection cache mounted; replication,
search labels, project tree, other workspaces and research library stay outside.
No insecure Windows/native fallback is permitted. Windows synthetic fixtures
test code only. CPU threads/worker memory/process/deadline limits remain enforced.

Important preserved fixes:

- Fused/bound-buffer ONNX removes callback overhead while parity preserves the
  frozen mathematical candidate; correction-only timing previously understated cost.
- Worker EOF now preserves sanitized stderr/watchdog reason instead of hiding
  the implementation cause. Serialize sandbox jobs to avoid namespace exhaustion.
- The exact transient pre-worker namespace allocation error gets at most3 retries
  under the same overall deadline/mount/network restrictions; candidate and
  permission errors are never retried. Failed scientific source hash is preserved.
- Paired bootstrap draws for compared populations, uncertainty on gradients and
  matched mean-one weight normalization repair misleading diagnostics.
- Sequence Parquet statistics may be absent: read/validate the ID column;
  reject mixed IDs rather than inventing an identity.
- Early sparse-convolution padding, state reset, causal prefix/stream parity and
  polynomial ZOH/filter parity are regression checked.
- Diversity trainer selection variables no longer overwrite fitting baselines;
  unequal fit/selection-length end-to-end regression catches the original error.
- Research-card revisions validate before publishing; duplicate-source rejection
  is not a scientific negative. Closed checkpoints verify immutable knowledge
  snapshots rather than the advancing latest knowledge file.
- MLEvolve code-only valid replies are accepted; draft locks only claim actual
  work and release on success/failure. Root model identity stays actual.
- Historical calibration omitted from the knowledge index is now recorded;
  matched-strength controls distinguish selector effects from feature gains.

Outstanding technical limitations: legacy launch configs retain an older scored
root and are not a resume plan for the practical incumbent; no verified in-place
MLEvolve journal resume exists. Historical full-run hash verifiers require the
original large local run/caches, which Git intentionally omits. The portable
verifier checks exported evidence/provenance and required external artifacts;
it does not pretend to validate absent bulk caches. Optional original artifacts
are listed separately. GPU software/device setup is optional and machine-specific;
CPU bootstrap is sufficient for integrity/regression. Do not rerun old finalize,
tests-recording, prepare or train phases in frozen directories.

## Durable knowledge and what remains open

Accumulated knowledge is**v25**, research-card library**v16**,39 primary/concept
cards, with earlier immutable revisions preserved. Important foundations include
covariate-shift importance weighting/identifiability, model-selection variance,
frozen representation probes, functional objective derivatives, deep ensembles,
random features, TCN architecture, LMU/ZOH dynamics, correlation ratio and queue
imbalance. Cards specify paper claims versus curator inference and task limits.
Particularly, signed anonymous volume-like values and unordered column identifiers
do not identify physical positive queue sizes, best quotes or physical order flow.

The recent local-correction neighborhood around the incumbent appears
**increasingly saturated**. This is an evidence-based prioritization judgment,
not a proven score ceiling, noise bound or exhaustion of compute/data.
Materially different causal representations remain open if a measurable
hypothesis supports them; arbitrary GRU/TCN/SSM selection is not the next answer.

Do not repeat without new measured mechanism evidence: broad loss/penalty grids;
tested additive EMA/diagonal bilinear memory; disagreement shrinking; position
gating; frozen-state nonlinear/readout tweaks; fixed-budget sequence-count tuning;
the tested86-row temporal/512-row polynomial mechanisms; decoder/supervision
rescues; mean/log-std prefix-gradient conditioning; prediction-only spline
calibration; tested eight volume-side aggregates. This closes specific tested
mechanisms, not all architecture, diversity, context or microstructure families.

Strongest still-open questions, not prescriptions:

1. Which **observable causal signal** predicts a stable incremental WP direction
   on correction-disjoint sequences after controlling for current inputs and
   frozen state? Require an explicit matched mechanism control.
2. Can label-free feature semantics or a materially different representation
   identify a transferable mapping missing from current features, without
   guessing anonymous physical units or tuning old temporal widths/windows?
3. Why does development/proxy correction utility fail official-population
   transport? Distinguish measurable selection/population effects from model
   selection noise; do not adapt to the reused official search population.
4. Can a prospective selection rule establish improvement independent of
   strength/seed fluctuations? Matched-strength counterfactuals are explanatory
   diagnostics, not permission to rescue a failed primary after seeing results.

No new model experiment is currently preselected or authorized during bootstrap.
Reserve roles, preregister an informative mechanism and success/failure gates,
use WP-aligned selection, freeze the candidate, and only then consider consuming
the untouched pool. Failed gates require diagnosis and knowledge updates rather
than tiny rescue sweeps. Wait for the user's approval before beginning that work.
