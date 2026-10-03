# Autonomous Connectome research report

Updated: 2026-10-01T17:57:52.607384+00:00

Status: Configured two-hour wall-clock window elapsed; user requested summary. Two distinct scientific experiments completed; frozen incumbent retained.

Starting frozen search benchmark: **0.658835322332**.
All intervals are descriptive paired sequence bootstraps on reused search evidence; adaptive selection is not corrected.
Training and inference use the existing production bubblewrap sandbox. No protected evaluation or submission was performed.

## Current campaign trajectory

| Experiment | Combined WP | Delta | Paired 95% interval | CPU µs/row | Status |
|---|---:|---:|---|---:|---|
| e01_additive_memory | 0.658397454552 | -0.000437868 | [-0.000948046, +0.000059625] | 19.57 | exploratory_search_only; residual only; engineering rehearsal |
| e01e_additive_bound_buffers | 0.658397454535 | -0.000437868 | [-0.000948046, +0.000059625] | 70.17 | exploratory_search_only; full isolated callback |
| e02_bilinear_memory | 0.658244825243 | -0.000590497 | [-0.001187504, -0.000004260] | 71.28 | exploratory_search_only; full isolated callback |

## Interrupted attempts and repairs

- {"error": "ValueError: callback throughput 134.8 us/row exceeds 90 us/row gate", "evidence": "reused_search_evidence_not_independent_validation", "failure_type": "implementation_or_infrastructure", "id": "e01b_additive_isolated", "kind": "experiment_interrupted", "previous_ledger_sha256": "2f2f74f090aea7fa45f845ad87c3a7cfc20f960d98002237f3998f7891343c0b", "seconds": 90.696568212, "utc": "2026-10-01T06:44:58.989363+00:00"}
- {"evidence": "reused_search_evidence_not_independent_validation", "id": "cpu_probe_v1", "kind": "cpu_integration_probe", "previous_ledger_sha256": "0215ca672019456d0a1f2301e4ff5246fb1f008df30853d671fe1934473033e8", "timings": {"baseline": [98.39344934999743, 94.57753725000879, 95.49518775000959], "linear": [130.11002500000276, 130.8606693499975, 130.43779405000373]}, "utc": "2026-10-01T06:46:50.212981+00:00"}
- {"error": "ValueError: callback throughput 91.1 us/row exceeds 90 us/row gate", "evidence": "reused_search_evidence_not_independent_validation", "failure_type": "implementation_or_infrastructure", "id": "e01c_additive_fused", "kind": "experiment_interrupted", "previous_ledger_sha256": "7ca7386e572ccb045bd048b60b0d64321c81d0a42a159838eaeb6ca0aa38960c", "seconds": 6.1070640119999995, "utc": "2026-10-01T06:49:45.299218+00:00"}
- {"error": "ValueError: callback throughput 90.2 us/row exceeds 90 us/row gate", "evidence": "reused_search_evidence_not_independent_validation", "failure_type": "implementation_or_infrastructure", "id": "e01d_additive_fused_coalesced", "kind": "experiment_interrupted", "previous_ledger_sha256": "73cf310c437ac8aed42a20af357592b8a800dc624baa40be6a23023a3bc32fbf", "seconds": 5.9748023770000005, "utc": "2026-10-01T06:51:55.829016+00:00"}
- {"cpu_observation": "Frozen Python combo 94.6-98.4 us per row and residual 130.1-130.9 in current isolated process; fixed 90 us gate retained", "evidence": "reused_search_evidence_not_independent_validation", "kind": "engineering_repair", "max_abs_train_coef_linux_vs_windows": 0.0, "previous_ledger_sha256": "83019ef699a5f2d75e16ef3220056003786f64189cc938817a87eaedbb17a27e", "reason": "Production isolation enforced for accepted candidate evidence; manual control requires isolated reproduction", "remedy": "Fused algebraic ONNX export with float64 memory and explicit parity tests; completed fits reused, original artifacts preserved", "tests": {"export_and_state_tests_passed": 12, "linux_production_and_retrieval_passed": 24, "windows_skipped_linux_isolation": 1, "windows_synthetic": 19}, "utc": "2026-10-01T06:52:18.924845+00:00"}
- {"error": "RuntimeError: callback exited or returned incomplete output", "evidence": "reused_search_evidence_not_independent_validation", "failure_type": "implementation_or_infrastructure", "id": "e01e_additive_bound_buffers", "kind": "experiment_interrupted", "previous_ledger_sha256": "e04f57f304a4b6246c3626de8375fbdc3acf87db9c95023682dbb29a75e66548", "seconds": 64.817553512, "utc": "2026-10-01T06:54:41.557874+00:00"}
- {"bug": "Inference pipe EOF discarded worker stderr and watchdog reason", "checkpoints": "e01e completed sequence predictions retained and fit reused", "evidence": "reused_search_evidence_not_independent_validation", "kind": "engineering_repair", "previous_ledger_sha256": "f4838d4cafaef329b13e1458748e1347a0c962805c8f89d24af880edf729feab", "regression": "tests/test_campaign_worker_failure.py passed; export/state/retrieval/production checks 36 passed on Linux", "repair": "Preserve final sandbox diagnostic through existing failure sanitizer; no scoring or isolation changes", "utc": "2026-10-01T06:57:27.921223+00:00"}
- {"error": "RuntimeError: callback exited or returned incomplete output; sandbox detail: bwrap: Creating new namespace failed: Resource temporarily unavailable", "evidence": "reused_search_evidence_not_independent_validation", "failure_type": "implementation_or_infrastructure", "id": "e01e_additive_bound_buffers", "kind": "experiment_interrupted", "previous_ledger_sha256": "fbb52ef4f8c1ca96f7ee0c27e63214788a319b3f2540035f73f1e9f9697af9a0", "seconds": 60.641150191, "utc": "2026-10-01T06:58:06.257960+00:00"}
- {"bug": "Concurrent numerical supervisors exhausted the existing sandbox process/namespace resource limit", "evidence": "Worker stderr: bwrap Creating new namespace failed Resource temporarily unavailable; production tests were concurrent with replay", "kind": "engineering_repair", "previous_ledger_sha256": "9116ed62ca63f105dff4594403ce608b8e36a074dc5505f44f16bc046d86fdef", "regression": "Worker error propagation test passed on both platforms; 19 Linux production tests passed after error repair; no scoring protocol changes", "repair": "Serialize sandbox jobs and set OMP/OPENBLAS/MKL/NUMEXPR supervisor threads to one before numerical imports; retain worker rlimits", "utc": "2026-10-01T06:59:47.618388+00:00"}

## Best distinct isolated search candidate

**e01e_additive_bound_buffers**: 0.658397454535, delta -0.000437868.
Paired 99% interval: [-0.0011598103879034395, 0.00016634274780787952].
The incumbent remains unchanged unless the fixed rule is met.

## Prior search trajectory

| Mechanism | Search WP | Interpretation |
|---|---:|---|
| linear_lags_10_100 | 0.657075365018 | See original search-only report. |
| histogram_nonlinear | 0.657968566710 | See original search-only report. |
| quadratic_basis | 0.657348114982 | See original search-only report. |
| gpu_multiscale_huber | 0.657075365018 | See original search-only report. |
| gpu_multiscale_wp | 0.657075365018 | See original search-only report. |
| ridge4096 | 0.658322601198 | See original search-only report. |
| causal_ema | 0.657075365018 | See original search-only report. |
| magnitude_ridge1024 | 0.658163572983 | See original search-only report. |
| magnitude_ridge4096 | 0.658513139777 | See original search-only report. |
| gated_ridge1024 | 0.657805317349 | See original search-only report. |
| gated_ridge4096 | 0.658428257435 | See original search-only report. |
| targetwise_combo | 0.658835322332 | See original search-only report. |
| mask_propensity_weighted_ridge4096 | 0.657743252674 | coarse search-mask propensity weighting did not beat prior 4096-sequence ridge or frozen targetwise combo; no protected evaluation |
| combo_spline_calibration1024 | 0.658664992698 | Nonlinear output-shape correction did not improve the combo; best tested strength was small on both targets. Persistent t0 errors likely require new causal information or conditional mechanism. |
| combo_innovation_gate1024 | 0.658095483776 | Causal prediction-change regime gating did not improve the combo. Short-lag residual persistence alone does not establish a useful global high/low innovation split. |
| combo_tiny_tree_t0_1024 | 0.658882537421 | Tiny nonlinear tree gives only a 0.000047 search gain, CI crosses zero, and stronger correction worsens; retain as exploratory evidence, not a promoted winner. CPU callback cost remains unverified. |
| combo_random_fourier_features1024 | 0.658380658872 | Smooth random-feature interactions did not improve the frozen combo. Both targets selected the weakest tested correction; global nonlinear expansion alone is unsupported. |
| combo_late_t0_expert4096 | 0.658428101061 | Late-only t0 linear correction worsened pooled search WP; a distinct late mapping of the same current features is insufficient at the tested regularization/strengths. |

## Literature

Library v2 preserves all 23 v1 cards and adds `nonlinear_vector_autoregression`.
[Gauthier et al., Next generation reservoir computing (2021)](http<LOCAL_PATH> motivates polynomial functions of causal history with a regularized readout. The featurewise EMA adaptation is curator inference, not a paper reproduction or evidence of task improvement.
Library v3 preserves v2 and adds `functional_gradient_fitting`: [Friedman (2001)](http<LOCAL_PATH> It motivates objective-aligned function fitting; the proposed clipped-correlation derivative is curator inference and has not been implemented or tested in this campaign.

## What was learned and next action

Neither additive multiscale memory nor featurewise current-by-history products improved the frozen combo. The nonlinear candidate also worsened late t0; this weakens the tested temporal mechanism rather than supporting more timescale tuning. Other nonlinear temporal architectures remain untested.
The best practical search incumbent remains 0.6588353223316641. The earlier tiny-tree point score 0.6588825374213778 remains uncertain (paired 95% delta interval [-0.0002228343, 0.0002908522]) and its measured residual-only CPU cost was 174.47 microseconds/row.
Recommended next experiment: a train-only, objective-aligned correction, compared with the matched additive control. Training t0 correlation-tangent slope was 0.8453; clipped prediction weight mass was 0.838% in sampled training versus 4.161% in search. This is a diagnostic lead, not evidence that the proposed correction works. Causal input normalization and learned cross-feature temporal interactions remain alternative hypotheses.
Engineering repairs: fused ONNX export and alternating preallocated buffers brought full callbacks below the unchanged 90-microsecond gate; worker EOF now preserves error diagnostics; numerical supervisor threads are bounded and sandbox jobs serialized. The final serialized Linux preflight had 32 passing tests.
Only two distinct scientific experiments completed. Control repetitions and failed callback representations were engineering verification, not extra scientific trials. No experiments were running during the later interaction gap; elapsed campaign-window time is not active compute time.

## Frozen candidates awaiting user review

None from this campaign.

## Reproduction

Each experiment has a pre-scoring config, source snapshots, retrieval query/card hashes, immutable isolated fit, model hashes, callback validation, and checkpointed search predictions. The JSON report contains the complete trajectory and exact deployment-file hashes.

| Artifact | SHA256 |
|---|---|
| competition_engineering\runs\autonomous_20261001\campaign.json | `765467201c114f60df33fdebb4e752f0a237e29aae26fb15cda96257d07b1021` |
| competition_engineering\runs\autonomous_20261001\library_manifest.json | `7f784fb5e4ad42540dfb810a46d7d33bde7acae7a8b0e9befe352e771c94a887` |
| competition_engineering\runs\autonomous_20261001\library_manifest_v3.json | `f6b0e186e5641defd9c84286babc838d3ff6af0f8a32b447034f92064ca89a82` |
| competition_engineering\runs\autonomous_20261001\input_hashes.json | `5180d17ef82cff929c1fce5e9399cba6785494d01c59028ce2a5559f572d3738` |
| competition_engineering\runs\autonomous_20261001\ledger.jsonl | `c74e85e3238da02813529d3d9015f38b85f48f4c5d2318f2d17cec56c913bf22` |

Use `python -m competition_engineering.isolated_campaign --id <incomplete-id> --mode <mode> --resume` to resume an interrupted scoring phase without refitting. Completed experiments reject reruns in their original directories.

The source, input-cache and library manifests identify the exact reproducibility inputs. Earlier failed workspaces are retained.
