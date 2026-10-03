# Alternative causal representation checkpoint — 3 October 2026

**Practical incumbent: 0.658835322. No candidate awaits promotion.**

Reserved before architecture results:512 fitting sequences,128 development sequences, four disjoint256-sequence replication blocks. Excluded4608 earlier assigned research/protected training-role groups. Exact row-group/sequence identities and content hashes are preserved. Frozen incumbent pretraining exposure is not claimed independent; each candidate replication block is disjoint from its correction fitting/selection and all earlier consumed blocks.

Consumed 5 replication blocks; 1 remain untouched. Additional blocks, if present, were reserved prospectively before the corresponding new experiment.

| Experiment / model | Uniform WP gain | Probability WP gain | Probability paired95 interval | Standalone uniform WP(t0,t1) |
|---|---:|---:|---|---|
| fixed_convolution/core_r0 | +0.000133095 | +0.000214573 | [+0.000064237, +0.000369086] | 0.329604 / 0.325172 |
| fixed_convolution/instant_r0 | +0.000164403 | +0.000264912 | [+0.000101416, +0.000433470] | 0.330452 / 0.333456 |
| fixed_convolution/gru_r0 | +0.000893005 | +0.000503719 | [-0.000550818, +0.001559171] | 0.439782 / 0.450819 |
| fixed_convolution/delay_r0 | +0.000094640 | +0.000197717 | [+0.000058666, +0.000337996] | 0.339495 / 0.333954 |
| fixed_convolution/tcn_r0 | +0.000363783 | +0.000432463 | [+0.000137838, +0.000736503] | 0.334268 / 0.338096 |
| fixed_convolution/core_r1 | -0.000013471 | +0.000009170 | [-0.000045320, +0.000066310] | 0.327252 / 0.323943 |
| fixed_convolution/instant_r1 | -0.000011194 | +0.000065074 | [-0.000055926, +0.000190994] | 0.328945 / 0.332717 |
| fixed_convolution/gru_r1 | +0.000397393 | +0.000316427 | [-0.000650826, +0.001285749] | 0.439038 / 0.450045 |
| fixed_convolution/delay_r1 | -0.000201474 | +0.000053558 | [-0.000294207, +0.000421966] | 0.336497 / 0.332555 |
| fixed_convolution/tcn_r1 | -0.000007186 | +0.000013576 | [-0.000119530, +0.000147981] | 0.334000 / 0.336027 |
| polynomial_memory/core_r0 | +0.000042671 | +0.000068479 | [-0.000090160, +0.000227328] | 0.354713 / 0.360822 |
| polynomial_memory/gru_r0 | +0.000986724 | +0.001237941 | [+0.000304795, +0.002171819] | 0.443825 / 0.459536 |
| polynomial_memory/smooth_r0 | +0.000442874 | +0.000288897 | [-0.000141268, +0.000711738] | 0.360128 / 0.364760 |
| polynomial_memory/ssm_r0 | +0.000098517 | +0.000074644 | [-0.000057173, +0.000210976] | 0.358982 / 0.363339 |
| polynomial_memory/core_r1 | -0.000022186 | +0.000025468 | [-0.000025598, +0.000078656] | 0.351742 / 0.357563 |
| polynomial_memory/gru_r1 | +0.000855773 | +0.001311450 | [+0.000490426, +0.002124834] | 0.445247 / 0.457426 |
| polynomial_memory/smooth_r1 | +0.000298066 | +0.000280151 | [-0.000205572, +0.000722796] | 0.359198 / 0.363327 |
| polynomial_memory/ssm_r1 | +0.000052400 | +0.000028645 | [-0.000034266, +0.000091954] | 0.358434 / 0.361331 |
| learned_convolution/current_r0 | +0.000068226 | +0.000077223 | [-0.000000835, +0.000154154] | 0.315546 / 0.332950 |
| learned_convolution/temporal_r0 | +0.000090709 | -0.000002096 | [-0.000270326, +0.000250568] | 0.219901 / 0.320456 |
| learned_convolution/gru_r0 | +0.000760905 | +0.001097058 | [-0.000021449, +0.002122307] | 0.456584 / 0.457650 |
| learned_convolution/current_r1 | +0.000049349 | +0.000064035 | [-0.000000334, +0.000127164] | 0.316972 / 0.326394 |
| learned_convolution/temporal_r1 | -0.000054177 | -0.000233415 | [-0.000465087, -0.000009243] | 0.224040 / 0.310994 |
| learned_convolution/gru_r1 | +0.000593137 | +0.001455200 | [+0.000495184, +0.002398948] | 0.457063 / 0.457458 |
| disjoint_readout/current_r0 | +0.000188131 | +0.000178516 | [-0.000163208, +0.000499266] | 0.323193 / 0.326457 |
| disjoint_readout/temporal_r0 | -0.000081217 | -0.000302308 | [-0.000910811, +0.000267189] | 0.325519 / 0.328631 |
| disjoint_readout/gru_r0 | -0.000270574 | +0.000255768 | [-0.000691521, +0.001168816] | 0.443696 / 0.444430 |
| disjoint_readout/tcn_r0 | +0.000105914 | +0.000015223 | [-0.000329798, +0.000351096] | 0.326411 / 0.334620 |
| disjoint_readout/current_r1 | -0.000528890 | -0.000348983 | [-0.001125474, +0.000405137] | 0.321338 / 0.326463 |
| disjoint_readout/temporal_r1 | -0.000382030 | -0.000597677 | [-0.001469607, +0.000255021] | 0.323342 / 0.328029 |
| disjoint_readout/gru_r1 | -0.000859837 | -0.000068906 | [-0.001027327, +0.000860597] | 0.439706 / 0.443288 |
| disjoint_readout/tcn_r1 | -0.000428647 | -0.000460756 | [-0.001278803, +0.000355489] | 0.322397 / 0.334895 |
| raw_target/current_r0 | -0.000103125 | +0.000161494 | [-0.000202627, +0.000518244] | 0.240173 / 0.313360 |
| raw_target/temporal_r0 | -0.000333063 | -0.000038438 | [-0.000470154, +0.000399834] | 0.237491 / 0.250551 |
| raw_target/gru_r0 | +0.000789539 | +0.001450868 | [+0.000466785, +0.002415438] | 0.413907 / 0.431517 |
| raw_target/current_r1 | +0.000118853 | +0.000054275 | [-0.000154792, +0.000253659] | 0.286648 / 0.311621 |
| raw_target/temporal_r1 | +0.000000000 | +0.000000000 | [+0.000000000, +0.000000000] | 0.296043 / 0.318358 |
| raw_target/gru_r1 | +0.000598242 | +0.001201083 | [+0.000306598, +0.002094672] | 0.414027 / 0.432309 |

These are training-derived development scores, not official competition scores. All prospective mechanism gates failed; no official-search access, protected evaluation, promotion, leaderboard evaluation or submission occurred. Standalone probes omit incumbent prediction features, but their learned/frozen representations may reflect incumbent-supervised training or GRU pretraining.

The first experiment compared equally sized current-only nonlinear, projected GRU, exact-delay and fixed dilated-convolution features. The convolution effect was seed-sensitive and weaker than the GRU controls. This rejects this frozen probe as a candidate, not trained TCNs in general.

The second experiment compared512-row order4 Legendre memory with matched smoothing and GRU features. It was weaker than smoothing and reliably weaker than GRU on the proxy population. No memory window/order sweep followed.

The third experiment trained a112-to32 projection and four width32 causal layers, comparing a parameter-matched current-only topology. Identical fitting rows, optimizer, checkpoint grid and WP strength selector isolated temporal connections from additional static capacity. Detailed target effects, contrasts, fitting histories and selected checkpoints are preserved in the JSON evidence.

All ensembles used simple prospective target-wise residual corrections with development-frozen strengths. Poor-row error diagnostics and sequence uncertainty are recorded. Lower fitting MSE and better poor-row MSE are not substitutes for replicated WP gains.

Infrastructure: Parquet omitted sequence-ID statistics; reservation now reads and verifies the ID column instead. Regression covers missing statistics and mixed identities. Numerical tests cover batch/stream convolution parity, early causal padding, resets, future independence, differentiable sparse endpoints, and polynomial-filter/state-update parity. Implementation failure is logged separately from scientific outcomes.

Research library: generic TCN card updated from primary architecture sections; LMU card added and refined after verifying equations1-4 and ZOH discretization in the primary paper. Libraries v13/v14 retain provenance and distinguish curator inference from claims.

No candidate qualified for deployment. Complete candidate callback latency was therefore not measured; compact parameter counts are plausibility evidence only. No deployment performance is claimed.

Best observed probability-population development gain: +0.001455200, learned_convolution/gru_r1. This adaptively observed maximum does not replace the incumbent.

Regression: Windows 101 passed; Linux 102 passed. Exact artifacts, source snapshots, ledger chain and hashes accompany this report.
A fourth experiment froze the learned encoders and reserved512 additional readout-fitting sequences, disjoint from encoder fitting, development and every replication block. Matched ridge readouts tested whether independent decoding rescues learned temporal information. The final reserved256-sequence replication block was consumed once. No encoder or checkpoint was changed.

A fifth experiment trained the same topology on original clipped targets, omitting incumbent prediction inputs, with a parameter-matched static model. The simple target-wise blend was chosen on development WP before a newly reserved256-sequence replication. Another256-sequence block was frozen at the same time and remains untouched. This counterfactual distinguishes supervision-dependent representation quality from another architecture sweep.


Next question: distinguish missing observed information, domain-dependent input-to-target mappings and target noise before another temporal-capacity campaign. The unused prospective block(s) remain reserved.

The raw-target primary blend lost0.000038 proxy WP; the paired repeat selected zero weight for both targets. Disjoint decoding improved temporal standalone decoding from~0.22 to~0.325 on different fresh populations but still produced negative incremental WP. Thus representation/readout dependence matters, without establishing a useful incumbent complement. These point comparisons span different populations and are not paired causal estimates.

Target-specific behavior: temporal poor-row MSE gains were most consistent on t0, yet its WP effects were weak or negative. Several t1 corrections were set to zero by development selection. Full target-wise effects and sequence bootstrap intervals accompany every experiment. Error cosines near1 occasionally exceed1 by float32 roundoff; they are descriptive, not evidence of perfect agreement.

Additional infrastructure failure: bubblewrap returned Resource temporarily unavailable while creating a namespace, before candidate execution. Failed work was preserved as raw_target/failed_fit_001. A new bounded retry helper permits at most three attempts only for that exact transient error, under the same overall deadline and identical sandbox command/mount/network restrictions. Permission and candidate errors are not retried. Regression covers recovery, exhaustion and non-retry cases; the scientific training source hash is identical before and after repair.

Primary sources: [TCN architecture paper](http<LOCAL_PATH> [LMU dynamics paper](http<LOCAL_PATH> Their benchmark successes do not imply Connectome improvements; the experiments and curator interpretations are separately recorded.

Best alternative development point: +0.000432463 proxy WP, fixed_convolution/tcn_r0; it failed the prospectively frozen mechanism/replication criteria. Best credible candidate remains the practical incumbent.

Checkpoint rationale: the tested representation mechanisms and both justified rescue counterfactuals failed fresh replication. The evidence does not justify another window, width, strength, loss or seed sweep. Compute and the overall eligible training dataset are not claimed exhausted. One reserved block is preserved for a materially different hypothesis backed by new diagnostics; no additional architecture experiment is selected from the present evidence.
