# Fitting-sequence diversity checkpoint — 3 October 2026

**Practical incumbent retained: 0.658835322. No candidate awaits promotion.**

Equal budget: 1024 sequences × 128 rows versus 2048 × 64, each 131,072 rows. Two paired sampling/initialization seeds; fixed causal GRU features, 32-unit correction head, optimizer, 1,280 updates and WP-based selection. Linear controls use the same sampled rows.

| Comparison: broad minus narrow | Uniform WP Δ | Probability-weighted WP Δ | Probability-weighted paired 95% interval |
|---|---:|---:|---|
| Original mixture, r0_linear | -0.000823569 | -0.000738056 | [-0.001590351, +0.000100085] |
| Original mixture, r0_neural | +0.000833035 | +0.000117416 | [-0.000872898, +0.001105228] |
| Original mixture, r1_linear | -0.000386497 | -0.000364957 | [-0.000804881, +0.000067198] |
| Original mixture, r1_neural | -0.000098395 | -0.000151213 | [-0.001167504, +0.000850532] |
| Original mixture, mean neural | +0.000367320 | -0.000016898 | [-0.000814198, +0.000753759] |
| Balanced 50:50 mixture, r0_linear | +0.000878322 | +0.000161153 | [-0.000456560, +0.000802727] |
| Balanced 50:50 mixture, r0_neural | +0.000302683 | -0.000007160 | [-0.000659849, +0.000662450] |
| Balanced 50:50 mixture, r1_linear | +0.000186865 | -0.000071492 | [-0.000459269, +0.000326057] |
| Balanced 50:50 mixture, r1_neural | +0.000992758 | +0.000624552 | [-0.000114086, +0.001386473] |
| Balanced 50:50 mixture, mean neural | +0.000647721 | +0.000308696 | [-0.000174377, +0.000810394] |

Neither comparison passed its prospectively frozen replication gate. The first comparison confounded sequence count with source mixture; the follow-up held that mixture at 50:50. Each used 512 fresh, disjoint training-derived replication sequences. These are development WP differences, not official competition scores. No new official search, protected evaluation, promotion, leaderboard evaluation or submission was run.

The initial trainer failed because a selection-loop baseline variable overwrote its fitting baseline. The failed artifacts were preserved, the baseline references corrected, and an end-to-end regression with unequal fitting and selection lengths added. This was an infrastructure failure, separate from the scientific negative results.

The designated 4,096-sequence pool now has no never-observed replication groups remaining. Further seed, strength or allocation tuning on these outcomes would be adaptive. The next justified step is to obtain an independently reserved training-derived sequence pool and test a materially different representation or data-coverage hypothesis with a frozen protocol. Sequence count alone is not a demonstrated improvement for this model; this does not establish that diversity is ineffective for every architecture.

Regression checks: Windows 92 passed; Linux 93 passed. Immutable protocols, model identities, per-sequence moments, source snapshots and evidence hashes accompany this report.
