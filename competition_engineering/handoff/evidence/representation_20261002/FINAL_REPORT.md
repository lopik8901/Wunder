# Connectome representation research checkpoint — 2 October 2026

**Practical incumbent retained: 0.658835322. No candidate awaits promotion.**

| Candidate | Search WP | Δ WP | Paired 99% interval | Callback µs/row | Decision |
|---|---:|---:|---|---:|---|
| mean | 0.658869403 | +0.000034081 | [-0.0013100, +0.0010851] | 72.46 | insufficient |
| matched_amplitude | 0.658898189 | +0.000062867 | [-0.0009662, +0.0008743] | 72.39 | insufficient |
| consensus | 0.659072103 | +0.000236781 | [-0.0008066, +0.0011116] | 77.49 | insufficient |

The consensus point is the best observed in this continuation, but it is not credible: its interval crosses zero and its 77.49 µs callback exceeds the 76.93 µs limit. It was not preserved for promotion.

Frozen GRU-state readouts recovered a positive effect on two training-derived replications, showing that learned causal states carry correction information beyond the current features. That information did not reliably transfer to the reused 64-sequence official search. Position effects also failed replication.

The final disagreement experiment used two disjoint readouts, fixed strength 0.25, and a predeclared soft-threshold rule. On a fresh 256-sequence replication it lost to the amplitude-matched control by −0.000063 uniform and −0.000134 probability-weighted combined WP. The rule is abandoned; the higher search point does not overturn its mechanism failure.

The campaign remains strictly search-only: no promotion, full validation, leaderboard, or submission was run. The incumbent archive is unchanged. New regression tests cover consensus algebra and fused ONNX parity. The research library now has v11 with the primary [deep ensembles paper](http<LOCAL_PATH> its limits for shared-readout clipped-WP use are explicit.

The next credible direction is a materially distinct causal architecture or feature family, evaluated first through the existing disjoint sequence replication protocol. Further selection, position, loss, or agreement shrinkage tuning is not justified by this evidence.
