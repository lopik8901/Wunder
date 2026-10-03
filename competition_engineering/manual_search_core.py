"""Search-only helpers for Codex-led causal experiments.

All paths here are the designated training and search caches. Never import a
supervisor promotion module or accept a protected cache as an input.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from competition_engineering.pipeline import cached
from competition_engineering.residual import features, from_stats, sufficient
from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
from wnn_connectome_starterpack.utils import GlobalAccumulator

ROOT = Path(__file__).resolve().parents[1]
COMBO = ROOT / "competition_engineering/deployment/manual_targetwise_combo_v1/combo.npz"
COMBO_SHA = "7aa3204522ef118f664e6677f5e9a6ed5ad296b0e9466d9884906abc5b8ddb0a"
TRAIN_1024 = ROOT / "competition_engineering/cache/gru_train_scale_phase2"
TRAIN_4096 = ROOT / "competition_engineering/cache/gru_train_medium_4096"
SEARCH = ROOT / "competition_engineering/cache/gru_tune"
SEARCH_ROOT_WP = 0.6588353223316641


def load_combo():
    if hashlib.sha256(COMBO.read_bytes()).hexdigest() != COMBO_SHA:
        raise ValueError("frozen search combo artifact changed")
    with np.load(COMBO) as archive:
        return {key: archive[key].copy() for key in archive.files}


def predict_combo(z, model):
    base = (z["p"] * model["base_scale"].astype(np.float32)
            + model["base_bias"].astype(np.float32)).astype(np.float32)
    f = features({**z, "p": base}, model["mean"], model["scale"])
    root = (base + model["root_strengths"] * (f @ model["root_coef"])).astype(np.float32)
    magnitude = (base + model["magnitude_strengths"] * (f @ model["magnitude_coef"])).astype(np.float32)
    high = np.max(np.abs(root), axis=1) >= 1.0
    low = f @ model["gated_coef"][0]
    high_part = f @ model["gated_coef"][1]
    t1 = base[:, 1] + model["gated_strengths"][1] * np.where(
        high, high_part[:, 1], low[:, 1])
    return np.column_stack((magnitude[:, 0], t1)).astype(np.float32)


def assess(predictor, *, diagnostics=True):
    """Exact pooled search WP and paired sequence-bootstrap against frozen combo."""
    combo = load_combo()
    base_acc, candidate_acc = GlobalAccumulator(), GlobalAccumulator()
    moments = []
    atlas = SearchErrorDiagnostics(root_predictor=lambda z: predict_combo(z, combo)) if diagnostics else None
    for _, z in cached(SEARCH):
        base = predict_combo(z, combo)
        prediction = np.asarray(predictor(z, base), dtype=np.float32)
        if prediction.shape != base.shape or not np.isfinite(prediction[z["need"]]).all():
            raise ValueError("candidate search prediction shape or finiteness invalid")
        base_acc.add(z["y"], base, z["mask"])
        candidate_acc.add(z["y"], prediction, z["mask"])
        mask = z["mask"]
        moments.append([sufficient(z["y"][mask], base[mask]),
                        sufficient(z["y"][mask], prediction[mask])])
        if atlas is not None:
            atlas.add(z, prediction)
    baseline, candidate = base_acc.result(), candidate_acc.result()
    if abs(baseline["weighted_pearson"] - SEARCH_ROOT_WP) > 2e-6:
        raise ValueError("frozen combo search WP did not reproduce")
    moments = np.asarray(moments)
    rng = np.random.default_rng(20261001)
    deltas = []
    for _ in range(1000):
        indices = rng.integers(len(moments), size=len(moments))
        sums = moments[indices].sum(axis=0)
        deltas.append(from_stats(sums[1]) - from_stats(sums[0]))
    deltas = np.asarray(deltas)
    result = {"reference": {k: baseline[k] for k in ("t0", "t1", "weighted_pearson")},
              "candidate": {k: candidate[k] for k in ("t0", "t1", "weighted_pearson")},
              "delta_per_target": [candidate[k] - baseline[k] for k in ("t0", "t1")],
              "delta_combined": candidate["weighted_pearson"] - baseline["weighted_pearson"],
              "paired_95ci_combined": np.quantile(deltas.mean(1), [.025, .975]).tolist(),
              "scored_rows": baseline["selected_rows"], "sequences": baseline["blocks"],
              "evidence": "reused_search_evidence_not_independent_validation"}
    if atlas is not None:
        result["search_error_diagnostics"] = atlas.result()
    return result
