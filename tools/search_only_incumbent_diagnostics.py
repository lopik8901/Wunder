"""Aggregate diagnostics from the designated 64-sequence search cache only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from competition_engineering.pipeline import cached
from competition_engineering.residual import from_stats, sufficient


ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "competition_engineering/cache/gru_tune"
RIDGE = ROOT / "competition_engineering/checkpoints/ridge1024_targetwise_v1/ridge.npz"


def main() -> None:
    identity = json.loads((SEARCH / "identity.json").read_text(encoding="utf-8"))
    if identity.get("split") != "tune" or len(identity.get("groups", [])) != 64:
        raise ValueError("diagnostics require exactly the designated search split")
    with np.load(RIDGE) as archive:
        ridge = {name: archive[name] for name in archive.files}

    root_stats = np.zeros((3, 6, 2), dtype=np.float64)
    gru_stats = np.zeros((3, 6, 2), dtype=np.float64)
    sequence_wp = []
    residual_cross = np.zeros((2, 2), dtype=np.float64)
    residual_lag = np.zeros((2, 3), dtype=np.float64)
    scored_rows = np.zeros(3, dtype=np.int64)

    for _, z in cached(SEARCH):
        raw = z["p"]
        base = (raw * ridge["base_scale"].astype(np.float32)
                + ridge["base_bias"].astype(np.float32)).astype(np.float32)
        feature = np.column_stack((
            np.ones(len(base), dtype=np.float32),
            np.clip((z["x"] - ridge["mean"]) / ridge["scale"], -8, 8),
            np.nan_to_num(base, nan=0),
        )).astype(np.float32)
        pred = (base + ridge["strengths"] * (feature @ ridge["coef"])).astype(np.float32)
        mask = z["mask"]
        if not np.isfinite(pred[mask]).all():
            raise ValueError("nonfinite search prediction")
        for third, indices in enumerate(np.array_split(np.arange(len(mask)), 3)):
            selected = indices[mask[indices]]
            scored_rows[third] += len(selected)
            if len(selected):
                root_stats[third] += sufficient(z["y"][selected], pred[selected])
                gru_stats[third] += sufficient(z["y"][selected], base[selected])
        sequence_wp.append(from_stats(sufficient(z["y"][mask], pred[mask])))
        residual = z["y"][mask].astype(np.float64) - pred[mask].astype(np.float64)
        residual_cross += residual.T @ residual
        if len(residual) > 1:
            residual_lag[:, 0] += (residual[:-1] * residual[1:]).sum(axis=0)
            residual_lag[:, 1] += (residual[:-1] ** 2).sum(axis=0)
            residual_lag[:, 2] += (residual[1:] ** 2).sum(axis=0)

    root_wp = from_stats(root_stats.sum(axis=0))
    if not np.allclose(root_wp, [0.6459635357038319, 0.6637674635202809], atol=2e-6):
        raise ValueError(f"root reproduction mismatch: {root_wp}")
    seq = np.asarray(sequence_wp)
    report = {
        "domain": "fixed 64-sequence search split only",
        "search_cache_identity_sha256": hashlib.sha256((SEARCH / "identity.json").read_bytes()).hexdigest(),
        "incumbent_ridge_sha256": hashlib.sha256(RIDGE.read_bytes()).hexdigest(),
        "root_wp": root_wp.tolist(),
        "thirds": [
            {"position": name, "scored_rows": int(scored_rows[i]),
             "root_wp": from_stats(root_stats[i]).tolist(),
             "calibrated_gru_wp": from_stats(gru_stats[i]).tolist()}
            for i, name in enumerate(("early", "middle", "late"))
        ],
        "per_sequence_wp_quantiles_10_50_90": np.quantile(seq, [.1, .5, .9], axis=0).tolist(),
        "residual_cross_target_uncentered_cosine": float(
            residual_cross[0, 1] / np.sqrt(residual_cross[0, 0] * residual_cross[1, 1])),
        "residual_adjacent_scored_row_uncentered_cosine": (
            residual_lag[:, 0] / np.sqrt(residual_lag[:, 1] * residual_lag[:, 2])).tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
