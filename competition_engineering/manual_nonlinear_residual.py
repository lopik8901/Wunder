"""Search-only test of nonlinear, causal residual correction on the frozen model."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from competition_engineering.manual_temporal_residual import frozen_prediction
from competition_engineering.pipeline import cached, write_json
from wnn_connectome_starterpack.utils import GlobalAccumulator


def design(z, model, indices, base=None):
    x = z["x"]
    current = np.clip((x[indices] - model["mean"]) / model["scale"], -8, 8)
    aux = []
    for lag in (10, 100):
        previous = np.zeros((len(indices), 8), dtype=np.float32)
        valid = indices >= lag
        previous[valid] = x[indices[valid] - lag, 104:112]
        previous[~valid] = model["mean"][104:112]
        aux.append(np.clip((previous - model["mean"][104:112]) / model["scale"][104:112], -8, 8))
    if base is None:
        base = frozen_prediction(z, model)
    return np.concatenate((current, *aux, base[indices]), axis=1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    for name in ("train", "tune", "reference", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("output exists")
    start = time.perf_counter()
    with np.load(args.reference) as f:
        root = {k: f[k].copy() for k in f.files}
    feats, targets, weights = [], [], []
    with threadpool_limits(limits=2):
        for _, z in cached(args.train):
            idx = np.flatnonzero(z["need"])[::20]
            base = frozen_prediction(z, root)
            feats.append(design(z, root, idx, base))
            y = np.clip(z["y"][idx], -2, 2)
            targets.append((y - base[idx]).astype(np.float32))
            weights.append(np.abs(y).astype(np.float32))
        x = np.concatenate(feats)
        y = np.concatenate(targets)
        w = np.concatenate(weights)
        del feats, targets, weights
        models = []
        for target in range(2):
            reg = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                 learning_rate=0.05, l2_regularization=10.0, min_samples_leaf=80,
                 max_bins=127, random_state=20260930, early_stopping=False)
            reg.fit(x, y[:, target], sample_weight=w[:, target])
            models.append(reg)
        strengths = (0.0, 0.25, 0.5, 1.0)
        accs = [GlobalAccumulator() for _ in strengths]
        for _, z in cached(args.tune):
            base = frozen_prediction(z, root)
            f = design(z, root, np.arange(len(z["x"])), base)
            correction = np.column_stack([reg.predict(f) for reg in models])
            for strength, acc in zip(strengths, accs):
                acc.add(z["y"], base + strength * correction, z["mask"])
    out.mkdir(parents=True)
    joblib.dump(models, out / "hist_residual.joblib")
    report = {
       "evidence": "reused_search_evidence_not_independent_validation",
       "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
       "model": "two 100-tree histogram boosting residuals on current features, frozen prediction, and lagged auxiliary features",
       "train_rows": int(len(x)), "train_sequences": 1024,
       "strengths": strengths, "search_scores": [a.result() for a in accs],
       "runtime_seconds": time.perf_counter() - start,
    }
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
