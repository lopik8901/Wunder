"""Search-only compact nonlinear t0 residual on the frozen combo."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from competition_engineering.manual_search_core import (COMBO, TRAIN_1024, assess,
                                                       load_combo, predict_combo)
from competition_engineering.pipeline import cached, write_json


def design(z, combo, base, idx):
    normalized = np.clip((z["x"][idx] - combo["mean"]) / combo["scale"], -8, 8)
    return np.column_stack((normalized, np.clip(base[idx], -2, 2))).astype(np.float32)


def main():
    out = Path("competition_engineering/runs/manual_tiny_tree_t0_20261001")
    if out.exists():
        raise ValueError("frozen experiment output exists")
    started = time.perf_counter()
    combo = load_combo()
    feat, residual, weight = [], [], []
    with threadpool_limits(limits=1):
        for _, z in cached(TRAIN_1024):
            idx = np.flatnonzero(z["need"])[::40]
            base = predict_combo(z, combo)
            feat.append(design(z, combo, base, idx))
            y = np.clip(z["y"][idx, 0], -2, 2)
            residual.append((y - base[idx, 0]).astype(np.float32))
            weight.append(np.abs(y).astype(np.float32))
        x = np.concatenate(feat); y = np.concatenate(residual); w = np.concatenate(weight)
        del feat, residual, weight
        model = HistGradientBoostingRegressor(
            max_iter=32, max_leaf_nodes=7, max_bins=127, learning_rate=.08,
            l2_regularization=20., min_samples_leaf=100,
            early_stopping=False, random_state=20261001)
        model.fit(x, y, sample_weight=w)

        def predictor(z, base, strength):
            prediction = base.copy()
            prediction[:, 0] += strength * model.predict(
                design(z, combo, base, np.arange(len(base)))).astype(np.float32)
            return prediction

        grid = {s: assess(lambda z, base, s=s: predictor(z, base, s), diagnostics=False)
                for s in (.25, .5, 1.)}
        strength = max(grid, key=lambda s: grid[s]["candidate"]["t0"])
        result = assess(lambda z, base: predictor(z, base, strength))
    out.mkdir(parents=True)
    joblib.dump(model, out / "tree.joblib")
    report = {"hypothesis": "t0 residual contains nonlinear current-feature interactions missed by linear/gated ridge",
              "mechanism": "32 shallow boosted histogram trees, t0 correction only",
              "research_card": "featurewise_nonlinear (inspiration, not a paper reproduction)",
              "training_sequences": 1024, "training_rows": len(x),
              "strength_search": {str(k): v["candidate"] for k, v in grid.items()},
              "chosen_strength": strength, "search_result": result,
              "runtime_seconds": time.perf_counter() - started,
              "artifact_sha256": hashlib.sha256((out / "tree.joblib").read_bytes()).hexdigest(),
              "deployment_note": "tree topology is compact but CPU callback cost requires explicit export/benchmark before promotion"}
    write_json(out / "report.json", report)
    print(json.dumps({"search_wp": result["candidate"]["weighted_pearson"],
                      "delta": result["delta_combined"], "strength": strength,
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
