"""Search-only smooth nonlinear residual with fixed CPU-feasible random features."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.manual_search_core import (COMBO, TRAIN_1024, assess,
                                                       load_combo, predict_combo)
from competition_engineering.pipeline import cached, write_json


def design(z, combo, base, idx, projection):
    x = np.clip((z["x"][idx] - combo["mean"]) / combo["scale"], -6, 6)
    b = np.clip(np.nan_to_num(base[idx], nan=0), -2, 2)
    signal = np.column_stack((x, b)).astype(np.float32)
    phase = signal @ projection
    return np.column_stack((np.ones(len(idx), np.float32), np.cos(phase),
                            np.sin(phase))).astype(np.float32)


def main():
    out = Path("competition_engineering/runs/manual_random_features1024_20261001")
    if out.exists():
        raise ValueError("frozen experiment output exists")
    started = time.perf_counter()
    combo = load_combo()
    rng = np.random.default_rng(20261001)
    projection = (rng.standard_normal((114, 32)) * (1.5 / np.sqrt(114))).astype(np.float32)
    dimensions = 65
    gram = np.zeros((2, dimensions, dimensions), np.float64)
    rhs = np.zeros((2, dimensions), np.float64)
    rows = 0
    with threadpool_limits(limits=1):
        for _, z in cached(TRAIN_1024):
            idx = np.flatnonzero(z["need"])[::20]
            base = predict_combo(z, combo)
            f = design(z, combo, base, idx, projection).astype(np.float64)
            y = np.clip(z["y"][idx], -2, 2)
            residual = y - base[idx]
            weight = np.abs(y)
            rows += len(idx)
            for k in range(2):
                w = weight[:, k]
                gram[k] += f.T @ (f * w[:, None])
                rhs[k] += f.T @ (residual[:, k] * w)
        coef = np.column_stack([
            np.linalg.solve(gram[k] + np.eye(dimensions) * rows * .002, rhs[k])
            for k in range(2)])

        def predictor(z, base, strengths):
            f = design(z, combo, base, np.arange(len(base)), projection)
            return base + (f @ coef) * np.asarray(strengths, np.float32)

        grid = {s: assess(lambda z, base, s=s: predictor(z, base, (s, s)), diagnostics=False)
                for s in (.25, .5, 1.)}
        chosen = [max(grid, key=lambda s: grid[s]["candidate"][target])
                  for target in ("t0", "t1")]
        result = assess(lambda z, base: predictor(z, base, chosen))
    out.mkdir(parents=True)
    np.savez(out / "random_features.npz", projection=projection, coef=coef,
             strengths=chosen, combo_sha256=hashlib.sha256(COMBO.read_bytes()).hexdigest())
    report = {"hypothesis": "smooth nonlinear combinations of current causal features explain residual structure missed by ridge and shallow trees",
              "mechanism": "fixed 32 random projections with sine/cosine expansion, weighted ridge residual",
              "research_card": "featurewise_nonlinear (mechanism inspiration only)",
              "training_sequences": 1024, "training_rows": rows,
              "chosen_strengths": chosen,
              "strength_search": {str(k): v["candidate"] for k, v in grid.items()},
              "search_result": result, "runtime_seconds": time.perf_counter() - started,
              "artifact_sha256": hashlib.sha256((out / "random_features.npz").read_bytes()).hexdigest(),
              "cpu_path": "incremental current-row matrix-vector product plus 64 trigonometric evaluations"}
    write_json(out / "report.json", report)
    print(json.dumps({"search_wp": result["candidate"]["weighted_pearson"],
                      "delta": result["delta_combined"], "strengths": chosen,
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
