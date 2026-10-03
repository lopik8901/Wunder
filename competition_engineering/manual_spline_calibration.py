"""Search-only test: low-cost nonlinear calibration of frozen combo outputs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.manual_search_core import (COMBO, SEARCH, TRAIN_1024,
                                                       assess, load_combo, predict_combo)
from competition_engineering.pipeline import cached, write_json

KNOTS = np.array([-1.5, -1, -.5, 0, .5, 1, 1.5], np.float32)


def basis(prediction):
    clipped = np.clip(prediction, -2, 2)
    return np.column_stack((np.ones(len(prediction)), clipped,
                            np.maximum(clipped[:, None] - KNOTS, 0))).astype(np.float64)


def main():
    out = Path("competition_engineering/runs/manual_spline_calibration_20261001")
    if out.exists():
        raise ValueError("frozen experiment output exists")
    started = time.perf_counter()
    combo = load_combo()
    gram = np.zeros((2, 9, 9), np.float64)
    rhs = np.zeros((2, 9), np.float64)
    rows = 0
    with threadpool_limits(limits=1):
        for _, z in cached(TRAIN_1024):
            idx = np.flatnonzero(z["need"])[::20]
            base = predict_combo(z, combo)[idx]
            target = np.clip(z["y"][idx], -2, 2)
            for k in range(2):
                f = basis(base[:, k])
                w = np.abs(target[:, k])
                gram[k] += f.T @ (f * w[:, None])
                rhs[k] += f.T @ ((target[:, k] - base[:, k]) * w)
            rows += len(idx)
        coef = np.stack([np.linalg.solve(gram[k] + np.eye(9) * rows * .02, rhs[k])
                         for k in range(2)])
        # The same fixed search split selects a per-target correction strength.
        candidates = {}
        for strength in (0.25, .5, 1.0):
            candidates[strength] = assess(lambda z, p, s=strength: p + s * np.column_stack(
                [basis(p[:, k]) @ coef[k] for k in range(2)]).astype(np.float32), diagnostics=False)
        chosen = [max(candidates, key=lambda s: candidates[s]["candidate"][k])
                  for k in ("t0", "t1")]
        result = assess(lambda z, p: p + np.column_stack(
            [chosen[k] * (basis(p[:, k]) @ coef[k]) for k in range(2)]).astype(np.float32))
    out.mkdir(parents=True)
    np.savez(out / "spline.npz", coef=coef, knots=KNOTS, strengths=chosen,
             combo_sha256=hashlib.sha256(COMBO.read_bytes()).hexdigest())
    report = {"hypothesis": "targetwise nonlinear output calibration may correct clipping-related residual shape",
              "mechanism": "train-only seven-knot hinge spline per target around frozen combo",
              "research_card": "classification_temperature_scaling (curator inference only)",
              "training_sequences": 1024, "training_rows": rows,
              "strength_search": {str(k): v["candidate"] for k, v in candidates.items()},
              "chosen_strengths": chosen, "search_result": result,
              "runtime_seconds": time.perf_counter() - started,
              "artifact_sha256": hashlib.sha256((out / "spline.npz").read_bytes()).hexdigest()}
    write_json(out / "report.json", report)
    print(json.dumps({"search_wp": result["candidate"]["weighted_pearson"],
                      "delta": result["delta_combined"], "strengths": chosen,
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
