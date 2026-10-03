"""Search-only late-position t0 expert on a frozen causal combo."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.manual_search_core import (COMBO, TRAIN_4096, assess,
                                                       load_combo, predict_combo)
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import features

LATE_START = 13333  # fixed atlas boundary, selected before this experiment


def design(z, combo):
    calibrated = (z["p"] * combo["base_scale"].astype(np.float32)
                  + combo["base_bias"].astype(np.float32)).astype(np.float32)
    return features({**z, "p": calibrated}, combo["mean"], combo["scale"])


def main():
    out = Path("competition_engineering/runs/manual_late_t0_expert4096_20261001")
    if out.exists():
        raise ValueError("frozen experiment output exists")
    started = time.perf_counter()
    combo = load_combo()
    gram = np.zeros((115, 115), np.float64)
    rhs = np.zeros(115, np.float64)
    rows = 0
    with threadpool_limits(limits=1):
        for _, z in cached(TRAIN_4096):
            selected = z["need"] & (z["step"] >= LATE_START)
            idx = np.flatnonzero(selected)[::20]
            base = predict_combo(z, combo)
            f = design(z, combo)[idx].astype(np.float64)
            y = np.clip(z["y"][idx, 0], -2, 2)
            w = np.abs(y)
            gram += f.T @ (f * w[:, None])
            rhs += f.T @ ((y - base[idx, 0]) * w)
            rows += len(idx)
        coef = np.linalg.solve(gram + np.eye(115) * rows * .1, rhs)

        def predictor(z, base, strength):
            prediction = base.copy()
            late = z["step"] >= LATE_START
            f = design(z, combo)[late]
            prediction[late, 0] += strength * (f @ coef).astype(np.float32)
            return prediction

        grid = {s: assess(lambda z, base, s=s: predictor(z, base, s), diagnostics=False)
                for s in (.25, .5, 1.)}
        strength = max(grid, key=lambda s: grid[s]["candidate"]["t0"])
        result = assess(lambda z, base: predictor(z, base, strength))
    out.mkdir(parents=True)
    np.savez(out / "late_t0.npz", coef=coef, strength=strength,
             late_start=LATE_START, combo_sha256=hashlib.sha256(COMBO.read_bytes()).hexdigest())
    report = {"hypothesis": "late-sequence t0 residuals need a position-specific mapping absent from the global combo",
              "mechanism": "train-only late-step t0 ridge residual, no changes to earlier rows or t1",
              "research_card": "sparse_expert_gating (conditional mechanism, not direct reproduction)",
              "training_sequences": 4096, "training_rows": rows,
              "late_start": LATE_START, "chosen_strength": strength,
              "strength_search": {str(k): v["candidate"] for k, v in grid.items()},
              "search_result": result, "runtime_seconds": time.perf_counter() - started,
              "artifact_sha256": hashlib.sha256((out / "late_t0.npz").read_bytes()).hexdigest(),
              "cpu_path": "one additional 115-feature dot product only after step 13333"}
    write_json(out / "report.json", report)
    print(json.dumps({"search_wp": result["candidate"]["weighted_pearson"],
                      "delta": result["delta_combined"], "strength": strength,
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
