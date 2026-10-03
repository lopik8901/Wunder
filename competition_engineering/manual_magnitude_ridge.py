"""Train-only ridge emphasizing high-prediction-magnitude rows.

The fixed weight shape is motivated by aggregate search-only mask diagnostics;
no search targets or mask values enter fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.manual_temporal_residual import frozen_prediction
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import calibrated_cached, evaluate, features


def main():
    p = argparse.ArgumentParser()
    for key in ("train", "tune", "reference", "output"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("output exists")
    started = time.perf_counter()
    ref = Path(args.reference)
    with np.load(ref) as q:
        root = {k: q[k].copy() for k in q.files}
    base_scale = root["base_scale"]
    base_bias = root["base_bias"]
    mean = root["mean"]
    scale = root["scale"]
    stride = 20
    penalty = 0.1
    gram = np.zeros((2, 115, 115), np.float64)
    rhs = np.zeros((2, 115), np.float64)
    rows = 0
    with threadpool_limits(limits=1):
        for _, z in cached(args.train):
            idx = np.flatnonzero(z["need"])[::stride]
            frozen = frozen_prediction(z, root)[idx]
            magnitude = np.clip(np.max(np.abs(frozen), axis=1), 0, 2)
            focus = 0.1 + magnitude**2
            calibrated = (z["p"] * base_scale.astype(np.float32)
                          + base_bias.astype(np.float32)).astype(np.float32)
            local = {**z, "p": calibrated}
            f = features(local, mean, scale)[idx].astype(np.float64)
            y = z["y"][idx].copy()
            y[:, 1] = np.clip(y[:, 1], -2, 2)
            residual = y - calibrated[idx]
            weights = np.abs(np.clip(z["y"][idx], -2, 2)) * focus[:, None]
            rows += len(idx)
            for k in range(2):
                gram[k] += f.T @ (f * weights[:, k, None])
                rhs[k] += f.T @ (residual[:, k] * weights[:, k])
        coef = np.column_stack([
            np.linalg.solve(gram[k] + np.eye(115) * rows * penalty, rhs[k])
            for k in range(2)
        ])
        tune = evaluate(args.tune, mean, scale, coef,
                        base_scale=base_scale, base_bias=base_bias)
        selected = evaluate(args.tune, mean, scale, coef,
                            strengths=tune["chosen_strengths"],
                            base_scale=base_scale, base_bias=base_bias)
    tune["search_error_diagnostics"] = selected["search_error_diagnostics"]
    out.mkdir(parents=True)
    np.savez(out / "ridge.npz", mean=mean, scale=scale, coef=coef,
             strengths=tune["chosen_strengths"], base_scale=base_scale, base_bias=base_bias)
    report = {"evidence": "reused_search_evidence_not_independent_validation",
              "hypothesis": "fit residuals more strongly where causal predictions indicate a high-magnitude regime",
              "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
              "train_sequences": len(json.loads((Path(args.train) / "identity.json").read_text())["groups"]),
              "sampled_train_rows": rows, "stride": stride, "penalty": penalty,
              "training_weight": "abs(clipped target) * (0.1 + min(max(abs(frozen prediction)),2)^2)",
              "tune": tune, "runtime_seconds": time.perf_counter() - started}
    write_json(out / "report.json", report)
    print(json.dumps({"train_sequences": report["train_sequences"],
                      "chosen_strengths": tune["chosen_strengths"],
                      "candidate_search_wp": selected["candidate"]["weighted_pearson"],
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
