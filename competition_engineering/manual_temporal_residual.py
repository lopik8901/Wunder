"""Search-only manual test: causal lagged-feature correction of a frozen ridge.

This script reads only designated training and search-tuning caches. It does not
create a deployment package or access supervisor-only evaluation resources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import features
from wnn_connectome_starterpack.utils import GlobalAccumulator


def frozen_prediction(z, model):
    base = (z["p"] * model["base_scale"].astype(np.float32)
            + model["base_bias"].astype(np.float32)).astype(np.float32)
    f = features({**z, "p": base}, model["mean"], model["scale"])
    return (base + model["strengths"] * (f @ model["coef"])).astype(np.float32)


def design(z, mean, scale, indices):
    """Current and strictly past features; each sequence starts with zero history."""
    x = np.asarray(z["x"], dtype=np.float32)
    parts = [np.ones((len(indices), 1), dtype=np.float32)]
    for lag in (0, 10, 100):
        valid = indices >= lag
        previous = np.zeros((len(indices), x.shape[1]), dtype=np.float32)
        previous[valid] = x[indices[valid] - lag]
        normalized = np.clip((previous - mean) / scale, -8, 8)
        normalized[~valid] = 0
        parts.append(normalized)
    return np.concatenate(parts, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--tune", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--penalty", type=float, default=0.1)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("output already exists")
    if not 1 <= args.stride <= 50 or not 1e-6 <= args.penalty <= 10:
        raise ValueError("invalid fixed resource bounds")
    started = time.perf_counter()
    with np.load(args.reference) as f:
        root = {k: f[k].copy() for k in f.files}
    mean, scale = root["mean"], root["scale"]
    dims = 1 + 112 * 3
    gram = np.zeros((2, dims, dims), dtype=np.float64)
    rhs = np.zeros((2, dims), dtype=np.float64)
    rows = 0
    with threadpool_limits(limits=1):
        for _, z in cached(args.train):
            idx = np.flatnonzero(z["need"])[::args.stride]
            f = design(z, mean, scale, idx).astype(np.float64)
            residual = np.clip(z["y"][idx], -2, 2) - frozen_prediction(z, root)[idx]
            weight = np.abs(np.clip(z["y"][idx], -2, 2)).astype(np.float64)
            rows += len(idx)
            for target in range(2):
                gram[target] += f.T @ (weight[:, target, None] * f)
                rhs[target] += f.T @ (weight[:, target] * residual[:, target])
        coef = np.column_stack([
            np.linalg.solve(gram[k] + np.eye(dims) * rows * args.penalty, rhs[k])
            for k in range(2)
        ])
        strengths = (0.0, 0.25, 0.5, 1.0)
        accumulators = [GlobalAccumulator() for _ in strengths]
        for _, z in cached(args.tune):
            base = frozen_prediction(z, root)
            correction = design(z, mean, scale, np.arange(len(z["x"]))) @ coef
            for strength, accumulator in zip(strengths, accumulators):
                accumulator.add(z["y"], base + strength * correction, z["mask"])
    scores = [acc.result() for acc in accumulators]
    report = {
        "evidence": "reused_search_evidence_not_independent_validation",
        "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        "design": "bias + current x + x[t-10] + x[t-100], normalized by frozen root statistics",
        "training_sequences": 1024,
        "training_rows": rows,
        "stride": args.stride,
        "penalty": args.penalty,
        "strengths": strengths,
        "search_scores": scores,
        "runtime_seconds": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    np.savez(out / "temporal_residual.npz", coef=coef, mean=mean, scale=scale,
             lags=np.array([0, 10, 100]), reference_sha256=report["reference_sha256"])
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
