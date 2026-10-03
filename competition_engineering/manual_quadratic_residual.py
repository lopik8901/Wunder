"""Search-only quadratic residual model with cheap causal CPU features."""
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
from wnn_connectome_starterpack.utils import GlobalAccumulator


def design(z, model, indices, base):
    x = np.clip((z["x"][indices] - model["mean"]) / model["scale"], -8, 8)
    # Fixed nonlinear basis: current signal, its magnitude, and coupling to the
    # causal GRU/ridge output. No target values or future feature rows appear.
    return np.column_stack((np.ones(len(indices)), x, np.abs(x),
                            x * np.clip(base[indices, 0:1], -2, 2))).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    for name in ("train", "tune", "reference", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("output exists")
    start = time.perf_counter()
    with np.load(args.reference) as q:
        root = {k: q[k].copy() for k in q.files}
    dims = 337
    gram = np.zeros((2, dims, dims), dtype=np.float64)
    rhs = np.zeros((2, dims), dtype=np.float64)
    rows = 0
    with threadpool_limits(limits=1):
        for _, z in cached(args.train):
            idx = np.flatnonzero(z["need"])[::20]
            base = frozen_prediction(z, root)
            f = design(z, root, idx, base).astype(np.float64)
            y = np.clip(z["y"][idx], -2, 2)
            residual = y - base[idx]
            weight = np.abs(y)
            rows += len(idx)
            for k in range(2):
                gram[k] += f.T @ (f * weight[:, k, None])
                rhs[k] += f.T @ (residual[:, k] * weight[:, k])
        coef = np.column_stack([
            np.linalg.solve(gram[k] + np.eye(dims) * rows * 0.1, rhs[k])
            for k in range(2)
        ])
        strengths = (0.0, 0.25, 0.5, 1.0)
        acc = [GlobalAccumulator() for _ in strengths]
        targetwise = GlobalAccumulator()
        for _, z in cached(args.tune):
            base = frozen_prediction(z, root)
            f = design(z, root, np.arange(len(z["x"])), base)
            correction = f @ coef
            for s, a in zip(strengths, acc):
                a.add(z["y"], base + s * correction, z["mask"])
            targetwise.add(z["y"], base + correction * [1, 0], z["mask"])
    out.mkdir(parents=True)
    np.savez(out / "quadratic_residual.npz", coef=coef, mean=root["mean"],
             scale=root["scale"], reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest())
    report = {"evidence": "reused_search_evidence_not_independent_validation",
              "model": "current x, abs(x), x * frozen t0 prediction ridge residual",
              "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
              "training_rows": rows, "training_sequences": 1024,
              "search_uniform_strengths": dict(zip(strengths, [a.result() for a in acc])),
              "search_t0_only_strength_1": targetwise.result(),
              "runtime_seconds": time.perf_counter() - start}
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
