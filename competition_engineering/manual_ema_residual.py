"""Search-only test of causal running-input context beyond fixed lag snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits

from competition_engineering.manual_temporal_residual import frozen_prediction
from competition_engineering.pipeline import cached, write_json
from wnn_connectome_starterpack.utils import GlobalAccumulator


def design(z, root, indices, base):
    current = np.clip((z["x"] - root["mean"]) / root["scale"], -8, 8).astype(np.float32)
    # Includes the current row, and never crosses a sequence boundary.
    running = lfilter([0.01], [1.0, -0.99], current, axis=0).astype(np.float32)
    return np.column_stack((np.ones(len(indices)), current[indices], running[indices],
                            np.nan_to_num(base[indices], nan=0))).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    for key in ("reference", "train", "tune", "output"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("output exists")
    started = time.perf_counter()
    ref = Path(args.reference)
    with np.load(ref) as q:
        root = {k: q[k].copy() for k in q.files}
    dims = 227
    gram = np.zeros((2, dims, dims), np.float64)
    rhs = np.zeros((2, dims), np.float64)
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
        for _, z in cached(args.tune):
            base = frozen_prediction(z, root)
            f = design(z, root, np.arange(len(z["x"])), base)
            correction = f @ coef
            for s, a in zip(strengths, acc):
                a.add(z["y"], base + s * correction, z["mask"])
    out.mkdir(parents=True)
    np.savez(out / "ema_residual.npz", coef=coef,
             reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
    report = {"evidence": "reused_search_evidence_not_independent_validation",
              "hypothesis": "a causal 100-row exponential input state explains residual context beyond current inputs",
              "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
              "training_sequences": 1024, "training_rows": rows,
              "ema_alpha": 0.01, "ridge_penalty": 0.1,
              "search_scores": dict(zip(map(str, strengths), [a.result() for a in acc])),
              "runtime_seconds": time.perf_counter() - started}
    write_json(out / "report.json", report)
    print(json.dumps({"scores": {str(s): {"t0": a.result()["t0"],
                                           "t1": a.result()["t1"],
                                           "combined": a.result()["weighted_pearson"]}
                                 for s, a in zip(strengths, acc)},
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
