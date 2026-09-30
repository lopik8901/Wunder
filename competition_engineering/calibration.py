"""Evaluate a bounded causal affine calibration of the frozen GRU on search data."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from competition_engineering.pipeline import cached, write_json
from wnn_connectome_starterpack.utils import GlobalAccumulator
from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("train", "tune", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--base-scale", type=float, nargs=2, required=True)
    parser.add_argument("--base-bias", type=float, nargs=2, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    if output.exists():
        raise ValueError("use a fresh output directory")
    output.mkdir(parents=True)
    acc = GlobalAccumulator()
    diagnostics = SearchErrorDiagnostics()
    scale = np.asarray(args.base_scale, dtype=np.float32)
    bias = np.asarray(args.base_bias, dtype=np.float32)
    for _, z in cached(args.tune):
        candidate = (z["p"] * scale + bias).astype(np.float32)
        acc.add(z["y"], candidate, z["mask"])
        diagnostics.add(z, candidate)
    score = acc.result()
    np.savez(output / "calibration.npz", base_scale=scale, base_bias=bias)
    write_json(output / "report.json", {
        "model": "causal affine calibration of frozen GRU output",
        "train_cache": args.train, "tune_cache": args.tune,
        "base_scale": args.base_scale, "base_bias": args.base_bias,
        "training_rows": 0, "runtime_seconds": time.perf_counter() - started,
        "tune": {"grid": [dict(strength=0.0, **score)],
                 "search_error_diagnostics": diagnostics.result()},
    })


if __name__ == "__main__":
    main()
