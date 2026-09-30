"""Supervisor-only complete-validation comparison of frozen deployment and incumbent.

Never import this file into the MLEvolve search executor or expose its output
to MLEvolve prompts, journals, retrieval, or candidate result objects.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from wnn_connectome_starterpack.utils import DataPoint, GlobalAccumulator

from competition_engineering.deployment.build_and_test import ROOT, optimized
from competition_engineering.pipeline import read_sequence
from competition_engineering.residual import from_stats, sufficient


def main():
    report_path = ROOT / "competition_engineering/reports/promoted_cpu_v1_complete_validation.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    zip_path = ROOT / "competition_engineering/submissions/promoted_20260929_014231_cpu_v1.zip"
    incumbent_zip = ROOT / "competition_engineering/submissions/ridge1024_targetwise.zip"
    incumbent_file = ROOT / "competition_engineering/checkpoints/ridge1024_targetwise_v1/ridge.npz"
    with np.load(incumbent_file) as z:
        ridge = {k: z[k] for k in z.files}
    data_path = ROOT / "competition_engineering/assets/wnn_connectome_starterpack/datasets/valid.parquet"
    pf = pq.ParquetFile(data_path)
    candidate = optimized()
    accumulators = [GlobalAccumulator(), GlobalAccumulator()]
    moments = []
    start = time.perf_counter()
    callback_time = 0.0
    for group in range(pf.num_row_groups):
        seq, need, x, y, mask = read_sequence(pf, group)
        predictions = [np.full_like(y, np.nan), np.full_like(y, np.nan)]
        step_start = time.perf_counter()
        for step, row in enumerate(x):
            point = DataPoint(seq, step, bool(need[step]), row)
            out = candidate.predict(point)
            if not need[step]:
                if out is not None:
                    raise ValueError("warm-up output was not None")
                continue
            if out is None or np.asarray(out).shape != (2,) or not np.isfinite(out).all():
                raise ValueError("invalid candidate output")
            raw = candidate.last_raw
            base = (raw * ridge["base_scale"].astype(np.float32) + ridge["base_bias"].astype(np.float32)).astype(np.float32)
            features = np.concatenate(([1.], np.clip((row-ridge["mean"])/ridge["scale"], -8, 8), base)).astype(np.float32)
            predictions[0][step] = (base + ridge["strengths"] * (features @ ridge["coef"])).astype(np.float32)
            predictions[1][step] = out
        callback_time += time.perf_counter()-step_start
        for k in range(2):
            accumulators[k].add(y, predictions[k], mask)
        moments.append([sufficient(y[mask], predictions[k][mask]) for k in range(2)])
        if (group+1) % 64 == 0:
            print(f"{group+1}/{pf.num_row_groups} sequences, {time.perf_counter()-start:.0f}s elapsed", flush=True)
    moments = np.asarray(moments)
    values = [from_stats(x) for x in moments.sum(0)]
    official = [a.result() for a in accumulators]
    for k in range(2):
        np.testing.assert_allclose(values[k], [official[k]["t0"], official[k]["t1"]], atol=1e-9)
    rng = np.random.default_rng(20260928)
    differences = []
    for _ in range(1000):
        index = rng.integers(len(moments), size=len(moments))
        subtotal = moments[index].sum(0)
        differences.append((from_stats(subtotal[1]) - from_stats(subtotal[0])).mean())
    report = {"metric_domain": "complete local validation; supervisor-only promotion",
              "sequences": pf.num_row_groups, "rows": pf.metadata.num_rows,
              "incumbent_wp": values[0].tolist(), "candidate_wp": values[1].tolist(),
              "incumbent_combined_wp": float(values[0].mean()),
              "candidate_combined_wp": float(values[1].mean()),
              "deltas": (values[1]-values[0]).tolist(),
              "combined_delta": float((values[1]-values[0]).mean()),
              "paired_95ci_combined": np.quantile(differences, [.025,.975]).tolist(),
              "wall_seconds": time.perf_counter()-start,
              "callback_seconds": callback_time,
              "callback_us_per_row": callback_time/pf.metadata.num_rows*1e6,
              "candidate_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
              "incumbent_zip_sha256": hashlib.sha256(incumbent_zip.read_bytes()).hexdigest(),
              "official_accumulator": official}
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
