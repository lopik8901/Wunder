"""Supervisor-only parallel sequence evaluation; results never enter search context."""
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from wnn_connectome_starterpack.utils import DataPoint

from competition_engineering.deployment.build_and_test import ROOT, optimized
from competition_engineering.pipeline import read_sequence
from competition_engineering.residual import from_stats, sufficient

DATA = ROOT / "competition_engineering/assets/wnn_connectome_starterpack/datasets/valid.parquet"
INCUMBENT = ROOT / "competition_engineering/checkpoints/ridge1024_targetwise_v1/ridge.npz"
ZIP = ROOT / "competition_engineering/submissions/promoted_20260929_014231_cpu_v1.zip"
REPORT = ROOT / "competition_engineering/reports/promoted_cpu_v1_complete_validation.json"


def worker(groups):
    with threadpool_limits(limits=1):
        pf = pq.ParquetFile(DATA)
        model = optimized()
        with np.load(INCUMBENT) as z:
            ridge = {k: z[k] for k in z.files}
        scale = ridge["base_scale"].astype(np.float32)
        bias = ridge["base_bias"].astype(np.float32)
        rows = []
        callback = 0.0
        for group in groups:
            seq, need, x, y, mask = read_sequence(pf, group)
            pred0 = np.full_like(y, np.nan)
            pred1 = np.full_like(y, np.nan)
            started = time.perf_counter()
            for step, state in enumerate(x):
                out = model.predict(DataPoint(seq, step, bool(need[step]), state))
                if not need[step]:
                    if out is not None:
                        raise ValueError("warm-up output not None")
                    continue
                if out is None or np.asarray(out).shape != (2,) or not np.isfinite(out).all():
                    raise ValueError("invalid candidate output")
                raw = model.last_raw
                base = (raw*scale+bias).astype(np.float32)
                feat = np.concatenate(([1.], np.clip((state-ridge["mean"])/ridge["scale"], -8, 8), base)).astype(np.float32)
                pred0[step] = (base+ridge["strengths"]*(feat@ridge["coef"])).astype(np.float32)
                pred1[step] = out
            callback += time.perf_counter()-started
            rows.append((group, sufficient(y[mask], pred0[mask]), sufficient(y[mask], pred1[mask])))
        return rows, callback


def main():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    pf = pq.ParquetFile(DATA)
    total = pf.num_row_groups
    chunks = [list(range(i, min(i+16, total))) for i in range(0, total, 16)]
    moments = [None]*total
    callback_seconds = 0.0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(worker, groups) for groups in chunks]
        done = 0
        for task in as_completed(pending):
            entries, callback = task.result()
            callback_seconds += callback
            for group, base, candidate in entries:
                moments[group] = [base, candidate]
            done += len(entries)
            if done % 64 == 0 or done == total:
                print(f"{done}/{total} sequences, {time.perf_counter()-started:.0f}s elapsed", flush=True)
    moments = np.asarray(moments)
    values = [from_stats(x) for x in moments.sum(0)]
    rng = np.random.default_rng(20260928)
    delta = []
    for _ in range(1000):
        ix = rng.integers(total, size=total)
        sums = moments[ix].sum(0)
        delta.append((from_stats(sums[1])-from_stats(sums[0])).mean())
    report = {"metric_domain": "complete local validation; supervisor-only promotion",
              "sequences": total, "rows": pf.metadata.num_rows,
              "incumbent_wp": values[0].tolist(), "candidate_wp": values[1].tolist(),
              "incumbent_combined_wp": float(values[0].mean()),
              "candidate_combined_wp": float(values[1].mean()),
              "deltas": (values[1]-values[0]).tolist(),
              "combined_delta": float((values[1]-values[0]).mean()),
              "paired_95ci_combined": np.quantile(delta, [.025,.975]).tolist(),
              "wall_seconds": time.perf_counter()-started,
              "aggregate_callback_seconds": callback_seconds,
              "candidate_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "incumbent_checkpoint_sha256": hashlib.sha256(INCUMBENT.read_bytes()).hexdigest()}
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
