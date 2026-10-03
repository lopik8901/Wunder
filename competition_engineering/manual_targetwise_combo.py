"""Search-only paired comparison of frozen target-specific correction models."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.compare_ridges import predict as ridge_predict
from competition_engineering.manual_gated_ridge import predict as gated_predict
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import from_stats, sufficient
from wnn_connectome_starterpack.utils import GlobalAccumulator


def load(path):
    with np.load(path) as q:
        return {k: q[k].copy() for k in q.files}


def main():
    p = argparse.ArgumentParser()
    for key in ("root", "magnitude", "gated", "output"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    root, magnitude, gated = (load(path) for path in (args.root, args.magnitude, args.gated))
    reference = GlobalAccumulator()
    candidate = GlobalAccumulator()
    moments = []
    with threadpool_limits(limits=1):
        for _, z in cached("competition_engineering/cache/gru_tune"):
            base = ridge_predict(z, root)
            high = ridge_predict(z, magnitude)
            alternate = gated_predict(z, root, gated["coef"], gated["strengths"])
            prediction = np.column_stack((high[:, 0], alternate[:, 1])).astype(np.float32)
            reference.add(z["y"], base, z["mask"])
            candidate.add(z["y"], prediction, z["mask"])
            m = z["mask"]
            moments.append([sufficient(z["y"][m], base[m]),
                            sufficient(z["y"][m], prediction[m])])
    moments = np.asarray(moments)
    rng = np.random.default_rng(20261001)
    differences = []
    for _ in range(1000):
        idx = rng.integers(len(moments), size=len(moments))
        selected = moments[idx].sum(0)
        differences.append(from_stats(selected[1]) - from_stats(selected[0]))
    differences = np.asarray(differences)
    scores = [reference.result(), candidate.result()]
    sources = {"root": args.root, "magnitude": args.magnitude, "gated": args.gated}
    report = {"evidence": "reused_search_evidence_not_independent_validation",
              "candidate": "t0 magnitude-weighted ridge4096; t1 high/low gated ridge4096",
              "source_sha256": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                for key, path in sources.items()},
              "reference_search_wp": scores[0], "candidate_search_wp": scores[1],
              "delta_combined": scores[1]["weighted_pearson"] - scores[0]["weighted_pearson"],
              "paired_95ci_combined": np.quantile(differences.mean(1), [.025, .975]).tolist(),
              "paired_95ci_per_target": np.quantile(differences, [.025, .975], axis=0).tolist(),
              "runtime_seconds": time.perf_counter() - started}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
