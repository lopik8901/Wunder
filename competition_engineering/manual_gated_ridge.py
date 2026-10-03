"""Train-only two-expert causal ridge, gated by frozen prediction magnitude."""
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
from competition_engineering.residual import features
from wnn_connectome_starterpack.utils import GlobalAccumulator


def predict(z, root, coef, strengths):
    calibrated = (z["p"] * root["base_scale"].astype(np.float32)
                  + root["base_bias"].astype(np.float32)).astype(np.float32)
    f = features({**z, "p": calibrated}, root["mean"], root["scale"])
    magnitude = np.max(np.abs(frozen_prediction(z, root)), axis=1)
    # Both expert outputs are causal; the frozen root itself is a causal model.
    low = f @ coef[0]
    high = f @ coef[1]
    correction = np.where((magnitude >= 1.0)[:, None], high, low)
    return calibrated + strengths * correction


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
    gram = np.zeros((2, 2, 115, 115), np.float64)
    rhs = np.zeros((2, 2, 115), np.float64)
    rows = np.zeros(2, np.int64)
    with threadpool_limits(limits=1):
        for _, z in cached(args.train):
            idx = np.flatnonzero(z["need"])[::20]
            root_p = frozen_prediction(z, root)[idx]
            gate = np.max(np.abs(root_p), axis=1) >= 1.0
            calibrated = (z["p"] * root["base_scale"].astype(np.float32)
                          + root["base_bias"].astype(np.float32)).astype(np.float32)
            f = features({**z, "p": calibrated}, root["mean"], root["scale"])[idx].astype(np.float64)
            target = z["y"][idx].copy()
            target[:, 1] = np.clip(target[:, 1], -2, 2)
            residual = target - calibrated[idx]
            weight = np.abs(np.clip(z["y"][idx], -2, 2))
            for expert, selected in enumerate((~gate, gate)):
                if not selected.any():
                    continue
                rows[expert] += int(selected.sum())
                ff = f[selected]
                for k in range(2):
                    ww = weight[selected, k]
                    gram[expert, k] += ff.T @ (ff * ww[:, None])
                    rhs[expert, k] += ff.T @ (residual[selected, k] * ww)
        coef = np.stack([
            np.column_stack([np.linalg.solve(gram[e, k] + np.eye(115) * rows[e] * 0.1,
                                             rhs[e, k]) for k in range(2)])
            for e in range(2)
        ])
        strengths = (0.0, 0.25, 0.5, 1.0)
        acc = [GlobalAccumulator() for _ in strengths]
        for _, z in cached(args.tune):
            for s, a in zip(strengths, acc):
                a.add(z["y"], predict(z, root, coef, s), z["mask"])
    scores = [a.result() for a in acc]
    chosen = [strengths[max(range(len(strengths)), key=lambda i: scores[i][target])]
              for target in ("t0", "t1")]
    out.mkdir(parents=True)
    np.savez(out / "gated_ridge.npz", coef=coef, strengths=chosen,
             reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
    report = {"evidence": "reused_search_evidence_not_independent_validation",
              "hypothesis": "low and high frozen-prediction regimes need different causal correction coefficients",
              "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
              "training_sequences": len(json.loads((Path(args.train) / "identity.json").read_text())["groups"]),
              "expert_rows": rows.tolist(), "gate": "max(abs(frozen_prediction)) >= 1.0",
              "search_scores": dict(zip(map(str, strengths), scores)),
              "chosen_target_strengths": chosen,
              "chosen_search_wp": float(np.mean([max(score[target] for score in scores)
                                                     for target in ("t0", "t1")])),
              "runtime_seconds": time.perf_counter() - started}
    write_json(out / "report.json", report)
    print(json.dumps({"training_sequences": report["training_sequences"],
                      "expert_rows": report["expert_rows"],
                      "chosen_target_strengths": chosen,
                      "chosen_search_wp": report["chosen_search_wp"],
                      "grid": {str(s): {"t0": v["t0"], "t1": v["t1"], "combined": v["weighted_pearson"]}
                               for s, v in zip(strengths, scores)}}, indent=2))


if __name__ == "__main__":
    main()
