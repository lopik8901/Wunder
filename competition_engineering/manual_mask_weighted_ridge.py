"""Train-only residual ridge with search-mask propensity transfer.

Research hypothesis: uniform training rows underrepresent states selected by the
official score mask. Search labels are never used for fitting this model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.mask_protocol import _bins, verify_frozen_spec
from competition_engineering.manual_temporal_residual import frozen_prediction
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import evaluate, features


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("train", "search", "reference", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("output already exists")
    spec = verify_frozen_spec()
    root_path = Path(args.reference)
    if hashlib.sha256(root_path.read_bytes()).hexdigest() != spec["frozen_root_sha256"]:
        raise ValueError("reference is not the frozen mask-protocol root")
    with np.load(root_path) as archive:
        root = {key: archive[key].copy() for key in archive.files}
    train_identity = json.loads((Path(args.train) / "identity.json").read_text())
    search_identity = json.loads((Path(args.search) / "identity.json").read_text())
    if train_identity["split"] != "train_medium" or len(train_identity["groups"]) != 4096:
        raise ValueError("expected frozen 4096-sequence training cache")
    if hashlib.sha256((Path(args.search) / "identity.json").read_bytes()).hexdigest() != spec["search_cache_identity_sha256"]:
        raise ValueError("search cache identity differs from frozen spec")
    if search_identity["split"] != "tune":
        raise ValueError("search evaluation must use designated tune split")
    counts = np.asarray(spec["search_required_counts"], np.float64)
    selected = np.asarray(spec["search_scored_counts"], np.float64)
    propensity = (selected + .5) / (counts + 1)
    baseline_rate = spec["search_scored_total"] / spec["search_required_total"]
    started = time.perf_counter()
    gram = np.zeros((2, 115, 115), np.float64)
    rhs = np.zeros((2, 115), np.float64)
    rows = 0
    with threadpool_limits(limits=1):
        for _, z in cached(args.train):
            index = np.flatnonzero(z["need"])[::20]
            calibrated = (z["p"] * root["base_scale"].astype(np.float32)
                          + root["base_bias"].astype(np.float32)).astype(np.float32)
            f = features({**z, "p": calibrated}, root["mean"], root["scale"])[index].astype(np.float64)
            position, magnitude = _bins(z, root, spec)
            # Fixed search-derived propensity is a *training weight* only. The
            # callback remains causal and has no access to is_scored.
            focus = propensity[position[index], magnitude[index]] / baseline_rate
            target = z["y"][index].copy()
            target[:, 1] = np.clip(target[:, 1], -2, 2)
            residual = target - calibrated[index]
            weights = np.abs(np.clip(z["y"][index], -2, 2)) * focus[:, None]
            rows += len(index)
            for k in range(2):
                w = weights[:, k]
                gram[k] += f.T @ (f * w[:, None])
                rhs[k] += f.T @ (residual[:, k] * w)
        coef = np.column_stack([
            np.linalg.solve(gram[k] + np.eye(115) * rows * .1, rhs[k])
            for k in range(2)
        ])
        search = evaluate(args.search, root["mean"], root["scale"], coef,
                          base_scale=root["base_scale"], base_bias=root["base_bias"])
        final = evaluate(args.search, root["mean"], root["scale"], coef,
                         strengths=search["chosen_strengths"],
                         base_scale=root["base_scale"], base_bias=root["base_bias"])
    out.mkdir(parents=True)
    np.savez(out / "ridge.npz", mean=root["mean"], scale=root["scale"], coef=coef,
             strengths=search["chosen_strengths"], base_scale=root["base_scale"],
             base_bias=root["base_bias"])
    report = {"evidence": "reused_search_evidence_not_independent_validation",
              "hypothesis": "transfer coarse causal score-mask propensity into train-only residual fitting",
              "protocol_spec_sha256": hashlib.sha256(
                  Path("competition_engineering/protocols/mask_aware_v1.json").read_bytes()).hexdigest(),
              "training_sequences": 4096, "sampled_train_rows": rows,
              "training_target_source": "training cache only",
              "search_split": "gru_tune", "search_result": search,
              "selected_result": final, "runtime_seconds": time.perf_counter() - started}
    write_json(out / "report.json", report)
    print(json.dumps({"selected_strengths": search["chosen_strengths"],
                      "search_t0": final["candidate"]["t0"],
                      "search_t1": final["candidate"]["t1"],
                      "search_combined": final["candidate"]["weighted_pearson"],
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
