"""Supervisor-only protected holdout check for an eligible frozen search winner.

This module is not imported by the MLEvolve executor and is not a kind in its
literal experiment specification. Its output must never enter search memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.compare_ridges import predict as ridge_predict
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import from_stats, sufficient

ROOT = Path(__file__).resolve().parents[1]
INCUMBENT_SEARCH_WP = 0.6548654996120564
MIN_SEARCH_GAIN = 0.001
HOLDOUT = ROOT / "competition_engineering/cache/gru_train_holdout_phase2"
INCUMBENT = ROOT / "competition_engineering/checkpoints/ridge1024_targetwise_v1/ridge.npz"


def _load_npz(path):
    with np.load(path) as values:
        return {name: values[name] for name in values.files}


def promote(search_result: Path, output: Path):
    result = json.loads(search_result.read_text(encoding="utf-8"))
    if result.get("status") != "success" or result.get("metric_domain") != "fixed development tuning sequences":
        raise ValueError("only successful fixed-search candidates may enter promotion")
    if result["primary_metric"] < INCUMBENT_SEARCH_WP + MIN_SEARCH_GAIN:
        raise ValueError("candidate did not meet the predetermined search gain")
    run_dir = search_result.parent
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    kind = config["kind"]
    artifact = Path(config["output"])
    if kind == "ridge":
        candidate = _load_npz(artifact / "ridge.npz")
        predict = lambda z: ridge_predict(z, candidate)
        artifact_file = artifact / "ridge.npz"
    elif kind == "calibration":
        candidate = _load_npz(artifact / "calibration.npz")
        predict = lambda z: z["p"] * candidate["base_scale"] + candidate["base_bias"]
        artifact_file = artifact / "calibration.npz"
    elif kind == "gpu_residual":
        import torch
        from competition_engineering.gpu_residual import ResidualTCN, input_features, predict_sequence
        state = torch.load(artifact / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = ResidualTCN()
        model.load_state_dict(state["state_dict"])
        model.eval()
        def predict(z):
            base = z["p"] * np.asarray(state["base_scale"]) + np.asarray(state["base_bias"])
            if state["base_ridge"] is not None:
                base = ridge_predict(z, state["base_ridge"])
            local = {**z, "p": base.astype(np.float32)}
            correction = predict_sequence(model, input_features(local, state["mean"], state["scale"]), torch.device("cpu"))
            return base + np.asarray(state["strengths"]) * correction
        artifact_file = artifact / "checkpoint.pt"
    else:
        raise ValueError("unsupported candidate family")
    incumbent = _load_npz(INCUMBENT)
    moments = []
    with threadpool_limits(limits=1):
        for _, z in cached(HOLDOUT):
            mask = z["mask"]
            moments.append([sufficient(z["y"][mask], prediction[mask]) for prediction in
                            (ridge_predict(z, incumbent), predict(z))])
    moments = np.asarray(moments)
    values = [from_stats(x) for x in moments.sum(0)]
    rng = np.random.default_rng(20260928)
    differences = []
    for _ in range(1000):
        indices = rng.integers(len(moments), size=len(moments))
        subtotal = moments[indices].sum(0)
        differences.append((from_stats(subtotal[1]) - from_stats(subtotal[0])).mean())
    interval = np.quantile(differences, [.025, .975])
    report = {"candidate_kind": kind, "search_metric": result["primary_metric"],
              "search_gain": result["primary_metric"] - INCUMBENT_SEARCH_WP,
              "reference_wp": values[0].tolist(), "candidate_wp": values[1].tolist(),
              "combined_gain": float((values[1] - values[0]).mean()),
              "paired_95ci_combined": interval.tolist(),
              "positive_paired_interval": bool(interval[0] > 0),
              "artifact_sha256": hashlib.sha256(artifact_file.read_bytes()).hexdigest(),
              "holdout_sequences": len(moments)}
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("search_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(promote(args.search_result, args.output), indent=2))
