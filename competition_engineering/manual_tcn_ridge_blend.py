"""Search-only complementarity test for two frozen, CPU-deployable models."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from competition_engineering.compare_ridges import predict as ridge_predict
from competition_engineering.gpu_residual import ResidualTCN, input_features, predict_sequence, system_cached
from competition_engineering.pipeline import cached, write_json
from wnn_connectome_starterpack.utils import GlobalAccumulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ridge", required=True)
    p.add_argument("--tcn", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    started = time.perf_counter()
    with np.load(args.ridge) as q:
        ridge = {k: q[k].copy() for k in q.files}
    state = torch.load(args.tcn, map_location="cpu", weights_only=False)
    model = ResidualTCN()
    model.load_state_dict(state["state_dict"])
    model.eval()
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    accumulators = [GlobalAccumulator() for _ in alphas]
    with threadpool_limits(limits=1):
        raw_cache = cached("competition_engineering/cache/gru_tune")
        tcn_cache = system_cached("competition_engineering/cache/gru_tune", state["base_scale"],
                                  state["base_bias"], state["base_ridge"])
        for (_, raw), (_, base) in zip(raw_cache, tcn_cache, strict=True):
            correction = predict_sequence(model, input_features(base, state["mean"], state["scale"]),
                                          torch.device("cpu"))
            tcn = base["p"] + np.asarray(state["strengths"]) * correction
            linear = ridge_predict(raw, ridge)
            for alpha, accumulator in zip(alphas, accumulators):
                accumulator.add(raw["y"], (1 - alpha) * linear + alpha * tcn, raw["mask"])
    scores = [a.result() for a in accumulators]
    chosen = [alphas[max(range(len(alphas)), key=lambda i: scores[i][target])]
              for target in ("t0", "t1")]
    report = {"evidence": "reused_search_evidence_not_independent_validation",
              "ridge_sha256": hashlib.sha256(Path(args.ridge).read_bytes()).hexdigest(),
              "tcn_sha256": hashlib.sha256(Path(args.tcn).read_bytes()).hexdigest(),
              "tcn_weight_grid": dict(zip(map(str, alphas), scores)),
              "chosen_target_tcn_weights": chosen,
              "chosen_search_wp": float(np.mean([max(score[target] for score in scores)
                                                     for target in ("t0", "t1")])),
              "runtime_seconds": time.perf_counter() - started}
    write_json(args.output, report)
    print(json.dumps({"chosen_target_tcn_weights": chosen,
                      "chosen_search_wp": report["chosen_search_wp"],
                      "grid": {str(a): {"t0": s["t0"], "t1": s["t1"], "combined": s["weighted_pearson"]}
                               for a, s in zip(alphas, scores)},
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
