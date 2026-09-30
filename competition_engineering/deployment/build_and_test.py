"""Export frozen weights and compare row-wise callbacks on search-only sequences."""
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from wnn_connectome_starterpack.utils import DataPoint

from competition_engineering.gpu_residual import IncrementalTCN, ResidualTCN, point_base
from competition_engineering.pipeline import cached, load_model

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "competition_engineering/checkpoints/promoted_20260929_014231_best_distinct/checkpoint.pt"
GRU_SOURCE = ROOT / "competition_engineering/submissions/ridge1024_targetwise"
CACHE = ROOT / "competition_engineering/cache/gru_tune"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    assert sha(CHECKPOINT) == "afead41dbbe93cd888c59da19b24a5d416a2f007e4fbcbba386de4e97c3992e1"
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = ResidualTCN()
    model.load_state_dict(state["state_dict"])
    model.eval()
    ridge = state["base_ridge"]
    assert ridge is not None
    arrays = {"mean": state["mean"], "scale": state["scale"],
              "strengths": np.asarray(state["strengths"]),
              "base_scale": np.asarray(state["base_scale"], np.float32),
              "base_bias": np.asarray(state["base_bias"], np.float32),
              "ridge_mean": ridge["mean"], "ridge_scale": ridge["scale"],
              "ridge_coef": ridge["coef"], "ridge_strengths": ridge["strengths"],
              "head": model.head.weight.detach().numpy()[:, :, 0],
              "head_bias": model.head.bias.detach().numpy()}
    for k, layer in enumerate(model.convs):
        w = layer.weight.detach().numpy()
        arrays[f"weights_{k}"] = w.transpose(0, 2, 1).reshape(w.shape[0], -1).copy()
        arrays[f"biases_{k}"] = layer.bias.detach().numpy().copy()
    np.savez(DEPLOY / "model.npz", **arrays)
    for name in ("gru.py", "baseline.onnx"):
        shutil.copyfile(GRU_SOURCE / name, DEPLOY / name)
    return state, model


def frozen(state, model):
    gru = load_model(GRU_SOURCE / "gru.py")
    tcn = IncrementalTCN(model)
    last = None
    def predict(point):
        nonlocal gru, tcn, last
        if last != point.seq_ix:
            if last is not None:
                tcn = IncrementalTCN(model)
            last = point.seq_ix
        raw = gru.predict(point)
        base = point_base(raw, point.state, state["base_scale"], state["base_bias"], state["base_ridge"])
        x = np.concatenate([np.clip((point.state-state["mean"])/state["scale"], -8, 8),
                            np.zeros(2, np.float32) if base is None else base]).astype(np.float32)
        correction = tcn.predict(x)
        return None if base is None else (base + np.asarray(state["strengths"]) * correction).astype(np.float32)
    return predict


def optimized():
    sys.path.insert(0, str(DEPLOY))
    spec = importlib.util.spec_from_file_location("optimized_solution", DEPLOY / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PredictionModel()


def replay(predict, z):
    out = np.full((len(z["x"]), 2), np.nan, np.float32)
    started = time.perf_counter()
    for step, row in enumerate(z["x"]):
        p = predict(DataPoint(int(z["seq"]), step, bool(z["need"][step]), row))
        if z["need"][step]:
            assert p is not None and np.asarray(p).shape == (2,) and np.asarray(p).dtype == np.float32 and np.isfinite(p).all()
            out[step] = p
        else:
            assert p is None
    return out, time.perf_counter() - started


def main():
    state, model = export()
    reference = frozen(state, model)
    candidate = optimized()
    rows = []
    with threadpool_limits(limits=1):
        for i, (_, z) in enumerate(cached(CACHE)):
            if i >= 4:
                break
            a, ta = replay(reference, z)
            b, tb = replay(candidate.predict, z)
            delta = np.abs(a[z["need"]] - b[z["need"]])
            rows.append({"sequence": int(z["seq"]), "rows": len(a), "frozen_seconds": ta,
                         "optimized_seconds": tb, "max_abs": float(delta.max()),
                         "mean_abs": float(delta.mean())})
            print(rows[-1], flush=True)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
