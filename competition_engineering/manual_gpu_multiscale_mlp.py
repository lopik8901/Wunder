"""Manual, search-only GPU test of a small causal multiscale residual MLP.

The source cache supplies the frozen official-GRU output and 112 inputs. Only
the designated 1024 training sequences and 64 search sequences are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from competition_engineering.manual_temporal_residual import frozen_prediction
from competition_engineering.pipeline import cached, write_json
from wnn_connectome_starterpack.utils import GlobalAccumulator


TRAIN = Path("competition_engineering/cache/gru_train_scale_phase2")
SEARCH = Path("competition_engineering/cache/gru_tune")


def design(z, root, indices, base):
    x = z["x"]
    blocks = []
    for lag in (0, 10, 100):
        previous = np.zeros((len(indices), 112), np.float32)
        valid = indices >= lag
        previous[valid] = x[indices[valid] - lag]
        norm = np.clip((previous - root["mean"]) / root["scale"], -8, 8)
        norm[~valid] = 0
        blocks.append(norm)
    return np.concatenate([*blocks, np.nan_to_num(base[indices], nan=0)], axis=1).astype(np.float32)


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(338, 64), nn.GELU(),
                                    nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--objective", choices=("huber", "weighted_pearson"), default="huber")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; do not fall back silently")
    out = Path(args.output)
    if out.exists():
        raise ValueError("output exists")
    ref = Path(args.reference)
    with np.load(ref) as q:
        root = {k: q[k].copy() for k in q.files}
    start = time.perf_counter()
    torch.manual_seed(20261001)
    rng = np.random.default_rng(20261001)
    feats, targets, weights, bases = [], [], [], []
    with threadpool_limits(limits=1):
        for _, z in cached(TRAIN):
            idx = np.flatnonzero(z["need"])[::20]
            base = frozen_prediction(z, root)
            y = np.clip(z["y"][idx], -2, 2)
            feats.append(design(z, root, idx, base))
            targets.append((y - base[idx]).astype(np.float32))
            weights.append(np.abs(y).astype(np.float32))
            bases.append(base[idx].astype(np.float32))
    train_x = np.concatenate(feats)
    train_y = np.concatenate(targets)
    train_w = np.concatenate(weights)
    train_b = np.concatenate(bases)
    del feats, targets, weights, bases
    device = torch.device("cuda:0")
    # The full sampled design is <2 GiB on this 16 GiB GPU. Check again before
    # copying so a competing workload cannot cause an uncontrolled OOM.
    free, total = torch.cuda.mem_get_info()
    required = train_x.nbytes + train_y.nbytes + train_w.nbytes + train_b.nbytes
    if free < required * 3:
        raise RuntimeError(f"insufficient free GPU memory: {free} for {required} input bytes")
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(train_y).to(device)
    w = torch.from_numpy(train_w).to(device)
    b = torch.from_numpy(train_b).to(device)
    model = ResidualMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    losses = []
    batch = 8192
    for epoch in range(5):
        order = torch.from_numpy(rng.permutation(len(train_x)).astype(np.int64)).to(device)
        total_loss = 0.0
        for start_idx in range(0, len(train_x), batch):
            index = order[start_idx:start_idx + batch]
            residual = model(x[index])
            if args.objective == "huber":
                loss = (w[index] * torch.nn.functional.smooth_l1_loss(
                    residual, y[index], reduction="none", beta=0.5)).mean()
            else:
                target = (y[index] + b[index]).clamp(-2, 2)
                prediction = (b[index] + residual).clamp(-2, 2)
                weight = w[index]
                mass = weight.sum(dim=0).clamp_min(1e-8)
                target_mean = (weight * target).sum(dim=0) / mass
                prediction_mean = (weight * prediction).sum(dim=0) / mass
                centered_target = target - target_mean
                centered_prediction = prediction - prediction_mean
                covariance = (weight * centered_target * centered_prediction).sum(dim=0)
                target_variance = (weight * centered_target.square()).sum(dim=0)
                prediction_variance = (weight * centered_prediction.square()).sum(dim=0)
                correlation = covariance / (target_variance * prediction_variance).clamp_min(1e-8).sqrt()
                loss = -correlation.mean() + 0.001 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
        losses.append(total_loss / len(train_x))
        print(f"epoch={epoch + 1} {args.objective}_loss={losses[-1]:.6f}", flush=True)
    torch.cuda.synchronize()
    model.eval()
    strengths = (0.0, 0.25, 0.5, 1.0)
    accumulators = [GlobalAccumulator() for _ in strengths]
    # Store only aggregate scores; search targets never enter the artifact.
    with torch.inference_mode(), threadpool_limits(limits=1):
        for _, z in cached(SEARCH):
            base = frozen_prediction(z, root)
            f = design(z, root, np.arange(len(z["x"])), base)
            correction = model(torch.from_numpy(f).to(device)).cpu().numpy()
            for strength, accumulator in zip(strengths, accumulators):
                accumulator.add(z["y"], base + strength * correction, z["mask"])
    scores = [accumulator.result() for accumulator in accumulators]
    chosen = [strengths[max(range(len(strengths)), key=lambda i: scores[i][target])]
              for target in ("t0", "t1")]
    out.mkdir(parents=True)
    arrays = {}
    for k in (0, 2, 4):
        layer = model.layers[k]
        arrays[f"weight_{k}"] = layer.weight.detach().cpu().numpy()
        arrays[f"bias_{k}"] = layer.bias.detach().cpu().numpy()
    np.savez(out / "mlp.npz", **arrays, strengths=chosen,
             reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
    report = {
        "evidence": "reused_search_evidence_not_independent_validation",
        "objective": args.objective,
        "hypothesis": "nonlinear interactions between current/10/100-step features and the frozen prediction explain persistent residuals",
        "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
        "train_sequences": 1024, "sampled_training_rows": len(train_x),
        "gpu": torch.cuda.get_device_name(0), "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        "training_losses": losses, "search_strengths": strengths,
        "search_scores": scores, "chosen_target_strengths": chosen,
        "chosen_search_wp": float(np.mean([max(score[target] for score in scores) for target in ("t0", "t1")])),
        "runtime_seconds": time.perf_counter() - start,
    }
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
