"""Search-only causal prediction-innovation regime experts."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits

from competition_engineering.manual_search_core import (COMBO, TRAIN_1024, assess,
                                                       load_combo, predict_combo)
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import features


def signal(base):
    # Cached GRU outputs are NaN on warm-up rows. In deployment, the callback
    # returns None there; use a fixed zero state until the first required row.
    finite = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    innovation = np.abs(np.diff(finite, axis=0, prepend=finite[:1])).mean(axis=1)
    return lfilter([.02], [1., -.98], innovation).astype(np.float32)


def design(z, combo):
    calibrated = (z["p"] * combo["base_scale"].astype(np.float32)
                  + combo["base_bias"].astype(np.float32)).astype(np.float32)
    return features({**z, "p": calibrated}, combo["mean"], combo["scale"])


def main():
    out = Path("competition_engineering/runs/manual_innovation_gate1024_20261001")
    if out.exists():
        raise ValueError("frozen experiment output exists")
    started = time.perf_counter()
    combo = load_combo()
    samples = []
    with threadpool_limits(limits=1):
        for n, (_, z) in enumerate(cached(TRAIN_1024)):
            if n == 128:
                break
            idx = np.flatnonzero(z["need"])[::20]
            samples.append(signal(predict_combo(z, combo))[idx])
        threshold = float(np.median(np.concatenate(samples)))
        gram = np.zeros((2, 2, 115, 115), np.float64)
        rhs = np.zeros((2, 2, 115), np.float64)
        rows = np.zeros(2, np.int64)
        for _, z in cached(TRAIN_1024):
            idx = np.flatnonzero(z["need"])[::20]
            base = predict_combo(z, combo)
            gate = signal(base)[idx] >= threshold
            f = design(z, combo)[idx].astype(np.float64)
            y = np.clip(z["y"][idx], -2, 2)
            residual = y - base[idx]
            weight = np.abs(y)
            for expert, selected in enumerate((~gate, gate)):
                ff = f[selected]
                rows[expert] += len(ff)
                for k in range(2):
                    w = weight[selected, k]
                    gram[expert, k] += ff.T @ (ff * w[:, None])
                    rhs[expert, k] += ff.T @ (residual[selected, k] * w)
        coef = np.stack([np.column_stack([
            np.linalg.solve(gram[e, k] + np.eye(115) * rows[e] * .1, rhs[e, k])
            for k in range(2)]) for e in range(2)])

        def predict(z, base, strengths):
            f = design(z, combo)
            gate = signal(base) >= threshold
            correction = np.where(gate[:, None], f @ coef[1], f @ coef[0])
            return base + correction * np.asarray(strengths, np.float32)

        grid = {s: assess(lambda z, base, s=s: predict(z, base, (s, s)), diagnostics=False)
                for s in (.25, .5, 1.)}
        chosen = [max(grid, key=lambda s: grid[s]["candidate"][target])
                  for target in ("t0", "t1")]
        result = assess(lambda z, base: predict(z, base, chosen))
    out.mkdir(parents=True)
    np.savez(out / "innovation_gate.npz", coef=coef, threshold=threshold,
             strengths=chosen, combo_sha256=hashlib.sha256(COMBO.read_bytes()).hexdigest())
    report = {"hypothesis": "persistent errors are conditional on recent causal prediction-change regime",
              "mechanism": "two residual ridge experts gated by 50-row EMA of absolute frozen prediction change",
              "research_cards": ["sparse_expert_gating", "online_polynomial_memory"],
              "threshold_source": "first 128 training sequences, unlabeled",
              "threshold": threshold, "training_sequences": 1024,
              "expert_rows": rows.tolist(), "chosen_strengths": chosen,
              "strength_search": {str(k): v["candidate"] for k, v in grid.items()},
              "search_result": result, "runtime_seconds": time.perf_counter() - started,
              "artifact_sha256": hashlib.sha256((out / "innovation_gate.npz").read_bytes()).hexdigest()}
    write_json(out / "report.json", report)
    print(json.dumps({"search_wp": result["candidate"]["weighted_pearson"],
                      "delta": result["delta_combined"], "threshold": threshold,
                      "strengths": chosen, "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
