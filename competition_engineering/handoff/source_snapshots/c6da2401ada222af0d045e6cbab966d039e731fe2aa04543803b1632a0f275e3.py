"""Train-only ridge residual baseline and held-out target-specific shrinkage."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.pipeline import cached, score_arrays, write_json
from wnn_connectome_starterpack.utils import GlobalAccumulator, weighted_pearson


def calibrated_cached(directory, scale, bias):
    for group,z in cached(directory):
        z["gru_p"] = z["p"]
        z["p"]=(z["p"]*np.asarray(scale,dtype=np.float32)+np.asarray(bias,dtype=np.float32)).astype(np.float32)
        yield group,z


def features(z, mean, scale):
    x = np.clip((z["x"]-mean)/scale, -8, 8)
    p = np.nan_to_num(z["p"], nan=0)
    return np.column_stack([np.ones(len(x)), x, p]).astype(np.float32)


def sufficient(y, p):
    y = np.clip(y, -2, 2).astype(np.float64)
    p = np.clip(p, -2, 2).astype(np.float64)
    w = np.abs(y)
    return np.stack([w.sum(0),(w*y).sum(0),(w*p).sum(0),
                     (w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)])


def from_stats(s):
    w, sy, sp, syy, spp, syp = s
    vy = np.maximum(syy-sy*sy/np.maximum(w, 1e-300),0)
    vp = np.maximum(spp-sp*sp/np.maximum(w, 1e-300),0)
    denom = np.sqrt(vy*vp)
    valid = (w >= 1e-8) & (vy/np.maximum(w,1e-300) > 1e-16) & (vp/np.maximum(w,1e-300) > 1e-16)
    result = np.zeros(2)
    result[valid] = (syp-sy*sp/np.maximum(w,1e-300))[valid]/denom[valid]
    return np.clip(result,-1,1)


def evaluate(directory, mean, scale, coef, strengths=None, base_scale=(1,1), base_bias=(0,0)):
    strengths_grid = [0., .05, .1, .25, .5, 1.]
    base_acc = GlobalAccumulator()
    scores = [GlobalAccumulator() for _ in strengths_grid]
    base_stats, candidate_stats, details = [], [], []
    diagnostics = None
    if strengths is not None:
        from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
        diagnostics = SearchErrorDiagnostics()
    for group,z in calibrated_cached(directory,base_scale,base_bias):
        correction = features(z,mean,scale) @ coef
        base_acc.add(z["y"], z["p"], z["mask"])
        if strengths is None:
            for strength,acc in zip(strengths_grid,scores):
                acc.add(z["y"],z["p"]+strength*correction,z["mask"])
        else:
            candidate = z["p"] + np.array(strengths)*correction
            diagnostics.add(z, candidate)
            scores[0].add(z["y"],candidate,z["mask"])
            base_stats.append(sufficient(z["y"][z["mask"]],z["p"][z["mask"]]))
            candidate_stats.append(sufficient(z["y"][z["mask"]],candidate[z["mask"]]))
            details.append({"group": group, "seq": int(z["seq"]),
                            "base": from_stats(base_stats[-1]).tolist(),
                            "candidate": from_stats(candidate_stats[-1]).tolist()})
    baseline = base_acc.result()
    if strengths is None:
        grid = [{"strength": strength, **acc.result()} for strength,acc in zip(strengths_grid,scores)]
        chosen = [max(grid,key=lambda r:r[k])["strength"] for k in ["t0","t1"]]
        return {"baseline": baseline, "grid": grid, "chosen_strengths": chosen}
    b, c = np.stack(base_stats), np.stack(candidate_stats)
    np.testing.assert_allclose(from_stats(b.sum(0)), [baseline["t0"],baseline["t1"]], atol=1e-10)
    rng = np.random.default_rng(20260928)
    differences = []
    for _ in range(500):
        idx = rng.integers(len(b),size=len(b))
        differences.append(from_stats(c[idx].sum(0))-from_stats(b[idx].sum(0)))
    differences = np.array(differences)
    return {"baseline": baseline, "candidate": scores[0].result(), "strengths": strengths,
            "paired_sequence_bootstrap_95ci_per_target": np.quantile(differences,[.025,.975],axis=0).tolist(),
            "paired_sequence_bootstrap_95ci_combined": np.quantile(differences.mean(1),[.025,.975]).tolist(),
            "sequences": details,
            "search_error_diagnostics": diagnostics.result()}


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ["train", "tune", "output"]:
        p.add_argument("--"+name,required=True)
    p.add_argument("--base-scale",type=float,nargs=2,default=[1,1])
    p.add_argument("--base-bias",type=float,nargs=2,default=[0,0])
    p.add_argument("--target-mode",choices=["raw","clipped","t0_raw_t1_clipped","t0_clipped_t1_raw"],default="raw")
    p.add_argument("--sample-stride",type=int,default=10)
    p.add_argument("--ridge-penalty",type=float,default=1e-3)
    args = p.parse_args()
    if not 1 <= args.sample_stride <= 50 or not 1e-7 <= args.ridge_penalty <= 1.0:
        raise ValueError("sample stride or ridge penalty outside bounded search range")
    out = Path(args.output)
    if out.exists():
        raise ValueError("use a fresh experiment output directory")
    out.mkdir(parents=True)
    started = time.perf_counter()
    count=0; sums=np.zeros(112); squares=np.zeros(112)
    for _,z in cached(args.train):
        x=z["x"][::args.sample_stride].astype(np.float64)
        count+=len(x); sums+=x.sum(0); squares+=(x*x).sum(0)
    mean=sums/count; scale=np.sqrt(np.maximum(squares/count-mean*mean,1e-6))
    # Same sampled observations for each target, with official target weights.
    matrices=np.zeros((2,115,115)); rhs=np.zeros((2,115)); rows=0
    with threadpool_limits(limits=1):
        for _,z in calibrated_cached(args.train,args.base_scale,args.base_bias):
            idx=np.flatnonzero(z["need"])[::args.sample_stride]
            x=features(z,mean,scale)[idx].astype(np.float64)
            targets=z["y"].copy()
            if args.target_mode in {"clipped","t0_clipped_t1_raw"}:
                targets[:,0]=np.clip(targets[:,0],-2,2)
            if args.target_mode in {"clipped","t0_raw_t1_clipped"}:
                targets[:,1]=np.clip(targets[:,1],-2,2)
            residual=(targets-z["p"])[idx].astype(np.float64)
            weights=np.abs(np.clip(z["y"][idx],-2,2)).astype(np.float64)
            rows+=len(idx)
            for k in range(2):
                matrices[k]+=x.T@(x*weights[:,k,None])
                rhs[k]+=x.T@(weights[:,k]*residual[:,k])
        coef=np.column_stack([np.linalg.solve(matrices[k]+np.eye(115)*rows*args.ridge_penalty,rhs[k]) for k in range(2)])
        tune=evaluate(args.tune,mean,scale,coef,base_scale=args.base_scale,base_bias=args.base_bias)
        diagnostic_result=evaluate(args.tune,mean,scale,coef,strengths=tune["chosen_strengths"],
                                   base_scale=args.base_scale,base_bias=args.base_bias)
        tune["search_error_diagnostics"]=diagnostic_result["search_error_diagnostics"]
    np.savez(out/"ridge.npz",mean=mean,scale=scale,coef=coef,strengths=tune["chosen_strengths"],
             base_scale=args.base_scale,base_bias=args.base_bias)
    report={"hypothesis":"A train-only linear correction captures information missed by frozen GRU",
            "model":"ridge on current standardized features and unchanged GRU outputs",
            "objective":"target-absolute-weighted residual MSE with configurable ridge regularization",
            "train_cache":args.train,"tune_cache":args.tune,
            "base_scale":args.base_scale,"base_bias":args.base_bias,
            "target_mode":args.target_mode,
            "sample_stride":args.sample_stride,"ridge_penalty":args.ridge_penalty,
            "training_rows":rows,"runtime_seconds":time.perf_counter()-started,
            "tune":tune}
    write_json(out/"report.json",report)
    print(json.dumps({"runtime_seconds":report["runtime_seconds"],"strengths":tune["chosen_strengths"],
                      "tune":tune["grid"]},indent=2))


if __name__ == "__main__":
    main()
