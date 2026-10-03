"""Sequence-streaming audit, aligned callback caches, and cheap diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn_connectome_starterpack.utils import (
    FEATURE_COLUMNS, TARGET_COLUMNS, DataPoint, GlobalAccumulator,
    validate_sequence, weighted_pearson,
)

ROOT = Path(__file__).resolve().parent


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path):
    path = Path(path).resolve()
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def model_identity(solution):
    return {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(solution).parent.iterdir())
            if p.is_file() and p.suffix in {".py", ".onnx", ".pt", ".pth", ".json", ".npz"}}


def load_model(solution):
    solution = Path(solution).resolve()
    # Official solution imports utils from the starter pack.
    sys.path.insert(0, str(solution.parent.parent))
    sys.path.insert(0, str(solution.parent))
    spec = importlib.util.spec_from_file_location("competition_solution", solution)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PredictionModel()


def read_sequence(parquet, group):
    table = parquet.read_row_group(int(group), use_threads=False)
    seq, need = validate_sequence(table)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite features/targets")
    mask = need.copy()
    if "is_scored" in table.column_names:
        mask &= table["is_scored"].to_numpy().astype(bool)
    return seq, need, x, y, mask


def audit(args):
    report = {"seed": 20260928, "datasets": {}}
    for split in ["train", "valid"]:
        path = Path(args.assets) / "datasets" / f"{split}.parquet"
        pf = pq.ParquetFile(path)
        expected = ["seq_ix", "step_in_seq", "need_prediction"]
        if split == "valid":
            expected += ["is_scored"]
        expected += FEATURE_COLUMNS + list(TARGET_COLUMNS)
        assert pf.schema_arrow.names == expected, pf.schema_arrow.names
        assert all(pf.metadata.row_group(i).num_rows == 20000 for i in range(pf.num_row_groups))
        ids=[]
        for i in range(pf.num_row_groups):
            stat=pf.metadata.row_group(i).column(0).statistics
            if stat is None or not stat.has_min_max:
                values=pf.read_row_group(i,columns=["seq_ix"])["seq_ix"].to_numpy()
                assert np.all(values==values[0]); ids.append(int(values[0]))
            else:
                assert stat.min==stat.max and stat.null_count==0
                ids.append(int(stat.min))
        assert len(set(ids))==len(ids), "sequence appears in multiple row groups"
        started = time.perf_counter()
        seq, need, x, y, mask = read_sequence(pf, 0)
        report["datasets"][split] = {
            **fingerprint(path), "rows": pf.metadata.num_rows, "groups": pf.num_row_groups,
            "columns": pf.schema_arrow.names, "sample_seq_id": seq,
            "unique_sequence_ids":len(set(ids)),
            "sample_read_seconds": time.perf_counter() - started,
            "sequence_feature_target_bytes": x.nbytes + y.nbytes,
            "full_feature_target_float32_bytes": pf.metadata.num_rows * 114 * 4,
        }
    rng = np.random.default_rng(report["seed"])
    train = rng.permutation(report["datasets"]["train"]["groups"]).tolist()
    valid = rng.permutation(report["datasets"]["valid"]["groups"]).tolist()
    assert len(train) >= 1024 and len(valid) > 128
    report["splits"] = {"train_pilot": train[:256], "train_scale": train[:1024],
                        "train_all": train, "tune": valid[:64],
                        "confirm": valid[64:128], "final_gate": valid[128:]}
    write_json(args.output, report)
    print(json.dumps({s: {k:v for k,v in d.items() if k != "columns"}
                      for s,d in report["datasets"].items()}, indent=2))


def cache(args):
    manifest = json.loads(Path(args.manifest).read_text())
    split = "train" if args.split.startswith("train") else "valid"
    data_path = manifest["datasets"][split]["path"]
    groups = manifest["splits"][args.split]
    if args.limit:
        groups = groups[:args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = {"data": fingerprint(data_path), "model": model_identity(args.solution),
                "split": args.split, "groups": groups}
    if args.sequence_onnx:
        identity["offline_sequence_onnx_sha256"]=hashlib.sha256(Path(args.sequence_onnx).read_bytes()).hexdigest()
    config = out / "identity.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("cache identity mismatch; use a fresh directory")
    write_json(config, identity)
    pf = pq.ParquetFile(data_path)
    model = load_model(args.solution)
    fast=None
    if args.sequence_onnx:
        from competition_engineering.gru_sequence import SequenceGRU
        fast=SequenceGRU(args.sequence_onnx)
    acc = GlobalAccumulator()
    started = time.perf_counter()
    sequence_reports = []
    for index, group in enumerate(groups):
        dest = out / f"{group:05d}.npz"
        if not dest.exists():
            seq, need, x, y, mask = read_sequence(pf, group)
            p = np.full_like(y, np.nan)
            x.setflags(write=False)
            replay_started = time.perf_counter()
            if fast is None or index==0:
                for step in range(len(x)):
                    value = model.predict(DataPoint(seq, step, bool(need[step]), x[step]))
                    if not need[step]:
                        if value is not None:
                            raise ValueError("warm-up output must be None")
                    else:
                        value = np.asarray(value, dtype=np.float32)
                        if value.shape != (2,) or not np.isfinite(value).all():
                            raise ValueError("invalid required prediction")
                        p[step] = value
            if fast is not None:
                batched=fast.predict(x[None])[0]
                if index==0:
                    np.testing.assert_allclose(batched[need],p[need],atol=3e-5,rtol=3e-5)
                p[need]=batched[need]
            elapsed = time.perf_counter() - replay_started
            temp = dest.with_suffix(".tmp")
            with temp.open("wb") as f:
                np.savez(f, x=x, y=y, p=p, mask=mask, need=need, seq=seq,
                         step=np.arange(20000), replay_seconds=elapsed)
            temp.replace(dest)
        with np.load(dest) as z:
            acc.add(z["y"], z["p"], z["mask"])
            local = GlobalAccumulator()
            local.add(z["y"], z["p"], z["mask"])
            sequence_reports.append({"group": group, "seq": int(z["seq"]),
                                     "replay_seconds": float(z["replay_seconds"]),
                                     **(local.result() if z["mask"].any() else {"selected_rows": 0})})
        if (index+1) % 8 == 0 or index == 0:
            print(f"{index+1}/{len(groups)} {acc.result()['weighted_pearson']:.6f}", flush=True)
    report = {**acc.result(), "wall_seconds": time.perf_counter()-started,
              "sequences": sequence_reports}
    write_json(out / "scores.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "sequences"}, indent=2))


def cached(path):
    directory = Path(path)
    identity = json.loads((directory / "identity.json").read_text())
    for group in identity["groups"]:
        with np.load(directory / f"{group:05d}.npz") as z:
            yield group, {k:z[k] for k in z.files}


def score_arrays(y, p):
    return [weighted_pearson(y[:,k], p[:,k]) for k in range(2)]


def diagnostic_arrays(path):
    blocks = list(cached(path))
    y = np.concatenate([z["y"][z["mask"]] for _,z in blocks])
    p = np.concatenate([z["p"][z["mask"]] for _,z in blocks])
    return blocks, y, p


def diagnostics(args):
    report = {}
    for split, directory in [("tune", args.tune), ("confirm", args.confirm)]:
        blocks, y, p = diagnostic_arrays(directory)
        if split == "tune":
            choices = []
            for k in range(2):
                candidates = [(weighted_pearson(y[:,k], p[:,k]*s+b), s, b)
                              for s in [0.75, 0.9, 1., 1.1, 1.25]
                              for b in [-0.1, 0., 0.1]]
                choices.append(max(candidates, key=lambda item: item[0]))
        calibrated = np.column_stack([p[:,k]*choices[k][1]+choices[k][2] for k in range(2)])
        # Lag pairs are adjacent scored observations in the same sequence only.
        correlations = {}
        for lag in [1, 10, 100]:
            a, b = [], []
            for _, z in blocks:
                r = z["y"] - z["p"]
                valid = z["mask"][lag:] & z["mask"][:-lag]
                a.append(r[:-lag][valid]); b.append(r[lag:][valid])
            a, b = np.concatenate(a), np.concatenate(b)
            correlations[str(lag)] = [float(np.corrcoef(a[:,k],b[:,k])[0,1]) for k in range(2)]
        # Correlations are descriptive, not evidence of a profitable correction.
        sample_x = np.concatenate([z["x"][z["mask"]][::20] for _,z in blocks])
        sample_r = np.concatenate([(z["y"]-z["p"])[z["mask"]][::20] for _,z in blocks])
        feature_corr = np.corrcoef(np.column_stack([sample_x, sample_r]), rowvar=False)[:112,112:]
        top = [[{"feature": FEATURE_COLUMNS[i], "correlation": float(feature_corr[i,k])}
                for i in np.argsort(-np.nan_to_num(np.abs(feature_corr[:,k])))[:8]] for k in range(2)]
        report[split] = {"gru_wp": score_arrays(y,p), "calibrated_wp": score_arrays(y, calibrated),
                         "calibration_scale_bias": [list(v[1:]) for v in choices],
                         "prediction_quantiles": np.quantile(p,[0,.01,.5,.99,1],axis=0).tolist(),
                         "residual_mean": (y-p).mean(axis=0).tolist(),
                         "residual_std": (y-p).std(axis=0).tolist(),
                         "residual_lag_correlation": correlations,
                         "top_feature_residual_correlation": top,
                         "cross_target_residual_correlation": float(np.corrcoef((y-p).T)[0,1])}
    write_json(args.output, report)
    print(json.dumps({s:{k:v for k,v in d.items() if k in ["gru_wp","calibrated_wp","calibration_scale_bias"]}
                      for s,d in report.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--assets", required=True); p.add_argument("--output", required=True)
    p.set_defaults(func=audit)
    p = sub.add_parser("cache")
    for name in ["manifest", "solution", "split", "output"]:
        p.add_argument("--"+name, required=True)
    p.add_argument("--limit", type=int); p.set_defaults(func=cache)
    p.add_argument("--sequence-onnx",help="Optional verified offline GRU sequence acceleration")
    p = sub.add_parser("diagnostics")
    for name in ["tune", "confirm", "output"]:
        p.add_argument("--"+name, required=True)
    p.set_defaults(func=diagnostics)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
