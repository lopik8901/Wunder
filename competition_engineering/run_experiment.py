"""Bounded GPU experiment entrypoint compatible with Connectome MLEvolve results."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from connectome.mlevolve_adapter import emit_result, RESULT_PREFIX
from competition_engineering.pipeline import write_json
from competition_engineering.gpu_environment import configure_miopen_cache


ROOT=Path(__file__).resolve().parents[1]


def emit_outcome(report):
    if report["status"]=="success":
        emit_result(report)
    else:
        # The historical score validator deliberately accepts successes only.
        # Failures still need a machine-readable record, without a fake score.
        print(RESULT_PREFIX+json.dumps(report,sort_keys=True,separators=(",",":")))


def validate_inputs(config):
    identities={}
    sequence_ids={}
    if "confirm" in config or "promotion" in config:
        raise ValueError("routine search cannot request holdout or promotion")
    for key in ["train","tune"]:
        directory=Path(config[key])
        identity=json.loads((directory/"identity.json").read_text())
        if "final" in identity["split"]:
            raise ValueError("completed final-gate cache cannot be used by the search runner")
        missing=[g for g in identity["groups"] if not (directory/f"{g:05d}.npz").is_file()]
        if missing: raise ValueError(f"incomplete {key} cache: {len(missing)} missing sequences")
        ids=[]
        for group in identity["groups"]:
            with np.load(directory/f"{group:05d}.npz") as z: ids.append(int(z["seq"]))
        if len(ids)!=len(set(ids)): raise ValueError(f"duplicate sequences within {key}")
        sequence_ids[key]=set(ids)
        identities[key]=identity
    if len(identities["train"]["groups"])<256:
        raise ValueError("competition search requires at least 256 training sequences")
    if identities["train"]["split"] not in {"train","train_pilot","train_scale","train_medium","train_all"}:
        raise ValueError("training cache is not a training split")
    if identities["tune"]["split"] != "tune":
        raise ValueError("routine evaluation requires the designated tuning split")
    for a,b in [("train","tune")]:
        if sequence_ids[a]&sequence_ids[b]:
            raise ValueError(f"sequence leakage between {a} and {b}")
    return identities


def run(config_path, timeout):
    configure_miopen_cache()
    if timeout<=0: raise ValueError("timeout must be positive")
    config=json.loads(Path(config_path).read_text())
    for key in ["train","tune","output","gru_solution","base_ridge"]:
        if key in config:
            path=Path(config[key])
            config[key]=str((ROOT/path).resolve() if not path.is_absolute() else path.resolve())
    output=Path(config["output"])
    if not output.is_relative_to(ROOT/"competition_engineering/checkpoints"):
        raise ValueError("experiment output must be inside competition_engineering/checkpoints")
    if output.exists(): raise ValueError("experiment output already exists")
    identities=validate_inputs(config)
    run_dir=ROOT/"competition_engineering/runs"/output.name
    if run_dir.exists(): raise ValueError("run name already exists")
    run_dir.mkdir(parents=True)
    write_json(run_dir/"config.json",config)
    write_json(run_dir/"cache_identities.json",identities)
    source_dir=run_dir/"source"; source_dir.mkdir()
    for name in ["run_experiment.py","gpu_residual.py","residual.py","calibration.py","pipeline.py","gru_sequence.py"]:
        shutil.copy2(ROOT/"competition_engineering"/name,source_dir/name)
    shutil.copy2(ROOT/"connectome/mlevolve_adapter.py",source_dir/"mlevolve_adapter.py")
    kind=config.get("kind","gpu_residual")
    if kind not in {"gpu_residual","ridge","calibration"}: raise ValueError("unsupported experiment kind")
    module={"gpu_residual":"gpu_residual","ridge":"residual","calibration":"calibration"}[kind]
    command=[sys.executable,"-m","competition_engineering."+module]
    keys=["train","tune","output"]
    if kind!="calibration": keys += ["target_mode"]
    if kind=="ridge": keys += ["sample_stride","ridge_penalty"]
    if kind=="gpu_residual": keys += ["gru_solution","base_ridge","epochs"]
    for key in keys:
        if key in config: command.extend(["--"+key.replace("_","-"),str(config[key])])
    for key in ["base_scale","base_bias"]:
        if key in config: command.extend(["--"+key.replace("_","-"),*map(str,config[key])])
    if kind=="gpu_residual" and config.get("amp"): command.append("--amp")
    preflight=[sys.executable,str(ROOT/"tools/verify_gpu.py")]
    started=time.perf_counter()
    report={"status":"error","metric_name":"combined_WP","maximize":True,
            "command":command,"training_sequences":len(identities["train"]["groups"]),
            "experiment_kind":kind,"gpu_required":kind=="gpu_residual",
            "timeout_seconds":timeout,
            "trainer_sha256":hashlib.sha256((ROOT/f"competition_engineering/{module}.py").read_bytes()).hexdigest()}
    try:
        if kind=="gpu_residual":
            report["stage"]="gpu_preflight"
            with (run_dir/"gpu_preflight.log").open("w",encoding="utf-8") as log:
                subprocess.run(preflight,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
        remaining=timeout-(time.perf_counter()-started)
        if remaining<=0: raise TimeoutError("budget consumed by GPU preflight")
        report["stage"]="training_and_evaluation"
        with (run_dir/"training.log").open("w",encoding="utf-8") as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining,check=True)
        trained=json.loads((output/"report.json").read_text())
        report["stage"]="completed"
        # Search feedback is tuning WP, never the previously consumed final gate.
        t0=max(r["t0"] for r in trained["tune"]["grid"])
        t1=max(r["t1"] for r in trained["tune"]["grid"])
        report.update(status="success",primary_metric=(t0+t1)/2,
                      metrics={"WP_t0":t0,"WP_t1":t1,"combined_WP":(t0+t1)/2},
                      metric_domain="fixed development tuning sequences",
                      artifact_bytes=trained.get("checkpoint_bytes",0) if kind=="gpu_residual" else (output/("ridge.npz" if kind=="ridge" else "calibration.npz")).stat().st_size,
                      peak_gpu_allocated_bytes=trained.get("peak_allocated_bytes"),
                      cpu_microseconds_per_row=trained.get("system_callback_microseconds_per_row"))
        report["search_error_diagnostics"] = trained.get("tune", {}).get("search_error_diagnostics")
    except (subprocess.TimeoutExpired,TimeoutError) as exc:
        report.update(status="timeout",error=str(exc))
    except Exception as exc:
        detail = repr(exc)
        training_log = run_dir / "training.log"
        if training_log.exists():
            lines = training_log.read_text(encoding="utf-8", errors="replace").splitlines()
            # Only this search-only trainer writes here; expose its final
            # exception, not raw logs or any supervisor-only evaluation.
            failures = [line.strip() for line in lines[-30:] if line.startswith(("RuntimeError:", "ValueError:", "AssertionError:"))]
            if failures:
                detail = failures[-1][:400]
        report.update(error=detail)
    report["runtime_seconds"]=time.perf_counter()-started
    write_json(run_dir/"result.json",report)
    emit_outcome(report)
    return report["status"]=="success"


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("config"); parser.add_argument("--timeout",type=int,default=600)
    args=parser.parse_args()
    try:
        success=run(args.config,args.timeout)
    except Exception as exc:
        emit_outcome({"status":"error","metric_name":"combined_WP","maximize":True,
                      "error":repr(exc),"stage":"input preflight"})
        success=False
    sys.exit(0 if success else 1)
