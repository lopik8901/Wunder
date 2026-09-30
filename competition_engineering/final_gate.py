"""One frozen ridge-candidate gate; reference scores over complete validation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from competition_engineering.pipeline import (
    cached, fingerprint, load_model, model_identity, read_sequence, write_json,
)
from competition_engineering.residual import features, sufficient, from_stats
from wnn_connectome_starterpack.utils import DataPoint


def init_worker(data, solution, ridge, directory, sequence_onnx):
    global PARQUET, MODEL, RIDGE, DIRECTORY, FAST, FAST_CHECKED
    threadpool_limits(limits=1)
    PARQUET=pq.ParquetFile(data); MODEL=load_model(solution)
    with np.load(ridge) as z: RIDGE={k:z[k] for k in z.files}
    DIRECTORY=Path(directory)
    FAST=None; FAST_CHECKED=False
    if sequence_onnx:
        from competition_engineering.gru_sequence import SequenceGRU
        FAST=SequenceGRU(sequence_onnx)


def moments(z, config):
    original=z["p"]
    calibrated=(original*config["base_scale"].astype(np.float32)+config["base_bias"].astype(np.float32)).astype(np.float32)
    z={**z,"p":calibrated}
    corrected=(calibrated+config["strengths"]*(features(z,config["mean"],config["scale"])@config["coef"])).astype(np.float32)
    return [sufficient(z["y"][z["mask"]],pred[z["mask"]]).tolist()
            for pred in [original,calibrated,corrected]]


def process_group(group):
    global FAST_CHECKED
    output=DIRECTORY/f"{group:05d}.json"
    if output.exists(): return json.loads(output.read_text())
    started=time.perf_counter()
    seq,need,x,y,mask=read_sequence(PARQUET,group)
    prediction=np.full_like(y,np.nan)
    parity=None
    if FAST is None or not FAST_CHECKED:
        for step,row in enumerate(x):
            value=MODEL.predict(DataPoint(seq,step,bool(need[step]),row))
            if need[step]: prediction[step]=value
            else: assert value is None
    if FAST is not None:
        batched=FAST.predict(x[None])[0]
        if not FAST_CHECKED:
            np.testing.assert_allclose(prediction[need],batched[need],atol=3e-5,rtol=3e-5)
            parity=float(np.max(np.abs(prediction[need]-batched[need])))
            FAST_CHECKED=True
        prediction[need]=batched[need]
    assert np.isfinite(prediction[need]).all()
    result={"group":group,"seq":seq,"selected_rows":int(mask.sum()),
            "moments":moments({"x":x,"y":y,"p":prediction,"mask":mask},RIDGE),
            "seconds":time.perf_counter()-started,"callback_parity_max_abs":parity}
    write_json(output,result)
    return result


def scores(values):
    return {name:{"t0":float(v[0]),"t1":float(v[1]),"combined":float(v.mean())}
            for name,v in zip(["official_gru","calibrated_gru","ridge_gru"],
                              [from_stats(s) for s in values])}


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ["manifest","solution","ridge","tune","confirm","output"]:
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--sequence-onnx")
    parser.add_argument("--reference-gate-cache")
    parser.add_argument("--validation-reuse",action="store_true")
    parser.add_argument("--report-output",default=str(Path(__file__).parent/"reports/final_gate.json"))
    args=parser.parse_args(); manifest=json.loads(Path(args.manifest).read_text())
    directory=Path(args.output); directory.mkdir(parents=True,exist_ok=True)
    if Path(args.report_output).exists(): raise FileExistsError("use a fresh report path; historical results are immutable")
    data=manifest["datasets"]["valid"]["path"]
    identity={"data":fingerprint(data),"model":model_identity(args.solution),
              "ridge":model_identity(Path(args.ridge).parent/"solution.py"),
              "groups":manifest["splits"]["final_gate"],
              "selection":"ridge selected on development evidence before opening final gate"}
    if args.sequence_onnx: identity["sequence_onnx"]=fingerprint(args.sequence_onnx)
    if args.validation_reuse: identity["validation_reuse"]=True
    frozen=directory/"frozen_identity.json"
    if frozen.exists() and json.loads(frozen.read_text())!=identity:
        raise ValueError("frozen final-gate identity mismatch")
    write_json(frozen,identity)
    results=[]; started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             initargs=(data,args.solution,args.ridge,directory,args.sequence_onnx)) as pool:
        for result in pool.map(process_group,identity["groups"],chunksize=4):
            results.append(result)
            if len(results)%100==0:
                print(f"gate completed {len(results)}/{len(identity['groups'])}",flush=True)
    gate=np.array([r["moments"] for r in results])
    prior=None
    if args.reference_gate_cache:
        previous=[]
        for result in results:
            old=json.loads((Path(args.reference_gate_cache)/f"{result['group']:05d}.json").read_text())
            assert old["seq"]==result["seq"] and old["selected_rows"]==result["selected_rows"]
            np.testing.assert_allclose(np.array(old["moments"])[:2],np.array(result["moments"])[:2],atol=1e-8,rtol=1e-10)
            previous.append(old["moments"][2])
        prior=np.array(previous)
    total=gate.sum(0)
    with np.load(args.ridge) as z: config={k:z[k] for k in z.files}
    for cache_path in [args.tune,args.confirm]:
        for _,z in cached(cache_path): total+=np.array(moments(z,config))
    rng=np.random.default_rng(20260928); improvement=[]; prior_improvement=[]
    for _ in range(1000):
        ix=rng.integers(len(gate),size=len(gate)); stats=gate[ix].sum(0)
        values=[from_stats(s) for s in stats]
        improvement.append([(values[2]-values[0]).mean(),(values[2]-values[1]).mean()])
        if prior is not None:
            prior_improvement.append((values[2]-from_stats(prior[ix].sum(0))).mean())
    report={"gate_sequences":len(results),"gate_scored_rows":sum(r["selected_rows"] for r in results),
            "gate":scores(gate.sum(0)),"complete_validation":scores(total),
            "bootstrap_95ci_ridge_gain_vs_raw_and_calibrated":np.quantile(improvement,[.025,.975],axis=0).tolist(),
            "workers":args.workers,"wall_seconds":time.perf_counter()-started,
            "note":"Parallel offline evaluation. Deployment timing is measured separately at one thread.",
            "validation_reuse":args.validation_reuse,
            "offline_sequence_acceleration":bool(args.sequence_onnx),
            "callback_parity_checks":[r["callback_parity_max_abs"] for r in results if r.get("callback_parity_max_abs") is not None]}
    if prior is not None:
        report["prior_system_gate_wp"]=from_stats(prior.sum(0)).tolist()
        report["gate_combined_gain_vs_prior"]=float(from_stats(gate.sum(0)[2]).mean()-from_stats(prior.sum(0)).mean())
        report["bootstrap_95ci_vs_prior"]=np.quantile(prior_improvement,[.025,.975]).tolist()
    write_json(args.report_output,report)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
