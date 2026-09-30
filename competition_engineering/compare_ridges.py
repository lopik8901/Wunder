"""Paired comparison of frozen ridge systems without tuning either model."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import features, sufficient, from_stats


def predict(z,config):
    p=(z["p"]*config["base_scale"].astype(np.float32)+config["base_bias"].astype(np.float32)).astype(np.float32)
    local={**z,"p":p}
    return (p+config["strengths"]*(features(local,config["mean"],config["scale"])@config["coef"])).astype(np.float32)


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ["reference","candidate","cache","output"]: parser.add_argument("--"+name,required=True)
    args=parser.parse_args(); started=time.perf_counter()
    configs=[]
    for path in [args.reference,args.candidate]:
        with np.load(path) as z: configs.append({k:z[k] for k in z.files})
    moments=[]; count=0
    with threadpool_limits(limits=1):
        for _,z in cached(args.cache):
            moments.append([sufficient(z["y"][z["mask"]],predict(z,c)[z["mask"]]) for c in configs])
            count+=int(z["mask"].sum())
    moments=np.asarray(moments); combined=moments.sum(0)
    values=[from_stats(s) for s in combined]
    rng=np.random.default_rng(20260928); deltas=[]
    for _ in range(1000):
        idx=rng.integers(len(moments),size=len(moments)); summed=moments[idx].sum(0)
        deltas.append(from_stats(summed[1])-from_stats(summed[0]))
    deltas=np.asarray(deltas)
    report={"cache":args.cache,"sequences":len(moments),"scored_rows":count,
            "reference_wp":values[0].tolist(),"candidate_wp":values[1].tolist(),
            "combined_gain":float((values[1]-values[0]).mean()),
            "bootstrap_95ci_combined":np.quantile(deltas.mean(1),[.025,.975]).tolist(),
            "bootstrap_95ci_per_target":np.quantile(deltas,[.025,.975],axis=0).tolist(),
            "model_sha256":{name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for name,path in [("reference",args.reference),("candidate",args.candidate)]},
            "runtime_seconds":time.perf_counter()-started}
    write_json(args.output,report); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
