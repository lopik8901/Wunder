"""Aligned GRU/TCN complementarity and tune-only target-specific blending."""
import argparse
import json
from pathlib import Path

import numpy as np

from competition_engineering.pipeline import cached, write_json
from wnn_connectome_starterpack.utils import GlobalAccumulator


def paired(first, second):
    ia=json.loads((Path(first)/"identity.json").read_text())
    ib=json.loads((Path(second)/"identity.json").read_text())
    if ia["data"]!=ib["data"] or ia["groups"]!=ib["groups"]:
        raise ValueError("cache source or sequence order differs")
    for (ga,a),(gb,b) in zip(cached(first),cached(second),strict=True):
        assert ga==gb and int(a["seq"])==int(b["seq"])
        for field in ["step","need","mask","y"]:
            np.testing.assert_array_equal(a[field],b[field])
        yield ga,a,b


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ["gru-tune","tcn-tune","gru-confirm","tcn-confirm","output"]:
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args(); report={}
    alphas=[0.,.25,.5,.75,.9,.95,1.]
    for split in ["tune","confirm"]:
        ga=[GlobalAccumulator(),GlobalAccumulator()]
        accs=[GlobalAccumulator() for _ in alphas]
        pred_a=[]; pred_b=[]; res_a=[]; res_b=[]; sequences=[]
        for group,a,b in paired(getattr(args,"gru_"+split),getattr(args,"tcn_"+split)):
            mask=a["mask"]; y=a["y"]
            ga[0].add(y,a["p"],mask); ga[1].add(y,b["p"],mask)
            local=[]
            for pred in [a["p"],b["p"]]:
                acc=GlobalAccumulator(); acc.add(y,pred,mask)
                local.append(acc.result() if mask.any() else None)
            sequences.append({"group":group,"gru":local[0],"tcn":local[1]})
            if split=="tune":
                for alpha,acc in zip(alphas,accs):
                    acc.add(y,alpha*a["p"]+(1-alpha)*b["p"],mask)
            else:
                alpha=np.asarray(chosen)
                accs[0].add(y,alpha*a["p"]+(1-alpha)*b["p"],mask)
            pred_a.append(a["p"][mask]); pred_b.append(b["p"][mask])
            res_a.append((y-a["p"])[mask]); res_b.append((y-b["p"])[mask])
        pa,pb,ra,rb=map(np.concatenate,[pred_a,pred_b,res_a,res_b])
        if split=="tune":
            grid=[{"alpha_gru":alpha,**acc.result()} for alpha,acc in zip(alphas,accs)]
            chosen=[max(grid,key=lambda r:r[k])["alpha_gru"] for k in ["t0","t1"]]
            result={"grid":grid,"chosen_alpha_gru":chosen}
        else: result={"frozen_alpha_gru":chosen,"blend":accs[0].result()}
        report[split]={**result,"gru":ga[0].result(),"tcn":ga[1].result(),
                       "prediction_correlation":[float(np.corrcoef(pa[:,k],pb[:,k])[0,1]) for k in range(2)],
                       "residual_correlation":[float(np.corrcoef(ra[:,k],rb[:,k])[0,1]) for k in range(2)],
                       "sequences":sequences}
    write_json(args.output,report)
    print(json.dumps({"chosen_alpha_gru":chosen,"confirm":report["confirm"]["blend"]},indent=2))


if __name__=="__main__": main()
