"""Small causal residual TCN, bounded sequence streaming, GPU-only training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from competition_engineering.pipeline import cached, write_json, load_model
from competition_engineering.residual import sufficient, from_stats, calibrated_cached, features
from wnn_connectome_starterpack.utils import GlobalAccumulator
from wnn_connectome_starterpack.utils import DataPoint
from threadpoolctl import threadpool_limits


def system_cached(directory, base_scale, base_bias, ridge=None):
    for group,z in calibrated_cached(directory,base_scale,base_bias):
        if ridge is not None:
            z["p"]=(z["p"]+ridge["strengths"]*(features(z,ridge["mean"],ridge["scale"])@ridge["coef"])).astype(np.float32)
        yield group,z


def point_base(raw, row, base_scale, base_bias, ridge=None):
    if raw is None: return None
    prediction=(raw*np.asarray(base_scale,dtype=np.float32)+np.asarray(base_bias,dtype=np.float32)).astype(np.float32)
    if ridge is not None:
        normalized=np.clip((row-ridge["mean"])/ridge["scale"],-8,8)
        phi=np.concatenate(([1.],normalized,prediction)).astype(np.float32)
        prediction=(prediction+ridge["strengths"]*(phi@ridge["coef"])).astype(np.float32)
    return prediction


class ResidualTCN(nn.Module):
    context = 14

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(114,32,3,dilation=1),
                                   nn.Conv1d(32,32,3,dilation=2),
                                   nn.Conv1d(32,32,3,dilation=4)])
        self.head = nn.Conv1d(32,2,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self,x):
        for conv in self.convs:
            x = F.relu(conv(F.pad(x,(2*conv.dilation[0],0))))
        return self.head(x)


def input_features(z,mean,scale):
    return np.column_stack([np.clip((z["x"]-mean)/scale,-8,8),
                            np.nan_to_num(z["p"],nan=0)]).astype(np.float32)


class IncrementalTCN:
    """NumPy one-row replay of the trained convolution stack, no GPU needed."""
    def __init__(self, model):
        self.layers=[]
        for conv in model.convs:
            w=conv.weight.detach().cpu().numpy()
            self.layers.append((w.transpose(0,2,1).reshape(w.shape[0],-1).copy(),
                                conv.bias.detach().cpu().numpy().copy(),conv.dilation[0],
                                np.zeros((2*conv.dilation[0]+1,w.shape[1]),np.float32)))
        self.head=model.head.weight.detach().cpu().numpy()[:,:,0].copy()
        self.bias=model.head.bias.detach().cpu().numpy().copy()
        self.step=0

    def predict(self,x):
        for weight,bias,dilation,history in self.layers:
            pos=self.step%len(history)
            history[pos]=x
            inputs=np.concatenate([history[(pos-2*dilation)%len(history)],
                                   history[(pos-dilation)%len(history)],history[pos]])
            x=np.maximum(weight@inputs+bias,0)
        self.step+=1
        return self.head@x+self.bias


def predict_sequence(model, x, device):
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),2048):
            left=max(0,start-model.context)
            window=torch.from_numpy(x[left:start+2048].T.copy()).unsqueeze(0).to(device)
            outputs.append(model(window)[0,:,start-left:].T.cpu().numpy())
    return np.concatenate(outputs)


def evaluate(model,directory,mean,scale,device,strengths=None,base_scale=(1,1),base_bias=(0,0),ridge=None):
    grid=[0.,.05,.1,.25,.5,1.]
    accs=[GlobalAccumulator() for _ in grid]
    base=GlobalAccumulator(); bstats=[]; cstats=[]
    diagnostics = None
    if strengths is not None:
        from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
        diagnostics = SearchErrorDiagnostics()
    for _,z in system_cached(directory,base_scale,base_bias,ridge):
        correction=predict_sequence(model,input_features(z,mean,scale),device)
        base.add(z["y"],z["p"],z["mask"])
        if strengths is None:
            for strength,acc in zip(grid,accs):
                acc.add(z["y"],z["p"]+strength*correction,z["mask"])
        else:
            pred=z["p"]+np.asarray(strengths)*correction
            diagnostics.add(z, pred)
            accs[0].add(z["y"],pred,z["mask"])
            bstats.append(sufficient(z["y"][z["mask"]],z["p"][z["mask"]]))
            cstats.append(sufficient(z["y"][z["mask"]],pred[z["mask"]]))
    if strengths is None:
        scores=[{"strength":s,**a.result()} for s,a in zip(grid,accs)]
        return {"grid":scores,"strengths":[max(scores,key=lambda r:r[k])["strength"] for k in ["t0","t1"]]}
    b,c=np.stack(bstats),np.stack(cstats); rng=np.random.default_rng(20260928)
    delta=[]
    for _ in range(500):
        ix=rng.integers(len(b),size=len(b))
        delta.append((from_stats(c[ix].sum(0))-from_stats(b[ix].sum(0))).mean())
    return {"baseline":base.result(),"candidate":accs[0].result(),
            "bootstrap_95ci_combined":np.quantile(delta,[.025,.975]).tolist(),
            "search_error_diagnostics":diagnostics.result()}


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ["train","tune","output","gru-solution"]:
        p.add_argument("--"+name,required=True)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--amp",action="store_true")
    p.add_argument("--base-scale",type=float,nargs=2,default=[1,1])
    p.add_argument("--base-bias",type=float,nargs=2,default=[0,0])
    p.add_argument("--base-ridge",help="Frozen ridge checkpoint to include in the residual base")
    p.add_argument("--target-mode",choices=["raw","clipped"],default="raw")
    args=p.parse_args(); out=Path(args.output)
    if out.exists(): raise ValueError("use a fresh output directory")
    if not torch.cuda.is_available(): raise RuntimeError("GPU required, no CPU fallback")
    torch.set_num_threads(1); torch.manual_seed(20260928)
    device=torch.device("cuda:0"); model=ResidualTCN().to(device)
    out.mkdir(parents=True)
    ridge=None
    if args.base_ridge:
        with np.load(args.base_ridge) as z: ridge={k:z[k] for k in z.files}
        np.testing.assert_array_equal(ridge["base_scale"],args.base_scale)
        np.testing.assert_array_equal(ridge["base_bias"],args.base_bias)
    rng=np.random.default_rng(20260928)
    n=0; sx=np.zeros(112); sxx=np.zeros(112)
    for _,z in cached(args.train):
        x=z["x"][::10].astype(np.float64); n+=len(x); sx+=x.sum(0); sxx+=(x*x).sum(0)
    mean=(sx/n).astype(np.float32); scale=np.sqrt(np.maximum(sxx/n-(sx/n)**2,1e-6)).astype(np.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    scaler=torch.amp.GradScaler("cuda",enabled=args.amp)
    started=time.perf_counter(); updates=0; rows=0; losses=[]
    for epoch in range(args.epochs):
        for group,z in system_cached(args.train,args.base_scale,args.base_bias,ridge):
            x=input_features(z,mean,scale)
            targets=np.clip(z["y"],-2,2) if args.target_mode=="clipped" else z["y"]
            residual=np.nan_to_num(targets-z["p"],nan=0)
            starts=rng.permutation(np.arange(0,len(x),512))
            for begin in range(0,len(starts),8):
                selected=starts[begin:begin+8]; bx=[]; by=[]; bw=[]
                for start in selected:
                    left=max(0,start-14); end=min(len(x),start+512)
                    xx=np.zeros((526,114),np.float32)
                    offset=14-(start-left)
                    xx[offset:offset+end-left]=x[left:end]
                    yy=np.zeros((512,2),np.float32); weight=np.zeros_like(yy)
                    yy[:end-start]=residual[start:end]
                    weight[:end-start]=np.abs(np.clip(z["y"][start:end],-2,2))*z["need"][start:end,None]
                    bx.append(xx.T); by.append(yy.T); bw.append(weight.T)
                    rows+=int(z["need"][start:end].sum())
                def upload(values):
                    return torch.from_numpy(np.stack(values)).pin_memory().to(device,non_blocking=True)
                tx,ty,tw=upload(bx),upload(by),upload(bw)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16,enabled=args.amp):
                    prediction=model(tx)[:,:,14:]
                    loss=((prediction.float()-ty).square()*tw).sum()/tw.sum().clamp_min(1e-8)
                if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
                updates+=1; losses.append(float(loss.detach()))
            if updates%80==0:
                print(f"epoch={epoch} group={group} updates={updates} loss={np.mean(losses[-80:]):.5f}",flush=True)
    torch.cuda.synchronize(); train_seconds=time.perf_counter()-started
    model.eval()
    tune=evaluate(model,args.tune,mean,scale,device,base_scale=args.base_scale,base_bias=args.base_bias,ridge=ridge)
    diagnostic_result=evaluate(model,args.tune,mean,scale,device,strengths=tune["strengths"],
                               base_scale=args.base_scale,base_bias=args.base_bias,ridge=ridge)
    tune["search_error_diagnostics"]=diagnostic_result["search_error_diagnostics"]
    # Test the exact incremental CPU implementation against GPU sequence outputs.
    _,z=next(system_cached(args.tune,args.base_scale,args.base_bias,ridge)); x=input_features(z,mean,scale)[:2000]
    expected=predict_sequence(model,x,device)
    replay=IncrementalTCN(model); replay_start=time.perf_counter()
    actual=np.stack([replay.predict(row) for row in x]); replay_seconds=time.perf_counter()-replay_start
    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-4)
    # Whole-sequence callback replay, including the unchanged official GRU.
    gru=load_model(args.gru_solution); incremental=IncrementalTCN(model)
    replayed=np.full_like(z["p"],np.nan)
    callback_started=time.perf_counter()
    with threadpool_limits(limits=1):
        for step,row in enumerate(z["x"]):
            gp=gru.predict(DataPoint(int(z["seq"]),step,bool(z["need"][step]),row))
            gp=point_base(gp,row,args.base_scale,args.base_bias,ridge)
            inputs=np.concatenate([np.clip((row-mean)/scale,-8,8),
                                   np.zeros(2,np.float32) if gp is None else gp]).astype(np.float32)
            rp=incremental.predict(inputs)
            if gp is not None: replayed[step]=gp+np.asarray(tune["strengths"])*rp
    callback_seconds=time.perf_counter()-callback_started
    expected_system=z["p"]+np.asarray(tune["strengths"])*predict_sequence(model,input_features(z,mean,scale),device)
    np.testing.assert_allclose(replayed[99:],expected_system[99:],atol=3e-5,rtol=3e-4)
    torch.save({"state_dict":model.cpu().state_dict(),"mean":mean,"scale":scale,
                "strengths":tune["strengths"],"base_scale":args.base_scale,"base_bias":args.base_bias,
                "base_ridge":ridge},out/"checkpoint.pt")
    report={"hypothesis":"Short causal nonlinear residual improves the frozen base system",
            "seed":20260928,"config":vars(args),"device":torch.cuda.get_device_name(0),"process_id":os.getpid(),
            "torch":torch.__version__,"train_seconds":train_seconds,"updates":updates,
            "training_rows":rows,"rows_per_second":rows/train_seconds,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "loss_first80":float(np.mean(losses[:80])),"loss_last80":float(np.mean(losses[-80:])),
            "tune":tune,"incremental_parity_max_abs":float(np.max(np.abs(actual-expected))),
            "correction_replay_microseconds_per_row":replay_seconds/len(x)*1e6,
            "system_callback_20000_rows_seconds":callback_seconds,
            "system_callback_microseconds_per_row":callback_seconds/20000*1e6,
            "estimated_max_rows_in_60min":int(3600/callback_seconds*20000),
            "checkpoint_bytes":(out/"checkpoint.pt").stat().st_size}
    if args.base_ridge:
        report["base_ridge_sha256"]=hashlib.sha256(Path(args.base_ridge).read_bytes()).hexdigest()
    write_json(out/"report.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
