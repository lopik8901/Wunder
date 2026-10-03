"""Exact endpoint computation for the fixed four-layer causal topology."""
import numpy as np

def endpoint_windows(projected,indices):
    offsets=np.array([0,1,4,5,16,17,20,21,64,65,68,69,80,81,84,85])
    positions=np.asarray(indices)[:,None]-offsets
    valid=positions>=0
    windows=projected[np.maximum(positions,0)].copy();windows[~valid]=0
    return windows.astype(np.float32),valid

def endpoint_features(windows,valid,weights,bias,instant=False):
    if instant:
        h=np.repeat(windows[:,:1],16,axis=1);mask=np.ones_like(valid)
    else: h=windows;mask=valid
    layers=[]
    for i in range(4):
        mask=mask[:,::2]
        h=np.tanh(h[:,::2]@weights[i,0]+h[:,1::2]@weights[i,1]+bias[i])*mask[:,:,None]
        layers.append(h[:,0])
    return np.concatenate(layers,axis=1).astype(np.float32)
