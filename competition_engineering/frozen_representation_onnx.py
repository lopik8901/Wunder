"""Reuse the existing forward GRU states in a frozen current-row readout."""
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from competition_engineering import campaign_onnx

def export(model_path,destination,work,force_latent=False):
    with np.load(model_path) as q:
        coef=q['coef'];mean=q['mean'];scale=q['scale']
    if coef.shape!=(371,2) or mean.shape!=(256,) or scale.shape!=(256,):
        raise ValueError('Unexpected frozen representation dimensions')
    if not all(np.isfinite(v).all() for v in [coef,mean,scale]) or np.any(scale<=0):
        raise ValueError('Invalid readout parameters')
    zero=work/'zero_readout.npz';np.savez(zero,coef=np.zeros(115,np.float32))
    campaign_onnx.export(zero,destination,'current')
    model=onnx.load(destination);graph=model.graph;frozen=graph.output[0].name
    combo=campaign_onnx.load_combo();counter=0
    def const(value,dtype):
        nonlocal counter
        counter+=1;name='representation_constant_'+str(counter)
        graph.initializer.append(nh.from_array(np.asarray(value,dtype=dtype),name));return name
    def op(kind,*inputs,**attrs):
        nonlocal counter
        counter+=1;name='representation_value_'+str(counter)
        graph.node.append(h.make_node(kind,list(inputs),[name],**attrs));return name
    def norm(value,mean,scale):
        value=op('Div',op('Sub',op('Cast',value,to=T.DOUBLE),const(mean,np.float64)),const(scale,np.float64))
        return op('Cast',op('Clip',value,const(-8,np.float64),const(8,np.float64)),to=T.FLOAT)
    x=op('Reshape','features',const([112],np.int64))
    p=op('Clip',frozen,const(-2,np.float32),const(2,np.float32))
    features=[const([1.],np.float32),p,norm(x,combo['mean'],combo['scale'])]
    if force_latent or np.count_nonzero(coef[115:]):
        hidden=op('Concat',op('Reshape','next_hidden_0',const([128],np.int64)),
            op('Reshape','next_hidden_1',const([128],np.int64)),axis=0)
        features.append(norm(hidden,mean,scale))
    else:
        coef=coef[:115]
    correction=op('MatMul',op('Concat',*features,axis=0),const(coef,np.float32))
    output=op('Add',frozen,correction);graph.output[0].name=output
    onnx.checker.check_model(model);onnx.save(model,destination)
