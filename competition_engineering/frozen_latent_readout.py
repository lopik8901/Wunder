"""Diagnose objective signal in the incumbent's already-frozen GRU states.

No recurrent parameters, update rules, or timescales are changed. Sequence
extraction exposes existing forward GRU activations; deployment uses those same
states already computed by the frozen rowwise graph.
"""
from competition_engineering import objective_alignment as oa
import json
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper,TensorProto
from threadpoolctl import threadpool_limits
from competition_engineering.manual_search_core import COMBO,TRAIN_1024,SEARCH,load_combo,predict_combo
from competition_engineering.pipeline import cached,write_json
from tools.diagnose_causal_scoring_population import stats_and_gradient


def latent_graph():
    model=onnx.load(COMBO.parent/'baseline.onnx')
    nodes=[node for node in model.graph.node if node.op_type=='GRU']
    if len(nodes)!=2:
        raise ValueError('Expected the two frozen128-unit GRU layers')
    for node in nodes:
        for attr in node.attribute:
            if attr.name=='direction' and attr.s!=b'forward':
                raise ValueError('Latent extractor must be forward causal')
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_param='steps'
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_param='steps'
    del model.graph.value_info[:]
    model.graph.output.extend([helper.make_tensor_value_info(node.output[0],TensorProto.FLOAT,['steps',1,1,128]) for node in nodes])
    onnx.checker.check_model(model)
    return model


class LatentExtractor:
    def __init__(self,model_bytes=None):
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(model_bytes or latent_graph().SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])

    def predict(self,x):
        x=np.asarray(x,np.float32)
        if x.ndim!=2 or x.shape[1]!=112 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Expected nonempty finite112-feature rows')
        values=self.session.run(None,{'features':x[None],'hidden_0':np.zeros((1,1,128),np.float32),
                                    'hidden_1':np.zeros((1,1,128),np.float32)})
        h=np.concatenate((values[3][:,0,0],values[4][:,0,0]),axis=1)
        return values[0][0],h


def gradient_summary(directory,extractor,combo,training):
    stats=np.zeros((6,2));feature_stats=np.zeros((3,257,2));sx=np.zeros(256);sxx=sx.copy();rows=0
    for count,(_,z) in enumerate(cached(directory)):
        prediction,h=extractor.predict(z['x'])
        if count==0:
            np.testing.assert_allclose(prediction[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
            changed=z['x'].copy();changed[10000:]+=.7
            mutated,changed_h=extractor.predict(changed)
            np.testing.assert_array_equal(prediction[:10000],mutated[:10000])
            np.testing.assert_array_equal(h[:10000],changed_h[:10000])
            repeated,repeated_h=extractor.predict(z['x'])
            np.testing.assert_array_equal(prediction,repeated);np.testing.assert_array_equal(h,repeated_h)
        idx=np.flatnonzero(z['need'])[::40] if training else np.flatnonzero(z['mask'])
        base=predict_combo(z,combo)[idx];p=np.clip(base,-2,2).astype(np.float64)
        y=np.clip(z['y'][idx],-2,2).astype(np.float64);w=np.abs(y);active=np.abs(base)<2
        f=np.column_stack((np.ones(len(idx)),h[idx])).astype(np.float64)
        stats+=np.stack((w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)))
        feature_stats+=np.stack((f.T@(w*active),f.T@(w*active*y),f.T@(w*active*p)))
        sx+=h[idx].sum(0);sxx+=(h[idx].astype(np.float64)**2).sum(0);rows+=len(idx)
        if training and (count+1)%256==0:
            print(json.dumps({'phase':'frozen_latent_diagnosis','training_sequences':count+1}),flush=True)
    return stats_and_gradient(stats,feature_stats),sx/rows,np.sqrt(np.maximum(sxx/rows-(sx/rows)**2,.05**2)),rows


def main():
    path=oa.OUT/'frozen_latent_diagnosis.json'
    if path.exists():
        return
    oa.event('hypothesis_planned',id='frozen_GRU_latent_objective_diagnostic',
        question='Do existing frozen latent features retain objective-aligned residual directions lost by the two output values?',
        rule='Fit a current readout only for targets with train/search latent-gradient cosine>=0.3 and positive training-side gradient norm',
        scope='No recurrent weight, state update, new memory mechanism or timescale tuning; existing forward hidden states only',
        boundary='Fixed training1024 and search64 diagnostics only')
    model=latent_graph();onnx.save(model,oa.OUT/'frozen_latent_extractor.onnx')
    extractor=LatentExtractor(model.SerializeToString());combo=load_combo()
    with threadpool_limits(limits=1):
        train,mean,scale,train_rows=gradient_summary(TRAIN_1024,extractor,combo,True)
        search,_,_,search_rows=gradient_summary(SEARCH,extractor,combo,False)
    tg=(train[1:]-mean[:,None]*train[:1])/scale[:,None]
    sg=(search[1:]-mean[:,None]*search[:1])/scale[:,None]
    norms=np.linalg.norm(tg,axis=0)
    cosine=(tg*sg).sum(0)/np.maximum(norms*np.linalg.norm(sg,axis=0),1e-20)
    eligible=[t for t in range(2) if cosine[t]>=.3 and norms[t]>1e-8]
    np.savez(oa.OUT/'frozen_latent_scaling.npz',mean=mean,scale=scale)
    report={'latent_gradient_cosine_per_target':cosine.tolist(),'training_latent_gradient_norm':norms.tolist(),
            'eligible_targets':eligible,'train_rows':train_rows,'search_rows':search_rows,
            'causal_prefix':True,'reset_determinism':True,'baseline_prediction_parity':True,
            'extractor_sha256':oa.digest(oa.OUT/'frozen_latent_extractor.onnx'),
            'decision':'A new fixed-feature readout fit requires the predeclared alignment rule; no temporal dynamics are tuned.'}
    write_json(path,report);oa.event('frozen_latent_diagnostic_completed',**report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
