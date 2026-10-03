"""Frozen activation extraction must match rowwise states and causal prefixes."""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from competition_engineering import frozen_latent_readout as latent


def test_sequence_hidden_features_match_rowwise_frozen_state(tmp_path,monkeypatch):
    rng=np.random.default_rng(2)
    init=[nh.from_array(np.array([1],np.int64),'squeeze_axis')]
    nodes=[h.make_node('Transpose',['features'],['sequence'],perm=[1,0,2])]
    previous='sequence'
    for i,input_size in enumerate([112,128]):
        for name,shape in [('W',(1,384,input_size)),('R',(1,384,128)),('B',(1,768))]:
            init.append(nh.from_array((rng.normal(size=shape)*.01).astype(np.float32),name+str(i)))
        nodes.append(h.make_node('GRU',[previous,'W'+str(i),'R'+str(i),'B'+str(i),'','hidden_'+str(i)],
                   ['layer'+str(i),'next_hidden_'+str(i)],hidden_size=128,linear_before_reset=1))
        nodes.append(h.make_node('Squeeze',['layer'+str(i),'squeeze_axis'],['squeezed'+str(i)]))
        previous='squeezed'+str(i)
    nodes.append(h.make_node('Transpose',[previous],['batch_hidden'],perm=[1,0,2]))
    init.append(nh.from_array(rng.normal(size=(128,2)).astype(np.float32)*.01,'readout'))
    nodes.append(h.make_node('MatMul',['batch_hidden','readout'],['prediction']))
    inputs=[h.make_tensor_value_info('features',T.FLOAT,[1,1,112])]+[
        h.make_tensor_value_info('hidden_'+str(i),T.FLOAT,[1,1,128]) for i in range(2)]
    outputs=[h.make_tensor_value_info('prediction',T.FLOAT,[1,1,2])]+[
        h.make_tensor_value_info('next_hidden_'+str(i),T.FLOAT,[1,1,128]) for i in range(2)]
    model=h.make_model(h.make_graph(nodes,'synthetic',inputs,outputs,init),opset_imports=[h.make_opsetid('',17)])
    model.ir_version=8;onnx.save(model,tmp_path/'baseline.onnx')
    monkeypatch.setattr(latent,'COMBO',tmp_path/'combo.npz')
    extractor=latent.LatentExtractor()
    x=rng.normal(size=(80,112)).astype(np.float32)
    prediction,hidden=extractor.predict(x)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    row=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    feed={'hidden_0':np.zeros((1,1,128),np.float32),'hidden_1':np.zeros((1,1,128),np.float32)}
    expected=[];states=[]
    for value in x:
        results=row.run(None,{**feed,'features':value[None,None]})
        expected.append(results[0][0,0]);states.append(np.concatenate([v.reshape(-1) for v in results[1:]]))
        feed['hidden_0'],feed['hidden_1']=results[1:]
    np.testing.assert_allclose(prediction,expected,atol=2e-7)
    np.testing.assert_allclose(hidden,states,atol=2e-7)
    changed=x.copy();changed[40:]+=.7
    changed_prediction,changed_hidden=extractor.predict(changed)
    np.testing.assert_array_equal(prediction[:40],changed_prediction[:40])
    np.testing.assert_array_equal(hidden[:40],changed_hidden[:40])
    repeated_prediction,repeated_hidden=extractor.predict(x)
    np.testing.assert_array_equal(prediction,repeated_prediction)
    np.testing.assert_array_equal(hidden,repeated_hidden)
