"""Synthetic export regression: algebra, state precision, and reset contracts."""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper as h, numpy_helper as nh, TensorProto as T

from competition_engineering import campaign_onnx
from competition_engineering.autonomous_campaign import design
from competition_engineering.manual_search_core import predict_combo
from connectome.mlevolve_generated import validate_source


@pytest.mark.parametrize('mode,dims', [('linear', 339), ('bilinear', 563), ('normalize', 451), ('current', 115), ('current',339)])
@pytest.mark.parametrize('late_only', [False, True])
@pytest.mark.parametrize('target', [0, 1])
def test_fused_graph_matches_numpy_with_double_memory(tmp_path, monkeypatch, mode, dims, late_only, target):
    rng = np.random.default_rng(29)
    combo = {'mean': rng.normal(size=112)*.1, 'scale': np.ones(112)*1.1,
             'base_scale': np.array([1.2, .9]), 'base_bias': np.array([.05, -.02]),
             'root_coef': rng.normal(size=(115,2))*.01, 'root_strengths': np.array([.8,.8]),
             'magnitude_coef': rng.normal(size=(115,2))*.01, 'magnitude_strengths': np.array([.8,.6]),
             'gated_coef': rng.normal(size=(2,115,2))*.01, 'gated_strengths': np.array([.8,.9])}
    np.savez(tmp_path/'combo.npz', **combo)
    monkeypatch.setattr(campaign_onnx, 'COMBO', tmp_path/'combo.npz')
    monkeypatch.setattr(campaign_onnx, 'load_combo', lambda: combo)
    initializers = [nh.from_array(np.array(v, np.int64), k) for k,v in
                    [('start',[0]),('end',[2]),('axis',[2])]]
    nodes = [h.make_node('Slice', ['features','start','end','axis'], ['prediction']),
             h.make_node('Identity',['hidden_0'],['next_hidden_0']),
             h.make_node('Identity',['hidden_1'],['next_hidden_1'])]
    graph = h.make_graph(nodes,'synthetic',
            [h.make_tensor_value_info('features',T.FLOAT,[1,1,112]),
             h.make_tensor_value_info('hidden_0',T.FLOAT,[1,1,128]),
             h.make_tensor_value_info('hidden_1',T.FLOAT,[1,1,128])],
            [h.make_tensor_value_info('prediction',T.FLOAT,[1,1,2]),
             h.make_tensor_value_info('next_hidden_0',T.FLOAT,[1,1,128]),
             h.make_tensor_value_info('next_hidden_1',T.FLOAT,[1,1,128])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('',17)])
    model.ir_version = 8
    onnx.save(model,tmp_path/'baseline.onnx')
    coef = rng.normal(size=dims).astype(np.float32)*.003
    if mode=='current' and dims==339:
        np.savez(tmp_path/'unsafe.npz',coef=coef)
        with pytest.raises(ValueError,match='cannot discard learned temporal'):
            campaign_onnx.export(tmp_path/'unsafe.npz',tmp_path/'unsafe.onnx','current')
        coef[115:]=0
    np.savez(tmp_path/'model.npz',coef=coef)
    campaign_onnx.export(tmp_path/'model.npz',tmp_path/'fused.onnx',mode,late_only,.5,target=target)
    validate_source(campaign_onnx.callback_source(mode),training=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    session = ort.InferenceSession(str(tmp_path/'fused.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    x = rng.normal(size=(200,112)).astype(np.float32)*2
    base = predict_combo({'x':x,'p':x[:,:2]},combo)
    correction = (design(x,base,combo['mean'],combo['scale'],'linear')[:,:115] if mode=='current' else design(x,base,combo['mean'],combo['scale'],mode)) @ (coef[:115] if mode=='current' else coef)
    expected = base.copy()
    steps = np.arange(13200,13400)
    active = steps>=13333 if late_only else np.ones(200,bool)
    expected[active,target] += .5*correction[active]
    feed={'hidden_0':np.zeros((1,1,128),np.float32),'hidden_1':np.zeros((1,1,128),np.float32),
          'campaign_fast':np.zeros(112,np.float64),'campaign_slow':np.zeros(112,np.float64)}
    if mode=='normalize':
        feed['campaign_second']=np.zeros(112,np.float64)
    if mode=='current':
        del feed['campaign_fast'],feed['campaign_slow']
        assert len(session.get_outputs())==3
    actual=[]
    for i,row in enumerate(x):
        feed['features']=row.reshape(1,1,112)
        feed['campaign_step']=np.array(steps[i],np.int64)
        result=session.run(None,feed)
        actual.append(result[0])
        if mode!='current':
            feed['campaign_fast'],feed['campaign_slow']=result[3:5]
        if mode=='normalize':
            feed['campaign_second']=result[5]
    np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6)
    # Exercise the exported Python wrapper and alternating prebound buffers.
    # Retained output arrays must not change after later callback invocations.
    from types import SimpleNamespace
    namespace = {'__file__': str(tmp_path/'solution.py')}
    exec(campaign_onnx.callback_source(mode), namespace)
    callback = namespace['PredictionModel']()
    def replay(seq, values):
        return [callback.predict(SimpleNamespace(seq_ix=seq,step_in_seq=i,
                    need_prediction=i>=7,state=row)) for i,row in enumerate(values)]
    observed = replay(1,x)
    retained = np.array(observed[7:])
    expected_callback = base.copy() if late_only else expected
    np.testing.assert_allclose(retained,expected_callback[7:],atol=3e-6,rtol=3e-6)
    assert all(item is None for item in observed[:7])
    np.testing.assert_array_equal(retained,np.array(replay(2,x)[7:]))
    np.testing.assert_array_equal(retained,np.array(replay(2,x)[7:]))
    changed=x.copy()
    changed[100:]+=10
    np.testing.assert_array_equal(retained[:93],np.array(replay(3,changed)[7:100]))
    np.testing.assert_array_equal(retained,np.array(observed[7:]))
