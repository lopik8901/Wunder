import numpy as np
import pytest
from competition_engineering.nonlinear_state_campaign import features,projections,source

def test_generated_fit_preserves_source_guard():
    from connectome.mlevolve_generated import validate_source
    validate_source(source(),training=True)

def test_joint_features_express_interactions_absent_from_separable_control():
    rng=np.random.default_rng(3);f=rng.normal(size=(4,371)).astype(np.float32)
    f[:]=f[0];f[1,3]+=2;f[2,130]+=2;f[3]=f[1]+f[2]-f[0]
    params=projections()
    separate=features(f,params,'separable');joint=features(f,params,'joint')
    np.testing.assert_allclose(separate[3]-separate[2]-separate[1]+separate[0],0,atol=5e-7)
    assert np.max(np.abs(joint[3]-joint[2]-joint[1]+joint[0]))>1e-4

def test_current_control_ignores_hidden_and_row_batching_is_invariant():
    rng=np.random.default_rng(5);f=rng.normal(size=(20,371)).astype(np.float32);p=projections()
    changed=f.copy();changed[:,115:]+=7
    np.testing.assert_array_equal(features(f,p,'current'),features(changed,p,'current'))
    for name,dim in [('linear',371),('current',243),('separable',499),('joint',499)]:
        a=features(f,p,name);b=np.concatenate([features(row[None],p,name) for row in f])
        assert a.shape==(20,dim)
        # GEMM/GEMV float32 accumulation differs; bound both against float64.
        reference=features(f.astype(np.float64),{k:v.astype(np.float64) for k,v in p.items()},name)
        np.testing.assert_allclose(a,reference,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(b,reference,atol=2e-6,rtol=2e-6)

def test_failed_or_missing_replication_blocks_search():
    from competition_engineering.nonlinear_state_evaluation import require_replication
    for report in [{},{'search_gate_passed':False},{'search_gate_passed':1}]:
        with pytest.raises(ValueError): require_replication(report)
    require_replication({'search_gate_passed':True})

def test_learned_controls_constrain_interactions_and_source_is_guarded():
    from competition_engineering.learned_state_readout import head_mask,source as learned_source,predict_head
    from connectome.mlevolve_generated import validate_source
    validate_source(learned_source(),training=True)
    rng=np.random.default_rng(31);f=np.tile(rng.normal(size=371),(4,1)).astype(np.float32)
    f[1,3]+=2;f[2,130]+=2;f[3]=f[1]+f[2]-f[0]
    for family in ['current','separable','joint']:
        model={'p_linear':np.zeros((371,1),np.float32),'p_w':rng.normal(size=(370,32)).astype(np.float32)*head_mask(family),
            'p_b':rng.normal(size=32).astype(np.float32),'p_o':rng.normal(size=(32,1)).astype(np.float32),'p_c':np.zeros(1,np.float32)}
        y=predict_head(f,model,'p');contrast=float((y[3]-y[2]-y[1]+y[0])[0])
        if family!='joint': assert abs(contrast)<2e-5
        else: assert abs(contrast)>1e-4

def test_histogram_tree_learns_thresholds_and_exports_exact_routing():
    import onnxruntime as ort
    from competition_engineering.state_tree_readout import build_tree,predict_tree,source as tree_source
    from competition_engineering.compact_tree_onnx import ensemble_model
    from connectome.mlevolve_generated import validate_source
    validate_source(tree_source(),training=True)
    rng=np.random.default_rng(5);x=rng.normal(size=(1024,2)).astype(np.float32)
    thresholds=np.zeros((2,1),np.float32);bins=(x>0).astype(np.uint8)
    y=np.where(x[:,0]>0,1.,-1.)+.5*np.where(x[:,1]>0,1.,-1.)
    tree=build_tree(bins,thresholds,y,np.ones(len(y)),min_leaf=16,l2=.0001,rate=1.)
    pred=predict_tree(x,tree);assert np.mean((pred-y)**2)<1e-10
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(ensemble_model([tree],0,2).SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    boundary=np.array([[0,0],[np.nextafter(np.float32(0),np.float32(1)),0],[-1,0],[1,1]],np.float32)
    np.testing.assert_allclose(session.run(None,{'x':boundary})[0][:,0],predict_tree(boundary,tree),atol=1e-7)

def test_state_innovations_reset_and_do_not_use_future_or_sampled_predecessor():
    from competition_engineering.state_innovation_readout import state_changes,source as innovation_source
    from connectome.mlevolve_generated import validate_source
    validate_source(innovation_source(),training=True)
    h=np.array([[1.,2.],[2.,4.],[5.,10.],[6.,11.]],np.float32)
    change=state_changes(h)
    np.testing.assert_array_equal(change,[[1,2],[1,2],[3,6],[1,1]])
    # Compute before row sampling: previous physical row, not prior sampled row.
    np.testing.assert_array_equal(change[[0,2]],[[1,2],[3,6]])
    altered=h.copy();altered[3]+=100
    np.testing.assert_array_equal(state_changes(altered)[:3],change[:3])
    np.testing.assert_array_equal(state_changes(h[2:])[0],h[2])

@pytest.mark.parametrize('family,dim',[('linear',371),('current',243),('separable',499),('joint',499)])
def test_nonlinear_onnx_head_matches_reference(tmp_path,monkeypatch,family,dim):
    import onnx
    import onnxruntime as ort
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    from competition_engineering import nonlinear_state_evaluation as exporter
    rng=np.random.default_rng(11);models=projections()
    models.update(mean=np.zeros(256),scale=np.ones(256));models[family]=rng.normal(size=(dim,2)).astype(np.float32)*.01
    def fixture(path,destination,work,force_latent):
        assert force_latent
        g=h.make_graph([h.make_node('MatMul',['features','coef'],['correction']),h.make_node('Add',['base','correction'],['prediction'])],
            'fixture',[h.make_tensor_value_info('features',T.FLOAT,[371]),h.make_tensor_value_info('base',T.FLOAT,[2])],
            [h.make_tensor_value_info('prediction',T.FLOAT,[2])],[nh.from_array(np.zeros((371,2),np.float32),'coef')])
        model=h.make_model(g,opset_imports=[h.make_opsetid('',17)]);model.ir_version=8;onnx.save(model,destination)
    monkeypatch.setattr(exporter.frozen_representation_onnx,'export',fixture)
    (tmp_path/'artifacts').mkdir();path=tmp_path/'model.onnx';strengths=[.5,.25]
    exporter.export(path,tmp_path,models,family,strengths)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    for _ in range(10):
        f=rng.normal(size=371).astype(np.float32);base=rng.normal(size=2).astype(np.float32)
        expected=base+features(f[None],models,family)[0]@models[family]*strengths
        np.testing.assert_allclose(session.run(None,{'features':f,'base':base})[0],expected,atol=5e-7,rtol=5e-7)
