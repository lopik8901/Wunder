import numpy as np
import pytest

def test_consensus_rejects_disagreement_and_preserves_smaller_agreement():
    from competition_engineering.consensus_ensemble import consensus
    a=np.array([2.,-3.,2.,-2.,0.,1.]);b=np.array([3.,-1.,-1.,2.,4.,1.])
    np.testing.assert_allclose(consensus(a,b),[2.,-1.,0.,0.,0.,1.])
    np.testing.assert_allclose(consensus(a,b),consensus(b,a))

def test_consensus_onnx_head_matches_numpy(tmp_path,monkeypatch):
    import onnx
    import onnxruntime as ort
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    from competition_engineering import consensus_ensemble as ensemble
    rng=np.random.default_rng(18)
    models={k:rng.normal(size=(371,2)).astype(np.float32)*.01 for k in ['latent_a','latent_b']}
    def fixture(path,destination,work,force_latent=False):
        assert force_latent
        graph=h.make_graph([h.make_node('MatMul',['features','coef'],['correction']),h.make_node('Add',['base','correction'],['prediction'])],
            'fixture',[h.make_tensor_value_info('features',T.FLOAT,[371]),h.make_tensor_value_info('base',T.FLOAT,[2])],
            [h.make_tensor_value_info('prediction',T.FLOAT,[2])],[nh.from_array((models['latent_a']+models['latent_b'])*.125,'coef')])
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=8;onnx.save(model,destination)
    monkeypatch.setattr(ensemble.frozen_representation_onnx,'export',fixture)
    path=tmp_path/'model.onnx';ensemble.export_consensus(None,path,tmp_path,models)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    for _ in range(30):
        f=rng.normal(size=371).astype(np.float32);base=rng.normal(size=2).astype(np.float32)
        expected=base+ensemble.corrections(f,models,[1,1])[2]
        np.testing.assert_allclose(session.run(None,{'features':f,'base':base})[0],expected,atol=3e-7,rtol=3e-7)

def test_checkpoint_evidence_paths_follow_actual_artifact_layout(tmp_path):
    from tools.finalize_representation import evidence_paths
    expected={'protocol.json','knowledge_v5.json','knowledge_v6.json','replicated_readout/replication.json',
        'replicated_readout/candidate_checks.json','consensus_ensemble/protocol.json',
        'consensus_ensemble/replication.json','consensus_ensemble/candidate_checks.json',
        'consensus_ensemble/mechanism_comparison.json','library_v10_manifest.json','library_v11_manifest.json'}
    assert {p.relative_to(tmp_path).as_posix() for p in evidence_paths(tmp_path)}==expected

def test_prospective_group_roles_are_disjoint_and_repeatable():
    from competition_engineering.representation_campaign import group_plan
    fit=list(range(1024));pool=list(range(4096));excluded=list(range(1024,1280))
    plan=group_plan(fit,pool,excluded)
    assert plan==group_plan(fit,pool,excluded)
    assert [len(plan[k]) for k in ['fit_a','fit_b','selection','replication']]==[512,512,256,256]
    roles=[set(v) for v in plan.values()]
    for i,left in enumerate(roles):
        for right in roles[i+1:]:
            assert not left&right
    assert not (roles[2]|roles[3])&set(excluded)
    assert roles[0]|roles[1]==set(fit)
    with pytest.raises(ValueError):
        group_plan(fit,pool[:1300],excluded)

def test_selector_requires_gain_in_both_populations_and_prefers_zero_ties():
    from competition_engineering.replicated_readout import choose_strengths,source
    from connectome.mlevolve_generated import validate_source
    stats=np.zeros((2,6,6,2));stats[:,:,0]=10;stats[:,:,3]=10;stats[:,:,4]=10;stats[:,:,5]=5
    stats[0,5,5]=7;stats[1,5,5]=4
    stats[:,3,5]=6
    assert choose_strengths(stats)['strengths']==[.25,.25]
    stats[:,:,5]=5
    assert choose_strengths(stats)['strengths']==[0.,0.]
    validate_source(source(),training=True)

@pytest.mark.parametrize('latent',[False,True])
def test_frozen_state_export_preserves_layer_order_and_normalization(tmp_path,monkeypatch,latent):
    import onnx
    import onnxruntime as ort
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    from competition_engineering import frozen_representation_onnx as exporter
    from competition_engineering.replicated_readout import readout_design
    rng=np.random.default_rng(145)
    combo={'mean':rng.normal(size=112),'scale':rng.uniform(.4,2,112)}
    def initial_export(path,destination,mode):
        initializers=[nh.from_array(np.array(v,np.int64),name) for name,v in [('shape',[112]),('idx',[0,1])]]
        nodes=[h.make_node('Reshape',['features','shape'],['x']),h.make_node('Gather',['x','idx'],['base'],axis=0),
            h.make_node('Identity',['hidden_0'],['next_hidden_0']),h.make_node('Identity',['hidden_1'],['next_hidden_1'])]
        graph=h.make_graph(nodes,'frozen_fixture',[
            h.make_tensor_value_info('features',T.FLOAT,[1,1,112]),
            h.make_tensor_value_info('hidden_0',T.FLOAT,[1,1,128]),h.make_tensor_value_info('hidden_1',T.FLOAT,[1,1,128])],
            [h.make_tensor_value_info('base',T.FLOAT,[2]),h.make_tensor_value_info('next_hidden_0',T.FLOAT,[1,1,128]),
             h.make_tensor_value_info('next_hidden_1',T.FLOAT,[1,1,128])],initializers)
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=8;onnx.save(model,destination)
    monkeypatch.setattr(exporter.campaign_onnx,'export',initial_export)
    monkeypatch.setattr(exporter.campaign_onnx,'load_combo',lambda:combo)
    mean=rng.normal(size=256)*.1;scale=rng.uniform(.05,.5,256)
    coef=rng.normal(size=(371,2)).astype(np.float32)*.003
    if not latent:
        coef[115:]=0
    np.savez(tmp_path/'model.npz',coef=coef,mean=mean,scale=scale)
    exporter.export(tmp_path/'model.npz',tmp_path/'fused.onnx',tmp_path)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(tmp_path/'fused.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    for _ in range(20):
        x=rng.normal(size=112).astype(np.float32)*3;hidden=rng.normal(size=256).astype(np.float32)
        base=x[:2]
        current=np.column_stack((np.ones(1),np.clip(base[None],-2,2),np.clip((x[None]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
        expected=base+readout_design(current,hidden[None],mean,scale)[0]@coef
        actual=session.run(None,{'features':x.reshape(1,1,112),'hidden_0':hidden[:128].reshape(1,1,128),'hidden_1':hidden[128:].reshape(1,1,128)})[0]
        np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=5e-7)
