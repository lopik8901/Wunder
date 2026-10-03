import numpy as np
import pytest
from competition_engineering.sequence_diversity import sampling_indices,source,replication_gate

def test_row_budgets_nested_sampling_and_no_extra_groups_in_narrow_arm():
    for replica in [0,1]:
        narrow=sampling_indices(500,123,replica,'narrow',0)
        broad=sampling_indices(500,123,replica,'broad',0)
        assert len(narrow)==128 and len(broad)==64 and set(broad)<=set(narrow)
        assert len(sampling_indices(500,234,replica,'narrow',1))==0
        assert len(sampling_indices(500,234,replica,'broad',1))==64
        assert 1024*len(narrow)==2048*len(broad)==131072
    assert not np.array_equal(sampling_indices(500,123,0,'narrow',0),sampling_indices(500,123,1,'narrow',0))
    with pytest.raises(ValueError): sampling_indices(500,123,0,'broad',3)

def test_generated_trainer_preserves_import_and_access_guard():
    from connectome.mlevolve_generated import validate_source
    validate_source(source(),training=True)

def test_real_trainer_keeps_fit_baselines_when_selection_has_different_length(tmp_path,monkeypatch):
    from competition_engineering import sequence_diversity as campaign
    rng=np.random.default_rng(27);files=[]
    for group,role in [(1,0),(2,0),(3,1),(4,1),(5,2)]:
        n=500 if role<2 else 17
        f=rng.normal(size=(n,371)).astype(np.float32)*.1;f[:,0]=1
        base=rng.normal(size=(n,2)).astype(np.float32)*.2
        path=tmp_path/f'{group}.npz'
        np.savez(path,f=f,base=base,y=base+rng.normal(size=(n,2)).astype(np.float32)*.2,
            focus=np.full(n,.3),indices=np.arange(n),group=group,role=role)
        files.append(str(path))
    def toy_budget(rows,groups,arm):
        assert rows==256 and groups==(2 if arm=='narrow' else 4)
    monkeypatch.setattr(campaign,'validate_allocation',toy_budget)
    output=tmp_path/'artifacts';output.mkdir()
    campaign.fit(files,str(output))
    import json
    report=json.loads((output/'training.json').read_text())
    assert set(report)=={'narrow_r0','broad_r0','narrow_r1','broad_r1'}
    for row in report.values():
        assert len(row['selected_uniform_fit_wp'])==2
        assert all(np.isfinite(row['selected_uniform_fit_wp']))
        assert [h['epoch'] for h in row['history']]==[0,5,10,20]

def test_replication_requires_diversity_effect_in_both_paired_fits():
    point=np.zeros((2,2,2,2,2));point[:,0,1]=.001;point[:,1,1]=.003
    draws=np.broadcast_to(point,(100,)+point.shape).copy();halves=[[.003,.003],[.003,.003]]
    passed,ci,_=replication_gate(point,draws,halves);assert passed
    np.testing.assert_allclose(ci,.002)
    point[1,1,1]=.0005;draws=np.broadcast_to(point,(100,)+point.shape).copy()
    assert not replication_gate(point,draws,halves)[0]
    point[:,1,1]=.003;draws=np.broadcast_to(point,(100,)+point.shape).copy()
    assert not replication_gate(point,draws,[[.003,.003],[-.001,.003]])[0]

@pytest.mark.parametrize('shared,linear',[(True,False),(False,False),(True,True)])
def test_fused_readout_preserves_single_and_separate_checkpoints(tmp_path,monkeypatch,shared,linear):
    import onnx
    import onnxruntime as ort
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    from competition_engineering import sequence_diversity_evaluation as evaluation
    from competition_engineering.sequence_diversity import corrections
    rng=np.random.default_rng(13);model={};name='broad_r0'
    model[name+'_linear']=rng.normal(size=(371,2)).astype(np.float32)*.01
    model[name+'_linear_strengths']=np.array([.5,.25],np.float32)
    for t in [0,1]:
        p=name+'_t'+str(t)
        model[p+'_linear']=model[name+'_linear'][:,t:t+1]
        model[p+'_w']=rng.normal(size=(370,32)).astype(np.float32)*.05
        model[p+'_b']=rng.normal(size=32).astype(np.float32)*.1
        model[p+'_o']=rng.normal(size=(32,1)).astype(np.float32)*.03
        model[p+'_c']=rng.normal(size=1).astype(np.float32)*.01
        model[p+'_strength']=np.float32([.5,.25][t])
    if shared:
        model[name+'_t1_w']=model[name+'_t0_w'].copy();model[name+'_t1_b']=model[name+'_t0_b'].copy()
    def fixture(path,destination,work,force_latent):
        with np.load(path) as q: coef=q['coef']
        graph=h.make_graph([h.make_node('MatMul',['f','coef'],['correction']),h.make_node('Add',['base','correction'],['prediction'])],
            'fixture',[h.make_tensor_value_info('f',T.FLOAT,[371]),h.make_tensor_value_info('base',T.FLOAT,[2])],
            [h.make_tensor_value_info('prediction',T.FLOAT,[2])],[nh.from_array(coef,'coef')])
        m=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);m.ir_version=8;onnx.save(m,destination)
    monkeypatch.setattr(evaluation.frozen_representation_onnx,'export',fixture)
    (tmp_path/'artifacts').mkdir();path=tmp_path/'model.onnx'
    evaluation.export(path,tmp_path,model,name,{'mean':np.zeros(256),'scale':np.ones(256)},linear)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    for _ in range(10):
        f=rng.normal(size=371).astype(np.float32);base=rng.normal(size=2).astype(np.float32)
        expected=base+corrections(f[None],model,name,linear)[0]
        np.testing.assert_allclose(session.run(None,{'f':f,'base':base})[0],expected,atol=5e-7,rtol=5e-7)
