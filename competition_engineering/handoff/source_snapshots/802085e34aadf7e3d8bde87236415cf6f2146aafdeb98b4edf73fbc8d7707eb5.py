import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from competition_engineering.alternative_representation import sequence_identity,temporal_features,source,design

def test_sequence_identity_without_parquet_statistics_and_mixed_identity(tmp_path):
    p=tmp_path/'train.parquet'
    pq.write_table(pa.table({'seq_ix':[27,27,27]}),p,write_statistics=False)
    assert sequence_identity(pq.ParquetFile(p),0)==27
    pq.write_table(pa.table({'seq_ix':[27,28,27]}),p,write_statistics=False)
    with pytest.raises(ValueError): sequence_identity(pq.ParquetFile(p),0)

def test_temporal_features_match_streaming_reset_and_ignore_future():
    rng=np.random.default_rng(17);x=rng.normal(size=(110,112)).astype(np.float32)
    b={'projection':rng.normal(size=(112,32)).astype(np.float32)*.1,'weights':rng.normal(size=(4,2,32,32)).astype(np.float32)*.1,'bias':rng.normal(size=(4,32)).astype(np.float32)*.1}
    delay,features=temporal_features(x,b);projected=x@b['projection'];hist=[[] for _ in range(4)];stream=[];delays=[]
    for t in range(len(x)):
        h=projected[t];layers=[];delays.append(np.concatenate([projected[t-d] if t>=d else np.zeros(32) for d in [1,4,16,64]]))
        for i,d in enumerate([1,4,16,64]):
            past=hist[i][t-d] if t>=d else np.zeros(32,dtype=np.float32);hist[i].append(h.copy())
            h=np.tanh(h@b['weights'][i,0]+past@b['weights'][i,1]+b['bias'][i]);layers.append(h)
        stream.append(np.concatenate(layers))
    np.testing.assert_allclose(stream,features,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(delays,delay,atol=2e-6,rtol=2e-6)
    changed=x.copy();changed[73:]+=9
    for a,c in zip(temporal_features(x,b),temporal_features(changed,b)): np.testing.assert_array_equal(a[:73],c[:73])
    for a,c in zip(temporal_features(x,b),temporal_features(x,b)): np.testing.assert_array_equal(a,c)

def test_standalone_omits_incumbent_predictions_and_generated_guard():
    from connectome.mlevolve_generated import validate_source
    validate_source(source(),training=True)
    z={'core':np.ones((5,115),np.float32),'tcn':np.ones((5,128),np.float32)}
    z['core'][:,1:3]=1000
    a=design(z,'tcn',np.zeros(128),np.ones(128),True)
    assert a.shape==(5,241) and np.max(a)==1

def test_polynomial_memory_filter_matches_causal_state_update_and_reset():
    from competition_engineering.orthogonal_memory_campaign import matrices,representations,source as memory_source
    from connectome.mlevolve_generated import validate_source
    a,b=matrices();assert np.max(np.abs(np.linalg.eigvals(a)))<1
    rng=np.random.default_rng(71);x=rng.normal(size=(1200,32));state=np.zeros((4,32));expected=[]
    for row in x:
        state=a@state+b[:,None]*row;expected.append(state.reshape(-1).copy())
    memory,smooth=representations(x,a,b)
    np.testing.assert_allclose(memory,expected,atol=2e-6,rtol=2e-5)
    changed=x.copy();changed[801:]+=7
    for v,w in zip(representations(x,a,b),representations(changed,a,b)): np.testing.assert_array_equal(v[:801],w[:801])
    for v,w in zip(representations(x,a,b),representations(x,a,b)): np.testing.assert_array_equal(v,w)
    validate_source(memory_source(),training=True)

def test_sparse_endpoint_convolution_preserves_early_padding_and_dense_parity():
    from competition_engineering.sparse_temporal import endpoint_windows,endpoint_features
    rng=np.random.default_rng(29);x=rng.normal(size=(137,112)).astype(np.float32)
    b={'projection':rng.normal(size=(112,32)).astype(np.float32)*.1,'weights':rng.normal(size=(4,2,32,32)).astype(np.float32)*.1,'bias':rng.normal(size=(4,32)).astype(np.float32)*.1}
    idx=np.arange(len(x));w,v=endpoint_windows(x@b['projection'],idx)
    actual=endpoint_features(w,v,b['weights'],b['bias'])
    np.testing.assert_allclose(actual,temporal_features(x,b)[1],atol=2e-6,rtol=2e-6)
    # Current-only topology is independent of every older input.
    altered=w.copy();altered[:,1:]+=77
    np.testing.assert_array_equal(endpoint_features(w,v,b['weights'],b['bias'],True),endpoint_features(altered,v,b['weights'],b['bias'],True))

def test_trainable_sparse_features_match_numpy_and_have_finite_gradients():
    import torch
    from competition_engineering.learned_causal_campaign import tensor_features,source as learned_source
    from competition_engineering.sparse_temporal import endpoint_windows,endpoint_features
    from connectome.mlevolve_generated import validate_source
    rng=np.random.default_rng(61);x=rng.normal(size=(103,112)).astype(np.float32);projection=rng.normal(size=(112,32)).astype(np.float32)*.1
    w=rng.normal(size=(4,2,32,32)).astype(np.float32)*.1;b=rng.normal(size=(4,32)).astype(np.float32)*.1
    windows,valid=endpoint_windows(x,np.arange(len(x)));pw,pv=endpoint_windows(x@projection,np.arange(len(x)))
    for instant in [False,True]:
        parameters=[torch.tensor(p,requires_grad=True) for p in [projection,w,b]]
        result=tensor_features(torch.from_numpy(windows),torch.from_numpy(valid),*parameters,instant)
        np.testing.assert_allclose(result.detach().numpy(),endpoint_features(pw,pv,w,b,instant),atol=3e-6,rtol=3e-6)
        result.square().sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    validate_source(learned_source(),training=True)

def test_disjoint_decoder_target_features_and_standalone_exclude_baseline():
    from competition_engineering.disjoint_readout_campaign import design,source as disjoint_source
    from connectome.mlevolve_generated import validate_source
    z={'core':np.ones((7,115),np.float32),'temporal_r0':np.stack([np.full((7,128),2),np.full((7,128),3)],axis=1)}
    z['core'][:,1:3]=100
    a=design(z,'temporal',0,0,np.zeros(128),np.ones(128),True);b=design(z,'temporal',0,1,np.zeros(128),np.ones(128),True)
    assert a.shape==b.shape==(7,241) and np.all(a[:,113:]==2) and np.all(b[:,113:]==3)
    assert np.max(a)==2 and np.max(b)==3
    validate_source(disjoint_source(),training=True)

def test_raw_target_model_omits_incumbent_inputs_and_source_guard():
    from competition_engineering.raw_target_campaign import direct,source as raw_source
    from connectome.mlevolve_generated import validate_source
    z={'core':np.ones((3,115),np.float32),'windows':np.zeros((3,16,112),np.float32),'valid':np.ones((3,16),bool)};model={}
    for t in [0,1]:
        suffix='temporal_r0_t'+str(t)
        for k,v in {'projection':np.zeros((112,32),np.float32),'weights':np.zeros((4,2,32,32),np.float32),'bias':np.zeros((4,32),np.float32),'head':np.zeros((128,1),np.float32),'linear':np.ones((113,1),np.float32)}.items(): model[suffix+'_'+k]=v
    before=direct(z,model,'temporal_r0');z['core'][:,1:3]+=99
    np.testing.assert_array_equal(before,direct(z,model,'temporal_r0'));assert np.all(before==113)
    validate_source(raw_source(),training=True)

def test_namespace_retry_is_bounded_and_never_retries_candidate_or_permission_errors():
    import time
    from types import SimpleNamespace
    from competition_engineering.isolation_retry import training_with_namespace_retry,NAMESPACE_TRANSIENT
    def run(errors):
        calls=[];retries=[]
        def launch():
            calls.append(1);error=errors[min(len(calls)-1,len(errors)-1)]
            process=SimpleNamespace(returncode=1 if error else 0,communicate=lambda timeout:(b'',error))
            return process,{},SimpleNamespace(join=lambda timeout:None)
        result=training_with_namespace_retry(launch,time.monotonic()+100,lambda attempt,error:retries.append(attempt),sleep=lambda seconds:None)
        return len(calls),retries,result
    calls,retries,result=run([NAMESPACE_TRANSIENT,b'']);assert calls==2 and retries==[1] and result[0].returncode==0
    calls,retries,result=run([NAMESPACE_TRANSIENT]);assert calls==3 and retries==[1,2] and result[0].returncode==1
    for error in [b'Traceback: candidate failed',b'bwrap: Creating new namespace failed: Operation not permitted']:
        calls,retries,result=run([error]);assert calls==1 and retries==[] and result[0].returncode==1
