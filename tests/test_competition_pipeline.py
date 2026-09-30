import numpy as np
from competition_engineering.residual import sufficient, from_stats, features
from wnn_connectome_starterpack.utils import weighted_pearson


def test_bootstrap_moments_match_official_clipping_and_global_aggregation():
    rng = np.random.default_rng(11)
    y = rng.normal(size=(2000,2))*3
    p = y*.2+rng.normal(size=y.shape)
    actual = from_stats(sufficient(y[:700],p[:700])+sufficient(y[700:],p[700:]))
    expected = [weighted_pearson(y[:,i],p[:,i]) for i in range(2)]
    np.testing.assert_allclose(actual,expected,atol=1e-8)
    np.testing.assert_array_equal(from_stats(sufficient(y,np.ones_like(y))),[0.,0.])


def test_residual_features_do_not_depend_on_future_or_labels():
    rng=np.random.default_rng(12)
    z={"x":rng.normal(size=(200,112)),"p":rng.normal(size=(200,2))}
    mean=np.zeros(112); scale=np.ones(112)
    original=features(z,mean,scale)
    z["x"][100:]=10000; z["p"][100:]=-10000
    np.testing.assert_array_equal(features(z,mean,scale)[:100],original[:100])


def test_tcn_causality_chunk_context_and_incremental_cpu_parity():
    import torch
    from competition_engineering.gpu_residual import ResidualTCN, IncrementalTCN, predict_sequence
    torch.manual_seed(31); torch.set_num_threads(1)
    model=ResidualTCN().eval()
    # Nonzero output weights ensure the parity check is meaningful.
    torch.nn.init.normal_(model.head.weight,std=.1)
    x=np.random.default_rng(31).normal(size=(2100,114)).astype(np.float32)
    with torch.no_grad():
        full=model(torch.from_numpy(x.T.copy())[None])[0].T.numpy()
    chunk=predict_sequence(model,x,torch.device('cpu'))
    np.testing.assert_allclose(full,chunk,atol=1e-6)
    replay=IncrementalTCN(model)
    actual=np.stack([replay.predict(row) for row in x])
    np.testing.assert_allclose(full,actual,atol=1e-6)
    changed=x.copy(); changed[1000:]+=1000
    np.testing.assert_array_equal(predict_sequence(model,changed,torch.device('cpu'))[:1000],full[:1000])
    reset=IncrementalTCN(model)
    np.testing.assert_allclose(np.stack([reset.predict(row) for row in x[:100]]),full[:100],atol=1e-6)


def test_ridge_system_point_features_match_batch_including_calibration():
    from competition_engineering.gpu_residual import point_base
    rng=np.random.default_rng(22)
    z={"x":rng.normal(size=(30,112)).astype(np.float32),"p":rng.normal(size=(30,2)).astype(np.float32)}
    ridge={"mean":rng.normal(size=112),"scale":np.ones(112),
           "strengths":np.array([.5,1.]),"coef":rng.normal(size=(115,2))*.01}
    scale=[.75,.75]; bias=[-.1,-.1]
    transformed={**z,"p":z["p"]*np.array(scale,np.float32)+np.array(bias,np.float32)}
    expected=(transformed["p"]+ridge["strengths"]*(features(transformed,ridge["mean"],ridge["scale"])@ridge["coef"])).astype(np.float32)
    actual=np.stack([point_base(p,x,scale,bias,ridge) for x,p in zip(z["x"],z["p"])])
    np.testing.assert_allclose(actual,expected,atol=1e-7)
    assert point_base(None,z["x"][0],scale,bias,ridge) is None
