import numpy as np
import pytest
from competition_engineering import book_side_campaign as c

def test_side_partition_symmetry_and_no_depth_order_assumption():
    rng=np.random.default_rng(19);x=rng.normal(size=(27,112));permuted=x.copy();swapped=x.copy()
    for offset in [0,52]:
        for start in [22,33]:
            cols=np.arange(offset+start,offset+start+11);permuted[:,cols]=x[:,cols[rng.permutation(11)]]
        swapped[:,offset+22:offset+33]=x[:,offset+33:offset+44]
        swapped[:,offset+33:offset+44]=x[:,offset+22:offset+33]
    for plus in [False,True]:
        np.testing.assert_allclose(c.side_features(x,plus),c.side_features(permuted,plus),atol=1e-7)
    np.testing.assert_allclose(c.side_features(swapped),-c.side_features(x),atol=1e-7)
    np.testing.assert_allclose(c.side_features(swapped,True),c.side_features(x,True),atol=1e-7)

def test_features_are_row_local_finite_and_ignore_unrelated_columns():
    rng=np.random.default_rng(20);x=rng.normal(size=(30,112));changed=x.copy();changed[15:]*=100
    np.testing.assert_array_equal(c.side_features(changed)[:15],c.side_features(x)[:15])
    stream=np.concatenate([c.side_features(row[None]) for row in x]);np.testing.assert_array_equal(stream,c.side_features(x))
    other=x.copy();other[:,104:]+=100;np.testing.assert_array_equal(c.side_features(other),c.side_features(x))
    for v in [0.,1e20,-1e20]:assert np.isfinite(c.side_features(np.full((3,112),v))).all()

def test_standalone_design_omits_incumbent_predictions_and_development_requires_freeze(tmp_path,monkeypatch):
    z={'core':np.zeros((4,115)),'gru':np.ones((4,128)),'minus':np.ones((4,8))}
    model={'gru_mean':np.zeros(128),'gru_scale':np.ones(128),'minus_mean':np.zeros(8),'minus_scale':np.ones(8)}
    before=c.design(z,'minus',model,True);z['core'][:,1:3]=99
    np.testing.assert_array_equal(before,c.design(z,'minus',model,True));assert before.shape==(4,249)
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'initialize',lambda:{})
    with pytest.raises(AssertionError,match='Freeze candidate'):c.prepare('development')
    assert not (tmp_path/'development').exists()
