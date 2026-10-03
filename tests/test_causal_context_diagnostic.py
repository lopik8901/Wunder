import numpy as np
import pytest
from competition_engineering.causal_context_diagnostic import prefix_context,gradient
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.residual import from_stats

def test_prefix_context_causal_and_reset_independent():
    rng=np.random.default_rng(45);x=rng.normal(size=(1000,112)).astype(np.float32);projection=rng.normal(size=(112,32)).astype(np.float32)*.1
    expected=prefix_context(x,projection);x[512:]+=99
    np.testing.assert_array_equal(expected,prefix_context(x,projection))
    np.testing.assert_array_equal(expected,prefix_context(x[:512].copy(),projection))
    with pytest.raises(ValueError):prefix_context(x[:511],projection)

def test_official_clipped_wp_gradient_matches_finite_differences():
    rng=np.random.default_rng(49);y=rng.normal(size=(500,2))*2;p=rng.normal(size=(500,2));f=rng.normal(size=(500,3));focus=rng.uniform(.1,1,size=500)
    yy=np.clip(y,-2,2);pp=np.clip(p,-2,2);a=np.abs(yy)*focus[:,None]*(np.abs(p)<2)
    moments=np.stack((f.T@a,f.T@(a*yy),f.T@(a*pp)));actual=gradient(focus_stats(y,p,focus),moments)
    for j in range(3):
        for t in [0,1]:
            delta=np.zeros_like(p);delta[:,t]=f[:,j]*1e-6
            numeric=(from_stats(focus_stats(y,p+delta,focus))[t]-from_stats(focus_stats(y,p-delta,focus))[t])/2e-6
            assert abs(actual[j,t]-numeric)<1e-7
