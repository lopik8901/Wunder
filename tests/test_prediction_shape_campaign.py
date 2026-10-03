import json
import numpy as np
import pytest
from competition_engineering import prediction_shape_campaign as c
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.residual import from_stats

def test_interpolant_and_streaming_are_bounded_and_target_local():
    knots=np.linspace(-2,2,17);values=np.column_stack((np.tanh(knots)*1.9,knots*.7))
    p=np.array([[-9,9],[-.13,.42],[2,-2],[.8,-.8]])
    basis=c.interpolation_basis(p[:,0],knots)
    np.testing.assert_allclose(basis.sum(1),1)
    np.testing.assert_allclose(basis@values[:,0],np.interp(np.clip(p[:,0],-2,2),knots,values[:,0]))
    batch=c.curve_predict(p,knots,values,[.5,1]);stream=np.concatenate([c.curve_predict(row[None],knots,values,[.5,1]) for row in p])
    np.testing.assert_allclose(batch,stream);assert np.max(np.abs(batch))<=2
    changed=p.copy();changed[:,1]*=-1
    np.testing.assert_array_equal(c.curve_predict(changed,knots,values,[.5,1])[:,0],batch[:,0])

def test_affine_control_preserves_clipped_wp_even_with_saturated_input():
    rng=np.random.default_rng(4);p=rng.normal(size=(1000,2))*2;y=rng.normal(size=(1000,2));knots=np.linspace(-2,2,17)
    values=knots[:,None]*np.array([.6,.7])+np.array([.1,-.2]);a=c.curve_predict(p,knots,values,[1,1])
    for w in [np.ones(1000),rng.uniform(.05,1,size=1000)]:
        np.testing.assert_allclose(from_stats(focus_stats(y,p,w)),from_stats(focus_stats(y,a,w)),atol=1e-12)

def test_generated_fit_recovers_nonlinear_signal_with_wp_selection(tmp_path):
    namespace={};exec(c.source(),namespace)
    rng=np.random.default_rng(9);files=[]
    for role in [0,1]:
        p=rng.uniform(-2,2,size=(5000,2));y=1.8*np.tanh(2*p)+rng.normal(0,.03,p.shape)
        path=tmp_path/f'{role}.npz';np.savez(path,base=p,y=y,focus=np.ones(len(p)),role=role);files.append(str(path))
    result=namespace['fit'](files,str(tmp_path));assert result['selected_strength']==[1.,1.]
    with np.load(tmp_path/'models.npz') as q:
        pred=c.curve_predict(p,q['knots'],q['values'],q['strength']);assert np.max(np.abs(q['values']))<=1.9
    gain=from_stats(focus_stats(y,pred,np.ones(len(p))))-from_stats(focus_stats(y,p,np.ones(len(p))))
    assert np.all(gain>.01)

def test_failed_mechanism_blocks_selection_and_candidate_training(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'reserve',lambda:{})
    directory=tmp_path/'shape_diagnostic_work/artifacts';directory.mkdir(parents=True)
    (directory/'crossfit.json').write_text(json.dumps({'gate_passed':False}))
    with pytest.raises(AssertionError,match='No selection access'):c.prepare('selection')
    with pytest.raises(AssertionError,match='Fresh shape evidence'):c.train()
    assert not (tmp_path/'fit_work').exists()
