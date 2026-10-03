import numpy as np
import json
import pytest
from competition_engineering import objective_alignment as oa
from competition_engineering.objective_alignment import objective_value_gradient, source, metrics
from connectome.mlevolve_generated import validate_source


def test_training_source_preserves_domain_guard():
    # Internal training sequence selection must not be confused with the
    # protected holdout domain, and serialization uses allowed Path methods.
    code = source()
    validate_source(code, training=True)
    assert 'internal_selection' in code
    assert 'holdout' not in code
    assert 'open(' not in code
    from competition_engineering.full_moment_objective import source as exact_source
    validate_source(exact_source(),training=True)


def test_clipped_wp_gradient_matches_finite_difference():
    rng=np.random.default_rng(42)
    f=rng.normal(size=(150,7)); y=np.clip(rng.normal(size=150),-2,2)
    base=rng.normal(size=150)*1.4; coef=rng.normal(size=7)*.03
    value,grad=objective_value_gradient(coef,f,base,y,.1)
    eps=1e-6
    numerical=[]
    for i in range(len(coef)):
        direction=np.eye(len(coef))[i]*eps
        plus=objective_value_gradient(coef+direction,f,base,y,.1)[0]
        minus=objective_value_gradient(coef-direction,f,base,y,.1)[0]
        numerical.append((plus-minus)/(2*eps))
    np.testing.assert_allclose(grad,numerical,atol=1e-8,rtol=1e-6)
    assert np.isfinite(value)


def test_wp_invariance_and_clipping_distinction():
    y=np.array([-1.5,-.5,.2,.7,1.3]); base=y*.4+.1
    rows=metrics(y,base,base,[0.,1.])
    np.testing.assert_allclose(rows[0]['wp'],rows[1]['wp'],atol=1e-12)
    assert rows[1]['weighted_raw_mse'] < rows[0]['weighted_raw_mse']


def test_resumed_fit_detects_modified_artifact(tmp_path):
    names=['train.py','combo.npz','artifacts/models.npz','artifacts/training.json']
    (tmp_path/'artifacts').mkdir()
    for name in names:
        (tmp_path/name).write_bytes(b'fixture')
    (tmp_path/'fit_identity.json').write_text(json.dumps({name:oa.digest(tmp_path/name) for name in names}))
    oa.verify_fit_identity(tmp_path)
    (tmp_path/'artifacts/models.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity changed'):
        oa.verify_fit_identity(tmp_path)


def test_group_risk_variance_gradient_matches_finite_difference():
    from competition_engineering.group_objective_alignment import group_value_gradient
    rng=np.random.default_rng(12)
    groups=[(rng.normal(size=(80,5)),rng.normal(size=80),np.clip(rng.normal(size=80),-2,2)) for _ in range(4)]
    zero=np.zeros(5)
    reference=[oa.objective_value_gradient(zero,*g,0.)[0] for g in groups]
    coef=rng.normal(size=5)*.01
    _,gradient=group_value_gradient(coef,groups,reference,100.,.1)
    numerical=[]
    for i in range(5):
        change=np.eye(5)[i]*1e-6
        plus=group_value_gradient(coef+change,groups,reference,100.,.1)[0]
        minus=group_value_gradient(coef-change,groups,reference,100.,.1)[0]
        numerical.append((plus-minus)/2e-6)
    np.testing.assert_allclose(gradient,numerical,atol=1e-8,rtol=1e-6)


def test_population_log_loss_accepts_boolean_scoring_indicators():
    from tools.diagnose_causal_scoring_population import binary_log_loss
    assert binary_log_loss(np.array([True,False]),np.array([.8,.2]))==pytest.approx(-np.log(.8))
    assert np.isfinite(binary_log_loss(np.array([False,True]),np.array([0.,1.])))


def test_causal_population_diagnostic_completes_on_synthetic_sequences(tmp_path,monkeypatch):
    from tools import diagnose_causal_scoring_population as diagnostic
    from pathlib import Path
    rng=np.random.default_rng(7)
    sequences=[]
    for group in range(16):
        x=rng.normal(size=(128,112)).astype(np.float32)
        sequences.append((group,{'x':x,'need':np.ones(128,bool),'mask':x[:,0]>.5,
            'step':np.arange(128)+99,'y':x[:,:2]*.3+rng.normal(size=(128,2))*.2}))
    monkeypatch.setattr(diagnostic,'cached',lambda _:iter(sequences))
    monkeypatch.setattr(diagnostic,'load_combo',lambda:{'mean':np.zeros(112),'scale':np.ones(112)})
    monkeypatch.setattr(diagnostic,'predict_combo',lambda z,_:z['x'][:,:2]*.4)
    monkeypatch.setattr(oa,'OUT',tmp_path)
    diagnostic.main()
    report=json.loads((tmp_path/'causal_population_diagnosis.json').read_text())
    assert report['source_sha256']==oa.digest(Path(diagnostic.__file__))
    assert report['diagnostic_rows']==2048
    assert 0<=report['cross_fitted_auc']<=1
    assert report['cross_fitted_log_loss']<report['prior_log_loss']


def test_aggregated_gradient_matches_exact_clipped_wp_derivative():
    from tools.diagnose_causal_scoring_population import stats_and_gradient
    rng=np.random.default_rng(5)
    f=rng.normal(size=(200,8));base=rng.normal(size=(200,2))*1.4
    y=np.clip(rng.normal(size=(200,2)),-2,2);p=np.clip(base,-2,2);w=np.abs(y);active=np.abs(base)<2
    stats=np.stack((w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)))
    features=np.stack((f.T@(w*active),f.T@(w*active*y),f.T@(w*active*p)))
    gradient=stats_and_gradient(stats,features)
    for target in range(2):
        _,negative_gradient=oa.objective_value_gradient(np.zeros(8),f,base[:,target],y[:,target],0.)
        np.testing.assert_allclose(gradient[:,target],-negative_gradient,atol=1e-12)


def test_exact_moment_training_matches_dense_ridge_on_synthetic_sequences(tmp_path):
    from competition_engineering.full_moment_objective import source as exact_source
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(71)
    combo={'mean':np.zeros(112),'scale':np.ones(112),'base_scale':np.ones(2),'base_bias':np.zeros(2),
           'root_coef':np.zeros((115,2)),'root_strengths':np.ones(2),'magnitude_coef':np.zeros((115,2)),
           'magnitude_strengths':np.ones(2),'gated_coef':np.zeros((2,115,2)),'gated_strengths':np.ones(2)}
    np.savez(tmp_path/'combo.npz',**combo)
    files=[];fit_f=[];fit_y=[];fit_base=[]
    for group in range(16):
        x=rng.normal(size=(128,112)).astype(np.float32);base=x[:,:2]*.3
        y=base+x[:,3,None]*.1+rng.normal(size=(128,2)).astype(np.float32)*.03
        path=tmp_path/f'{group}.npz';np.savez(path,x=x,p=base,y=y,need=np.ones(128,bool));files.append(str(path))
        if group%5:
            fit_f.append(np.column_stack((np.ones(128),base,x)).astype(np.float64))
            fit_y.append(np.clip(y[:,0],-2,2).astype(np.float64));fit_base.append(base[:,0].astype(np.float64))
    output=tmp_path/'artifacts';output.mkdir()
    namespace={'__file__':str(tmp_path/'train.py')};exec(exact_source(),namespace)
    with threadpool_limits(limits=1):
        namespace['train'](files,str(output))
        f=np.concatenate(fit_f);y=np.concatenate(fit_y);base=np.concatenate(fit_base);w=np.abs(y)
        expected=np.linalg.solve(f.T@(w[:,None]*f)+np.eye(115)*len(y)*.1,f.T@(w*(y-base)))
    with np.load(output/'models.npz') as model:
        np.testing.assert_allclose(model['raw_residual_0.1'][:115],expected,atol=1e-7)
        assert np.count_nonzero(model['raw_residual_0.1'][115:])==0
    report=json.loads((output/'training.json').read_text())
    assert report['fit_rows']==1536 and report['selection_rows']==512


@pytest.mark.parametrize('module',['competition_engineering.full_moment_objective',
    'competition_engineering.group_objective_alignment','tools.check_objective_candidates'])
def test_supervisor_thread_limits_precede_numeric_imports(module):
    import os
    import subprocess
    import sys
    environment={**os.environ,'OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OMP_NUM_THREADS':'8'}
    code=f'import {module}; import json; from threadpoolctl import threadpool_info; print(json.dumps(threadpool_info()))'
    completed=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True,env=environment,timeout=30)
    libraries=json.loads(completed.stdout)
    assert libraries and all(library['num_threads']==1 for library in libraries)
