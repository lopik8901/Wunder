"""Synthetic causality checks; no competition evaluation data."""
import numpy as np
import pytest

from competition_engineering.autonomous_campaign import design, StreamingCorrection


@pytest.mark.parametrize('mode', ['linear', 'bilinear', 'normalize'])
def test_stateful_features_match_batch_and_reset_without_future(mode):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(150, 112)).astype(np.float32)
    p = rng.normal(size=(150, 2)).astype(np.float32)
    p[:7] = np.nan
    mean = np.zeros(112, np.float32)
    scale = np.ones(112, np.float32)
    f = design(x, p, mean, scale, mode)
    coef = rng.normal(size=f.shape[1]).astype(np.float32) * .001
    model = StreamingCorrection(mean, scale, mode, coef, .5, False)
    def replay(seq, values):
        return np.array([model.predict(seq, i, row, p[i]) for i, row in enumerate(values)])
    a = replay(1, x)
    expected = p.copy()
    expected[:, 0] += .5 * (f @ coef)
    np.testing.assert_allclose(a, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(a, replay(2, x))
    np.testing.assert_array_equal(a, replay(2, x))
    changed = x.copy()
    changed[90:] += 100
    np.testing.assert_array_equal(a[:90], replay(3, changed)[:90])
    np.testing.assert_array_equal(f[:90], design(changed, p, mean, scale, mode)[:90])


def test_late_expert_updates_memory_before_activation():
    mean = np.zeros(112, np.float32)
    scale = np.ones(112, np.float32)
    coef = np.ones(339, np.float32) * .001
    model = StreamingCorrection(mean, scale, 'linear', coef, .5, True)
    x = np.ones(112, np.float32)
    base = np.zeros(2, np.float32)
    np.testing.assert_array_equal(model.predict(1, 0, x, base), base)
    assert model.slow[0] > 0
    assert model.predict(1, 13333, x, base)[0] > 0


def test_generated_candidate_sources_pass_existing_restrictions():
    from competition_engineering.isolated_campaign import train_source, callback_source
    for mode in ['linear', 'bilinear', 'normalize']:
        for late in [False, True]:
            assert 'def train(' in train_source(mode, late)
            source = callback_source(mode, late)
            assert 'class PredictionModel:' in source
            assert "z['y']" not in source
            assert "z['mask']" not in source


def test_resume_rejects_artifact_tampering(tmp_path):
    import json
    from competition_engineering.isolated_campaign import verify_work_identity, digest
    for directory in ['artifacts', 'deploy']:
        (tmp_path / directory).mkdir()
    (tmp_path / 'train.py').write_text('original')
    (tmp_path / 'artifacts/model.npz').write_bytes(b'weights')
    (tmp_path / 'deploy/solution.py').write_text('callback')
    (tmp_path / 'candidate.zip').write_bytes(b'archive')
    identity = {
        'source_sha256': {'train.py': digest(tmp_path / 'train.py')},
        'artifact_sha256': {'model.npz': digest(tmp_path / 'artifacts/model.npz')},
        'deployment_sha256': {'solution.py': digest(tmp_path / 'deploy/solution.py')},
        'package_sha256': digest(tmp_path / 'candidate.zip'),
    }
    (tmp_path / 'identity.json').write_text(json.dumps(identity))
    verify_work_identity(tmp_path)
    (tmp_path / 'artifacts/model.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity changed'):
        verify_work_identity(tmp_path)


def test_supervisor_bounds_blas_threads_before_import():
    import json
    import os
    import subprocess
    import sys
    variables=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']
    code="import competition_engineering.isolated_campaign; import numpy as np,os,psutil,json; np.ones((32,32)) @ np.ones((32,32)); print(json.dumps({'threads':psutil.Process().num_threads(),'configured':[os.environ[k] for k in " + repr(variables) + "]}))"
    completed=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True,
                             timeout=30,env={**os.environ,**{k:'64' for k in variables}})
    result=json.loads(completed.stdout)
    assert result['configured']==['1']*4
    assert result['threads'] < 16
