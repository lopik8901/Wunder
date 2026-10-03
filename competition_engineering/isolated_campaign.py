"""Run campaign candidates through existing production bubblewrap workers.

Supervisor scoring keeps the same fixed search cache, pooled scorer and atlas.
Training sees only the fixed training cache and frozen model files. Inference
receives one row at a time, never labels, masks, caches, or research documents.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import time
from pathlib import Path

# Bound supervisor thread creation before importing numerical runtimes. Worker
# rlimits count same-user threads too; concurrent unbounded BLAS supervisors can
# prevent bubblewrap from creating namespaces even when inference is idle.
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering import autonomous_campaign as lab
from competition_engineering import campaign_onnx
from competition_engineering.generated_runner import (
    GeneratedSandbox, WORKER, _launch, _check_artifacts, _package,
    validate_callback, replay, digest,
)
from competition_engineering.manual_search_core import (
    ROOT, COMBO, COMBO_SHA, SEARCH, TRAIN_1024, load_combo, predict_combo, assess,
)
from competition_engineering.pipeline import cached, write_json
from connectome.mlevolve_generated import validate_source

DEPLOY = ROOT / 'competition_engineering/deployment/manual_targetwise_combo_v1'


def verify_work_identity(work):
    """Fail closed if a resumed fit or scored callback has changed."""
    identity = json.loads((work / 'identity.json').read_text())
    for key, directory in [('source_sha256', work), ('artifact_sha256', work / 'artifacts'),
                           ('deployment_sha256', work / 'deploy')]:
        for name, expected in identity[key].items():
            if digest(directory / name) != expected:
                raise ValueError('Frozen isolated work identity changed: ' + key + '/' + name)
    if digest(work / 'candidate.zip') != identity['package_sha256']:
        raise ValueError('Frozen package identity changed')


def train_source(mode, late_only):
    combo_code = inspect.getsource(predict_combo).replace(
        'f = features({**z, "p": base}, model["mean"], model["scale"])',
        'f = np.column_stack((np.ones(len(base), np.float32), np.clip((z["x"]-model["mean"])/model["scale"], -8, 8), base)).astype(np.float32)')
    source = '''import numpy as np
from pathlib import Path
from scipy.signal import lfilter
'''
    source += inspect.getsource(lab.design) + '\n' + combo_code + '\n'
    source += f'''
def train(train_files, output_dir):
    with np.load(Path(__file__).with_name('combo.npz')) as q:
        combo = {{k: q[k].copy() for k in q.files}}
    gram = rhs = None
    rows = 0
    for count, path in enumerate(train_files):
        with np.load(path) as q:
            z = {{k: q[k] for k in q.files}}
        selected = z['need'] & (z['step'] >= 13333) if {late_only!r} else z['need']
        idx = np.flatnonzero(selected)[::40]
        base = predict_combo(z, combo)
        f = design(z['x'], base, combo['mean'], combo['scale'], {mode!r}, idx).astype(np.float64)
        y = np.clip(z['y'][idx, 0], -2, 2)
        w = np.abs(y)
        if gram is None:
            gram = np.zeros((f.shape[1], f.shape[1]), np.float64)
            rhs = np.zeros(f.shape[1], np.float64)
        gram += f.T @ (f * w[:, None])
        rhs += f.T @ ((y - base[idx, 0]) * w)
        rows += len(idx)
        if (count+1) % 256 == 0:
            print('TRAIN_PROGRESS=' + str(count+1), flush=True)
    coef = np.linalg.solve(gram + np.eye(len(rhs)) * rows * .1, rhs).astype(np.float32)
    np.savez(output_dir + '/model.npz', coef=coef, mean=combo['mean'], scale=combo['scale'])
    return {{'training_rows': rows, 'sequences': count+1}}
'''
    validate_source(source, training=True)
    return source


def callback_source(mode, late_only, strength=1.):
    # Copy the frozen baseline implementation into one self-contained module,
    # omitting host environment changes and the standalone scoring CLI.
    gru = (DEPLOY / 'gru.py').read_text()
    gru = gru[gru.index('class PredictionModel:'):gru.index('\n\nif __name__')]
    gru = gru.replace('class PredictionModel:', 'class GRU:')
    combo = (DEPLOY / 'solution.py').read_text()
    combo = combo[combo.index('class PredictionModel:'):]
    combo = combo.replace('class PredictionModel:', 'class Combo:')
    stream = inspect.getsource(lab.StreamingCorrection)
    source = 'from pathlib import Path\nimport numpy as np\nimport onnxruntime as ort\n\n'
    source += gru + '\n\n' + combo + '\n\n' + stream
    source += f'''
class PredictionModel:
    def __init__(self):
        self.combo = Combo()
        with np.load(Path(__file__).with_name('model.npz')) as q:
            self.correction = StreamingCorrection(q['mean'], q['scale'], {mode!r}, q['coef'], {strength!r}, {late_only!r})
    def predict(self, point):
        if point.step_in_seq == 0:
            self.combo.gru.current_seq_ix = None
        base = self.combo.predict(point)
        if base is None:
            self.correction.predict(point.seq_ix, point.step_in_seq, point.state, np.zeros(2, np.float32))
            return None
        return self.correction.predict(point.seq_ix, point.step_in_seq, point.state, base)
'''
    validate_source(source, training=False)
    return source


def isolated_fit(work, mode, late_only, deadline, reuse_fit=None):
    work.mkdir(parents=True)
    (work / 'train.py').write_text(train_source(mode, late_only), encoding='utf-8')
    (work / 'callback.py').write_text(campaign_onnx.callback_source(mode), encoding='utf-8')
    shutil.copyfile(COMBO, work / 'combo.npz')
    original = {name: digest(work / name) for name in ['train.py', 'callback.py', 'combo.npz']}
    sandbox = GeneratedSandbox()
    sandbox.preflight(WORKER, work)
    if reuse_fit is None:
        process, status, watcher = _launch(sandbox, work, 'train', deadline, TRAIN_1024)
        try:
            stdout, stderr = process.communicate(timeout=max(1, deadline-time.monotonic()))
        finally:
            if process.poll() is None:
                process.kill()
            watcher.join(timeout=1)
        (work / 'train_stdout.log').write_bytes(stdout)
        (work / 'train_stderr.log').write_bytes(stderr)
        if process.returncode != 0 or status:
            raise RuntimeError(status.get('reason') or stderr.decode(errors='replace')[-1500:])
        train_result = None
        for line in stdout.decode().splitlines():
            if line.startswith('TRAIN_RESULT='):
                train_result = json.loads(line.removeprefix('TRAIN_RESULT='))
    else:
        source_work = work.parent / reuse_fit
        verify_work_identity(source_work)
        if digest(source_work / 'train.py') != original['train.py']:
            raise ValueError('Fit reuse requires identical scientific training source')
        (work / 'artifacts').mkdir()
        shutil.copyfile(source_work / 'artifacts/model.npz', work / 'artifacts/model.npz')
        train_result = {'reused_fit': reuse_fit, 'model_sha256': digest(work / 'artifacts/model.npz')}
    campaign_onnx.export(work / 'artifacts/model.npz', work / 'artifacts/fused.onnx', mode, late_only)
    artifacts = _check_artifacts(work, original)
    archive, deploy_hashes = _package(work)
    write_json(work / 'identity.json', {'source_sha256': original, 'artifact_sha256': artifacts,
               'deployment_sha256': deploy_hashes, 'package_sha256': digest(archive)})
    return sandbox, train_result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', required=True)
    p.add_argument('--mode', choices=['linear', 'bilinear', 'normalize'], required=True)
    p.add_argument('--late-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--reuse-fit')
    args = p.parse_args()
    lab.deadline_check()
    campaign = lab.initialize()
    seconds_left = campaign['started_unix'] + campaign['overall_time_limit_seconds'] - time.time()
    deadline = time.monotonic() + min(1200, seconds_left)
    out = lab.CAMPAIGN / args.id
    work = ROOT / 'competition_engineering/mlevolve_runs/autonomous_20261001' / args.id
    if (out / 'report.json').exists():
        raise ValueError('Completed experiment is immutable')
    if out.exists() and not args.resume:
        raise ValueError('Use --resume for incomplete checkpoint')
    out.mkdir(exist_ok=True)
    started = time.perf_counter()
    if not (out / 'config.json').exists():
        library_hash = lab.retrieve_for(out, args.mode)
        config = {'id': args.id, 'mode': args.mode, 'late_only': args.late_only,
                  'tier': 1024, 'stride': 40, 'penalty': .1, 'strengths': [.25, .5, 1.],
                  'hypothesis': {'linear': 'Accumulated multiscale input state explains persistent t0 residuals; matched additive control.',
                                 'bilinear': 'Current-by-causal-history interactions explain t0 residuals missed by additive memory.',
                                 'normalize': 'Causal local location and variance normalization corrects sequence-dependent t0 feature drift.'}[args.mode],
                  'support': 'Gain over frozen combo with positive paired lower bound and late-t0 diagnostic improvement.',
                  'weaken': 'Nonpositive gain; for bilinear, no gain over additive control.',
                  'library_sha256': library_hash, 'reference_sha256': COMBO_SHA,
                  'runner_sha256': digest(Path(__file__)), 'core_sha256': digest(Path(lab.__file__)),
                  'exporter_sha256': digest(Path(campaign_onnx.__file__)), 'reuse_fit': args.reuse_fit,
                  'isolation': 'Existing production GeneratedSandbox; training-only mount; rowwise label-free inference',
                  'rule_sha256': digest(lab.CAMPAIGN / 'campaign.json')}
        write_json(out / 'config.json', config)
        shutil.copyfile(__file__, out / 'runner_snapshot.py')
        shutil.copyfile(lab.__file__, out / 'core_snapshot.py')
        shutil.copyfile(campaign_onnx.__file__, out / 'exporter_snapshot.py')
        lab.event('experiment_planned', **config)
    else:
        config = json.loads((out / 'config.json').read_text())
        if (config['mode'], config['late_only']) != (args.mode, args.late_only):
            raise ValueError('Resume configuration differs')
        lab.event('experiment_resumed', id=args.id)
    try:
        if not (work / 'identity.json').exists():
            if work.exists():
                raise RuntimeError('Incomplete isolated fit preserved; create explicitly linked repair attempt')
            sandbox, train_result = isolated_fit(work, args.mode, args.late_only, deadline, args.reuse_fit)
            write_json(out / 'training.json', train_result)
        else:
            sandbox = GeneratedSandbox()
            sandbox.preflight(WORKER, work)
        verify_work_identity(work)
        if not (out / 'validation.json').exists():
            validation = validate_callback(sandbox, work, TRAIN_1024, deadline)
            write_json(out / 'validation.json', validation)
        else:
            validation = json.loads((out / 'validation.json').read_text())
        print(json.dumps({'phase': 'isolated_fit_validated', 'validation': validation}), flush=True)
        pred_dir = out / 'search_predictions'
        pred_dir.mkdir(exist_ok=True)
        combo = load_combo()
        with threadpool_limits(limits=1):
            for seq_i, (group, z) in enumerate(cached(SEARCH)):
                dest = pred_dir / f'{group:05d}.npz'
                if not dest.exists():
                    values, timing = replay(sandbox, work, [z], deadline)
                    prediction = values[0]
                    base = predict_combo(z, combo)
                    with np.load(work / 'artifacts/model.npz') as q:
                        coef = q['coef']
                    correction = lab.design(z['x'], base, combo['mean'], combo['scale'], args.mode) @ coef
                    expected = base.copy()
                    active = z['step'] >= 13333 if args.late_only else np.ones(len(base), bool)
                    expected[active, 0] += correction[active]
                    np.testing.assert_allclose(prediction[z['need']], expected[z['need']], atol=3e-5, rtol=3e-5)
                    np.savez(dest, prediction=prediction, seq=z['seq'])
                if (seq_i+1) % 8 == 0:
                    print(json.dumps({'phase': 'isolated_search_replay', 'sequences': seq_i+1, 'seconds': time.perf_counter()-started}), flush=True)
            predictions = {}
            for path in sorted(pred_dir.glob('*.npz')):
                with np.load(path) as q:
                    predictions[int(q['seq'])] = q['prediction'].copy()
            def predictor(z, base, strength):
                # Only supervisor combines immutable causal predictions. Candidate
                # never receives this dictionary or target/mask fields.
                full = predictions[int(z['seq'])]
                value = base.copy()
                value[:, 0] += strength * (full[:, 0] - base[:, 0])
                return value
            grid = {s: assess(lambda z, b, s=s: predictor(z, b, s), diagnostics=False) for s in config['strengths']}
            strength = max(grid, key=lambda s: grid[s]['delta_combined'])
            result = assess(lambda z, b: predictor(z, b, strength))
            ci99 = lab.paired99(lambda z, b: predictor(z, b, strength))
        rule = campaign['rule']
        verify_work_identity(work)
        # Keep original >90us full callback gate unchanged in validate_callback.
        eligible = (result['delta_combined'] >= rule['minimum_combined_delta'] and ci99[0] > 0
                    and min(result['delta_per_target']) >= rule['minimum_target_delta']
                    and validation['callback_us_per_row'] <= 51.92753780833335 + rule['cpu_incremental_budget_us_per_row'])
        selected = out / 'selected_deploy'
        if not selected.exists():
            shutil.copytree(work / 'deploy', selected)
            campaign_onnx.export(work / 'artifacts/model.npz', selected / 'fused.onnx', args.mode, args.late_only, strength)
        report = {'config': config, 'chosen_strength': strength, 'strength_search': grid,
                  'search_result': result, 'paired_99ci_combined': ci99, 'validation': validation,
                  'status': 'AWAITING PROMOTION' if eligible else 'exploratory_search_only',
                  'runtime_seconds': time.perf_counter()-started, 'artifact_sha256': digest(work / 'artifacts/model.npz'),
                  'selected_deployment_sha256': {q.name: digest(q) for q in sorted(selected.iterdir())},
                  'config_sha256': digest(out / 'config.json'),
                  'identity': json.loads((work / 'identity.json').read_text()),
                  'interpretation': 'Fixed search rule met; further deployment verification required before external evaluation.' if eligible else 'Fixed search rule not met; frozen combo retained.'}
        write_json(out / 'report.json', report)
        lab.event('experiment_completed', id=args.id, hypothesis=config['hypothesis'],
                  score=result['candidate'], delta=result['delta_combined'],
                  paired_95ci=result['paired_95ci_combined'], paired_99ci=ci99,
                  validation=validation, runtime_seconds=report['runtime_seconds'],
                  artifact_sha256=report['artifact_sha256'], report_sha256=digest(out / 'report.json'),
                  interpretation=report['interpretation'])
        print(json.dumps({'id': args.id, 'score': result['candidate'], 'delta': result['delta_combined'],
                          'paired95': result['paired_95ci_combined'], 'paired99': ci99, 'status': report['status']}), flush=True)
    except Exception as exc:
        lab.event('experiment_interrupted', id=args.id, failure_type='implementation_or_infrastructure',
                  error=type(exc).__name__+': '+str(exc)[-1600:], seconds=time.perf_counter()-started)
        raise


if __name__ == '__main__':
    main()
