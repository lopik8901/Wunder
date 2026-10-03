"""Checkpointed, strictly search-only residual research laboratory.

The only data readers are the three fixed caches in manual_search_core. The
candidate feature interface contains inputs and frozen predictions, never labels
or scoring masks. Existing frozen runs and the shared ledger are not rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits

from competition_engineering.manual_search_core import (
    COMBO_SHA, ROOT, SEARCH, SEARCH_ROOT_WP, TRAIN_1024, TRAIN_4096,
    assess, load_combo, predict_combo,
)
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import sufficient, from_stats
from connectome.research_foundation import build_search_query, EVIDENCE
from connectome.research_retrieval import load_library, retrieve
from wnn_connectome_starterpack.utils import GlobalAccumulator

CAMPAIGN = ROOT / 'competition_engineering/runs/autonomous_20261001'
LIBRARY = ROOT / 'competition_engineering/research_cards/v3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(kind, **payload):
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    path = CAMPAIGN / 'ledger.jsonl'
    previous = sha(path) if path.exists() else None
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'kind': kind,
              'evidence': EVIDENCE, 'previous_ledger_sha256': previous, **payload}
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
        f.flush()


def initialize():
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    path = CAMPAIGN / 'campaign.json'
    if path.exists():
        return json.loads(path.read_text())
    config = {
        'started_unix': time.time(), 'overall_time_limit_seconds': 7200,
        'source_time_limit': 'competition_engineering/mlevolve_search_config.yaml agent.time_limit',
        'starting_incumbent': SEARCH_ROOT_WP, 'starting_incumbent_sha256': COMBO_SHA,
        'rule': {'minimum_combined_delta': .0002, 'paired_99ci_lower_must_exceed': 0,
                 'minimum_target_delta': -.0002, 'required': ['causal_prefix', 'reset',
                 'determinism', 'stream_batch_parity', 'label_mask_independence',
                 'finite_outputs', 'cpu_measurement'],
                 'cpu_incremental_budget_us_per_row': 25,
                 'note': 'Search evidence only; bootstrap intervals are descriptive, not corrected for adaptive reuse.'},
        'boundary': 'Fixed training and 64-sequence search caches only. No external evaluation operation.',
        'baseline_history': 'competition_engineering/reports/manual_research_search_only_20261001.json',
        'deduplication': 'Mechanism, state representation, training tier, target and objective checked before launch.',
    }
    write_json(path, config)
    event('campaign_started', config=config, config_sha256=sha(path))
    return config


def deadline_check():
    c = initialize()
    if time.time() >= c['started_unix'] + c['overall_time_limit_seconds']:
        raise TimeoutError('Configured overall campaign time limit reached')


def retrieve_for(out, mode):
    cards, library_hash = load_library(LIBRARY)
    atlas = json.loads((ROOT / 'competition_engineering/reports/manual_combo_search_atlas_20261001.json').read_text())
    observation = {'target': 't0', 'position': 'late', 'signal': 'residual_persistence',
                   'lag_rows': 100, 'feature_group': None}
    query = build_search_query(atlas, [])
    terms = (('objective','gradient','correlation','clipping','residual') if mode == 'pearson_tangent' else
             ('temporal', 'nonlinear', 'memory', 'normalization') if mode == 'normalize' else
             ('temporal', 'nonlinear', 'polynomial', 'memory'))
    result = retrieve(cards, observation, query, max_cards=5, max_utf8_bytes=5500,
                      hypothesis_terms=terms)
    write_json(out / 'retrieval.json', result)
    return library_hash


def design(x, base, mean, scale, mode, indices=None):
    """Zero-initialized causal moments; no sequence length dependent features."""
    x = np.clip((np.asarray(x, np.float32) - mean) / scale, -8, 8).astype(np.float64)
    base = np.clip(np.nan_to_num(base, nan=0), -2, 2)
    fast = lfilter([.01], [1., -.99], x, axis=0)
    slow = lfilter([.001], [1., -.999], x, axis=0)
    idx = slice(None) if indices is None else indices
    parts = [np.ones((len(x[idx]), 1)), base[idx], x[idx], fast[idx], slow[idx]]
    if mode == 'bilinear':
        parts += [x[idx] * fast[idx], x[idx] * slow[idx]]
    elif mode == 'normalize':
        variance = lfilter([.001], [1., -.999], x * x, axis=0) - slow * slow
        parts += [np.clip((x[idx] - slow[idx]) / np.sqrt(np.maximum(variance[idx], .1)), -8, 8)]
    elif mode != 'linear':
        raise ValueError('unknown mechanism')
    return np.column_stack(parts).astype(np.float32)


class StreamingCorrection:
    def __init__(self, mean, scale, mode, coef, strength, late_only):
        self.mean, self.scale, self.mode = mean, scale, mode
        self.coef, self.strength, self.late_only = coef, strength, late_only
        self.seq = None

    def predict(self, seq, step, x, base):
        if seq != self.seq or step == 0:
            self.fast = np.zeros(len(x), np.float64)
            self.slow = np.zeros(len(x), np.float64)
            self.second = np.zeros(len(x), np.float64)
            self.seq = seq
        current = np.clip((np.asarray(x, np.float32) - self.mean) / self.scale, -8, 8).astype(np.float64)
        self.fast = .01 * current + .99 * self.fast
        self.slow = .001 * current + .999 * self.slow
        p = np.clip(np.nan_to_num(base, nan=0), -2, 2)
        parts = [np.ones(1), p, current, self.fast, self.slow]
        if self.mode == 'bilinear':
            parts += [current * self.fast, current * self.slow]
        elif self.mode == 'normalize':
            self.second = .001 * current * current + .999 * self.second
            parts += [np.clip((current - self.slow) / np.sqrt(np.maximum(self.second - self.slow ** 2, .1)), -8, 8)]
        f = np.concatenate(parts).astype(np.float32)
        value = base.copy()
        if not self.late_only or step >= 13333:
            value[0] += np.float32(self.strength * (f @ self.coef))
        return value


def validate_cpu(combo, mode, coef, strength, late_only):
    _, z = next(cached(SEARCH))
    base = predict_combo(z, combo)
    def make():
        return StreamingCorrection(combo['mean'], combo['scale'], mode, coef, strength, late_only)
    def replay(model, x, seq):
        return np.array([model.predict(seq, i, row, base[i]) for i, row in enumerate(x)])
    model = make()
    started = time.perf_counter()
    actual = replay(model, z['x'], 1)
    duration = time.perf_counter() - started
    expected = base.copy()
    correction = design(z['x'], base, combo['mean'], combo['scale'], mode) @ coef
    active = z['step'] >= 13333 if late_only else np.ones(len(base), bool)
    expected[active, 0] += (strength * correction[active]).astype(np.float32)
    np.testing.assert_allclose(actual[z['need']], expected[z['need']], atol=3e-6, rtol=3e-6)
    again = replay(model, z['x'], 2)
    np.testing.assert_array_equal(actual[z['need']], again[z['need']])
    fresh = replay(make(), z['x'], 2)
    np.testing.assert_array_equal(actual[z['need']], fresh[z['need']])
    changed = z['x'].copy()
    changed[15000:] += .7
    future = replay(make(), changed, 1)
    np.testing.assert_array_equal(actual[:15000][z['need'][:15000]], future[:15000][z['need'][:15000]])
    # Functional candidate API accepts only inputs and frozen outputs. No target/mask argument exists.
    return {'causal_prefix': True, 'reset': True, 'determinism': True, 'stream_batch_parity': True,
            'label_mask_independence': True, 'finite_outputs': bool(np.isfinite(actual[z['need']]).all()),
            'max_abs_parity': float(np.max(np.abs(actual[z['need']] - expected[z['need']]))),
            'incremental_cpu_us_per_row': duration / len(base) * 1e6,
            'cpu_scope': 'Python streaming residual only; cached incumbent output, one BLAS thread, no core pinning',
            'platform': platform.platform(), 'rows': len(base)}


def paired99(predictor):
    combo = load_combo()
    moments = []
    for _, z in cached(SEARCH):
        base = predict_combo(z, combo)
        pred = predictor(z, base)
        m = z['mask']
        moments.append([sufficient(z['y'][m], base[m]), sufficient(z['y'][m], pred[m])])
    moments = np.array(moments)
    rng = np.random.default_rng(20261001)
    deltas = []
    for _ in range(10000):
        sums = moments[rng.integers(len(moments), size=len(moments))].sum(0)
        deltas.append(from_stats(sums[1]) - from_stats(sums[0]))
    return np.quantile(np.asarray(deltas).mean(1), [.005, .995]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', required=True)
    p.add_argument('--mode', choices=['linear', 'bilinear', 'normalize'], required=True)
    p.add_argument('--tier', type=int, choices=[1024, 4096], default=1024)
    p.add_argument('--late-only', action='store_true')
    args = p.parse_args()
    deadline_check()
    out = CAMPAIGN / args.id
    if out.exists():
        raise ValueError('Experiment already exists; preserve immutable evidence')
    out.mkdir()
    library_hash = retrieve_for(out, args.mode)
    hypotheses = {
        'linear': 'Accumulated multiscale input state contributes beyond the frozen combo; control for nonlinear interaction test.',
        'bilinear': 'Featurewise current-by-history interactions explain persistent t0 errors beyond additive memory.',
        'normalize': 'Sequence-dependent input location and variance drift, estimated causally, explain residual t0 error.',
    }
    config = {'id': args.id, 'mode': args.mode, 'tier': args.tier, 'late_only': args.late_only,
              'hypothesis': hypotheses[args.mode], 'support': 'Positive combined delta with paired lower bound above zero; late t0 improves without widespread sequence regressions.',
              'weaken': 'No improvement or gains disappear versus additive-memory control.',
              'stride': 40, 'ridge_penalty': .1, 'strengths': [.25, .5, 1.],
              'state_alphas': [.01, .001], 'reference_sha256': COMBO_SHA,
              'library_sha256': library_hash, 'source_sha256': sha(__file__),
              'decision_rule_sha256': sha(CAMPAIGN / 'campaign.json')}
    write_json(out / 'config.json', config)
    shutil.copyfile(__file__, out / 'runner_snapshot.py')
    event('experiment_planned', **config)
    started = time.perf_counter()
    combo = load_combo()
    gram = rhs = None
    rows = 0
    path = TRAIN_1024 if args.tier == 1024 else TRAIN_4096
    with threadpool_limits(limits=1):
        for seq_i, (_, z) in enumerate(cached(path)):
            deadline_check()
            selected = z['need'] & (z['step'] >= 13333) if args.late_only else z['need']
            idx = np.flatnonzero(selected)[::40]
            base = predict_combo(z, combo)
            f = design(z['x'], base, combo['mean'], combo['scale'], args.mode, idx).astype(np.float64)
            y = np.clip(z['y'][idx, 0], -2, 2)
            w = np.abs(y)
            if gram is None:
                gram = np.zeros((f.shape[1], f.shape[1]), np.float64)
                rhs = np.zeros(f.shape[1], np.float64)
            gram += f.T @ (f * w[:, None])
            rhs += f.T @ ((y - base[idx, 0]) * w)
            rows += len(idx)
            if (seq_i + 1) % 256 == 0:
                print(json.dumps({'phase': 'fit', 'sequences': seq_i + 1, 'seconds': time.perf_counter()-started}), flush=True)
                np.savez(out / 'fit_checkpoint.npz', gram=gram, rhs=rhs, rows=rows, completed=seq_i+1)
        coef = np.linalg.solve(gram + np.eye(len(rhs)) * rows * .1, rhs).astype(np.float32)
        np.savez(out / 'model.npz', coef=coef, mean=combo['mean'], scale=combo['scale'])
        def predictor(z, base, strength):
            output = base.copy()
            correction = design(z['x'], base, combo['mean'], combo['scale'], args.mode) @ coef
            active = z['step'] >= 13333 if args.late_only else np.ones(len(base), bool)
            output[active, 0] += (strength * correction[active]).astype(np.float32)
            return output
        grid = {s: assess(lambda z, b, s=s: predictor(z, b, s), diagnostics=False) for s in config['strengths']}
        strength = max(grid, key=lambda s: grid[s]['delta_combined'])
        result = assess(lambda z, b: predictor(z, b, strength))
        ci99 = paired99(lambda z, b: predictor(z, b, strength))
        gates = validate_cpu(combo, args.mode, coef, strength, args.late_only)
    rule = initialize()['rule']
    eligible = (result['delta_combined'] >= rule['minimum_combined_delta'] and ci99[0] > 0
                and min(result['delta_per_target']) >= rule['minimum_target_delta']
                and gates['finite_outputs'] and gates['incremental_cpu_us_per_row'] <= rule['cpu_incremental_budget_us_per_row'])
    report = {'config': config, 'training_rows': rows, 'training_sequences': seq_i+1,
              'chosen_strength': strength, 'strength_search': {str(s): r for s, r in grid.items()},
              'search_result': result, 'paired_99ci_combined': ci99, 'validation': gates,
              'runtime_seconds': time.perf_counter()-started, 'artifact_sha256': sha(out / 'model.npz'),
              'config_sha256': sha(out / 'config.json'), 'retrieval_sha256': sha(out / 'retrieval.json'),
              'status': 'AWAITING PROMOTION' if eligible else 'exploratory_search_only',
              'interpretation': 'Supported under fixed search rule' if eligible else 'Fixed search-incumbent rule not met; retain incumbent.'}
    write_json(out / 'report.json', report)
    event('experiment_completed', id=args.id, report_sha256=sha(out / 'report.json'),
          report_path=str(out.relative_to(ROOT) / 'report.json'),
          hypothesis=config['hypothesis'], score=result['candidate'], delta=result['delta_combined'],
          paired_95ci=result['paired_95ci_combined'], paired_99ci=ci99,
          validation=gates, runtime_seconds=report['runtime_seconds'],
          artifact_sha256=report['artifact_sha256'], interpretation=report['interpretation'])
    print(json.dumps({k: report[k] for k in ['chosen_strength', 'paired_99ci_combined', 'validation', 'runtime_seconds', 'status']}), flush=True)
    print(json.dumps({k: result[k] for k in ['candidate', 'delta_combined', 'paired_95ci_combined']}), flush=True)


if __name__ == '__main__':
    main()
