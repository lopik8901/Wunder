"""Search-only continuation; frozen inputs and prospective diagnostic decisions."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.special import expit
from competition_engineering.manual_search_core import ROOT, SEARCH, COMBO, COMBO_SHA, load_combo, predict_combo
from competition_engineering.pipeline import cached, write_json
from competition_engineering.gradient_uncertainty import gradient, cosine, bootstrap_counts, resample
from competition_engineering.nonlinear_selection_diagnostic import weighted_primitives
from competition_engineering.nonlinear_weighted_readout import tree_predict
from competition_engineering.prospective_selection_diagnostic import shortlist
from competition_engineering.search_population_audit import pooled_correlations
from tools.diagnose_causal_scoring_population import classifier_features

OUT = ROOT / 'competition_engineering/runs/root_cause_20261002'
PREVIOUS = ROOT / 'competition_engineering/runs/reassessment_20261002'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def event(kind, **payload):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'ledger.jsonl'
    entry = dict(utc=datetime.now(timezone.utc).isoformat(), kind=kind,
                 previous_ledger_sha256=sha(path) if path.exists() else None,
                 boundary='search-only', **payload)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n')

def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'research_state_map.json').exists():
        return
    assert sha(COMBO) == COMBO_SHA
    knowledge = json.loads((ROOT / 'competition_engineering/search_knowledge.json').read_text())
    write_json(OUT / 'starting_knowledge.json', knowledge)
    state = json.loads((PREVIOUS / 'research_state_map.json').read_text())
    state['continuation_updates'] = {
        'supported': [entry['lesson'] for entry in knowledge['established']],
        'weakened': knowledge['weakened'],
        'remaining_explanations': [
            'Clipped scoring-probability weights distort the selected population.',
            'Observable scoring propensity does not capture conditional correction efficacy.',
            'Python tree inference overhead prevented a fair deployable test of compact interactions.',
            'Adaptive reuse and sequence variance limit credible detection of small improvements.'],
        'decisions': ['First compare frozen capped and uncapped weights without fitting a correction.',
                      'Keep objective grids and old temporal mechanisms closed.',
                      'A matched tree execution repair may justify one training-selected compact interaction fit.'],
        'previous_final_report_sha256': sha(PREVIOUS / 'FINAL_REPORT.json'),
        'scope': 'Prior map plus completed six-diagnostic reassessment and weighted-readout evidence; no protected feedback.'}
    write_json(OUT / 'research_state_map.json', state)
    event('synthesis_completed', state_sha256=sha(OUT / 'research_state_map.json'),
          incumbent_sha256=COMBO_SHA, starting_knowledge_sha256=sha(OUT / 'starting_knowledge.json'))

def probability_from_trees(features, trees, fold):
    value = np.full(len(features), float(trees[f'fold{fold}_baseline'].ravel()[0]))
    for index in range(64):
        value += tree_predict(features, trees[f'fold{fold}_tree{index}'])
    return expit(value)

def bootstrap_moments(moments, counts):
    return (counts @ moments.reshape(len(moments), -1)).reshape(len(counts), *moments.shape[1:])

def diagnose_weights(matched=False):
    initialize()
    suffix = '_matched' if matched else ''
    destination = OUT / ('weight_diagnosis'+suffix+'.json')
    if destination.exists():
        print(destination.read_text()); return
    protocol = {
        'observation': 'Existing nonlinear weights clip probability/prior to [0.2,5], altering about one third of training rows.',
        'hypothesis': 'This truncation explains part of the discrepancy between predicted and official correction efficacy.',
        'comparison': ['official', 'capped', 'probability', 'uncapped_ratio'] if matched else ['official', 'capped', 'probability'],
        'classifier': 'Previously frozen four-fold nonlinear scoring classifier, no refit and no target labels in classifier.',
        'probability': 'Use probability directly, equivalent to probability/prior within each fold; avoid artificial fold-specific prior rescaling.',
        'fixed_models': ['incumbent', 'raw_t0_1024', 'raw_t0_4096', 'raw_t1_1024', 'raw_t1_4096', 'historical_exploratory_point'],
        'launch_rule': 'For a target, uncapped gradient alignment has positive paired99 lower bound AND paired99 lower improvement over capped is positive. Otherwise no uncapped correction fit.',
        'additional_diagnostics': 'Paired differences in frozen-model efficacy error, target moments, effective sequence mass, cap-hit fractions.',
        'uncertainty': '10000 shared sequence draws, seed 20261011; descriptive adaptive search only.'}
    if matched:
        protocol['attribution_followup'] = 'The first diagnostic changed both clipping and fold-prior scaling. Add an uncapped ratio with the identical fold prior to isolate clipping; freeze before collecting.'
    config = OUT / ('weight_protocol'+suffix+'.json')
    if not config.exists():
        write_json(config, protocol); event('experiment_planned', id='weight_truncation'+suffix, protocol_sha256=sha(config))
    combo = load_combo(); models = shortlist()
    with np.load(PREVIOUS / 'nonlinear_mask_trees.npz') as q:
        trees = {k:q[k] for k in q.files}
    with np.load(ROOT / 'competition_engineering/runs/objective_alignment_20261002/causal_population_classifier.npz') as q:
        rates = q['rates'].copy()
    names = protocol['comparison']; stats = {k:[] for k in names}; cross = {k:[] for k in names}
    moments = []; cap_counts = np.zeros(3, dtype=np.int64)
    for index, (_, z) in enumerate(cached(SEARCH)):
        idx = np.flatnonzero(z['need']); base = predict_combo(z, combo)[idx]; y = z['y'][idx]
        cf = classifier_features(z['x'][idx], base, z['step'][idx], combo['mean'], combo['scale'])
        probability = probability_from_trees(cf[:,1:], trees, index%4)
        ratio = probability/rates[index%4]
        focuses = [z['mask'][idx].astype(float), np.clip(ratio, .2, 5), probability]
        if matched:
            focuses.append(ratio)
        cap_counts += [int((ratio<.2).sum()), int((ratio>5).sum()), len(idx)]
        f = np.column_stack((np.ones(len(idx)), np.clip(base,-2,2), cf[:,1:113]))
        for name, focus in zip(names, focuses):
            s,c = weighted_primitives(f, y, base, focus); stats[name].append(s); cross[name].append(c)
        row = []
        for name, coef, target, previous in models:
            p = base.copy(); p[:,target] += f@coef
            # Empty feature matrix collects exact moments without redundant gradients.
            row.append([weighted_primitives(f[:,:0], y, p, focus)[0] for focus in focuses])
        moments.append(row)
        if (index+1)%16 == 0:
            print('WEIGHT_DIAGNOSTIC_SEQUENCE='+str(index+1), flush=True)
    stats = {k:np.array(v) for k,v in stats.items()}; cross = {k:np.array(v) for k,v in cross.items()}
    moments = np.asarray(moments)
    moments_path=OUT/('weight_moments'+suffix+'.npz')
    np.savez(moments_path, moments=moments, **{k+'_stats':v for k,v in stats.items()}, **{k+'_cross':v for k,v in cross.items()})
    counts = bootstrap_counts(64, np.random.default_rng(20261011), draws=10000)
    boot = {k:resample(stats[k],cross[k],None,counts=counts) for k in names}
    point = {k:gradient(stats[k].sum(0),cross[k].sum(0)) for k in names}
    capped = cosine(boot['capped'],boot['official']); raw = cosine(boot['probability'],boot['official'])
    interval = np.quantile(raw,[.005,.995],axis=0); improvement = np.quantile(raw-capped,[.005,.995],axis=0)
    eligible = [t for t in range(2) if interval[0,t]>0 and improvement[0,t]>0]
    scores = pooled_correlations(moments.sum(0)).mean(-1)
    draws = pooled_correlations(bootstrap_moments(moments,counts)).mean(-1)
    delta = scores-scores[:1]; draw_delta = draws-draws[:,:1]
    results = {}
    for i,model in enumerate(models):
        error_change = abs(draw_delta[:,i,2]-draw_delta[:,i,0])-abs(draw_delta[:,i,1]-draw_delta[:,i,0])
        results[model[0]] = {'delta_official_capped_probability':delta[i].tolist(),
                            'absolute_efficacy_error_change99':np.quantile(error_change,[.005,.995]).tolist()}
    result = {'eligible_targets':eligible,'gradient_cosine_capped':cosine(point['capped'],point['official']).tolist(),
              'gradient_cosine_probability':cosine(point['probability'],point['official']).tolist(),
              'probability_cosine99':interval.tolist(),'cosine_improvement99':improvement.tolist(),
              'incumbent_wp_official_capped_probability':scores[0].tolist(),
              'effective_sequence_mass':{k:((stats[k][:,0].sum(0)**2)/(stats[k][:,0]**2).sum(0)).tolist() for k in names},
              'cap_hit_counts_low_high_total':cap_counts.tolist(),'models':results,
              'protocol_sha256':sha(config),'moments_sha256':sha(moments_path),
              'classifier_sha256':sha(PREVIOUS/'nonlinear_mask_trees.npz'),
              'scope':'Frozen conditional classifier; sequence bootstrap omits classifier-learning and adaptive-selection uncertainty.',
              'decision':'Bounded uncapped fit justified only for eligible targets; cap removal alone is not a scientific repair.'}
    if matched:
        ratio_cos=cosine(boot['uncapped_ratio'],boot['official'])
        result['uncapped_ratio_cosine']=cosine(point['uncapped_ratio'],point['official']).tolist()
        result['uncapped_ratio_minus_capped_cosine99']=np.quantile(ratio_cos-capped,[.005,.995],axis=0).tolist()
        result['probability_minus_uncapped_ratio_cosine99']=np.quantile(raw-ratio_cos,[.005,.995],axis=0).tolist()
        result['population_order']=names
    write_json(destination,result); event('experiment_completed',id='weight_truncation'+suffix,result_sha256=sha(destination),eligible_targets=eligible)
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['initialize','weights','weights_matched'],required=True)
    args=parser.parse_args()
    initialize() if args.phase=='initialize' else diagnose_weights(matched=args.phase=='weights_matched')
