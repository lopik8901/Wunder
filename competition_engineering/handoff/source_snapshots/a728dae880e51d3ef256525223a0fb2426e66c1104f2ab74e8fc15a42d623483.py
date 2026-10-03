"""Persistent, search-only evidence synthesis and prospective research journal."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from competition_engineering.manual_search_core import ROOT, COMBO, COMBO_SHA, SEARCH_ROOT_WP
from competition_engineering.pipeline import write_json
from connectome.mlevolve_history import load_prior_search_records, SEARCH_METRIC_DOMAIN, proposal_signature
from connectome.research_retrieval import load_library

OUT=ROOT/'competition_engineering/runs/reassessment_20261002'
KNOWLEDGE=ROOT/'competition_engineering/search_knowledge.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def event(kind, **payload):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'ledger.jsonl'
    record={'utc':datetime.now(timezone.utc).isoformat(),'kind':kind,
            'previous_ledger_sha256':sha(path) if path.exists() else None,
            'boundary':'search-only; no protected feedback','evidence':'reused_search_not_independent_validation',**payload}
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')

def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    if sha(COMBO)!=COMBO_SHA:
        raise ValueError('Frozen incumbent changed; investigate identity before experiments')
    if (OUT/'campaign.json').exists():
        return
    write_json(OUT/'campaign.json',{'started_unix':time.time(),'maximum_wall_seconds':7200,
        'incumbent_wp':SEARCH_ROOT_WP,'incumbent_sha256':COMBO_SHA,
        'boundary':'Only designated training and gru_tune search caches; protected feedback excluded',
        'rules':['Official WP for selection','No broad loss/grid or old temporal tuning without new evidence',
                 'Freeze experiment protocol before data analysis','No new incumbent from highest adaptive point',
                 'Sequence-level uncertainty; pooled official metric unchanged',
                 'Repair infrastructure with regression tests; preserve scientific negatives separately']})
    knowledge={'schema_version':1,'version':1,'evidence':'search-only; adaptive reuse disclosed',
        'incumbent':{'wp':SEARCH_ROOT_WP,'sha256':COMBO_SHA},
        'established':[{'lesson':'Use official clipped weighted Pearson for model and hyperparameter selection whenever feasible. Lower residual MSE does not imply higher WP.',
                        'source':'runs/objective_alignment_20261002/final_review.json'}],
        'weakened':[{'hypothesis':'Objective misalignment alone is the main remaining bottleneck.',
                     'reason':'60 fits in eight suites; objective-aware selection helped, but no convincing replacement.'}],
        'closed_branches':['Broad objective-loss/grid tuning','Previously tested additive EMA and diagonal bilinear temporal tuning'],
        'exploratory_only':[{'wp':0.6590796311188727,'paired99_delta':[-0.0005840183,0.0010744564],
                             'reason':'Adaptively selected point; interval crosses zero'}],
        'reopen_requirement':'New measured evidence and a predeclared falsifiable mechanism.'}
    if KNOWLEDGE.exists():
        raise ValueError('Persistent knowledge already exists; use versioned merge')
    write_json(KNOWLEDGE,knowledge);write_json(OUT/'starting_knowledge.json',knowledge)
    event('knowledge_updated',knowledge_sha256=sha(KNOWLEDGE),incumbent_sha256=COMBO_SHA)

def synthesize():
    initialize()
    if (OUT/'research_state_map.json').exists():
        return
    records=load_prior_search_records(ROOT/'competition_engineering/mlevolve_runs')
    safe=[]
    for r in records:
        result=record_mapping(r,'result');spec=record_mapping(r,'spec');config=record_mapping(r,'config')
        scores=search_scores(result)
        safe.append({'node_id':r['node_id'],'status':r['status'],'mechanism':spec.get('kind','generated' if 'train_source' in spec else 'unknown'),
            'hypothesis':spec.get('hypothesis','')[:300],'search_scores':scores,
            'settings':{k:config[k] for k in ('target_mode','sample_stride','ridge_penalty','epochs','base_model','base_scale','base_bias') if k in config},
            'signature':proposal_signature('EXPERIMENT = '+repr(spec)) if 'train_source' not in spec else None,
            'training_diagnostics':{k:v for k,v in record_mapping(r,'training_diagnostics').items() if k in ('training_rows','updates','train_seconds','loss_first80','loss_last80')},
            'cpu_us':record_mapping(r,'resource_estimate').get('cpu_microseconds_per_row'),
            'atlas':result.get('search_error_diagnostics') if scores else None})
    write_json(OUT/'mlevolve_search_history.json',safe)
    cards,library_hash=load_library(ROOT/'competition_engineering/research_cards/v6')
    write_json(OUT/'library_review.json',{'version':'v6','manifest_sha256':library_hash,'count':len(cards),
        'cards':[{'id':c['card_id'],'mechanism':c['mechanism'],'assumptions':c['assumptions'],'limitations':c['limitations']} for c in cards]})
    sources=[ROOT/'competition_engineering/reports/manual_research_search_only_20261001.json',
        ROOT/'competition_engineering/reports/manual_search_loop_five_20261001.json',
        ROOT/'competition_engineering/reports/manual_combo_search_atlas_20261001.json',
        ROOT/'competition_engineering/runs/autonomous_20261001/FINAL_REPORT.json',
        ROOT/'competition_engineering/runs/objective_alignment_20261002/final_review.json']
    sources+=list((ROOT/'competition_engineering/mlevolve_runs').glob('*/experiments/history.jsonl'))
    state={
        'strongly_supported':[
            'Current-feature residual ridge and target-specific combinations improved historical search roots.',
            'Official WP selection can materially outperform residual-MSE selection.',
            'Clipping, sequence resets and all-required-row callbacks matter; fused ONNX with preallocated buffers makes current readouts feasible.',
            'Small adaptively selected point gains do not establish generalizable improvement.'],
        'partially_supported':[
            'More training sequences helped original ridge/magnitude/gated branches, but 4096-sequence incremental aligned fits did not reliably help.',
            'Nonlinear t0 trees showed a tiny late-position signal, but uncertainty crossed zero and Python inference was expensive.',
            'Scoring selection changes observable populations; causal inputs predict selection, but weighting did not align target gradients.',
            'TCNs helped an older ridge parent modestly, but were inferior to the current combo and often marginal on CPU.'],
        'weakened':[
            'Objective mismatch alone explains the remaining bottleneck: 60 controlled fits failed to establish replacement.',
            'Global nonlinear expansions, output splines, innovation gates or a late-only linear map resolve current residuals.',
            'Tested additive EMA/lag and featurewise bilinear temporal corrections improve the combo.',
            'Coarse propensity or the tested current-input classifier makes scoring selection ignorable.',
            'Simple fixed-GRU latent readout deserves a fit solely from point-gradient transfer.'],
        'not_adequately_tested':[
            'Sequence-level uncertainty and dimensional noise of training/search correction-gradient alignment.',
            'Conditional target shift after matching causal observable populations, versus random search-sequence variation.',
            'Fresh training-derived, correction-disjoint prospective selection under scoring-like regimes.',
            'End-to-end learned dynamics, learned cross-feature causal interactions, input normalization under shift; broad architectural sweeps are not yet justified.'],
        'systematic_errors':[
            'Incumbent t0 WP 0.647991712 is lower than t1 0.669678933; earlier combo improvements were smallest in late t0.',
            'Large per-sequence score variation and highly persistent clipped residuals coexist with weak current/lagged feature-residual correlations.',
            'Training/uniform-search WP near 0.46-0.49 differs sharply from officially scored search WP near 0.66.',
            'Training WP gains often fail to transfer; observable scoring-selection AUC is not sufficient evidence for target transport.'],
        'contradictions':[
            'More data helped original ridge but not new corrections: marginal information, selection bias and population mismatch compete.',
            'Persistent residuals did not yield useful temporal correction: persistence is not causal predictability.',
            'Better loss and better training WP both sometimes worsen search WP: optimization cannot identify missing transferable information.',
            'High scoring-selector AUC coexists with poor gradient alignment: prediction of mask is not proof of ignorability.',
            'A low high-dimensional gradient cosine may reflect finite-sequence noise rather than a stable directional conflict.'],
        'methodological_limitations':[
            '82 MLEvolve attempts plus many manual fits reused one 64-sequence search; bootstrap intervals do not correct adaptive selection.',
            'Splitting already-exposed search into new halves cannot restore independence.',
            'Correction-only internal selection is not independent validation of the already-trained/selected incumbent.',
            'Point gradient-cosine launch rules lacked sequence-level uncertainty; quantify uncertainty before continuing their use.',
            'Early MLEvolve history/root is stale relative to manual incumbent and newer research; dedicated persistent lessons must be visible.',
            'CPU measurements differed with concurrency and callback representation; serialize comparable actual callbacks.',
            'Legacy global ledger mixes protected/search feedback and is excluded from this synthesis.'],
        'mlevolve_attempts':len(safe),'status_counts':dict(Counter(r['status'] for r in safe)),
        'routine_metric_records':sum(bool(r['search_scores']) for r in safe),
        'history_best_search_wp':max(r['search_scores'].get('combined_WP',-1) for r in safe),
        'incumbent':{'wp':SEARCH_ROOT_WP,'sha256':COMBO_SHA},
        'reviewed_sources_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'library_manifest_sha256':library_hash,
        'first_experiment':'Diagnose sequence-level gradient uncertainty and scoring-population contrast; no new model fit or parameter sweep.'}
    write_json(OUT/'research_state_map.json',state)
    lines=['# Search-only research-state synthesis','',f'Frozen incumbent: {SEARCH_ROOT_WP:.9f}. {len(safe)} MLEvolve attempts reviewed; protected feedback excluded.']
    for key in ('strongly_supported','partially_supported','weakened','not_adequately_tested','systematic_errors','contradictions','methodological_limitations'):
        lines+=['','## '+key.replace('_',' ').capitalize(),'']+['- '+x for x in state[key]]
    lines+=['','Next: '+state['first_experiment'],'']
    (OUT/'research_state_map.md').write_text('\n'.join(lines),encoding='utf-8')
    event('synthesis_completed',map_sha256=sha(OUT/'research_state_map.json'),reviewed_attempts=len(safe),library_hash=library_hash)
    print(json.dumps({'synthesis_complete':True,'attempts':len(safe),'incumbent':SEARCH_ROOT_WP,'cards':len(cards)}))

def record_mapping(record, key):
    value=record.get(key)
    return value if isinstance(value,dict) else {}

def search_scores(result):
    if result.get('metric_domain')!=SEARCH_METRIC_DOMAIN:
        return {}
    metrics=record_mapping(result,'metrics')
    return {k:metrics[k] for k in ('WP_t0','WP_t1','combined_WP')
            if isinstance(metrics.get(k),(int,float)) and __import__('math').isfinite(metrics[k])}

if __name__=='__main__':
    synthesize()
