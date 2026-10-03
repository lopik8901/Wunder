"""Freeze and verify the shape/semantic-side search-only research cycle."""
import argparse
import json
import platform
import re
import subprocess
import sys
from competition_engineering import prediction_shape_campaign as c
from competition_engineering import book_side_campaign as b
from competition_engineering.pipeline import write_json
from tools.nonlinear_campaign_checkpoint import TESTS,verify_ledger
from connectome.research_retrieval import load_library

def tests():
    files=TESTS+['test_sequence_diversity.py','test_alternative_representation.py','test_causal_context_diagnostic.py','test_prediction_shape_campaign.py','test_book_side_campaign.py']
    p=subprocess.run([sys.executable,'-m','pytest',*['tests/'+f for f in files],'-q','-p','no:cacheprovider'],cwd=c.ROOT,capture_output=True,text=True)
    output=p.stdout+'\n'+p.stderr;system=platform.system().lower();log=c.OUT/('regression_'+system+'.log');log.write_text(output,encoding='utf-8')
    assert p.returncode==0,output[-3000:]
    write_json(c.OUT/('tests_'+system+'.json'),{'passed':int(re.search(r'(\d+) passed',output)[1]),'platform':platform.platform(),'files':files,'log_sha256':c.sha(log)})
    print(output[-1000:])

def finalize():
    assert not (c.OUT/'FINAL_REPORT.json').exists(),'Checkpoint immutable'
    shape=json.loads((c.OUT/'shape_diagnostic_work/artifacts/crossfit.json').read_text());side=json.loads((b.OUT/'development.json').read_text())
    assert not shape['gate_passed'] and not side['development_gate_passed'],'Qualifying candidate needs independent frozen replication and deployment checks'
    assert not (c.OUT/'prepared_selection.json').exists() and not (c.OUT/'prepared_development.json').exists()
    k=json.loads((c.ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==24;k['version']=25
    k['weakened'].append({'hypothesis':'The tested eight semantic side-asymmetry features provide a material WP complement beyond projected frozen state and common-mode controls.','reason':'Prospective development gate failed; precise contrasts and target effects retained. This does not reject all microstructure representations or identify physical volume semantics.','source':'runs/prediction_shape_20261003/book_side/development.json'})
    k['closed_branches'].append('Tuning the tested eight semantic volume-side aggregate features without new incremental signal evidence')
    k['established'].append({'lesson':'Bounded17-knot prediction-only curves improve fitting WP but fail both cross-sequence transfer directions. A small parameter count does not establish correction transport.','source':'runs/prediction_shape_20261003/shape_generalization_diagnostic.json'})
    k['established'].append({'lesson':'Side-aggregate averaged contrasts initially appeared positive because repeat-minus chose t0 strength0.05 while controls chose0. Applying the same selection-frozen strength vector to all maps removes the apparent advantage: both fits trail both controls in combined uniform/proxy WP, intervals cross zero. Distinguish representation gain from selector/amplitude differences.','source':'runs/prediction_shape_20261003/book_side/selection_counterfactual.json'})
    k['next_research_question']='Identify a measurable incremental signal with justified observable feature semantics; current fixed prediction-shape and side-aggregate mechanisms do not qualify. Preserve untouched replication for a candidate passing prospective development evidence.'
    k['status']='Prediction-shape and semantic-side research cycle closed; incumbent retained, no candidate awaits promotion.'
    write_json(c.OUT/'knowledge_v25.json',k);write_json(c.ROOT/'competition_engineering/search_knowledge.json',k)
    sources=['competition_engineering/prediction_shape_campaign.py','competition_engineering/book_side_campaign.py','tests/test_prediction_shape_campaign.py','tests/test_book_side_campaign.py','tools/diagnose_prediction_shape.py','tools/diagnose_side_selection.py','tools/prediction_mechanism_checkpoint.py']
    directory=c.OUT/'source_snapshot';directory.mkdir(exist_ok=True)
    for name in sources:(directory/name.replace('/','__')).write_bytes((c.ROOT/name).read_bytes())
    tests={s:json.loads((c.OUT/('tests_'+s+'.json')).read_text()) for s in ['windows','linux']}
    prior=json.loads((c.context.OUT/'protocol.json').read_text());pool=prior['roles']['replication_1']+prior['roles']['replication_2']
    protocol=json.loads((c.OUT/'protocol.json').read_text());assert not set(pool)&set(sum(protocol['roles'].values(),[]))
    assert c.sha(c.prior.COMBO)==c.prior.COMBO_SHA
    archive=c.ROOT/'competition_engineering/submissions/manual_targetwise_combo_v1.zip';assert c.sha(archive)=='c8acc7153f60b7540c3b988877f30af8a1ec799d87026d09b46a9c3a37521076'
    libraries={str(v):load_library(c.ROOT/f'competition_engineering/research_cards/v{v}')[1] for v in [15,16]}
    lines=['# Prediction mechanism research checkpoint — 3 October 2026','','**Practical incumbent retained: 0.658835322. No candidate awaits promotion. All original512 replication sequences remain untouched.**','','The research cycle reviewed accumulated architecture, objective, population, fitting-diversity and context results; the choice and rejected alternatives are recorded in decision_review.json. No reused official-search labels or protected evaluation outcomes were consulted.','','## Prediction-only shape','','A source audit found the older seven-knot calibration experiment omitted from the knowledge map. The new diagnostic was therefore gated before fitting a candidate: two complementary192-sequence crossfits on newly reserved384 fitting sequences had to establish nonlinear-minus-affine combined WP>=0.0002 and paired95 lower>0 in both populations. Curves used a fixed bounded17-knot conditional-mean approximation; there was no knot, penalty, seed or weighting sweep.','','| Crossfit direction | Uniform WP contrast | Proxy WP contrast |','|---|---:|---:|']
    for i,f in enumerate(shape['folds']):lines.append(f'| {i+1} | {f["nonlinear_minus_affine_combined"][0]:+.9f} | {f["nonlinear_minus_affine_combined"][1]:+.9f} |')
    lines+=['','Both gates failed. Own-fitting gains were uniform+0.000923/+0.000474 and proxy+0.001142/+0.000976, demonstrating a fitting/transfer gap even in a compact curve. No shape selector or development labels were read. Affine controls preserve clipped WP exactly; synthetic nonlinear-data regression confirms that the fitter can recover known signal.','','## Semantic bid/ask asymmetry','','The dataset contract identifies bid/ask feature partitions but explicitly denies depth ordering by index. An eight-group feature-only audit found49.61% negative volume-like values. Physical queue imbalance and order-flow quantities are therefore not reconstructed. The tested surrogate uses eight bounded, permutation-invariant side contrasts and cross-instrument activity interactions. It is a curator hypothesis motivated by primary microstructure evidence, not a reproduction of physical queue imbalance.','','The384 fitting groups were reused from the failed shape diagnostic;128 selection and256 development groups were still unread when this mechanism was registered. Each of six matched readouts used the same98304 fitting rows for its paired seed, fixed mean-one target weights and ridge penalty. Current inputs and128 projected frozen-GRU features were shared. Bid-plus-ask summaries and the shared-feature readout control for added nonlinear capacity. Per-target strengths maximized minimum uniform/proxy WP on selection and were frozen before development access.','','| Readout | Uniform development WP gain | Proxy development WP gain | Strength(t0,t1) |','|---|---:|---:|---|']
    for key,v in side['variants'].items():lines.append(f'| {key} | {v["combined_deltas"][0]:+.9f} | {v["combined_deltas"][1]:+.9f} | {v["strength"]} |')
    lines+=['','The paired repeat minus_r1 is the highest new point (+0.000417 uniform/+0.000241 proxy), but its95% intervals cross zero and it cannot replace the predeclared primary. Its apparent averaged advantage over controls is also confounded by selecting t0 strength0.05 while controls select0. A separately recorded diagnostic applied that original selection-frozen vector to all six maps on consumed data, without new fitting or selection. Both minus fits then trail both controls in combined WP for both populations; all contrast intervals cross zero. Thus the larger point does not establish side-feature information. Exact matched-strength and selection-uncertainty evidence is in book_side/selection_counterfactual.json.']
    lines+=['','The predeclared development gate failed. Full target effects, paired sequence intervals, both halves, contrasts against both controls, standalone probes omitting incumbent prediction features and poor-row error diagnostics are preserved. These are training-derived scores and proxy populations, not official competition scores. No fresh replication was consumed.','','## Decision and verification','','Both proposed mechanisms are closed in their tested forms. No strength, curve, aggregation, loss or architecture rescue sweep follows these negatives. The results do not establish an irreducible score ceiling or exhaust available computation. A new candidate needs measurable incremental development evidence before consuming the original untouched pool. The present evidence does not select another model experiment.','','Research library advanced to v16 with properly sourced correlation-ratio and queue-imbalance cards, each explicitly separating paper claims from weighted-objective/anonymous-feature adaptations. The source audit repaired missing historical knowledge. No implementation failure occurred in the scientific runs; their negative outcomes are not counted as infrastructure errors.', '',f'Regression: Windows{tests["windows"]["passed"]} passed, Linux{tests["linux"]["passed"]} passed. New checks cover bounded interpolation, affine scoring invariance, known synthetic nonlinear recovery, failed-gate access blocking, side swaps, within-side permutations, row-local causality and standalone feature omission. Frozen evidence hashes, source snapshots and ledger chain accompany this report.','','Incumbent artifact and archive hashes are unchanged. No official search, protected evaluation, promotion, full validation, leaderboard evaluation, submission or submission feedback access occurred. No deployment latency claim is made because no candidate qualified.','','Primary sources: [correlation-ratio paper](https://arxiv.org/pdf/1811.03918), [queue-imbalance paper](https://arxiv.org/pdf/1512.03492), [order-flow impact paper reviewed for semantics constraints](https://arxiv.org/pdf/1011.6402).']
    (c.OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in c.OUT.rglob('*') if p.is_file() and p.name not in ['ledger.jsonl','FINAL_REPORT.json','FINAL_REPORT.md']}
    r={'incumbent_wp':.6588353223316641,'incumbent_sha256':c.prior.COMBO_SHA,'archive_sha256':c.sha(archive),'awaiting_promotion':False,'official_search_passes':0,'untouched_replication_sequences':512,'shape':shape,'book_side':side,'tests':tests,'library_manifests':libraries,'evidence_sha256':hashes,'ledger_prefix_sha256':c.sha(c.OUT/'ledger.jsonl')}
    write_json(c.OUT/'FINAL_REPORT.json',r);c.event('campaign_checkpointed',json_sha256=c.sha(c.OUT/'FINAL_REPORT.json'),markdown_sha256=c.sha(c.OUT/'FINAL_REPORT.md'));verify()

def verify():
    r=json.loads((c.OUT/'FINAL_REPORT.json').read_text())
    for p,h in r['evidence_sha256'].items():assert c.sha(c.ROOT/p)==h,p
    events=verify_ledger(c.OUT/'ledger.jsonl');last=events[-1]
    assert last['json_sha256']==c.sha(c.OUT/'FINAL_REPORT.json') and last['markdown_sha256']==c.sha(c.OUT/'FINAL_REPORT.md') and last['previous_ledger_sha256']==r['ledger_prefix_sha256']
    for v,h in r['library_manifests'].items():assert load_library(c.ROOT/f'competition_engineering/research_cards/v{v}')[1]==h
    assert c.sha(c.prior.COMBO)==r['incumbent_sha256']
    assert c.sha(c.ROOT/'competition_engineering/submissions/manual_targetwise_combo_v1.zip')==r['archive_sha256']
    print(json.dumps({'verified_hashes':len(r['evidence_sha256']),'ledger_events':len(events),'incumbent_unchanged':True,'untouched_replication_sequences':512}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tests','finalize','verify'],required=True)
    {'tests':tests,'finalize':finalize,'verify':verify}[p.parse_args().phase]()
