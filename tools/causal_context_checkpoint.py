"""Freeze and verify the search-only causal context diagnostic checkpoint."""
import argparse
import json
import platform
import subprocess
import sys
from competition_engineering import causal_context_diagnostic as c
from competition_engineering.pipeline import write_json
from tools.nonlinear_campaign_checkpoint import TESTS, verify_ledger

def tests():
    files=TESTS+['test_sequence_diversity.py','test_alternative_representation.py','test_causal_context_diagnostic.py']
    p=subprocess.run([sys.executable,'-m','pytest',*['tests/'+f for f in files],'-q','-p','no:cacheprovider'],cwd=c.ROOT,capture_output=True,text=True)
    output=p.stdout+'\n'+p.stderr
    (c.OUT/('regression_'+platform.system().lower()+'.log')).write_text(output,encoding='utf-8')
    print(output[-2000:]);assert p.returncode==0

def finalize():
    assert not (c.OUT/'FINAL_REPORT.json').exists(),'Checkpoint immutable'
    d=json.loads((c.OUT/'development_diagnostic.json').read_text())
    assert not d['development_gate_passed']
    k=json.loads((c.ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==21
    k['version']=22;k['status']='Causal prefix-context diagnostic closed; incumbent retained; no candidate awaits promotion.'
    k['weakened'].append({'hypothesis':'The tested 64-dimensional label-free prefix context predicts transferable sequence-level WP gradients.','reason':'All four development error reductions negative, both halves negative. This does not reject other context representations or establish irreducible noise.','source':'runs/causal_context_20261003/development_diagnostic.json'})
    k['closed_branches'].append('Tuning this prefix-context gradient predictor without new observable transport evidence')
    k['next_research_question']='Establish an observable, transferable source of mapping heterogeneity before fitting another conditional model; no next model experiment currently justified.'
    write_json(c.OUT/'knowledge_v22.json',k);write_json(c.ROOT/'competition_engineering/search_knowledge.json',k)
    snapshots=c.OUT/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for name in ['competition_engineering/causal_context_diagnostic.py','tests/test_causal_context_diagnostic.py','tools/causal_context_checkpoint.py']:
        (snapshots/name.replace('/','__')).write_bytes((c.ROOT/name).read_bytes())
    lines=['# Causal context diagnostic checkpoint — 3 October 2026','','**Best credible candidate remains the frozen incumbent: 0.658835322. No candidate awaits promotion.**','','Reserved prospectively: 512 fitting sequences, 128 development sequences and two 256-sequence replication blocks. Only fitting and development were read. Both replication blocks remain untouched, as does the earlier reserved block. Independence concerns new correction fitting and selection; incumbent pretraining exposure is not claimed independent.','','The diagnostic tested whether feature-only first-512-row context (32 projected-input means and 32 log standard deviations) predicts sequence-level derivatives of clipped weighted correlation for a fixed current-input correction basis. All assessed rows follow the context prefix. Reference scoring moments and context normalization were estimated on fitting data only. A mean-gradient predictor served as the control. The probability-weighted population is a proxy, not the official scoring mask.','','| Population | Target 0 error reduction | Target 1 error reduction |','|---|---:|---:|']
    for label,row in zip(['Uniform','Probability proxy'],d['gradient_error_reduction_population_target']):lines.append(f'| {label} | {row[0]:+.2%} | {row[1]:+.2%} |')
    lines+=['','Every point estimate and both development halves worsened. Three of four paired 95% intervals exclude improvement; the uniform target-0 interval crosses zero. The predeclared gate failed, so no replication, conditional model, or official search was run. Complete intervals are in development_diagnostic.json.','','Within-sequence interleaved gradient reliability was about 0.88 for uniform weighting and 0.65–0.68 for the proxy. Temporal dependence can inflate this descriptive measure; it is not an estimate of irreducible noise. Stable sequence differences alone do not establish that the tested observable context predicts them.','','The exact clipped correlation gradient passed finite-difference regression, including clipping saturation and probability weights. Prefix-context regression covers causality, future independence and reset behavior. Windows and Linux suite output is preserved alongside this report. No implementation failure occurred in this diagnostic.','','Checkpoint decision: abandon this specific predictor. Prior temporal, readout and supervision counterfactuals also failed; another architecture or context-window sweep lacks a new mechanism justification. Resources are not claimed exhausted. Preserve independent sequences until evidence identifies a materially different, observable and transferable signal.','','No protected evaluation, promotion, full validation, leaderboard evaluation or submission occurred. The incumbent and its archive are unchanged.']
    (c.OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in c.OUT.rglob('*') if p.is_file() and p.name not in ['ledger.jsonl','FINAL_REPORT.json','FINAL_REPORT.md']}
    write_json(c.OUT/'FINAL_REPORT.json',{'incumbent_wp':.6588353223316641,'awaiting_promotion':False,'official_search_passes':0,'diagnostic':d,'untouched_replication_sequences':512,'evidence_sha256':hashes,'ledger_prefix_sha256':c.sha(c.OUT/'ledger.jsonl')})
    c.event('campaign_checkpointed',json_sha256=c.sha(c.OUT/'FINAL_REPORT.json'),markdown_sha256=c.sha(c.OUT/'FINAL_REPORT.md'));verify()

def verify():
    r=json.loads((c.OUT/'FINAL_REPORT.json').read_text())
    for p,h in r['evidence_sha256'].items():assert c.sha(c.ROOT/p)==h,p
    last=verify_ledger(c.OUT/'ledger.jsonl')[-1]
    assert last['json_sha256']==c.sha(c.OUT/'FINAL_REPORT.json')
    assert last['markdown_sha256']==c.sha(c.OUT/'FINAL_REPORT.md')
    assert last['previous_ledger_sha256']==r['ledger_prefix_sha256']
    print('Verified',len(r['evidence_sha256']),'evidence hashes and ledger chain')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tests','finalize','verify'],required=True)
    {'tests':tests,'finalize':finalize,'verify':verify}[p.parse_args().phase]()
