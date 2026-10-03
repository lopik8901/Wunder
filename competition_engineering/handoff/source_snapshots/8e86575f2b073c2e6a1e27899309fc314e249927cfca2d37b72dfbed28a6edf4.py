"""Regression records and durable verification for alternative representations."""
import argparse
import json
import platform
import re
import subprocess
import sys
from competition_engineering import alternative_representation as c
from competition_engineering.pipeline import write_json
from tools.nonlinear_campaign_checkpoint import TESTS,verify_ledger
from connectome.research_retrieval import load_library

def tests():
    files=TESTS+['test_sequence_diversity.py','test_alternative_representation.py']
    p=subprocess.run([sys.executable,'-m','pytest',*['tests/'+f for f in files],'-q','-p','no:cacheprovider'],cwd=c.ROOT,capture_output=True,text=True)
    output=p.stdout+'\n'+p.stderr;system=platform.system().lower();log=c.OUT/('regression_'+system+'.log');log.write_text(output,encoding='utf-8')
    assert p.returncode==0,output[-3000:]
    write_json(c.OUT/('tests_'+system+'.json'),{'passed':int(re.search(r'(\d+) passed',output)[1]),'platform':platform.platform(),'files':files,'log_sha256':c.sha(log)})
    print(output[-1000:])

def finalize():
    path=c.OUT/'FINAL_REPORT.json';assert not path.exists(),'Checkpoint immutable'
    stages={name:json.loads((c.OUT/f).read_text()) for name,f in [('fixed_convolution','probe_replication.json'),('polynomial_memory','orthogonal_memory/replication.json'),('learned_convolution','learned_causal/replication.json')]}
    if (c.OUT/'disjoint_readout/replication.json').exists(): stages['disjoint_readout']=json.loads((c.OUT/'disjoint_readout/replication.json').read_text())
    if (c.OUT/'raw_target/replication.json').exists(): stages['raw_target']=json.loads((c.OUT/'raw_target/replication.json').read_text())
    assert not any(r['search_gate_passed'] for r in stages.values()),'Qualified experiment must complete frozen deployment/search first'
    k=json.loads((c.ROOT/'competition_engineering/search_knowledge.json').read_text());k['version']+=1
    k['status']='Alternative causal representation checkpoint complete; incumbent retained, no candidate awaits promotion.'
    k['weakened'].append({'hypothesis':'The tested alternative causal features yield a robust deployable improvement over the GRU incumbent.','reason':'All completed prospective mechanism gates failed. Fixed probes do not reject learned architecture families.','source':'runs/alternative_representation_20261003/FINAL_REPORT.json'})
    k['established'].append({'lesson':'The tested learned temporal encoder gains~0.08-0.09 unshrunk proxy WP on fitting rows and loses~0.054-0.056 on development. Its capacity-matched temporal-minus-current replication interval is below zero. Better fitting is not a transferable representation result.','source':'runs/alternative_representation_20261003/learned_causal/generalization_diagnostic.json'})
    k['closed_branches']+=['Tuning the tested86-row temporal topology or512-row fixed polynomial memory without new mechanism evidence','Rescuing the tested temporal encoders by disjoint decoding or original-target supervision without new evidence']
    k['weakened'].append({'hypothesis':'Original-target supervision and a simple heterogeneous blend recover useful transfer from this compact TCN.','reason':'Primary proxy gain-0.000038; repeated model selected zero blend weight. Both GRU contrasts negative with95 intervals excluding zero.','source':'runs/alternative_representation_20261003/raw_target/replication.json'})
    k['next_research_question']='Diagnose whether errors are driven by information outside the current112 observed inputs, domain-dependent feature mappings, or target noise; avoid further blind temporal-capacity sweeps.'
    snapshot=c.OUT/('knowledge_v'+str(k['version'])+'.json');write_json(snapshot,k);write_json(c.ROOT/'competition_engineering/search_knowledge.json',k)
    test={system:json.loads((c.OUT/('tests_'+system+'.json')).read_text()) for system in ['windows','linux']}
    reservation=json.loads((c.OUT/'reservation.json').read_text());used=len(stages);reserved=6 if 'raw_target' in stages else 4
    sources=['competition_engineering/alternative_representation.py','competition_engineering/orthogonal_memory_campaign.py','competition_engineering/learned_causal_campaign.py','competition_engineering/sparse_temporal.py','tests/test_alternative_representation.py','tools/alternative_campaign_checkpoint.py','tools/verify_memory_paper.py']
    sources+=['competition_engineering/disjoint_readout_campaign.py','competition_engineering/raw_target_campaign.py','competition_engineering/isolation_retry.py','tools/diagnose_learned_causal_gap.py']
    directory=c.OUT/'source_snapshot';directory.mkdir(exist_ok=True)
    for f in sources: (directory/f.replace('/','__')).write_bytes((c.ROOT/f).read_bytes())
    hashes={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in c.OUT.rglob('*') if p.is_file() and p.name not in ['ledger.jsonl','FINAL_REPORT.json','FINAL_REPORT.md']}
    candidates=[]
    for stage,r in stages.items():
        for name,v in r['variants'].items(): candidates.append((v.get('combined_delta',[0,0])[1],stage,name))
    best=max(candidates)
    alternatives=[x for x in candidates if not x[2].startswith(('gru','core'))];best_alternative=max(alternatives)
    report={'incumbent_wp':.6588353223316641,'incumbent_sha256':c.COMBO_SHA,'awaiting_promotion':False,'official_search_passes':0,'stages':stages,'sequence_reservation':reservation,'consumed_replication_blocks':used,'unused_replication_blocks':reserved-used,'best_observed_alternative_point':{'probability_wp_gain':best_alternative[0],'stage':best_alternative[1],'candidate':best_alternative[2]},'best_observed_development_point':{'probability_wp_gain':best[0],'stage':best[1],'candidate':best[2],'interpretation':'Exploratory maximum across separate training-derived populations; not an official-search score or credible replacement.'},'tests':test,'knowledge_snapshot':snapshot.name,'research_library_v14_manifest_sha256':load_library(c.ROOT/'competition_engineering/research_cards/v14')[1],'evidence_sha256':hashes,'ledger_prefix_sha256':c.sha(c.OUT/'ledger.jsonl')}
    write_json(path,report)
    lines=['# Alternative causal representation checkpoint — 3 October 2026','','**Practical incumbent: 0.658835322. No candidate awaits promotion.**','','Reserved before architecture results:512 fitting sequences,128 development sequences, four disjoint256-sequence replication blocks. Excluded4608 earlier assigned research/protected training-role groups. Exact row-group/sequence identities and content hashes are preserved. Frozen incumbent pretraining exposure is not claimed independent; each candidate replication block is disjoint from its correction fitting/selection and all earlier consumed blocks.','',f'Consumed {used} replication blocks; {reserved-used} remain untouched. Additional blocks, if present, were reserved prospectively before the corresponding new experiment.','', '| Experiment / model | Uniform WP gain | Probability WP gain | Probability paired95 interval | Standalone uniform WP(t0,t1) |','|---|---:|---:|---|---|']
    for stage,r in stages.items():
        for name,v in r['variants'].items():
            x=v['combined_delta'];ci=v['paired95'];stand=v.get('standalone_uniform_wp')
            lines.append(f'| {stage}/{name} | {x[0]:+.9f} | {x[1]:+.9f} | [{ci[0][1]:+.9f}, {ci[1][1]:+.9f}] | '+(' / '.join(f'{s:.6f}' for s in stand) if stand else 'not measured')+' |')
    lines+=['','These are training-derived development scores, not official competition scores. All prospective mechanism gates failed; no official-search access, protected evaluation, promotion, leaderboard evaluation or submission occurred. Standalone probes omit incumbent prediction features, but their learned/frozen representations may reflect incumbent-supervised training or GRU pretraining.','', 'The first experiment compared equally sized current-only nonlinear, projected GRU, exact-delay and fixed dilated-convolution features. The convolution effect was seed-sensitive and weaker than the GRU controls. This rejects this frozen probe as a candidate, not trained TCNs in general.','', 'The second experiment compared512-row order4 Legendre memory with matched smoothing and GRU features. It was weaker than smoothing and reliably weaker than GRU on the proxy population. No memory window/order sweep followed.','', 'The third experiment trained a112-to32 projection and four width32 causal layers, comparing a parameter-matched current-only topology. Identical fitting rows, optimizer, checkpoint grid and WP strength selector isolated temporal connections from additional static capacity. Detailed target effects, contrasts, fitting histories and selected checkpoints are preserved in the JSON evidence.','', 'All ensembles used simple prospective target-wise residual corrections with development-frozen strengths. Poor-row error diagnostics and sequence uncertainty are recorded. Lower fitting MSE and better poor-row MSE are not substitutes for replicated WP gains.','', 'Infrastructure: Parquet omitted sequence-ID statistics; reservation now reads and verifies the ID column instead. Regression covers missing statistics and mixed identities. Numerical tests cover batch/stream convolution parity, early causal padding, resets, future independence, differentiable sparse endpoints, and polynomial-filter/state-update parity. Implementation failure is logged separately from scientific outcomes.','', 'Research library: generic TCN card updated from primary architecture sections; LMU card added and refined after verifying equations1-4 and ZOH discretization in the primary paper. Libraries v13/v14 retain provenance and distinguish curator inference from claims.','', 'No candidate qualified for deployment. Complete candidate callback latency was therefore not measured; compact parameter counts are plausibility evidence only. No deployment performance is claimed.','',f'Best observed probability-population development gain: {best[0]:+.9f}, {best[1]}/{best[2]}. This adaptively observed maximum does not replace the incumbent.', '',f'Regression: Windows {test["windows"]["passed"]} passed; Linux {test["linux"]["passed"]} passed. Exact artifacts, source snapshots, ledger chain and hashes accompany this report.','', 'Next question: distinguish missing observed information, domain-dependent input-to-target mappings and target noise before another temporal-capacity campaign. The unused prospective block(s) remain reserved.']
    if 'disjoint_readout' in stages: lines.insert(-2,'A fourth experiment froze the learned encoders and reserved512 additional readout-fitting sequences, disjoint from encoder fitting, development and every replication block. Matched ridge readouts tested whether independent decoding rescues learned temporal information. The final reserved256-sequence replication block was consumed once. No encoder or checkpoint was changed.\n')
    if 'raw_target' in stages: lines.insert(-2,'A fifth experiment trained the same topology on original clipped targets, omitting incumbent prediction inputs, with a parameter-matched static model. The simple target-wise blend was chosen on development WP before a newly reserved256-sequence replication. Another256-sequence block was frozen at the same time and remains untouched. This counterfactual distinguishes supervision-dependent representation quality from another architecture sweep.\n')
    lines+=['','The raw-target primary blend lost0.000038 proxy WP; the paired repeat selected zero weight for both targets. Disjoint decoding improved temporal standalone decoding from~0.22 to~0.325 on different fresh populations but still produced negative incremental WP. Thus representation/readout dependence matters, without establishing a useful incumbent complement. These point comparisons span different populations and are not paired causal estimates.','', 'Target-specific behavior: temporal poor-row MSE gains were most consistent on t0, yet its WP effects were weak or negative. Several t1 corrections were set to zero by development selection. Full target-wise effects and sequence bootstrap intervals accompany every experiment. Error cosines near1 occasionally exceed1 by float32 roundoff; they are descriptive, not evidence of perfect agreement.','', 'Additional infrastructure failure: bubblewrap returned Resource temporarily unavailable while creating a namespace, before candidate execution. Failed work was preserved as raw_target/failed_fit_001. A new bounded retry helper permits at most three attempts only for that exact transient error, under the same overall deadline and identical sandbox command/mount/network restrictions. Permission and candidate errors are not retried. Regression covers recovery, exhaustion and non-retry cases; the scientific training source hash is identical before and after repair.','', 'Primary sources: [TCN architecture paper](https://arxiv.org/html/1803.01271v2), [LMU dynamics paper](https://papers.nips.cc/paper_files/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf). Their benchmark successes do not imply Connectome improvements; the experiments and curator interpretations are separately recorded.','',f'Best alternative development point: {best_alternative[0]:+.9f} proxy WP, {best_alternative[1]}/{best_alternative[2]}; it failed the prospectively frozen mechanism/replication criteria. Best credible candidate remains the practical incumbent.','', 'Checkpoint rationale: the tested representation mechanisms and both justified rescue counterfactuals failed fresh replication. The evidence does not justify another window, width, strength, loss or seed sweep. Compute and the overall eligible training dataset are not claimed exhausted. One reserved block is preserved for a materially different hypothesis backed by new diagnostics; no additional architecture experiment is selected from the present evidence.']
    (c.OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');c.event('campaign_checkpointed',json_sha256=c.sha(path),markdown_sha256=c.sha(c.OUT/'FINAL_REPORT.md'));verify()

def verify():
    r=json.loads((c.OUT/'FINAL_REPORT.json').read_text())
    for p,h in r['evidence_sha256'].items(): assert c.sha(c.ROOT/p)==h,p
    events=verify_ledger(c.OUT/'ledger.jsonl');last=events[-1]
    assert last['json_sha256']==c.sha(c.OUT/'FINAL_REPORT.json') and last['markdown_sha256']==c.sha(c.OUT/'FINAL_REPORT.md') and last['previous_ledger_sha256']==r['ledger_prefix_sha256']
    assert not any(e['kind']=='official_search_started' for e in events)
    roles=r['sequence_reservation']['roles'];seen=set()
    for groups in roles.values(): assert not seen&set(groups);seen.update(groups)
    assert not seen&set(r['sequence_reservation']['excluded_groups'])
    if (c.OUT/'disjoint_readout/protocol.json').exists():
        extra=json.loads((c.OUT/'disjoint_readout/protocol.json').read_text());groups=set(extra['groups']['readout_fit'])
        assert len(groups)==512 and not groups&seen and not groups&set(extra['excluded_groups'])
        assert c.sha(c.OUT/'learned_causal/fit/identity.json')==extra['encoder_identity_sha256']
        seen.update(groups)
    if (c.OUT/'raw_target/protocol.json').exists():
        extra=json.loads((c.OUT/'raw_target/protocol.json').read_text())
        for groups in extra['groups'].values():
            assert len(groups)==256 and not set(groups)&seen and not set(groups)&set(extra['excluded_groups']);seen.update(groups)
        assert not (c.OUT/'raw_target/future_reserved').exists()
    assert c.sha(c.COMBO)==c.COMBO_SHA
    assert load_library(c.ROOT/'competition_engineering/research_cards/v14')[1]==r['research_library_v14_manifest_sha256']
    print(json.dumps({'verified_hashes':len(r['evidence_sha256']),'ledger_events':len(events),'reservation_disjoint':True,'incumbent_unchanged':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tests','finalize','verify'],required=True);args=p.parse_args();{'tests':tests,'finalize':finalize,'verify':verify}[args.phase]()
