"""Record regressions and close the search-only diversity checkpoint."""
import argparse
import json
import platform
import re
import subprocess
import sys
from competition_engineering import sequence_diversity as c
from competition_engineering import balanced_sequence_diversity as b
from competition_engineering.pipeline import write_json
from tools.nonlinear_campaign_checkpoint import TESTS,verify_ledger

def tests():
    files=TESTS+['test_sequence_diversity.py']
    p=subprocess.run([sys.executable,'-m','pytest',*['tests/'+f for f in files],'-q','-p','no:cacheprovider'],cwd=c.ROOT,capture_output=True,text=True)
    output=p.stdout+'\n'+p.stderr;system=platform.system().lower()
    (c.OUT/('regression_'+system+'.log')).write_text(output,encoding='utf-8')
    assert p.returncode==0,output[-4000:]
    write_json(c.OUT/('tests_'+system+'.json'),{'passed':int(re.search(r'(\d+) passed',output)[1]),'platform':platform.platform(),'files':files})
    print(output[-1000:])

def finalize():
    a=json.loads((c.OUT/'replication.json').read_text());z=json.loads((b.OUT/'replication.json').read_text())
    assert not a['search_gate_passed'] and not z['search_gate_passed'],'Passing gate requires completing the predeclared search before checkpointing'
    k=json.loads((c.ROOT/'competition_engineering/search_knowledge.json').read_text())
    assert k['version']==13
    k['version']=14;k['status']='Sequence-diversity checkpoint complete; no candidate awaits promotion.'
    k['weakened'].append({'hypothesis':'More fitting sequences at fixed rows improves this frozen-state correction model.','reason':z['decision'],'source':'runs/sequence_diversity_20261003/balanced_followup/replication.json'})
    write_json(c.OUT/'knowledge_v14.json',k);write_json(c.ROOT/'competition_engineering/search_knowledge.json',k)
    tests={s:json.loads((c.OUT/('tests_'+s+'.json')).read_text()) for s in ['windows','linux']}
    snapshots=c.OUT/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for f in ['competition_engineering/sequence_diversity.py','competition_engineering/balanced_sequence_diversity.py','competition_engineering/sequence_diversity_evaluation.py','tests/test_sequence_diversity.py','tools/sequence_diversity_checkpoint.py']:
        (snapshots/f.replace('/','__')).write_bytes((c.ROOT/f).read_bytes())
    evidence={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in c.OUT.rglob('*') if p.is_file() and p.name not in ['ledger.jsonl','FINAL_REPORT.json','FINAL_REPORT.md']}
    report={'incumbent_wp':.6588353223316641,'awaiting_promotion':False,'official_search_passes':0,'original':a,'balanced':z,'tests':tests,'evidence_sha256':evidence,'ledger_prefix_sha256':c.sha(c.OUT/'ledger.jsonl')}
    write_json(c.OUT/'FINAL_REPORT.json',report)
    lines=['# Fitting-sequence diversity checkpoint — 3 October 2026','','**Practical incumbent retained: 0.658835322. No candidate awaits promotion.**','','Equal budget: 1024 sequences × 128 rows versus 2048 × 64, each 131,072 rows. Two paired sampling/initialization seeds; fixed causal GRU features, 32-unit correction head, optimizer, 1,280 updates and WP-based selection. Linear controls use the same sampled rows.','','| Comparison: broad minus narrow | Uniform WP Δ | Probability-weighted WP Δ | Probability-weighted paired 95% interval |','|---|---:|---:|---|']
    for title,result,key in [('Original mixture',a,'broad_minus_narrow'),('Balanced 50:50 mixture',z,'broad_minus_balanced_narrow')]:
        contrasts=result.get(key)
        if contrasts is None:
            contrasts=result.get('contrasts',{})
        for name,v in contrasts.items():
            x=v['combined_uniform_probability'];ci=v['paired95'];lines.append(f'| {title}, {name} | {x[0]:+.9f} | {x[1]:+.9f} | [{ci[0][1]:+.9f}, {ci[1][1]:+.9f}] |')
        v=result['mean_paired_neural_contrast'];x=v['combined_uniform_probability'];ci=v['paired95'];lines.append(f'| {title}, mean neural | {x[0]:+.9f} | {x[1]:+.9f} | [{ci[0][1]:+.9f}, {ci[1][1]:+.9f}] |')
    lines+=['','Neither comparison passed its prospectively frozen replication gate. The first comparison confounded sequence count with source mixture; the follow-up held that mixture at 50:50. Each used 512 fresh, disjoint training-derived replication sequences. These are development WP differences, not official competition scores. No new official search, protected evaluation, promotion, leaderboard evaluation or submission was run.','','The initial trainer failed because a selection-loop baseline variable overwrote its fitting baseline. The failed artifacts were preserved, the baseline references corrected, and an end-to-end regression with unequal fitting and selection lengths added. This was an infrastructure failure, separate from the scientific negative results.','','The designated 4,096-sequence pool now has no never-observed replication groups remaining. Further seed, strength or allocation tuning on these outcomes would be adaptive. The next justified step is to obtain an independently reserved training-derived sequence pool and test a materially different representation or data-coverage hypothesis with a frozen protocol. Sequence count alone is not a demonstrated improvement for this model; this does not establish that diversity is ineffective for every architecture.','',f'Regression checks: Windows {tests["windows"]["passed"]} passed; Linux {tests["linux"]["passed"]} passed. Immutable protocols, model identities, per-sequence moments, source snapshots and evidence hashes accompany this report.']
    (c.OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.event('campaign_checkpointed',json_sha256=c.sha(c.OUT/'FINAL_REPORT.json'),markdown_sha256=c.sha(c.OUT/'FINAL_REPORT.md'))
    verify()

def verify():
    r=json.loads((c.OUT/'FINAL_REPORT.json').read_text())
    for p,h in r['evidence_sha256'].items(): assert c.sha(c.ROOT/p)==h,p
    e=verify_ledger(c.OUT/'ledger.jsonl');verify_ledger(b.OUT/'ledger.jsonl')
    assert e[-1]['json_sha256']==c.sha(c.OUT/'FINAL_REPORT.json')
    assert e[-1]['markdown_sha256']==c.sha(c.OUT/'FINAL_REPORT.md')
    assert e[-1]['previous_ledger_sha256']==r['ledger_prefix_sha256']
    assert not any(x['kind']=='official_search_started' for x in e)
    assert c.sha(c.previous.COMBO)==c.previous.COMBO_SHA
    a=json.loads((c.OUT/'protocol.json').read_text())['groups'];z=json.loads((b.OUT/'protocol.json').read_text())['groups']
    assert not set(a['replication'])&set(z['replication'])
    assert not set(z['replication'])&(set(a['original_fit'])|set(a['extra_fit'])|set(a['selection']))
    assert len(set(z['fit'])&set(a['original_fit']))==512
    assert len(set(z['fit'])&set(a['extra_fit']))==512
    c.load_models();b.load_models()
    assert c.sha(c.OUT/'fit/identity.json')==json.loads((b.OUT/'protocol.json').read_text())['broad_identity_sha256']
    print(json.dumps({'verified_hashes':len(r['evidence_sha256']),'ledger_events':len(e),'incumbent_unchanged':True,'fresh_replication_disjoint':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tests','finalize','verify'],required=True);args=p.parse_args()
    {'tests':tests,'finalize':finalize,'verify':verify}[args.phase]()
