"""Durable regression records and read-only verification for this campaign."""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from competition_engineering import nonlinear_state_campaign as campaign
from competition_engineering.pipeline import write_json
from connectome.research_retrieval import load_library

TESTS=['test_nonlinear_state_campaign.py','test_representation_campaign.py','test_frozen_latent_readout.py',
    'test_root_cause_campaign.py','test_reassessment_campaign.py','test_mlevolve_search_history.py',
    'test_research_foundation.py','test_research_retrieval.py','test_mlevolve_search_boundary.py',
    'test_campaign_onnx.py','test_campaign_worker_failure.py']

def tests():
    command=[sys.executable,'-m','pytest',*['tests/'+name for name in TESTS],'-q','-p','no:cacheprovider']
    result=subprocess.run(command,cwd=campaign.ROOT,capture_output=True,text=True)
    text=result.stdout+'\n'+result.stderr;system=platform.system().lower()
    (campaign.OUT/('regression_'+system+'.log')).write_text(text,encoding='utf-8')
    match=re.search(r'(\d+) passed',text)
    if result.returncode or not match: raise RuntimeError(text[-4000:])
    skipped=re.search(r'(\d+) skipped',text)
    report={'platform':platform.platform(),'python':sys.version,'passed':int(match[1]),'skipped':int(skipped[1]) if skipped else 0,
        'files':TESTS,'log_sha256':campaign.sha(campaign.OUT/('regression_'+system+'.log'))}
    write_json(campaign.OUT/('tests_'+system+'.json'),report);print(json.dumps(report))

def verify_ledger(path):
    prefix=b'';events=[]
    for line in path.read_bytes().splitlines(keepends=True):
        row=json.loads(line);assert row['previous_ledger_sha256']==(hashlib.sha256(prefix).hexdigest() if prefix else None)
        assert row['boundary']=='search-only';events.append(row);prefix+=line
    return events

def verify():
    out=campaign.OUT;report=json.loads((out/'FINAL_REPORT.json').read_text())
    for path,expected in report['reproducibility_sha256'].items(): assert campaign.sha(campaign.ROOT/path)==expected,path
    assert campaign.sha(campaign.COMBO)==campaign.COMBO_SHA
    assert load_library(campaign.ROOT/'competition_engineering/research_cards/v12')[1]==report['library_manifest_sha256']
    events=verify_ledger(out/'ledger.jsonl');last=events[-1]
    assert last['kind']=='campaign_checkpointed'
    assert last['json_sha256']==campaign.sha(out/'FINAL_REPORT.json')
    assert last['markdown_sha256']==campaign.sha(out/'FINAL_REPORT.md')
    assert last['previous_ledger_sha256']==report['ledger_prefix_sha256']
    # The immutable snapshot remains verifiable when subsequent work advances live knowledge.
    knowledge=json.loads((out/'knowledge_v11.json').read_text());assert knowledge['incumbent']['sha256']==campaign.COMBO_SHA
    initial=json.loads((out/'protocol.json').read_text());roles=[set(initial['excluded_groups'])]
    roles.extend(set(g) for g in initial['replication_groups'].values())
    roles.append(set(json.loads((out/'state_trees/protocol.json').read_text())['replication_groups']))
    roles.append(set(json.loads((out/'state_innovation/protocol.json').read_text())['groups']['replication']))
    for i,role in enumerate(roles):
        for other in roles[i+1:]: assert not role&other
    assert len([e for e in events if e['kind']=='official_search_started'])==report['official_search_passes']
    assert report['tests']['linux']['passed']>0 and report['tests']['windows']['passed']>0
    print(json.dumps({'evidence_hashes_verified':len(report['reproducibility_sha256']),'ledger_events_verified':len(events),
        'fresh_replication_groups_disjoint':True,'incumbent_unchanged':True,'official_search_passes':report['official_search_passes']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['tests','verify'],required=True)
    args=parser.parse_args();{'tests':tests,'verify':verify}[args.phase]()
