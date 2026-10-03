"""Read-only closure and artifact identity verification."""
import hashlib
import json
from competition_engineering.reassessment_campaign import OUT,ROOT,sha
from competition_engineering.manual_search_core import COMBO,COMBO_SHA
from connectome.research_retrieval import load_library

def verify_frozen_knowledge(path,incumbent_sha256,incumbent_wp):
    """A closed campaign depends on its hashed snapshot, not the mutable pointer."""
    knowledge=json.loads(path.read_text())
    assert knowledge['version']==3
    assert knowledge['incumbent']=={'sha256':incumbent_sha256,'wp':incumbent_wp}

def main():
    report=json.loads((OUT/'FINAL_REPORT.json').read_text())
    for path,expected in report['reproducibility_sha256'].items():
        assert sha(ROOT/path)==expected,path
    assert sha(COMBO)==COMBO_SHA
    # The loop above already checks the exact knowledge snapshot hash. Later
    # campaigns may advance search_knowledge.json without invalidating this one.
    verify_frozen_knowledge(OUT/'knowledge_v3.json',COMBO_SHA,report['starting_incumbent'])
    assert load_library(ROOT/'competition_engineering/research_cards/v8')[1]==report['library_manifest_sha256']
    prefix=b'';count=0;last=None;before_last=None
    for line in (OUT/'ledger.jsonl').read_bytes().splitlines(keepends=True):
        record=json.loads(line)
        expected=hashlib.sha256(prefix).hexdigest() if prefix else None
        assert record['previous_ledger_sha256']==expected,count
        before_last=expected;prefix+=line;count+=1;last=record
    assert last['kind']=='campaign_ended'
    assert sha(OUT/'FINAL_REPORT.json')==last['review_sha256']
    assert sha(OUT/'FINAL_REPORT.md')==last['report_sha256']
    assert before_last==report['ledger_prefix_sha256']
    print(json.dumps({'evidence_hashes_verified':len(report['reproducibility_sha256']),
        'ledger_events_verified':count,'incumbent_unchanged':True,'diagnostics':report['diagnostic_experiments'],
        'readouts':report['current_row_readout_fits'],'awaiting':report['awaiting_promotion'],'new_cards':len(report['new_papers'])}))

if __name__=='__main__':
    main()
