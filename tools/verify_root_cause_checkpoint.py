"""Read-only integrity check for the completed search-only continuation."""
import hashlib
import json
from competition_engineering import root_cause_campaign as campaign
from competition_engineering.manual_search_core import COMBO,COMBO_SHA
from connectome.research_retrieval import load_library

def main():
    out=campaign.OUT;report=json.loads((out/'FINAL_REPORT.json').read_text())
    for relative,expected in report['reproducibility_sha256'].items():
        assert campaign.sha(campaign.ROOT/relative)==expected,relative
    assert campaign.sha(COMBO)==COMBO_SHA
    assert load_library(campaign.ROOT/'competition_engineering/research_cards/v9')[1]==report['library_manifest_sha256']
    snapshot=json.loads((out/'knowledge_v4.json').read_text())
    assert snapshot['incumbent']['sha256']==COMBO_SHA and snapshot['version']==4
    prefix=b'';count=0;before_last=None;last=None
    for line in (out/'ledger.jsonl').read_bytes().splitlines(keepends=True):
        row=json.loads(line);expected=hashlib.sha256(prefix).hexdigest() if prefix else None
        assert row['previous_ledger_sha256']==expected,count
        before_last=expected;prefix+=line;last=row;count+=1
    assert last['kind']=='campaign_checkpointed'
    assert before_last==report['ledger_prefix_sha256']
    assert last['review_sha256']==campaign.sha(out/'FINAL_REPORT.json')
    assert last['report_sha256']==campaign.sha(out/'FINAL_REPORT.md')
    assert report['tests']['passed']==67 and not report['awaiting_promotion']
    print(json.dumps({'evidence_hashes_verified':len(report['reproducibility_sha256']),
        'ledger_events_verified':count,'incumbent_unchanged':True,'tests':67,'awaiting':[]}))

if __name__=='__main__':
    main()
