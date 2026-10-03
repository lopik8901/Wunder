"""Persist model findings before the next mechanistic test."""
import json
from competition_engineering import representation_campaign as campaign
from competition_engineering.pipeline import write_json

def main():
    out=campaign.OUT;path=out/'knowledge_v5.json'
    if path.exists():
        return
    latest=campaign.ROOT/'competition_engineering/search_knowledge.json'
    knowledge=json.loads(latest.read_text());assert knowledge['version']==4
    knowledge['version']=5
    knowledge['previous_snapshot']={'path':'runs/root_cause_20261002/knowledge_v4.json','sha256':campaign.sha(campaign.PREVIOUS/'knowledge_v4.json')}
    knowledge['established'].append({'lesson':'A fitted frozen-GRU-state readout improves training-derived replication beyond current features: fixed0.25 probability-weighted combined gain0.000859, paired99[0.000234,0.001471]. Official-search gain is essentially zero with wide uncertainty. Representation information exists in development populations, but official transfer is not established.',
        'source':'runs/representation_20261002/replicated_readout/replication.json'})
    knowledge['weakened'].append({'hypothesis':'Point-gradient rejection was sufficient evidence that frozen GRU states contain no useful correction information.',
        'reason':'Actual matched fitting recovered replicated development signal; it still did not establish an official-score improvement.'})
    knowledge['status']='Active representation campaign; state probe completed, position-transport diagnostic planned.'
    write_json(path,knowledge);write_json(latest,knowledge)
    write_json(out/'CHECKPOINT.json',{'status':'active','completed':'Frozen-state representation experiment and replication',
        'next':'Test position-pattern transport before a conditional fit','incumbent':knowledge['incumbent'],'knowledge_version':5})
    campaign.event('knowledge_updated',version=5,snapshot_sha256=campaign.sha(path))
    campaign.event('research_decision',id='frozen_state_readout',decision='Useful replicated development signal but no credible search improvement. Investigate the recurring position-specific failure before selecting another model mechanism.')

if __name__=='__main__':
    main()
