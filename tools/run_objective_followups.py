"""Serialize authorized search-only scientific and callback follow-ups."""
import json
import subprocess
import sys
import time
from competition_engineering import objective_alignment as oa


def main():
    config=json.loads((oa.output_directory()/'campaign.json').read_text())
    stages=[('tools.review_objective_alignment',['--tier','4096','--target','1']),
            ('tools.check_objective_candidates',['--target','1']),
            ('tools.check_objective_candidates',['--tier','4096','--target','1']),
            ('competition_engineering.group_objective_alignment',['--target','0']),
            ('tools.check_objective_candidates',['--group','--target','0']),
            ('competition_engineering.group_objective_alignment',['--target','1']),
            ('tools.check_objective_candidates',['--group','--target','1'])]
    for module,args in stages:
        if time.time()>=config['started_unix']+config['budget_seconds']:
            oa.event('resource_window_exhausted',next_stage=[module,*args]); return
        if '--group' in args:
            target=int(args[args.index('--target')+1])
            directory=oa.ROOT/('competition_engineering/runs/group_objective_t'+str(target)+'_20261002')
            if not (directory/'diagnosis.json').exists():
                print('No group fit justified; callback stage skipped',flush=True); continue
        print(json.dumps({'stage':module,'args':args}),flush=True)
        result=subprocess.run([sys.executable,'-m',module,*args])
        if result.returncode:
            oa.event('followup_infrastructure_failure',stage=[module,*args],exit_code=result.returncode,
                     interpretation='No scientific outcome assigned by launcher; diagnose and repair the preserved failure')
            raise RuntimeError('Follow-up stage requires infrastructure repair')


if __name__=='__main__':
    main()
