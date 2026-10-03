"""Compare two frozen search-only campaign candidates on aligned sequences."""
import argparse
import json

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.autonomous_campaign import CAMPAIGN, sha, event
from competition_engineering.manual_search_core import SEARCH, load_combo, predict_combo
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import sufficient, from_stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('left')
    p.add_argument('right')
    args=p.parse_args()
    out=CAMPAIGN/'comparisons'/f'{args.right}_vs_{args.left}.json'
    if out.exists():
        raise ValueError('Comparison already frozen')
    records=[json.loads((CAMPAIGN/name/'report.json').read_text()) for name in [args.left,args.right]]
    combo=load_combo()
    moments=[]
    with threadpool_limits(limits=1):
        for group,z in cached(SEARCH):
            base=predict_combo(z,combo)
            pair=[]
            for name,report in zip([args.left,args.right],records):
                with np.load(CAMPAIGN/name/'search_predictions'/f'{group:05d}.npz') as q:
                    full=q['prediction']
                prediction=base.copy()
                prediction[:,0]+=report['chosen_strength']*(full[:,0]-base[:,0])
                pair.append(sufficient(z['y'][z['mask']],prediction[z['mask']]))
            moments.append(pair)
    moments=np.asarray(moments)
    sums=moments.sum(0)
    point=from_stats(sums[1])-from_stats(sums[0])
    rng=np.random.default_rng(20261001)
    deltas=[]
    for _ in range(10000):
        sums=moments[rng.integers(len(moments),size=len(moments))].sum(0)
        deltas.append(from_stats(sums[1])-from_stats(sums[0]))
    distribution=np.asarray(deltas).mean(1)
    result={'left':args.left,'right':args.right,'delta_combined':float(point.mean()),
            'delta_per_target':point.tolist(),'paired_95ci':np.quantile(distribution,[.025,.975]).tolist(),
            'paired_99ci':np.quantile(distribution,[.005,.995]).tolist(),
            'report_sha256':[sha(CAMPAIGN/name/'report.json') for name in [args.left,args.right]],
            'evidence':'reused_search_evidence_not_independent_validation'}
    write_json(out,result)
    event('paired_mechanism_comparison',**result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
