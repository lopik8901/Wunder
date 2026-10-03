"""Check whether deterministic row thinning changes train objective direction."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from competition_engineering import objective_alignment as oa
from competition_engineering.manual_search_core import TRAIN_1024, load_combo, predict_combo
from competition_engineering.pipeline import cached, write_json
from competition_engineering.residual import from_stats
from tools.diagnose_causal_scoring_population import stats_and_gradient


def main():
    path=oa.OUT/'training_thinning_diagnosis.json'
    if path.exists():
        return
    combo=load_combo(); phases=['full',0,10,20,30]
    totals={k:np.zeros((6,2)) for k in phases};features={k:np.zeros((3,115,2)) for k in phases}
    counts={k:0 for k in phases}
    oa.event('hypothesis_planned',id='deterministic_training_thinning_diagnostic',
        question='Does stride40 training substantially alter the full-training clipped-WP gradient?',
        rule='Consider a full-moment replication only if phase/full cosine falls below0.9 or relative gradient error exceeds0.2',
        boundary='Fixed training1024 only; no fit and no search selection')
    with threadpool_limits(limits=1):
        for number,(_,z) in enumerate(cached(TRAIN_1024)):
            indices=np.flatnonzero(z['need']);base=predict_combo(z,combo)
            for phase in phases:
                idx=indices if phase=='full' else indices[phase::40]
                p=np.clip(base[idx],-2,2).astype(np.float64); y=np.clip(z['y'][idx],-2,2).astype(np.float64)
                w=np.abs(y);active=np.abs(base[idx])<2
                f=np.column_stack((np.ones(len(idx)),p,np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8)))
                totals[phase]+=np.stack((w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)))
                features[phase]+=np.stack((f.T@(w*active),f.T@(w*active*y),f.T@(w*active*p)))
                counts[phase]+=len(idx)
            if (number+1)%256==0:
                print(json.dumps({'phase':'training_full_gradient','sequences':number+1}),flush=True)
    gradients={k:stats_and_gradient(totals[k],features[k]) for k in phases}
    full=gradients['full'];rows=[]
    for phase in phases[1:]:
        g=gradients[phase]
        cosine=(g*full).sum(0)/(np.linalg.norm(g,axis=0)*np.linalg.norm(full,axis=0))
        error=np.linalg.norm(g-full,axis=0)/np.linalg.norm(full,axis=0)
        rows.append({'offset':phase,'rows':counts[phase],'wp':from_stats(totals[phase]).tolist(),
                     'cosine_to_full':cosine.tolist(),'relative_gradient_error':error.tolist()})
    eligible=[t for t in range(2) if min(r['cosine_to_full'][t] for r in rows)<.9 or max(r['relative_gradient_error'][t] for r in rows)>.2]
    report={'full_rows':counts['full'],'full_training_wp':from_stats(totals['full']).tolist(),
            'phase_comparisons':rows,'eligible_targets_for_full_moment_replication':eligible,
            'decision':'Replicate exact training moments only for eligible targets; otherwise abandon row thinning as a material explanation.'}
    write_json(path,report);oa.event('training_thinning_diagnostic_completed',**report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
