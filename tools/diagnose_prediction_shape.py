"""Post-failure explanation using consumed fitting data only; no refitting."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from competition_engineering import prediction_shape_campaign as c
from competition_engineering.pipeline import write_json

def run():
    path=c.OUT/'shape_generalization_diagnostic.json'
    if path.exists():return
    groups=sorted(json.loads((c.OUT/'protocol.json').read_text())['roles']['fit']);results=[]
    for fold in [0,1]:
        with np.load(c.OUT/'shape_diagnostic_work/artifacts'/('crossfit_'+str(fold)+'.npz')) as q:model={k:q[k] for k in q.files}
        populations={}
        for role,ids in [('fitting',groups[fold*192:(fold+1)*192]),('crossfit',groups[(1-fold)*192:(2-fold)*192])]:
            stats=np.zeros((3,2,6,2));mass=np.zeros((2,17,2));mse=np.zeros((3,2,2))
            for group in ids:
                with np.load(c.OUT/'training'/f'{group:05d}.npz') as q:z={k:q[k] for k in q.files}
                curve=c.curve_predict(z['base'],model['knots'],model['values'],[1,1]);affine=c.curve_predict(z['base'],model['knots'],model['affine'],[1,1]);yy=np.clip(z['y'],-2,2)
                for pop,w in enumerate([np.ones(len(curve)),z['focus']]):
                    weights=np.abs(yy)*w[:,None]
                    for j,p in enumerate([z['base'],curve,affine]):stats[j,pop]+=c.focus_stats(z['y'],p,w);mse[j,pop]+=(weights*(yy-np.clip(p,-2,2))**2).sum(0)
                    for t in [0,1]:mass[pop,:,t]+=c.interpolation_basis(z['base'][:,t],model['knots']).T@weights[:,t]
            scores=c.pooled_correlations(stats);populations[role]={'curve_minus_incumbent_target':(scores[1]-scores[0]).tolist(),'curve_minus_affine_combined':(scores[1]-scores[2]).mean(-1).tolist(),'weighted_clipped_mse':(mse/stats[0,:,0]).tolist(),'knot_weight_fraction':(mass/mass.sum(1)[:,None]).tolist()}
        results.append(populations)
    write_json(path,{'folds':results,'scope':'Diagnostic on already consumed fitting rows only; no fits, selector changes, development or replication access. Fitting gains are not independent evidence.','decision':'The failed prospectively registered crossfit gate remains binding.'});c.event('shape_failure_diagnosed',report_sha256=c.sha(path));print(json.dumps({'folds':[{k:v['curve_minus_affine_combined'] for k,v in r.items()} for r in results]}))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
