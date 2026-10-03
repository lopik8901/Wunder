"""Prospective test of observable sequence context and WP-gradient transport."""
from competition_engineering import alternative_representation as prior
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from competition_engineering.pipeline import read_sequence,write_json,cached
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.probability_readout import probability_focus
from competition_engineering.frozen_latent_readout import LatentExtractor

ROOT=prior.ROOT;OUT=ROOT/'competition_engineering/runs/causal_context_20261003';sha=prior.sha

def event(kind,**payload):
    OUT.mkdir(exist_ok=True);p=OUT/'ledger.jsonl'
    from datetime import datetime,timezone
    row={'utc':datetime.now(timezone.utc).isoformat(),'boundary':'search-only','kind':kind,'previous_ledger_sha256':sha(p) if p.exists() else None,**payload}
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def prefix_context(x,projection):
    if len(x)<512: raise ValueError('Context requires512 past/current rows')
    p=x[:512]@projection
    return np.concatenate((p.mean(0),np.log(np.maximum(p.std(0),.05))))

def gradient(reference,moments):
    mass,sy,sp,syy,spp,syp=reference
    my=sy/mass;mp=sp/mass;vy=syy-sy*my;vp=spp-sp*mp;cov=syp-sy*mp
    return (moments[1]-my*moments[0]-(cov/vp)*(moments[2]-mp*moments[0]))/np.sqrt(vy*vp)

def reserve():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    previous=json.loads((prior.OUT/'reservation.json').read_text());decoder=json.loads((prior.OUT/'disjoint_readout/protocol.json').read_text());raw=json.loads((prior.OUT/'raw_target/protocol.json').read_text())
    excluded=set(previous['excluded_groups'])|set(sum(previous['roles'].values(),[]))|set(decoder['groups']['readout_fit'])|set(sum(raw['groups'].values(),[]))
    pf=pq.ParquetFile(prior.DATA);fresh=np.random.default_rng(20261070).permutation(sorted(set(range(pf.num_row_groups))-excluded)).tolist();assert len(fresh)>=1152
    roles={'fit':fresh[:512],'development':fresh[512:640],'replication_1':fresh[640:896],'replication_2':fresh[896:1152]}
    identities=[{'group':g,'seq_ix':prior.sequence_identity(pf,g),'role':role} for role,groups in roles.items() for g in groups];assert len({v['seq_ix'] for v in identities})==1152
    r={'observation':'Five alternative representation stages failed; temporal fitting greatly exceeded transfer. Residual mapping may vary across sequences in ways that are either observable from inputs or not identifiable.',
       'hypothesis':'Feature-only context from the first512rows predicts stable sequence-specific official-WP gradient contributions on later rows.',
       'mechanism':'Freeze a64-dimensional prefix descriptor of mean/log-standard-deviation of32 fixed projected inputs. Fit a ridge linear predictor of33-dimensional later-row WP gradients; compare the fitted mean-gradient predictor on disjoint sequences.',
       'experiment':'New512 fitting/128 development sequences; sample500 required rows after index512. Uniform and frozen probability-weighted populations. Same gradient target treatment/reference, fixed ridge0.1, no context/window/feature sweep.',
       'criterion':'Development predictive reduction in gradient squared error positive for both targets/populations with sequence95 lower>0; both64-sequence halves positive. A qualifying context predictor is frozen and checked once on replication_1 under the same criterion. No new candidate or official-search access based on diagnostic alone.',
       'scope':'Gradient predictability is a diagnostic, not proof of a better score or irreducible target noise. Gradient reference moments and context normalization estimated only on fitting data. Context uses no labels and is causal for every evaluated row. No position gating or EMA retuning.',
       'roles':roles,'identities':identities,'excluded_groups':sorted(excluded),'untouched_prior_block_preserved':raw['groups']['future_reserved'],
       'followup':'Only replicated predictable gradients justify a bounded conditional-readout experiment. Replication_2 remains reserved for a separately frozen modeling hypothesis.',
       'incumbent_sha256':prior.COMBO_SHA}
    write_json(path,r)
    with np.load(prior.OUT/'basis.npz') as q:np.savez(OUT/'projection.npz',projection=q['projection'])
    k=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==20;k['version']=21
    k['status']='New causal context diagnostic; prior alternative-representation checkpoint remains closed.'
    k['next_research_question']='Do label-free causal contexts predict stable WP-gradient heterogeneity across independently reserved sequences?'
    write_json(OUT/'knowledge_v21.json',k);write_json(ROOT/'competition_engineering/search_knowledge.json',k)
    event('experiment_preregistered',protocol_sha256=sha(path),knowledge_sha256=sha(OUT/'knowledge_v21.json'));return r

def prepare():
    r=reserve();pf=pq.ParquetFile(prior.DATA);combo=prior.load_combo();extractor=LatentExtractor()
    with np.load(OUT/'projection.npz') as q: projection=q['projection']
    teacher=ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q:trees={k:q[k] for k in q.files}
    records=[]
    for role in ['fit','development']:
        directory=OUT/role;directory.mkdir(exist_ok=True)
        for count,group in enumerate(r['roles'][role]):
            path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                row=json.loads(meta.read_text());assert sha(path)==row['derived_sha256'];records.append(row);continue
            seq,need,x,y,mask=read_sequence(pf,group);p,_=extractor.predict(x);z={'x':x,'y':y,'p':p,'step':np.arange(len(x)),'need':need};base=prior.predict_combo(z,combo)
            nx=np.clip((x-combo['mean'])/combo['scale'],-8,8).astype(np.float32);context=prefix_context(nx,projection)
            eligible=np.flatnonzero(need&(np.arange(len(x))>=512));idx=np.sort(np.random.default_rng(20261071+group).choice(eligible,500,replace=False))
            f=np.column_stack((np.ones(500),nx[idx]@projection)).astype(float);yy=np.clip(y[idx],-2,2).astype(float);pp=np.clip(base[idx],-2,2).astype(float);active=np.abs(base[idx])<2
            focus=probability_focus(x[idx],base[idx],z['step'][idx],combo,trees)[:,1];stats=[];moments=[];halves=[]
            for pop in [np.ones(500),focus]:
                w=np.abs(yy)*pop[:,None];stats.append(focus_stats(y[idx],base[idx],pop));a=w*active
                moments.append(np.stack((f.T@a,f.T@(a*yy),f.T@(a*pp))))
                half=[]
                for ids in [np.arange(0,500,2),np.arange(1,500,2)]:half.append(np.stack((f[ids].T@a[ids],f[ids].T@(a[ids]*yy[ids]),f[ids].T@(a[ids]*pp[ids]))))
                halves.append(half)
            np.savez(path,context=context,stats=stats,moments=moments,halves=halves,group=group,seq_ix=seq,indices=idx)
            row={'group':group,'seq_ix':seq,'role':role,'source_sequence_sha256':prior.hashlib.sha256(x.tobytes()+y.tobytes()+need.tobytes()).hexdigest(),'derived_sha256':sha(path)};write_json(meta,row);records.append(row)
            if (count+1)%128==0:print(role+' prepared '+str(count+1),flush=True)
        write_json(directory/'identity.json',{'groups':r['roles'][role]})
    path=OUT/'prepared_manifest.json'
    if not path.exists():write_json(path,{'records':records});event('data_prepared',manifest_sha256=sha(path))

def diagnose():
    path=OUT/'development_diagnostic.json'
    if path.exists(): print(path.read_text());return
    for row in json.loads((OUT/'prepared_manifest.json').read_text())['records']:
        assert sha(OUT/row['role']/f"{row['group']:05d}.npz")==row['derived_sha256']
    fit=[z for _,z in cached(OUT/'fit')];dev=[z for _,z in cached(OUT/'development')]
    reference=np.sum([z['stats'] for z in fit],axis=0);context=np.array([z['context'] for z in fit]);mean=context.mean(0);scale=np.maximum(context.std(0),.05)
    f=np.column_stack((np.ones(512),np.clip((context-mean)/scale,-8,8)));df=np.column_stack((np.ones(128),np.clip((np.array([z['context'] for z in dev])-mean)/scale,-8,8)))
    targets=np.array([[gradient(reference[p],z['moments'][p])*512 for p in [0,1]] for z in fit]);truth=np.array([[gradient(reference[p],z['moments'][p])*512 for p in [0,1]] for z in dev])
    center=targets.mean(0);coef=np.linalg.solve(f.T@f+np.eye(65)*512*.1,f.T@(targets-center).reshape(512,-1));prediction=(df@coef).reshape(128,2,33,2)+center
    constant_error=((truth-center)**2).sum(2);context_error=((truth-prediction)**2).sum(2);difference=constant_error-context_error
    counts=bootstrap_counts(128,np.random.default_rng(20261072),draws=10000);draw=(counts@difference.reshape(128,-1)).reshape(10000,2,2)/(counts@constant_error.reshape(128,-1)).reshape(10000,2,2)
    point=difference.sum(0)/constant_error.sum(0);ci=np.quantile(draw,[.025,.975],axis=0);halves=[difference[:64].sum(0)/constant_error[:64].sum(0),difference[64:].sum(0)/constant_error[64:].sum(0)]
    split=np.array([[[gradient(reference[p],z['halves'][p,h])*1024 for h in [0,1]] for p in [0,1]] for z in fit])
    reliability=((split[:,:,0]-center)*(split[:,:,1]-center)).sum((0,2))/np.sqrt(((split[:,:,0]-center)**2).sum((0,2))*((split[:,:,1]-center)**2).sum((0,2)))
    gate=bool(np.all(point>0) and np.all(ci[0]>0) and np.all(np.array(halves)>0))
    np.savez(OUT/'frozen_context_predictor.npz',mean=mean,scale=scale,reference=reference,center=center,coef=coef)
    report={'gradient_error_reduction_population_target':point.tolist(),'paired95':ci.tolist(),'development_halves':np.array(halves).tolist(),'interleaved_gradient_reliability':reliability.tolist(),'development_gate_passed':gate,'model_sha256':sha(OUT/'frozen_context_predictor.npz'),'interpretation':'Interleaved half reliability is descriptive and may include temporal dependence. This does not estimate irreducible target noise.','decision':'Proceed to frozen diagnostic replication' if gate else 'Abandon prefix-context gradient predictor; no modeling/official-search experiment justified.'}
    write_json(path,report);event('development_diagnostic_completed',gate=gate,report_sha256=sha(path));print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reserve','prepare','diagnose'],required=True);a=p.parse_args()
    with threadpool_limits(limits=1):{'reserve':reserve,'prepare':prepare,'diagnose':diagnose}[a.phase]()
