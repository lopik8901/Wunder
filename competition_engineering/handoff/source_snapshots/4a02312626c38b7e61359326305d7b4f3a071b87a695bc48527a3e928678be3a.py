"""Frozen representation versus readout leakage diagnostic using disjoint sequences."""
from competition_engineering import alternative_representation as c
from competition_engineering import learned_causal_campaign as l
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from competition_engineering.pipeline import read_sequence,write_json,cached
from competition_engineering.sparse_temporal import endpoint_windows
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.replicated_readout import grid_stats,choose_strengths
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations
from connectome.mlevolve_generated import validate_source

OUT=c.OUT/'disjoint_readout'

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    assert not json.loads((l.OUT/'replication.json').read_text())['search_gate_passed']
    diagnostic=json.loads((l.OUT/'generalization_diagnostic.json').read_text())
    previous=c.reserve();excluded=set(previous['excluded_groups'])|set(sum(previous['roles'].values(),[]))
    pf=pq.ParquetFile(c.DATA);groups=np.random.default_rng(20261060).permutation(sorted(set(range(pf.num_row_groups))-excluded))[:512].tolist()
    assert len(groups)==512 and not set(groups)&excluded
    identities=[{'group':g,'seq_ix':c.sequence_identity(pf,g)} for g in groups];assert len({r['seq_ix'] for r in identities})==512
    r={'observation':'Frozen selected temporal models show unshrunk fitting proxy gains~0.08-0.09 but development losses~0.054-0.056; learned-state raw-target decoding also loses transfer.',
       'hypothesis':'An independently fitted low-capacity readout can recover transferable signal from the existing learned temporal encoder despite its overfit jointly trained head.',
       'mechanism':'Freeze all learned encoders/checkpoints from experiment3. Only readout fitting changes to512 newly reserved sequences disjoint from encoder fitting/development and every prior replication block.',
       'groups':{'readout_fit':groups,'development':previous['roles']['development'],'replication':previous['roles']['replication_4']},'identities':identities,'excluded_groups':sorted(excluded),
       'experiment':'Same paired128-row samplers,65536rows/readout; mean-one clipped-target absolute-weight ridge0.1. Per-target same243-feature current115+representation128. Compare frozen learned temporal, learned current-only, GRU and fixed random TCN. Same WP strength selection, no body/architecture tuning.',
       'primary':'temporal_r0; repeat temporal_r1, no best seed.',
       'success':'Both temporal readout fits improve both populations; primary probability95 lower>0 and both128-sequence halves positive. Temporal-minus-current and temporal-minus-GRU probability contrast positive in both paired fits and average95 lower>0.',
       'failure':'Failure closes this encoder/readout rescue hypothesis and blocks official-search access. This does not reject every learned convolution architecture.',
       'replication':'Fourth and final pre-reserved block consumed once. Development reuse is disclosed; fresh replication is not used to tune anything.',
       'target_treatment':'Use each target-specific encoder checkpoint already frozen by experiment3 WP selection. Both targets use equal128-state readout dimensions; no checkpoint changes.',
       'diagnostic_sha256':c.sha(l.OUT/'generalization_diagnostic.json'),'encoder_identity_sha256':c.sha(l.OUT/'fit/identity.json')}
    write_json(path,r)
    k=json.loads((c.ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==17;k['version']=18
    k['weakened'].append({'hypothesis':'Learned86-row temporal corrections generalize better than a matched static model.','reason':'Strong train/development gap; mean temporal-minus-static proxy95 interval[-0.000383,-0.000005].','source':'runs/alternative_representation_20261003/learned_causal/replication.json'})
    k['status']='Learned temporal branch negative; disjoint readout diagnostic prospectively frozen.'
    write_json(c.OUT/'knowledge_v18.json',k);write_json(c.ROOT/'competition_engineering/search_knowledge.json',k)
    c.event('research_decision',protocol_sha256=c.sha(path),knowledge_sha256=c.sha(c.OUT/'knowledge_v18.json'));return r

def prepare():
    r=initialize();pf=pq.ParquetFile(c.DATA);combo=c.load_combo();extractor=c.LatentExtractor()
    with np.load(c.OUT/'basis.npz') as q: basis={k:q[k] for k in q.files}
    with np.load(l.OUT/'fit/artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    assert all(c.sha(l.OUT/'fit'/p)==h for p,h in json.loads((l.OUT/'fit/identity.json').read_text()).items())
    teacher=c.ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    records=[]
    for role,groups in r['groups'].items():
        directory=OUT/('replication' if role=='replication' else 'training');directory.mkdir(exist_ok=True)
        for count,group in enumerate(groups):
            path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                row=json.loads(meta.read_text());assert c.sha(path)==row['derived_sha256'];records.append(row);continue
            seq,need,x,y,mask=read_sequence(pf,group);nx=np.clip((x-combo['mean'])/combo['scale'],-8,8).astype(np.float32)
            if role=='development':
                with np.load(c.OUT/'training'/f'{group:05d}.npz') as q: z={k:q[k] for k in q.files if k in ['core','gru','tcn','base','y','focus','indices','group','seq_ix']}
                idx=z['indices'];assert int(z['seq_ix'])==seq
            else:
                p,hidden=extractor.predict(x);full={'x':x,'y':y,'p':p,'step':np.arange(len(x)),'need':need};base_full=c.predict_combo(full,combo)
                idx=np.sort(np.random.default_rng(20261061+group).choice(np.flatnonzero(need),500,replace=False));base=base_full[idx]
                _,random_tcn=c.temporal_features(nx,basis)
                z={'core':np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),nx[idx])).astype(np.float32),'gru':hidden[idx]@basis['gru'],'tcn':random_tcn[idx],'base':base,'y':y[idx],'focus':c.probability_focus(x[idx],base,full['step'][idx],combo,trees)[:,1],'indices':idx,'group':group,'seq_ix':seq}
            windows,valid=endpoint_windows(nx,idx)
            for replica in [0,1]:
                for family in ['current','temporal']:
                    states=[]
                    for target in [0,1]:
                        suffix=family+'_r'+str(replica)+'_t'+str(target)
                        with torch.no_grad(): states.append(l.tensor_features(torch.from_numpy(windows),torch.from_numpy(valid),*[torch.from_numpy(model[suffix+'_'+v]) for v in ['projection','weights','bias']],family=='current').numpy())
                    z[family+'_r'+str(replica)]=np.stack(states,axis=1)
            z['role']=0 if role=='readout_fit' else 1 if role=='development' else 2
            np.savez(path,**z);row={'group':group,'seq_ix':seq,'role':role,'source_sequence_sha256':c.hashlib.sha256(x.tobytes()+y.tobytes()+need.tobytes()).hexdigest(),'derived_sha256':c.sha(path)};write_json(meta,row);records.append(row)
            if (count+1)%128==0: print(role+' prepared '+str(count+1),flush=True)
        write_json(directory/'identity.json',{'groups':r['groups']['readout_fit']+r['groups']['development'] if role!='replication' else groups})
    path=OUT/'prepared_manifest.json'
    if not path.exists(): write_json(path,{'records':records});c.event('disjoint_readout_data_prepared',manifest_sha256=c.sha(path))

def features(z,family,replica,target):
    return z[family+'_r'+str(replica)][:,target] if family in ['current','temporal'] else z[family]

def design(z,family,replica,target,mean,scale,standalone=False):
    core=z['core'] if not standalone else np.column_stack((z['core'][:,0],z['core'][:,3:]))
    return np.column_stack((core,np.clip((features(z,family,replica,target)-mean)/scale,-8,8))).astype(np.float32)

def fit(train_files,output_dir):
    models={};report={}
    for replica in [0,1]:
        for family in ['current','temporal','gru','tcn']:
            key=family+'_r'+str(replica);choice_moments=np.zeros((2,6,6,2));coefs=[];stand=[];norm=[]
            for target in [0,1]:
                sx=np.zeros(128);sxx=np.zeros(128);count=0
                for path in train_files:
                    with np.load(path) as z:
                        if int(z['role'])!=0: continue
                        h=features(z,family,replica,target).astype(float);sx+=h.sum(0);sxx+=(h*h).sum(0);count+=len(h)
                mean=sx/count;scale=np.sqrt(np.maximum(sxx/count-mean*mean,.05**2));norm.append((mean,scale));gram=np.zeros((243,243));rhs=np.zeros(243);sg=np.zeros((241,241));sr=np.zeros(241);mass=0.;rows=0
                for path in train_files:
                    with np.load(path) as q: z={k:q[k] for k in q.files}
                    if int(z['role'])!=0: continue
                    idx=np.random.default_rng(20261053+int(z['group'])+replica*100000).permutation(len(z['base']))[:128]
                    f=design(z,family,replica,target,mean,scale)[idx].astype(float);sf=design(z,family,replica,target,mean,scale,True)[idx].astype(float);y=np.clip(z['y'][idx,target],-2,2).astype(float);w=np.abs(y)
                    gram+=f.T@(w[:,None]*f);rhs+=f.T@(w*(y-z['base'][idx,target]));sg+=sf.T@(w[:,None]*sf);sr+=sf.T@(w*y);mass+=w.sum();rows+=len(idx)
                assert rows==65536
                coef=np.linalg.solve(gram/mass*rows+np.eye(243)*rows*.1,rhs/mass*rows);direct=np.linalg.solve(sg/mass*rows+np.eye(241)*rows*.1,sr/mass*rows)
                models[key+'_t'+str(target)+'_mean']=mean;models[key+'_t'+str(target)+'_scale']=scale;models[key+'_t'+str(target)+'_coef']=coef.astype(np.float32);models[key+'_t'+str(target)+'_standalone']=direct.astype(np.float32);coefs.append(coef)
            for path in train_files:
                with np.load(path) as q: z={k:q[k] for k in q.files}
                if int(z['role'])!=1: continue
                correction=np.column_stack([design(z,family,replica,t,*norm[t])@coefs[t] for t in [0,1]])
                for i,pop in enumerate([np.ones(len(z['base'])),z['focus']]): choice_moments[i]+=grid_stats(z['y'],z['base'],correction,pop)
            choice=choose_strengths(choice_moments);models[key+'_strength']=np.array(choice['strengths']);report[key]=choice;print('frozen '+key,flush=True)
    np.savez(output_dir+'/models.npz',**models);Path(output_dir+'/selection.json').write_text(json.dumps(report),encoding='utf-8');return {'models':8,'rows_per_model':65536}

def source():
    code='import numpy as np\nimport json\nfrom pathlib import Path\n'
    for f in [from_stats,focus_stats,grid_stats,choose_strengths,features,design,fit]: code+=inspect.getsource(f)+'\n'
    code+='def train(train_files,output_dir):\n    return fit(train_files,output_dir)\n';validate_source(code,training=True);return code

def train():
    initialize();work=OUT/'fit';work.mkdir(exist_ok=True)
    for row in json.loads((OUT/'prepared_manifest.json').read_text())['records']:
        if row['role']!='replication': assert c.sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    if not (work/'artifacts/models.npz').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');c.event('disjoint_readout_fit_started',source_sha256=c.sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,OUT/'training')
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1);(work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            c.event('implementation_failure',component='disjoint readout',error=stderr.decode(errors='replace')[-1500:]);raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={p:c.sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/selection.json']}
    if (work/'identity.json').exists(): assert json.loads((work/'identity.json').read_text())==identity
    else: write_json(work/'identity.json',identity);c.event('disjoint_readout_models_frozen',identity=identity)

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    work=OUT/'fit';assert all(c.sha(work/p)==h for p,h in json.loads((work/'identity.json').read_text()).items())
    with np.load(work/'artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    moments=[];stand=[];expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_manifest.json').read_text())['records'] if r['role']=='replication'}
    names=[]
    for group,z in cached(OUT/'replication'):
        assert c.sha(OUT/'replication'/f'{group:05d}.npz')==expected[group]
        predictions=[z['base']];standalone=[];names=[]
        for replica in [0,1]:
            for family in ['current','temporal','gru','tcn']:
                key=family+'_r'+str(replica);names.append(key);correction=[];direct=[]
                for t in [0,1]:
                    suffix=key+'_t'+str(t);mean=model[suffix+'_mean'];scale=model[suffix+'_scale']
                    correction.append(design(z,family,replica,t,mean,scale)@model[suffix+'_coef']);direct.append(design(z,family,replica,t,mean,scale,True)@model[suffix+'_standalone'])
                p=z['base']+np.column_stack(correction)*model[key+'_strength'];predictions.append(p);standalone.append(np.column_stack(direct))
        moments.append([[focus_stats(z['y'],p,pop) for pop in [np.ones(len(z['base'])),z['focus']]] for p in predictions]);stand.append([focus_stats(z['y'],p,np.ones(len(z['base']))) for p in standalone])
    moments=np.array(moments);stand=np.array(stand);np.savez(OUT/'replication_moments.npz',moments=moments,standalone=stand)
    counts=bootstrap_counts(256,np.random.default_rng(20261062),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0));delta=boot-boot[:,:1];point=scores-scores[:1]
    variants={name:{'combined_delta':point[j].mean(-1).tolist(),'targets_uniform_probability':point[j].tolist(),'paired95':np.quantile(delta[:,j].mean(-1),[.025,.975],axis=0).tolist(),'standalone_uniform_wp':pooled_correlations(stand[:,j-1].sum(0)).tolist()} for j,name in enumerate(names,1)}
    contrasts={};gate=all(np.all(point[j].mean(-1)>0) for j in [2,6])
    for control,k in [('current',1),('gru',3)]:
        for r in [0,1]:
            dist=delta[:,2+r*4]-delta[:,k+r*4];v=point[2+r*4]-point[k+r*4];contrasts['temporal_minus_'+control+'_r'+str(r)]={'combined_delta':v.mean(-1).tolist(),'paired95':np.quantile(dist.mean(-1),[.025,.975],axis=0).tolist()};gate=gate and v.mean(-1)[1]>0
        average=((delta[:,2]-delta[:,k])+(delta[:,6]-delta[:,k+4]))*.5;gate=gate and np.quantile(average.mean(-1),.025,axis=0)[1]>0
    halves=[(pooled_correlations(x.sum(0))[2]-pooled_correlations(x.sum(0))[0]).mean(-1).tolist() for x in [moments[:128],moments[128:]]];gate=gate and np.all(np.array(halves)>0) and variants['temporal_r0']['paired95'][0][1]>0
    result={'variants':variants,'contrasts':contrasts,'primary_halves':halves,'search_gate_passed':bool(gate),'baseline_wp_uniform_probability':scores[0].tolist(),'replication_sequences':256,'decision':'Qualified for one frozen search comparison' if gate else 'Disjoint linear decoding did not rescue the learned temporal representation; no official search.'}
    write_json(path,result);c.event('disjoint_readout_replication_completed',report_sha256=c.sha(path),gate=bool(gate),block='replication_4');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','prepare','train','replicate'],required=True);args=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1): {'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
