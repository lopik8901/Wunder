"""Original-target supervision versus residual-supervised temporal representation."""
from competition_engineering import alternative_representation as c
from competition_engineering import learned_causal_campaign as l
from competition_engineering import disjoint_readout_campaign as d
import argparse
import json
import time
import numpy as np
import torch
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from competition_engineering.sparse_temporal import endpoint_windows
from competition_engineering.pipeline import read_sequence,write_json,cached
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations
from competition_engineering.prospective_selection_diagnostic import focus_stats
from connectome.mlevolve_generated import validate_source
from competition_engineering.isolation_retry import training_with_namespace_retry

OUT=c.OUT/'raw_target'

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    assert not json.loads((d.OUT/'replication.json').read_text())['search_gate_passed']
    old=c.reserve();extra=json.loads((d.OUT/'protocol.json').read_text());excluded=set(old['excluded_groups'])|set(sum(old['roles'].values(),[]))|set(extra['groups']['readout_fit'])
    pf=pq.ParquetFile(c.DATA);groups=np.random.default_rng(20261063).permutation(sorted(set(range(pf.num_row_groups))-excluded))[:512].tolist();assert len(groups)==512
    roles={'replication_5':groups[:256],'future_reserved':groups[256:]};identities=[{'group':g,'seq_ix':c.sequence_identity(pf,g)} for g in groups]
    assert len({r['seq_ix'] for r in identities})==512
    r={'observation':'Residual-trained temporal features failed transfer, including independent linear decoding. Raw-target standalone decoding was much poorer than static or GRU controls. Residual-specific supervision may shape representations around unstable incumbent errors.',
       'hypothesis':'Training the same compact temporal model directly on original clipped targets yields more transferable complementary information than residual-supervised feature learning.',
       'mechanism':'Same learned input112-to32 and four width32 dilated layers; remove incumbent predictions from model inputs. Train raw target prediction, then freeze a simple development-selected target-wise blend with incumbent. Parameter-matched current-only raw-target model isolates temporal representation.',
       'experiment':'Same512 fitting/128 development groups and65536 sampled rows; paired seeds/initialization,20epochs,batch1024,AdamW lr0.002 decay0.001. Mean-one abs(clipped target)-weighted MSE. Only supervision/prediction target changes; no architecture, loss-function or hyperparameter sweep.',
       'selection':'Per-target same checkpoints5,10,20 and strengths0,.05,.1,.25,.5,1, optimizing minimum uniform/probability blend WP. Blend=(1-strength)*incumbent+strength*raw-target model, frozen before replication5.',
       'primary':'temporal_r0; paired temporal_r1 repeats effect; no best seed.',
       'success':'Both temporal blends improve both populations; primary probability95 lower>0 and both128-sequence halves positive. Temporal-minus-current probability positive in both paired fits and average95 lower>0; primary probability point beats the frozen GRU readout benchmark.',
       'failure':'Failed gate blocks official search and closes this compact topology/supervision rescue. No loss tuning on replication.',
       'groups':roles,'identities':identities,'excluded_groups':sorted(excluded),'training_manifest_sha256':c.sha(l.OUT/'prepared_manifest.json'),
       'scope':'New blocks prospectively reserved from training only; frozen incumbent pretraining independence not claimed. Previous development reuse disclosed. Future block stays untouched.'}
    write_json(path,r)
    k=json.loads((c.ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==18;k['version']=19
    k['weakened'].append({'hypothesis':'Disjoint decoding rescues the residual-trained temporal encoder.','reason':'Prospective independent decoder experiment failed its mechanism gate.','source':'runs/alternative_representation_20261003/disjoint_readout/replication.json'})
    k['status']='Residual-supervised representation branch negative; one raw-target supervision counterfactual frozen.'
    write_json(c.OUT/'knowledge_v19.json',k);write_json(c.ROOT/'competition_engineering/search_knowledge.json',k);c.event('research_decision',protocol_sha256=c.sha(path),knowledge_sha256=c.sha(c.OUT/'knowledge_v19.json'));return r

def prepare():
    r=initialize();directory=OUT/'replication';directory.mkdir(exist_ok=True);pf=pq.ParquetFile(c.DATA);combo=c.load_combo();extractor=c.LatentExtractor()
    teacher=c.ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    with np.load(c.OUT/'basis.npz') as q: basis={k:q[k] for k in q.files}
    records=[]
    for count,group in enumerate(r['groups']['replication_5']):
        path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            row=json.loads(meta.read_text());assert c.sha(path)==row['derived_sha256'];records.append(row);continue
        seq,need,x,y,mask=read_sequence(pf,group);p,hidden=extractor.predict(x);z={'x':x,'y':y,'p':p,'step':np.arange(len(x)),'need':need};base_full=c.predict_combo(z,combo)
        idx=np.sort(np.random.default_rng(20261064+group).choice(np.flatnonzero(need),500,replace=False));nx=np.clip((x-combo['mean'])/combo['scale'],-8,8).astype(np.float32);base=base_full[idx];windows,valid=endpoint_windows(nx,idx)
        np.savez(path,core=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),nx[idx])).astype(np.float32),gru=hidden[idx]@basis['gru'],windows=windows,valid=valid,base=base,y=y[idx],focus=c.probability_focus(x[idx],base,z['step'][idx],combo,trees)[:,1],group=group,indices=idx,role=2,seq_ix=seq)
        row={'group':group,'seq_ix':seq,'source_sequence_sha256':c.hashlib.sha256(x.tobytes()+y.tobytes()+need.tobytes()).hexdigest(),'derived_sha256':c.sha(path)};write_json(meta,row);records.append(row)
        if (count+1)%128==0: print('replication5 prepared '+str(count+1),flush=True)
    write_json(directory/'identity.json',{'groups':r['groups']['replication_5']})
    path=OUT/'prepared_manifest.json'
    if not path.exists(): write_json(path,{'records':records});c.event('raw_target_data_prepared',manifest_sha256=c.sha(path))

def source():
    code=l.source()
    replacements={
      "core=torch.from_numpy(arrays['core'])":"core=torch.from_numpy(np.column_stack((arrays['core'][:,0],arrays['core'][:,3:])))",
      "residual=torch.from_numpy((y-arrays['base']).astype(np.float32))":"residual=torch.from_numpy(y.astype(np.float32))",
      "linear=torch.nn.Parameter(torch.zeros(115,2))":"linear=torch.nn.Parameter(torch.zeros(113,2))",
      "correction=z['core']@linear.detach().numpy()+f.numpy()@head.detach().numpy()":"correction=np.column_stack((z['core'][:,0],z['core'][:,3:]))@linear.detach().numpy()+f.numpy()@head.detach().numpy()-z['base']"}
    for old,new in replacements.items(): assert code.count(old)==1,old;code=code.replace(old,new)
    validate_source(code,training=True);return code

def train():
    initialize();work=OUT/'fit';work.mkdir(exist_ok=True)
    for row in json.loads((l.OUT/'prepared_manifest.json').read_text())['records']:
        if row['role']!='replication_3': assert c.sha(l.OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    if not (work/'artifacts/models.npz').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');c.event('raw_target_fit_started',source_sha256=c.sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);deadline=time.monotonic()+1800
        def retry(attempt,stderr):
            (work/('namespace_attempt_'+str(attempt)+'.stderr')).write_bytes(stderr)
            c.event('infrastructure_retry',component='bubblewrap namespace allocation',attempt=attempt,deadline_unchanged=True,isolation_unchanged=True)
        process,status,stdout,stderr=training_with_namespace_retry(lambda:_launch(sandbox,work,'train',deadline,l.OUT/'training'),deadline,retry)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            c.event('implementation_failure',component='raw target fit',error=stderr.decode(errors='replace')[-1500:]);raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={p:c.sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/training.json']}
    if (work/'identity.json').exists(): assert json.loads((work/'identity.json').read_text())==identity
    else: write_json(work/'identity.json',identity);c.event('raw_target_models_frozen',identity=identity)

def direct(z,model,key):
    predictions=[]
    for target in [0,1]:
        suffix=key+'_t'+str(target)
        with torch.no_grad(): h=l.tensor_features(torch.from_numpy(z['windows']),torch.from_numpy(z['valid']),*[torch.from_numpy(model[suffix+'_'+v]) for v in ['projection','weights','bias']],key.startswith('current')).numpy()
        predictions.append((np.column_stack((z['core'][:,0],z['core'][:,3:]))@model[suffix+'_linear']+h@model[suffix+'_head'])[:,0])
    return np.column_stack(predictions)

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    work=OUT/'fit';assert all(c.sha(work/p)==h for p,h in json.loads((work/'identity.json').read_text()).items())
    with np.load(work/'artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    with np.load(c.OUT/'probe_fit/artifacts/models.npz') as q: control={k:q[k] for k in q.files}
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_manifest.json').read_text())['records']};moments=[];stand=[];diagnostics=[]
    for group,z in cached(OUT/'replication'):
        assert c.sha(OUT/'replication'/f'{group:05d}.npz')==expected[group];predictions=[z['base']];standalone=[];diag=[]
        for replica in [0,1]:
            for family in ['current','temporal','gru']:
                key=family+'_r'+str(replica)
                if family=='gru':
                    p=z['base']+c.design(z,'gru',control['gru_mean'],control['gru_scale'])@control[key+'_coef']*control[key+'_strength'];sp=c.design(z,'gru',control['gru_mean'],control['gru_scale'],True)@control[key+'_standalone']
                else:
                    sp=direct(z,model,key);strength=np.array([model[key+'_t'+str(t)+'_strength'] for t in [0,1]]);p=z['base']+strength*(sp-z['base'])
                predictions.append(p);standalone.append(sp)
                y=np.clip(z['y'],-2,2);e=y-np.clip(z['base'],-2,2);ce=y-np.clip(p,-2,2);se=y-np.clip(sp,-2,2);w=np.abs(y);poor=np.abs(e)>=model['poor_threshold']
                diag.append(np.stack([(w*e*e).sum(0),(w*ce*ce).sum(0),(w*e*ce).sum(0),(w*poor*(e*e-ce*ce)).sum(0),(w*poor).sum(0),(w*se*se).sum(0),(w*e*se).sum(0)]))
        moments.append([[focus_stats(z['y'],p,pop) for pop in [np.ones(len(z['base'])),z['focus']]] for p in predictions]);stand.append([focus_stats(z['y'],p,np.ones(len(z['base']))) for p in standalone]);diagnostics.append(diag)
    moments=np.array(moments);stand=np.array(stand);diag=np.array(diagnostics);np.savez(OUT/'replication_moments.npz',moments=moments,standalone=stand,diagnostics=diag)
    counts=bootstrap_counts(256,np.random.default_rng(20261065),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0));delta=boot-boot[:,:1];point=scores-scores[:1];db=bootstrap_stats(diag,counts)
    names=['current_r0','temporal_r0','gru_r0','current_r1','temporal_r1','gru_r1'];variants={};contrasts={}
    for j,name in enumerate(names,1):
        x=diag[:,j-1].sum(0);variants[name]={'combined_delta':point[j].mean(-1).tolist(),'targets_uniform_probability':point[j].tolist(),'wp_uniform_probability':scores[j].tolist(),'paired95':np.quantile(delta[:,j].mean(-1),[.025,.975],axis=0).tolist(),'standalone_uniform_wp':pooled_correlations(stand[:,j-1].sum(0)).tolist(),'error_cosine':(x[2]/np.sqrt(x[0]*x[1])).tolist(),'standalone_error_cosine':(x[6]/np.sqrt(x[0]*x[5])).tolist(),'poor_row_mse_improvement':(x[3]/x[4]).tolist(),'poor_row_mse_improvement_paired95':np.quantile(db[:,j-1,3]/db[:,j-1,4],[.025,.975],axis=0).tolist()}
    for replica in [0,1]:
        for name,k in [('current',1+replica*3),('gru',3+replica*3)]:
            j=2+replica*3;dist=delta[:,j]-delta[:,k];contrasts['temporal_minus_'+name+'_r'+str(replica)]={'combined_delta':(point[j]-point[k]).mean(-1).tolist(),'paired95':np.quantile(dist.mean(-1),[.025,.975],axis=0).tolist()}
    avg=((delta[:,2]-delta[:,1])+(delta[:,5]-delta[:,4]))*.5;ci=np.quantile(avg.mean(-1),[.025,.975],axis=0)
    halves=[(pooled_correlations(x.sum(0))[2]-pooled_correlations(x.sum(0))[0]).mean(-1).tolist() for x in [moments[:128],moments[128:]]]
    gate=bool(all(np.all(point[j].mean(-1)>0) for j in [2,5]) and variants['temporal_r0']['paired95'][0][1]>0 and np.all(np.array(halves)>0) and all(contrasts['temporal_minus_current_r'+str(r)]['combined_delta'][1]>0 for r in [0,1]) and ci[0,1]>0 and contrasts['temporal_minus_gru_r0']['combined_delta'][1]>0)
    result={'variants':variants,'contrasts':contrasts,'mean_temporal_minus_current_paired95':ci.tolist(),'primary_halves':halves,'search_gate_passed':gate,'baseline_wp_uniform_probability':scores[0].tolist(),'replication_sequences':256,'decision':'Qualified for one frozen search comparison' if gate else 'Original-target supervision failed prospective complementary representation gate; no official search.'}
    write_json(path,result);c.event('raw_target_replication_completed',report_sha256=c.sha(path),gate=gate,block='replication_5');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','prepare','train','replicate'],required=True);args=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1): {'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
