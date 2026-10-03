"""One learned temporal topology versus a matched current-only topology."""
from competition_engineering import alternative_representation as c
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from competition_engineering.sparse_temporal import endpoint_windows
from competition_engineering.pipeline import read_sequence,write_json,cached
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from competition_engineering.replicated_readout import grid_stats,choose_strengths
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations
from connectome.mlevolve_generated import validate_source

OUT=c.OUT/'learned_causal'

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    r=json.loads((c.OUT/'orthogonal_memory/replication.json').read_text());assert not r['search_gate_passed']
    protocol={'observation':'Fixed temporal convolution and fixed polynomial memory probes failed matched GRU contrasts. These probes did not optimize raw-input feature extraction; frozen random projection/mixing may discard task-relevant channels.',
      'hypothesis':'A learned bounded temporal representation adds reproducible causal correction information beyond an equally sized learned current-only model.',
      'mechanism':'Learn input112-to32 projection and4 width32 kernel2 tanh layers at dilations1,4,16,64;128 concatenated states and same115-feature linear core feed a two-target linear correction. Current-only control repeats current input at every temporal tap, with identical parameters and optimizer.',
      'experiment':'512 reserved fit groups,128rows each, same two paired sampling seeds as probes. AdamW lr0.002 decay0.001, batch1024,20 epochs1280updates; mean-one clipped-target absolute-weight residual MSE. Paired init/shuffle seeds20261057 and20261058. No architecture sweep.',
      'selection':'Same128 development groups500rows; per-target checkpoint5,10,20 and strengths0,.05,.1,.25,.5,1 maximize minimum uniform/probability WP gain. All choices frozen before replication3.',
      'primary':'temporal_r0. Paired replica1 repeats mechanism, no best-seed choice.',
      'success':'Both temporal fits have positive gains on both populations; primary probability95 lower>0 and both128-sequence halves positive. Temporal-minus-current proxy positive in each paired fit and average95 lower>0. Primary also beats frozen matched-row GRU readout in probability point score.',
      'failure':'Failed gate blocks official search; no seed/architecture/epoch tuning against replication.',
      'replication':'Only reserved replication_3. Block4 remains unused for an independently justified mechanism.',
      'complementarity':'Direct residual correction is the prospective simple ensemble. Poor-row thresholds fixed at development75th percentile, with paired sequence uncertainty. Report error structure and supervised fitting loss. GRU benchmark uses fewer trainable parameters; temporal-vs-current is the capacity-matched causal contrast.',
      'causal':'Sparse endpoint graph uses16 required raw past/current taps, zeros both missing inputs and intermediate states. Projection uses no sequence-wide normalization; all fixed input normalization from incumbent.',
      'deployment':'Learned layers have~12000 parameters; incremental cached convolution can plausibly fit CPU budget, requiring complete callback measurement if qualified.'}
    write_json(path,protocol)
    k=json.loads((c.ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==16;k['version']=17
    k['weakened'].append({'hypothesis':'Fixed512-row order4 polynomial memory adds information beyond GRU or simple smoothing.','reason':'Both proxy effects tiny; both GRU contrasts negative with95 intervals excluding zero. No official search access.','source':'runs/alternative_representation_20261003/orthogonal_memory/replication.json'})
    k['status']='Fixed representation probes complete; one capacity-matched learned convolution test frozen.'
    write_json(c.OUT/'knowledge_v17.json',k);write_json(c.ROOT/'competition_engineering/search_knowledge.json',k)
    c.event('research_decision',protocol_sha256=c.sha(path),knowledge_sha256=c.sha(c.OUT/'knowledge_v17.json'));return protocol

def prepare():
    initialize();r=c.reserve();pf=pq.ParquetFile(c.DATA);combo=c.load_combo();extractor=c.LatentExtractor()
    with np.load(c.OUT/'basis.npz') as q: basis={k:q[k] for k in q.files}
    teacher=c.ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    records=[]
    for role in ['fit','development','replication_3']:
        directory=OUT/('training' if role!='replication_3' else 'replication');directory.mkdir(exist_ok=True)
        for count,group in enumerate(r['roles'][role]):
            path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                row=json.loads(meta.read_text());assert c.sha(path)==row['derived_sha256'];records.append(row);continue
            seq,need,x,y,mask=read_sequence(pf,group);nx=np.clip((x-combo['mean'])/combo['scale'],-8,8).astype(np.float32)
            if role!='replication_3':
                origin=c.OUT/'training'/f'{group:05d}.npz'
                with np.load(origin) as q: z={k:q[k] for k in q.files if k in ['core','gru','base','y','focus','indices','group','seq_ix','role']}
                idx=z['indices'];assert int(z['seq_ix'])==seq
            else:
                p,hidden=extractor.predict(x);full={'x':x,'y':y,'p':p,'step':np.arange(len(x)),'need':need};base_full=c.predict_combo(full,combo)
                idx=np.sort(np.random.default_rng(20261052+group).choice(np.flatnonzero(need),500,replace=False));base=base_full[idx]
                z={'core':np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),nx[idx])).astype(np.float32),'gru':hidden[idx]@basis['gru'],'base':base,'y':y[idx],'focus':c.probability_focus(x[idx],base,full['step'][idx],combo,trees)[:,1],'indices':idx,'group':group,'seq_ix':seq,'role':2}
            windows,valid=endpoint_windows(nx,idx);np.savez(path,**z,windows=windows,valid=valid)
            row={'group':group,'role':role,'source_sequence_sha256':c.hashlib.sha256(x.tobytes()+y.tobytes()+need.tobytes()).hexdigest(),'derived_sha256':c.sha(path)};write_json(meta,row);records.append(row)
            if (count+1)%128==0: print(role+' prepared '+str(count+1),flush=True)
        write_json(directory/'identity.json',{'groups':r['roles']['fit']+r['roles']['development'] if role!='replication_3' else r['roles'][role]})
    path=OUT/'prepared_manifest.json'
    if not path.exists(): write_json(path,{'records':records});c.event('learned_data_prepared',manifest_sha256=c.sha(path))

def tensor_features(windows,valid,projection,weights,bias,instant):
    if instant:
        h=(windows[:,:1]@projection).expand(-1,16,-1);mask=torch.ones_like(valid)
    else: h=windows@projection;mask=valid
    layers=[]
    for i in range(4):
        mask=mask[:,::2];h=torch.tanh(h[:,::2]@weights[i,0]+h[:,1::2]@weights[i,1]+bias[i])*mask[:,:,None]
        layers.append(h[:,0])
    return torch.cat(layers,dim=1)

def numpy_correction(z,model,key,target):
    suffix=key+'_t'+str(target);windows=torch.from_numpy(z['windows']);valid=torch.from_numpy(z['valid'])
    with torch.no_grad():
        f=tensor_features(windows,valid,*[torch.from_numpy(model[suffix+'_'+v]) for v in ['projection','weights','bias']],key.startswith('current'))
        prediction=z['core']@model[suffix+'_linear']+f.numpy()@model[suffix+'_head']
    return prediction[:,0]*model[suffix+'_strength']

def numpy_standalone(z,model,key,target):
    suffix=key+'_t'+str(target)
    with torch.no_grad():
        h=tensor_features(torch.from_numpy(z['windows']),torch.from_numpy(z['valid']),*[torch.from_numpy(model[suffix+'_'+v]) for v in ['projection','weights','bias']],key.startswith('current')).numpy()
    f=np.column_stack((z['core'][:,0],z['core'][:,3:],np.clip((h-model[suffix+'_state_mean'])/model[suffix+'_state_scale'],-8,8)))
    return (f@model[suffix+'_standalone'])[:,0]

def fit(train_files,output_dir):
    torch.set_num_threads(1);models={};reports={}
    for replica in [0,1]:
        parts={k:[] for k in ['core','windows','valid','y','base']};development=[]
        for path in train_files:
            with np.load(path) as q: z={k:q[k] for k in q.files}
            role=int(z['role'])
            if role==1:
                development.append({k:z[k] for k in ['core','windows','valid','y','base','focus']});continue
            if role!=0: raise ValueError('Replication cannot enter fitting sandbox')
            idx=np.random.default_rng(20261053+int(z['group'])+replica*100000).permutation(len(z['base']))[:128]
            for k in parts: parts[k].append(z[k][idx])
        arrays={k:np.concatenate(v) for k,v in parts.items()};assert len(arrays['base'])==65536
        core=torch.from_numpy(arrays['core']);windows=torch.from_numpy(arrays['windows']);valid=torch.from_numpy(arrays['valid'])
        y=np.clip(arrays['y'],-2,2);weight=np.abs(y);weight=weight/weight.mean(0);w=torch.from_numpy(weight.astype(np.float32));residual=torch.from_numpy((y-arrays['base']).astype(np.float32))
        for family in ['current','temporal']:
            torch.manual_seed(20261057+replica);rng=np.random.default_rng(20261057+replica)
            projection=torch.nn.Parameter(torch.randn(112,32)/np.sqrt(112));weights=torch.nn.Parameter(torch.randn(4,2,32,32)/8);bias=torch.nn.Parameter(torch.zeros(4,32));head=torch.nn.Parameter(torch.randn(128,2)*.01);linear=torch.nn.Parameter(torch.zeros(115,2))
            optimizer=torch.optim.AdamW([projection,weights,bias,head,linear],lr=.002,weight_decay=.001);snapshots={};choices=[];history=[]
            for epoch in range(1,21):
                order=rng.permutation(len(core));total=0.
                for start in range(0,len(core),1024):
                    idx=order[start:start+1024];optimizer.zero_grad();f=tensor_features(windows[idx],valid[idx],projection,weights,bias,family=='current')
                    pred=core[idx]@linear+f@head;loss=(w[idx]*(pred-residual[idx])**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_([projection,weights,bias,head,linear],10.);optimizer.step();total+=float(loss.detach())*len(idx)
                history.append({'epoch':epoch,'weighted_residual_mse':total/len(core)})
                if epoch in [5,10,20]:
                    moment=np.zeros((2,6,6,2))
                    with torch.no_grad():
                        for z in development:
                            f=tensor_features(torch.from_numpy(z['windows']),torch.from_numpy(z['valid']),projection,weights,bias,family=='current');correction=z['core']@linear.detach().numpy()+f.numpy()@head.detach().numpy()
                            for i,pop in enumerate([np.ones(len(z['base'])),z['focus']]): moment[i]+=grid_stats(z['y'],z['base'],correction,pop)
                    choice=choose_strengths(moment);choices.append((epoch,choice));snapshots[epoch]={k:v.detach().numpy().copy() for k,v in [('projection',projection),('weights',weights),('bias',bias),('head',head),('linear',linear)]}
                    print(family+'_r'+str(replica)+' epoch '+str(epoch),flush=True)
            key=family+'_r'+str(replica);selected=[]
            for target in [0,1]:
                gains=[max(choice['minimum_population_gains'][i][target] for i in range(6)) for _,choice in choices];j=int(np.argmax(gains));epoch,choice=choices[j];suffix=key+'_t'+str(target)
                for name,value in snapshots[epoch].items(): models[suffix+'_'+name]=value[:,target:target+1] if name in ['head','linear'] else value
                models[suffix+'_strength']=np.array(choice['strengths'][target]);selected.append({'epoch':epoch,'strength':choice['strengths'][target]})
                state=snapshots[epoch];latent=[]
                with torch.no_grad():
                    for start in range(0,len(core),1024):
                        latent.append(tensor_features(windows[start:start+1024],valid[start:start+1024],*[torch.from_numpy(state[v]) for v in ['projection','weights','bias']],family=='current').numpy())
                latent=np.concatenate(latent);mean=latent.mean(0);scale=np.maximum(latent.std(0),.05)
                sf=np.column_stack((arrays['core'][:,0],arrays['core'][:,3:],np.clip((latent-mean)/scale,-8,8))).astype(float);sw=weight[:,target]
                standalone=np.linalg.solve(sf.T@(sw[:,None]*sf)+np.eye(241)*len(core)*.1,sf.T@(sw*y[:,target]))
                models[suffix+'_state_mean']=mean;models[suffix+'_state_scale']=scale;models[suffix+'_standalone']=standalone[:,None].astype(np.float32)
            reports[key]={'selected':selected,'history':history,'choices':choices,'rows':65536,'parameters':sum(p.numel() for p in [projection,weights,bias,head,linear])}
        models['poor_threshold']=np.quantile(np.concatenate([np.abs(np.clip(z['y'],-2,2)-np.clip(z['base'],-2,2)) for z in development]),.75,axis=0)
    np.savez(output_dir+'/models.npz',**models);Path(output_dir+'/training.json').write_text(json.dumps(reports),encoding='utf-8');return {'models':4,'rows_per_model':65536}

def source():
    code='import numpy as np\nimport torch\nimport json\nfrom pathlib import Path\n'
    for f in [from_stats,focus_stats,grid_stats,choose_strengths,tensor_features,fit]: code+=inspect.getsource(f)+'\n'
    code+='def train(train_files,output_dir):\n    return fit(train_files,output_dir)\n';validate_source(code,training=True);return code

def train():
    initialize();work=OUT/'fit';work.mkdir(exist_ok=True)
    for row in json.loads((OUT/'prepared_manifest.json').read_text())['records']:
        if row['role']!='replication_3': assert c.sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    if not (work/'artifacts/models.npz').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');c.event('learned_fit_started',source_sha256=c.sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,OUT/'training')
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1);(work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            c.event('implementation_failure',component='learned causal fit',error=stderr.decode(errors='replace')[-1500:]);raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={p:c.sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/training.json']}
    if (work/'identity.json').exists(): assert json.loads((work/'identity.json').read_text())==identity
    else: write_json(work/'identity.json',identity);c.event('learned_models_frozen',identity=identity)

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    work=OUT/'fit';assert all(c.sha(work/p)==h for p,h in json.loads((work/'identity.json').read_text()).items())
    with np.load(work/'artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    with np.load(c.OUT/'probe_fit/artifacts/models.npz') as q: controls={k:q[k] for k in q.files}
    moments=[];diagnostics=[];standalone_moments=[]
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_manifest.json').read_text())['records'] if r['role']=='replication_3'}
    for group,z in cached(OUT/'replication'):
        assert c.sha(OUT/'replication'/f'{group:05d}.npz')==expected[group]
        predictions=[z['base']];diag=[];standalone=[]
        for replica in [0,1]:
            for family in ['current','temporal','gru']:
                key=family+'_r'+str(replica)
                if family=='gru':
                    p=z['base']+c.design(z,'gru',controls['gru_mean'],controls['gru_scale'])@controls[key+'_coef']*controls[key+'_strength']
                    sp=c.design(z,'gru',controls['gru_mean'],controls['gru_scale'],True)@controls[key+'_standalone']
                else:
                    p=z['base']+np.column_stack([numpy_correction(z,model,key,t) for t in [0,1]])
                    sp=np.column_stack([numpy_standalone(z,model,key,t) for t in [0,1]])
                standalone.append(focus_stats(z['y'],sp,np.ones(len(sp))))
                predictions.append(p);y=np.clip(z['y'],-2,2);e=y-np.clip(z['base'],-2,2);ce=y-np.clip(p,-2,2);w=np.abs(y);poor=np.abs(e)>=model['poor_threshold']
                diag.append(np.stack([(w*e*e).sum(0),(w*ce*ce).sum(0),(w*e*ce).sum(0),(w*poor*(e*e-ce*ce)).sum(0),(w*poor).sum(0)]))
        moments.append([[focus_stats(z['y'],p,pop) for pop in [np.ones(len(z['base'])),z['focus']]] for p in predictions]);diagnostics.append(diag);standalone_moments.append(standalone)
    moments=np.array(moments);diag=np.array(diagnostics);standalone_moments=np.array(standalone_moments);np.savez(OUT/'replication_moments.npz',moments=moments,diagnostics=diag,standalone=standalone_moments)
    counts=bootstrap_counts(256,np.random.default_rng(20261059),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0));delta=boot-boot[:,:1];point=scores-scores[:1];db=bootstrap_stats(diag,counts)
    names=['current_r0','temporal_r0','gru_r0','current_r1','temporal_r1','gru_r1'];variants={};contrasts={}
    for j,name in enumerate(names,1):
        d=diag[:,j-1].sum(0);variants[name]={'combined_delta':point[j].mean(-1).tolist(),'targets_uniform_probability':point[j].tolist(),'wp_uniform_probability':scores[j].tolist(),'paired95':np.quantile(delta[:,j].mean(-1),[.025,.975],axis=0).tolist(),'error_cosine':(d[2]/np.sqrt(d[0]*d[1])).tolist(),'poor_row_mse_improvement':(d[3]/d[4]).tolist(),'poor_row_mse_improvement_paired95':np.quantile(db[:,j-1,3]/db[:,j-1,4],[.025,.975],axis=0).tolist()}
        variants[name]['standalone_uniform_wp']=pooled_correlations(standalone_moments[:,j-1].sum(0)).tolist()
    for r in [0,1]:
        for name,k in [('current',1+r*3),('gru',3+r*3)]:
            j=2+r*3;dist=delta[:,j]-delta[:,k];contrasts['temporal_minus_'+name+'_r'+str(r)]={'combined_delta':(point[j]-point[k]).mean(-1).tolist(),'paired95':np.quantile(dist.mean(-1),[.025,.975],axis=0).tolist()}
    average=((delta[:,2]-delta[:,1])+(delta[:,5]-delta[:,4]))*.5;average_ci=np.quantile(average.mean(-1),[.025,.975],axis=0)
    halves=[(pooled_correlations(x.sum(0))[2]-pooled_correlations(x.sum(0))[0]).mean(-1).tolist() for x in [moments[:128],moments[128:]]]
    gate=bool(all(np.all(point[j].mean(-1)>0) for j in [2,5]) and variants['temporal_r0']['paired95'][0][1]>0 and np.all(np.array(halves)>0) and all(contrasts['temporal_minus_current_r'+str(r)]['combined_delta'][1]>0 for r in [0,1]) and average_ci[0,1]>0 and contrasts['temporal_minus_gru_r0']['combined_delta'][1]>0)
    result={'variants':variants,'contrasts':contrasts,'mean_temporal_minus_current_paired95':average_ci.tolist(),'primary_halves':halves,'search_gate_passed':gate,'baseline_wp_uniform_probability':scores[0].tolist(),'replication_sequences':256,'decision':'Qualified for one frozen search comparison' if gate else 'Learned causal topology failed prospective replication; no official search.'}
    write_json(path,result);c.event('learned_replication_completed',report_sha256=c.sha(path),gate=gate,block='replication_3');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','prepare','train','replicate'],required=True);args=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): {'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
