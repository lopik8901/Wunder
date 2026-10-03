"""Bounded histogram trees test threshold regimes in frozen causal states."""
from competition_engineering import nonlinear_state_campaign as campaign
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
from competition_engineering.replicated_readout import readout_design,grid_stats,choose_strengths,verify_training_cache
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.pipeline import cached,write_json
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from connectome.mlevolve_generated import validate_source
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations

OUT=campaign.OUT/'state_trees'

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    parent=campaign.initialize();excluded=set(parent['excluded_groups'])
    for groups in parent['replication_groups'].values(): excluded.update(groups)
    pool=json.loads((campaign.TRAIN_4096/'identity.json').read_text())['groups']
    groups=np.random.default_rng(20261037).permutation(sorted(set(pool)-excluded))[:256].tolist()
    assert len(groups)==256 and not set(groups)&excluded
    result={'observation':'Both smooth nonlinear families failed to establish a transferable increment over linear; the incumbent contains a successful threshold gate and compact trees have a feasible compiled runtime.',
        'hypothesis':'Shallow threshold partitions of frozen state isolate residual regimes missed by smooth readouts.',
        'fit':'Same1024 groups,128 rows/group. Fit32 depth2 trees per target,32 quantile bins, minimum512 rows/leaf, leaf L2=100, learning rate0.05, atop matched fixed linear ridge. No epoch or depth sweep.',
        'families':['current','state'],'control':'Current-only trees+current linear versus frozen-state/current trees+state linear; fixed state linear alone uses previously frozen strengths0.5,0.25.',
        'selection':'Original256 correction-disjoint selection groups; pooled uniform/probability WP chooses per-target strength0,.05,.1,.25,.5,1.',
        'replication_groups':groups,'gate':'State model positive in both populations and both halves; probability paired95 lower>0; state-minus-linear positive on both populations and probability paired95 lower>0; state-minus-current positive both populations.',
        'search':'Only if gate passes; freeze32-tree models and strength before one official-search comparison.',
        'interpretation':'Split-feature counts and mixed current/hidden paths test use of the frozen representation. Replication failure closes this implementation, not all threshold models.',
        'isolation':'Pure NumPy trainer under unchanged source guard and network-disabled sandbox; fresh replication is never mounted.'}
    write_json(path,result)
    knowledge=json.loads((campaign.ROOT/'competition_engineering/search_knowledge.json').read_text());assert knowledge['version']==8
    knowledge['version']=9;knowledge['previous_snapshot']={'path':'runs/nonlinear_state_20261003/knowledge_v8.json','sha256':campaign.sha(campaign.OUT/'knowledge_v8.json')}
    knowledge['weakened'].append({'hypothesis':'Learned32-unit smooth joint readouts add a robust correction beyond frozen-state linear probes.',
        'reason':'Optimization lowered MSE; fresh joint-minus-linear uniform gain-0.000223, proxy+0.000079; one uniform half negative and proxy below separable. Official search correctly blocked.'})
    knowledge['status']='Two smooth nonlinear implementations completed; final distinct threshold-regime test on fresh replication.'
    write_json(campaign.OUT/'knowledge_v9.json',knowledge);write_json(campaign.ROOT/'competition_engineering/search_knowledge.json',knowledge)
    campaign.event('experiment_decision',id='learned32',decision='Failed prospective replication; no official search. Test a distinct bounded threshold function class.',knowledge_sha256=campaign.sha(campaign.OUT/'knowledge_v9.json'))
    campaign.event('state_tree_planned',protocol_sha256=campaign.sha(path));return result

def predict_tree(x,nodes):
    indices=np.zeros(len(x),dtype=np.int64)
    for _ in range(len(nodes)):
        active=np.flatnonzero(nodes['is_leaf'][indices]==0)
        if len(active)==0: break
        node=nodes[indices[active]]
        indices[active]=np.where(x[active,node['feature_idx']]<=node['num_threshold'],node['left'],node['right'])
    if np.any(nodes['is_leaf'][indices]==0): raise ValueError('Invalid tree graph')
    return nodes['value'][indices].astype(np.float32)

def build_tree(bins,thresholds,gradient,weight,min_leaf=512,l2=100.,rate=.05,max_depth=2):
    dtype=np.dtype([('is_leaf','i1'),('feature_idx','i4'),('num_threshold','f8'),('left','i4'),('right','i4'),('value','f8'),('missing_go_to_left','i1'),('is_categorical','i1')])
    rows=[];n_bins=thresholds.shape[1]+1
    def grow(indices,depth):
        node_id=len(rows);g=gradient[indices];w=weight[indices];gs=float(g.sum());ws=float(w.sum())
        rows.append((1,0,0.,0,0,rate*gs/(ws+l2),0,0))
        if depth>=max_depth or len(indices)<2*min_leaf: return node_id
        best_gain=0.;best=None
        for feature in range(bins.shape[1]):
            b=bins[indices,feature];count=np.cumsum(np.bincount(b,minlength=n_bins))[:-1]
            hg=np.cumsum(np.bincount(b,weights=g,minlength=n_bins))[:-1]
            hw=np.cumsum(np.bincount(b,weights=w,minlength=n_bins))[:-1]
            gain=hg**2/(hw+l2)+(gs-hg)**2/(ws-hw+l2)-gs**2/(ws+l2)
            gain[(count<min_leaf)|((len(indices)-count)<min_leaf)]=-np.inf
            split=int(np.argmax(gain))
            if gain[split]>best_gain: best_gain=float(gain[split]);best=(feature,split)
        if best is None: return node_id
        feature,split=best;left=bins[indices,feature]<=split
        a=grow(indices[left],depth+1);b=grow(indices[~left],depth+1)
        rows[node_id]=(0,feature,float(thresholds[feature,split]),a,b,0.,0,0);return node_id
    grow(np.arange(len(bins)),0);return np.array(rows,dtype=dtype)

def fit(train_files,output_dir):
    with np.load('seed.npz') as q: seed={k:q[k] for k in q.files}
    xs=[];ys=[];ws=[];selection=[]
    for path in train_files:
        with np.load(path) as q: z={k:q[k] for k in q.files}
        role=int(z['role'])
        if role not in [0,1,2]: raise ValueError('Replication in trainer')
        if role==2:
            selection.append((readout_design(z['current'],z['hidden'],seed['mean'],seed['scale']),z['y'],z['base'],z['focus']));continue
        idx=np.sort(np.random.default_rng(20261031+int(z['group'])).choice(len(z['y']),128,replace=False))
        f=readout_design(z['current'][idx],z['hidden'][idx],seed['mean'],seed['scale']);y=np.clip(z['y'][idx],-2,2)
        xs.append(f);ys.append(y-z['base'][idx]);ws.append(np.abs(y))
    f=np.concatenate(xs);y=np.concatenate(ys).astype(float);weight=np.concatenate(ws).astype(float);del xs,ys,ws
    thresholds=np.quantile(f[:,1:],np.arange(1,32)/32,axis=0).T.astype(np.float32)
    bins=np.column_stack([np.searchsorted(thresholds[j],f[:,j+1],side='left') for j in range(370)]).astype(np.uint8)
    artifacts={'mean':seed['mean'],'scale':seed['scale']};metrics={};choices={};moments={}
    for family,width in [('current',114),('state',370)]:
        linear=seed['current'] if family=='current' else seed['linear'];prediction=f@linear
        metrics[family]={'before_mse':(np.sum(weight*(y-prediction)**2,0)/weight.sum(0)).tolist()}
        models=[[],[]]
        for iteration in range(32):
            for t in [0,1]:
                tree=build_tree(bins[:,:width],thresholds[:width],weight[:,t]*(y[:,t]-prediction[:,t]),weight[:,t])
                prediction[:,t]+=predict_tree(f[:,1:1+width],tree)
                models[t].append(tree);artifacts[family+'_'+str(t)+'_'+str(iteration)]=tree
            if (iteration+1)%8==0: print('TREE_FIT='+family+':'+str(iteration+1),flush=True)
        metrics[family]['after_mse']=(np.sum(weight*(y-prediction)**2,0)/weight.sum(0)).tolist()
        metrics[family]['hidden_splits']=int(sum(np.count_nonzero((tree['is_leaf']==0)&(tree['feature_idx']>=114)) for trees in models for tree in trees))
        metrics[family]['total_splits']=int(sum(np.count_nonzero(tree['is_leaf']==0) for trees in models for tree in trees))
        moments[family]=[]
        for x,sy,base,focus in selection:
            correction=x@linear
            for t in [0,1]:
                for tree in models[t]: correction[:,t]+=predict_tree(x[:,1:1+width],tree)
            moments[family].append([grid_stats(sy,base,correction,pop) for pop in [np.ones(len(x)),focus]])
        choices[family]=choose_strengths(np.array(moments[family]).sum(0));artifacts[family+'_linear']=linear
    np.savez(output_dir+'/models.npz',**artifacts);np.savez(output_dir+'/selection_moments.npz',**moments)
    Path(output_dir+'/training.json').write_text(json.dumps({'metrics':metrics,'choices':choices,'fit_rows':len(f)}))
    return {'trees':128,'rows':len(f)}

def source():
    code='import numpy as np\nimport json\nfrom pathlib import Path\n'
    code+='\n'.join(inspect.getsource(f) for f in [from_stats,focus_stats,readout_design,grid_stats,choose_strengths,predict_tree,build_tree])+'\n'
    code+=inspect.getsource(fit).replace('def fit(','def train(');validate_source(code,training=True);return code

def train():
    initialize();verify_training_cache();work=OUT/'fit';work.mkdir(exist_ok=True)
    from competition_engineering.learned_state_readout import load_models
    learned=load_models();original,_=campaign.load_models()
    if not (work/'artifacts/training.json').exists():
        np.savez(work/'seed.npz',mean=original['mean'],scale=original['scale'],linear=original['linear'],current=learned['current_baseline'])
        (work/'train.py').write_text(source(),encoding='utf-8');campaign.event('tree_fit_started',source_sha256=campaign.sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work)
        process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,campaign.prior.TRAIN)
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            campaign.event('implementation_failure',id='state_trees',status=status,error=stderr.decode(errors='replace')[-1200:]);raise RuntimeError(stderr.decode(errors='replace')[-1200:])
    identity={p:campaign.sha(work/p) for p in ['seed.npz','train.py','artifacts/models.npz','artifacts/training.json','artifacts/selection_moments.npz']}
    if (work/'identity.json').exists(): assert identity==json.loads((work/'identity.json').read_text())
    else: write_json(work/'identity.json',identity);campaign.event('tree_models_frozen',identity=identity)
    r=json.loads((work/'artifacts/training.json').read_text());print(json.dumps({'metrics':r['metrics'],'choices':{k:v['strengths'] for k,v in r['choices'].items()}}),flush=True)

def prepare():
    protocol=initialize();directory=OUT/'replication';directory.mkdir(exist_ok=True);records=[]
    combo=campaign.load_combo();extractor=campaign.LatentExtractor()
    teacher=campaign.ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    for count,group in enumerate(protocol['replication_groups']):
        path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json');origin=campaign.TRAIN_4096/f'{group:05d}.npz'
        if path.exists() and meta.exists():
            row=json.loads(meta.read_text());assert campaign.sha(path)==row['derived_sha256'];records.append(row);continue
        with np.load(origin) as q: z={k:q[k] for k in q.files}
        p,h=extractor.predict(z['x']);np.testing.assert_allclose(p[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
        idx=np.sort(np.random.default_rng(20261037+group).choice(np.flatnonzero(z['need']),500,replace=False));base=campaign.predict_combo(z,combo)[idx]
        current=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
        focus=campaign.probability_focus(z['x'][idx],base,z['step'][idx],combo,trees)[:,1]
        np.savez(path,current=current,hidden=h[idx],base=base,y=z['y'][idx],focus=focus,group=group,indices=idx)
        row={'group':group,'source_sha256':campaign.sha(origin),'derived_sha256':campaign.sha(path)};write_json(meta,row);records.append(row)
        if (count+1)%64==0: print('PREPARED='+str(count+1),flush=True)
    write_json(directory/'identity.json',{'groups':protocol['replication_groups']})
    if not (OUT/'replication_manifest.json').exists():
        write_json(OUT/'replication_manifest.json',{'records':records,'teacher_sha256':campaign.sha(teacher)});campaign.event('tree_replication_prepared',sha256=campaign.sha(OUT/'replication_manifest.json'))

def load_models():
    work=OUT/'fit';identity=json.loads((work/'identity.json').read_text());assert all(campaign.sha(work/k)==v for k,v in identity.items())
    with np.load(work/'artifacts/models.npz') as q: models={k:q[k] for k in q.files}
    return models,json.loads((work/'artifacts/training.json').read_text())

def corrections(f,models,family):
    c=f@models[family+'_linear']
    for t in [0,1]:
        for i in range(32): c[:,t]+=predict_tree(f[:,1:],models[family+'_'+str(t)+'_'+str(i)])
    return c

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    models,training=load_models();names=['linear','current','state'];moments=[]
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'replication_manifest.json').read_text())['records']}
    for _,z in cached(OUT/'replication'):
        assert campaign.sha(OUT/'replication'/f"{int(z['group']):05d}.npz")==expected[int(z['group'])]
        f=readout_design(z['current'],z['hidden'],models['mean'],models['scale'])
        cs=[f@models['state_linear']*np.array([.5,.25],np.float32)]+[corrections(f,models,k)*np.array(training['choices'][k]['strengths'],np.float32) for k in names[1:]]
        moments.append([[focus_stats(z['y'],p,w) for w in [np.ones(len(f)),z['focus']]] for p in [z['base']]+[z['base']+c for c in cs]])
    moments=np.array(moments);np.savez(OUT/'replication_moments.npz',moments=moments)
    counts=bootstrap_counts(256,np.random.default_rng(20261038),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0));delta=boot[:,1:]-boot[:,:1];point=scores[1:]-scores[:1]
    variants={name:{'combined_uniform_probability':point[i].mean(-1).tolist(),'targets_uniform_probability':point[i].tolist(),
        'paired95':np.quantile(delta[:,i].mean(-1),[.025,.975],axis=0).tolist(),'paired99':np.quantile(delta[:,i].mean(-1),[.005,.995],axis=0).tolist()} for i,name in enumerate(names)}
    contrasts={name:{'combined_uniform_probability':(point[2]-point[i]).mean(-1).tolist(),'paired95':np.quantile((delta[:,2]-delta[:,i]).mean(-1),[.025,.975],axis=0).tolist()} for i,name in enumerate(names[:2])}
    halves=[]
    for m in [moments[:128],moments[128:]]:
        s=pooled_correlations(m.sum(0));halves.append((s[3]-s[0]).mean(-1).tolist())
    gate=bool(np.all(point[2].mean(-1)>0) and np.all(np.array(halves)>0) and variants['state']['paired95'][0][1]>0
        and all(np.all(np.array(contrasts[k]['combined_uniform_probability'])>0) for k in ['linear','current']) and contrasts['linear']['paired95'][0][1]>0)
    result={'variants':variants,'state_minus_controls':contrasts,'state_halves_uniform_probability':halves,'search_gate_passed':gate,
        'decision':'Eligible for frozen comparison' if gate else 'No official search; threshold-regime increment failed replication.'}
    write_json(path,result);campaign.event('state_tree_replication_completed',gate=gate,report_sha256=campaign.sha(path));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','train','prepare','replicate'],required=True)
    args=p.parse_args();{'initialize':initialize,'train':train,'prepare':prepare,'replicate':replicate}[args.phase]()
