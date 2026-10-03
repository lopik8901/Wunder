"""Control source-population composition while testing fitting-sequence count."""
from competition_engineering import sequence_diversity as first
import argparse
import json
import shutil
import time
from datetime import datetime,timezone
import numpy as np
from competition_engineering.pipeline import cached,write_json
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from connectome.mlevolve_generated import validate_source
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations

ROOT=first.ROOT
OUT=first.OUT/'balanced_followup'
sha=first.sha
corrections=first.corrections

def event(kind,**payload):
    OUT.mkdir(exist_ok=True);path=OUT/'ledger.jsonl'
    row={'utc':datetime.now(timezone.utc).isoformat(),'boundary':'search-only','kind':kind,'previous_ledger_sha256':sha(path) if path.exists() else None,**payload}
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    original=first.initialize();result=json.loads((first.OUT/'replication.json').read_text());assert not result['search_gate_passed']
    rng=np.random.default_rng(20261044)
    narrow=rng.permutation(original['groups']['original_fit'])[:512].tolist()+rng.permutation(original['groups']['extra_fit'])[:512].tolist()
    prior=json.loads((first.previous.OUT/'protocol.json').read_text());excluded=set(prior['excluded_groups'])
    excluded.update(original['groups']['extra_fit']);excluded.update(original['groups']['replication'])
    pool=json.loads((first.previous.TRAIN_4096/'identity.json').read_text())['groups']
    fresh=rng.permutation(sorted(set(pool)-excluded)).tolist();assert len(fresh)==512
    assert not set(narrow)&set(fresh) and not set(fresh)&excluded
    shutil.copyfile(first.OUT/'normalization.npz',OUT/'normalization.npz')
    protocol={'observation':'Original-only1024 versus mixed2048 did not replicate a positive diversity effect; both wider linear controls regressed. The comparison changes source-population mixture as well as sequence count.',
        'hypothesis':'With a fixed50:50 mixture of original and additional fitting sequences,2048×64 generalizes better than1024×128 at equal rows and model budget.',
        'groups':{'fit':narrow,'selection':original['groups']['selection'],'replication':fresh},
        'narrow_composition':{'original':512,'extra':512},'broad_composition':{'original':1024,'extra':1024},
        'budget':'131072 rows and1280 optimizer updates/model. Refit only balanced narrow arms using the exact same trainer, two paired seeds, normalization,32-unit head, optimizer, checkpoint/strength selector. Broad artifacts remain frozen from the original experiment.',
        'primary':'The same broad neural replica0; no best-seed or model selection from replication.',
        'gate':original['gate'],'official_search':original['official_search'],
        'scope':'This controls broad source-mixture proportions, not every covariate or label distribution. Fresh replication is disjoint from all earlier correction fit/selection/development groups.',
        'broad_identity_sha256':sha(first.OUT/'fit/identity.json'),'original_replication_sha256':sha(first.OUT/'replication.json')}
    write_json(path,protocol)
    knowledge=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text());assert knowledge['version']==12
    knowledge['version']=13;knowledge['previous_snapshot']={'path':'runs/sequence_diversity_20261003/knowledge_v12.json','sha256':sha(first.OUT/'knowledge_v12.json')}
    knowledge['weakened'].append({'hypothesis':'Adding1024 observed development sequences improves learned-head generalization at a fixed sampled-row budget.',
        'reason':'Paired neural proxy gains+0.000117/-0.000151; mean-0.000017,95 interval[-0.000814,+0.000754]. Both linear controls regress. Source mixture and count are not separated by that comparison.'})
    knowledge['status']='First diversity comparison negative; source-mixture-balanced count comparison prospectively frozen.'
    write_json(first.OUT/'knowledge_v13.json',knowledge);write_json(ROOT/'competition_engineering/search_knowledge.json',knowledge)
    first.event('research_decision',decision='Original comparison failed. Control source composition before closing diversity hypothesis.',protocol_sha256=sha(path),knowledge_sha256=sha(first.OUT/'knowledge_v13.json'))
    event('experiment_preregistered',protocol_sha256=sha(path));return protocol

def prepare():
    protocol=initialize();training=OUT/'training';training.mkdir(exist_ok=True);replication=OUT/'replication';replication.mkdir(exist_ok=True);records=[]
    for role,name in [(0,'fit'),(2,'selection')]:
        for group in protocol['groups'][name]:
            path=training/f'{group:05d}.npz';meta=path.with_suffix('.json');origin=first.OUT/'training'/f'{group:05d}.npz'
            if path.exists() and meta.exists():
                row=json.loads(meta.read_text());assert sha(path)==row['derived_sha256'];records.append(row);continue
            with np.load(origin) as q: z={k:q[k] for k in q.files}
            z['role']=np.array(role);np.savez(path,**z)
            row={'group':group,'role':name,'source_path':origin.relative_to(ROOT).as_posix(),'source_sha256':sha(origin),'derived_sha256':sha(path)};write_json(meta,row);records.append(row)
    print('BALANCED_FIT_PREPARED',flush=True)
    with np.load(OUT/'normalization.npz') as q: mean=q['mean'];scale=q['scale']
    teacher=ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    extractor=first.previous.LatentExtractor();combo=first.previous.load_combo()
    for count,group in enumerate(protocol['groups']['replication']):
        path=replication/f'{group:05d}.npz';meta=path.with_suffix('.json');origin=first.previous.TRAIN_4096/f'{group:05d}.npz'
        if path.exists() and meta.exists():
            row=json.loads(meta.read_text());assert sha(path)==row['derived_sha256'];records.append(row);continue
        with np.load(origin) as q: z={k:q[k] for k in q.files}
        p,hidden=extractor.predict(z['x']);np.testing.assert_allclose(p[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
        idx=np.sort(np.random.default_rng(20261044+group).choice(np.flatnonzero(z['need']),500,replace=False));base=first.previous.predict_combo(z,combo)[idx]
        current=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
        f=first.readout_design(current,hidden[idx],mean,scale);focus=first.previous.probability_focus(z['x'][idx],base,z['step'][idx],combo,trees)[:,1]
        np.savez(path,f=f,base=base,y=z['y'][idx],focus=focus,group=group,indices=idx,role=3)
        row={'group':group,'role':'replication','source_path':origin.relative_to(ROOT).as_posix(),'source_sha256':sha(origin),'derived_sha256':sha(path)};write_json(meta,row);records.append(row)
        if (count+1)%128==0: print('BALANCED_REPLICATION_PREPARED='+str(count+1),flush=True)
    write_json(training/'identity.json',{'groups':protocol['groups']['fit']+protocol['groups']['selection']})
    write_json(replication/'identity.json',{'groups':protocol['groups']['replication']})
    if not (OUT/'prepared_manifest.json').exists():
        write_json(OUT/'prepared_manifest.json',{'records':records,'teacher_sha256':sha(teacher)});event('data_prepared',manifest_sha256=sha(OUT/'prepared_manifest.json'))

def source():
    code=first.source()
    needle="for arm in ['narrow','broad']:"
    assert code.count(needle)==1
    code=code.replace(needle,"for arm in ['narrow']:").replace("return {'fits':8,'rows_per_model':131072}","return {'fits':4,'rows_per_model':131072}")
    validate_source(code,training=True);return code

def train():
    protocol=initialize();expected=set(protocol['groups']['fit']+protocol['groups']['selection'])
    assert set(json.loads((OUT/'training/identity.json').read_text())['groups'])==expected and not expected&set(protocol['groups']['replication'])
    for row in json.loads((OUT/'prepared_manifest.json').read_text())['records']:
        if row['role']!='replication': assert sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    work=OUT/'fit';work.mkdir(exist_ok=True)
    if not (work/'artifacts/training.json').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');event('fit_started',source_sha256=sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,OUT/'training')
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            event('implementation_failure',status=status,error=stderr.decode(errors='replace')[-1200:]);raise RuntimeError(stderr.decode(errors='replace')[-1200:])
    identity={p:sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/training.json','artifacts/selection_moments.npz']}
    if (work/'identity.json').exists(): assert identity==json.loads((work/'identity.json').read_text())
    else: write_json(work/'identity.json',identity);event('models_frozen',identity=identity)
    r=json.loads((work/'artifacts/training.json').read_text());print(json.dumps({k:{'chosen':v['chosen'],'history':v['history']} for k,v in r.items()}),flush=True)

def load_models():
    original,training=first.load_models();work=OUT/'fit';identity=json.loads((work/'identity.json').read_text())
    assert all(sha(work/k)==v for k,v in identity.items())
    with np.load(work/'artifacts/models.npz') as q: original.update({k:q[k] for k in q.files})
    training.update(json.loads((work/'artifacts/training.json').read_text()));return original,training

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    model,training=load_models();moments=[]
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_manifest.json').read_text())['records'] if r['role']=='replication'}
    for _,z in cached(OUT/'replication'):
        assert sha(OUT/'replication'/f"{int(z['group']):05d}.npz")==expected[int(z['group'])]
        predictions=[z['base']]
        for r in [0,1]:
            for arm in ['narrow','broad']:
                for linear in [True,False]: predictions.append(z['base']+corrections(z['f'],model,arm+'_r'+str(r),linear))
        moments.append([[focus_stats(z['y'],p,pop) for pop in [np.ones(len(z['f'])),z['focus']]] for p in predictions])
    moments=np.array(moments);np.savez(OUT/'replication_moments.npz',moments=moments)
    counts=bootstrap_counts(512,np.random.default_rng(20261045),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0))
    point=(scores[1:]-scores[:1]).reshape(2,2,2,2,2);delta=(boot[:,1:]-boot[:,:1]).reshape(10000,2,2,2,2,2)
    variants={};contrasts={}
    for r in [0,1]:
        for a,arm in enumerate(['narrow','broad']):
            for m,name in enumerate(['linear','neural']):
                variants[arm+'_r'+str(r)+'_'+name]={'combined_uniform_probability':point[r,a,m].mean(-1).tolist(),'targets_uniform_probability':point[r,a,m].tolist(),
                    'paired95':np.quantile(delta[:,r,a,m].mean(-1),[.025,.975],axis=0).tolist()}
        for m,name in enumerate(['linear','neural']):
            contrasts['r'+str(r)+'_'+name]={'combined_uniform_probability':(point[r,1,m]-point[r,0,m]).mean(-1).tolist(),
                'paired95':np.quantile((delta[:,r,1,m]-delta[:,r,0,m]).mean(-1),[.025,.975],axis=0).tolist()}
    halves=[]
    for chunk in [moments[:256],moments[256:]]:
        s=pooled_correlations(chunk.sum(0));halves.append((s[4]-s[0]).mean(-1).tolist())
    gate,average_ci,_=first.replication_gate(point,delta,halves)
    result={'variants':variants,'broad_minus_balanced_narrow':contrasts,'mean_paired_neural_contrast':{
        'combined_uniform_probability':(point[:,1,1]-point[:,0,1]).mean(axis=(0,2)).tolist(),'paired95':average_ci.tolist()},
        'primary_halves_uniform_probability':halves,'search_gate_passed':gate,
        'decision':'New composition-controlled evidence qualifies frozen primary for one search comparison' if gate else 'No official search: composition-balanced diversity effect failed replication.',
        'fresh_sequences':512,'remaining_never_observed_groups_in_designated4096_pool':0}
    write_json(path,result);event('replication_completed',gate=gate,report_sha256=sha(path));first.event('balanced_followup_completed',gate=gate,report_sha256=sha(path));print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['initialize','prepare','train','replicate','search'],required=True)
    args=parser.parse_args()
    if args.phase=='search':
        from competition_engineering.sequence_diversity_evaluation import main
        main(owner=__import__(__name__,fromlist=['*']))
    else: {'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
