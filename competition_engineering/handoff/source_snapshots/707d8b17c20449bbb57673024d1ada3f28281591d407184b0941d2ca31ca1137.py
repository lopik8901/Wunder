"""Compact ordered-memory probe, separately reserved replication block."""
from competition_engineering import alternative_representation as c
import argparse
import json
import time
import numpy as np
import pyarrow.parquet as pq
from scipy.linalg import expm
from scipy.signal import ss2tf,lfilter
from threadpoolctl import threadpool_limits
from competition_engineering.pipeline import read_sequence,write_json,cached
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations
from competition_engineering.prospective_selection_diagnostic import focus_stats
from connectome.mlevolve_generated import validate_source

OUT=c.OUT/'orthogonal_memory';FAMILIES=['core','gru','smooth','ssm']

def matrices():
    a=np.array([[(2*i+1)*(-1 if i<j else (-1)**(i-j+1)) for j in range(4)] for i in range(4)],float)
    b=np.array([(2*i+1)*(-1)**i for i in range(4)],float)
    aug=np.zeros((5,5));aug[:4,:4]=a/512;aug[:4,4]=b/512
    discrete=expm(aug);return discrete[:4,:4],discrete[:4,4]

def representations(x,a,b):
    outputs=[]
    for i in range(4):
        # output after assimilating current input, with zero initial state.
        numerator,denominator=ss2tf(a,b[:,None],a[i:i+1],np.array([[b[i]]]))
        outputs.append(lfilter(numerator[0],denominator,x,axis=0))
    ssm=np.concatenate(outputs,axis=1).astype(np.float32)
    smooth=np.concatenate([lfilter([1-np.exp(-1/h)],[1,-np.exp(-1/h)],x,axis=0) for h in [128,256,512,1024]],axis=1).astype(np.float32)
    return ssm,smooth

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    first=json.loads((c.OUT/'probe_replication.json').read_text());assert not first['search_gate_passed']
    r={'observation':'Frozen finite-field TCN gains failed paired seed/GRU contrasts; standalone raw-history probes lag GRU. Finite-field random mixing and recurrent compression remain different explanations.',
       'hypothesis':'A512-row orthogonal polynomial memory of raw projected inputs preserves ordered-history information missing from the GRU readout.',
       'mechanism':'Order4 Legendre linear dynamical state with exact zero-order-hold discretization,32 shared projected input channels;128 state coefficients. This is a structured fixed SSM feature probe, not a learned S4/Mamba architecture.',
       'experiment':'Reuse exactly512 fit/128 development groups and paired128-row samplers, frozen input projection, mean-one target-weighted ridge0.1, target-wise WP strength grid. Compare128 SSM with128 independent EMA smoothing coefficients and128 projected GRU features.',
       'replication':'Reserved replication_2,256 previously unobserved groups; no block1 tuning.',
       'primary':'ssm_r0; second paired sampler repeats mechanism, no best-seed selection.',
       'success':'Both SSM fits beat incumbent in both populations. Primary probability95 lower>0 and both128-sequence halves positive. SSM-minus-GRU and SSM-minus-smoothing probability contrasts positive in both fits and average paired95 lower>0.',
       'failure':'Failed gate blocks official-search access. No window/order tuning on replication.',
       'complementarity':'Same prospective development75th-percentile poor-row threshold and low-capacity residual ensemble. Standalone raw-feature target ridge is diagnostic.',
       'source':'research_cards/v13/legendre_memory_unit.json','causal':'Update with current input only; state zeroed per sequence. No convolution with future rows.'}
    write_json(path,r);a,b=matrices();assert np.max(np.abs(np.linalg.eigvals(a)))<1
    np.savez(OUT/'dynamics.npz',a=a,b=b)
    k=json.loads((c.ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==15;k['version']=16
    k['weakened'].append({'hypothesis':'Fixed86-row random dilated convolutions produce robust incremental information over GRU.','reason':'Primary proxy gain0.000432 but paired repeat0.000014; both GRU contrasts negative. No official search access.','source':'runs/alternative_representation_20261003/probe_replication.json'})
    k['status']='Fixed convolution probe complete; structured ordered-memory experiment prospectively frozen.'
    write_json(c.OUT/'knowledge_v16.json',k);write_json(c.ROOT/'competition_engineering/search_knowledge.json',k)
    c.event('research_decision',protocol_sha256=c.sha(path),knowledge_sha256=c.sha(c.OUT/'knowledge_v16.json'));return r

def prepare():
    initialize();r=c.reserve();pf=pq.ParquetFile(c.DATA);combo=c.load_combo();extractor=c.LatentExtractor()
    with np.load(c.OUT/'basis.npz') as q: basis={k:q[k] for k in q.files}
    a,b=matrices();teacher=c.ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    records=[]
    for role in ['fit','development','replication_2']:
        directory=OUT/('training' if role!='replication_2' else 'replication');directory.mkdir(exist_ok=True)
        for count,group in enumerate(r['roles'][role]):
            path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                row=json.loads(meta.read_text());assert c.sha(path)==row['derived_sha256'];records.append(row);continue
            seq,need,x,y,mask=read_sequence(pf,group);nx=np.clip((x-combo['mean'])/combo['scale'],-8,8).astype(np.float32)
            ssm,smooth=representations(nx@basis['projection'],a,b)
            if role!='replication_2':
                origin=c.OUT/'training'/f'{group:05d}.npz'
                with np.load(origin) as q: z={k:q[k] for k in q.files if k in ['core','gru','base','y','focus','indices','group','seq_ix','role']}
                idx=z['indices'];assert int(z['seq_ix'])==seq
            else:
                p,hidden=extractor.predict(x);full={'x':x,'y':y,'p':p,'step':np.arange(len(x)),'need':need};base_full=c.predict_combo(full,combo)
                idx=np.sort(np.random.default_rng(20261052+group).choice(np.flatnonzero(need),500,replace=False));base=base_full[idx]
                z={'core':np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),nx[idx])).astype(np.float32),'gru':hidden[idx]@basis['gru'],'base':base,'y':y[idx],'focus':c.probability_focus(x[idx],base,full['step'][idx],combo,trees)[:,1],'indices':idx,'group':group,'seq_ix':seq,'role':2}
            np.savez(path,**z,ssm=ssm[idx],smooth=smooth[idx]);row={'group':group,'role':role,'source_sequence_sha256':c.hashlib.sha256(x.tobytes()+y.tobytes()+need.tobytes()).hexdigest(),'derived_sha256':c.sha(path)};write_json(meta,row);records.append(row)
            if (count+1)%128==0: print(role+' prepared '+str(count+1),flush=True)
        write_json(directory/'identity.json',{'groups':r['roles']['fit']+r['roles']['development'] if role!='replication_2' else r['roles'][role]})
    path=OUT/'prepared_manifest.json'
    if not path.exists(): write_json(path,{'records':records});c.event('ssm_data_prepared',manifest_sha256=c.sha(path))

def source():
    code=c.source().replace("families=['core','instant','gru','delay','tcn']","families=['core','gru','smooth','ssm']").replace("'models':10","'models':8")
    validate_source(code,training=True);return code

def train():
    initialize();work=OUT/'fit';work.mkdir(exist_ok=True)
    for row in json.loads((OUT/'prepared_manifest.json').read_text())['records']:
        if row['role']!='replication_2': assert c.sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    if not (work/'artifacts/models.npz').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');c.event('ssm_fit_started',source_sha256=c.sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,OUT/'training')
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1);(work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            c.event('implementation_failure',component='ssm fit',error=stderr.decode(errors='replace')[-1500:]);raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={p:c.sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/selection.json']}
    if (work/'identity.json').exists(): assert json.loads((work/'identity.json').read_text())==identity
    else: write_json(work/'identity.json',identity);c.event('ssm_models_frozen',identity=identity)

def replicate():
    path=OUT/'replication.json'
    if path.exists(): print(path.read_text());return
    work=OUT/'fit';assert all(c.sha(work/p)==h for p,h in json.loads((work/'identity.json').read_text()).items())
    with np.load(work/'artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_manifest.json').read_text())['records'] if r['role']=='replication_2'}
    moments=[];stand=[];diagnostics=[]
    for group,z in cached(OUT/'replication'):
        assert c.sha(OUT/'replication'/f'{group:05d}.npz')==expected[group]
        predictions=[z['base']];standalone=[];diag=[]
        for replica in [0,1]:
            for name in FAMILIES:
                key=name+'_r'+str(replica);f=c.design(z,name,model[name+'_mean'],model[name+'_scale']);p=z['base']+f@model[key+'_coef']*model[key+'_strength'];predictions.append(p)
                s=c.design(z,name,model[name+'_mean'],model[name+'_scale'],True)@model[key+'_standalone'];standalone.append(s)
                y=np.clip(z['y'],-2,2);e=y-np.clip(z['base'],-2,2);ce=y-np.clip(p,-2,2);se=y-np.clip(s,-2,2);w=np.abs(y);poor=np.abs(e)>=model['poor_threshold']
                diag.append(np.stack([(w*e*e).sum(0),(w*ce*ce).sum(0),(w*e*ce).sum(0),(w*poor*(e*e-ce*ce)).sum(0),(w*poor).sum(0),(w*se*se).sum(0),(w*e*se).sum(0)]))
        moments.append([[focus_stats(z['y'],p,pop) for pop in [np.ones(len(z['base'])),z['focus']]] for p in predictions]);stand.append([focus_stats(z['y'],p,np.ones(len(z['base']))) for p in standalone]);diagnostics.append(diag)
    moments=np.array(moments);stand=np.array(stand);diag=np.array(diagnostics);np.savez(OUT/'replication_moments.npz',moments=moments,standalone=stand,diagnostics=diag)
    counts=bootstrap_counts(256,np.random.default_rng(20261055),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0));delta=boot-boot[:,:1];point=scores-scores[:1];db=bootstrap_stats(diag,counts)
    variants={};contrasts={};gate=True
    for replica in [0,1]:
        for i,name in enumerate(FAMILIES):
            j=1+replica*4+i;d=diag[:,j-1].sum(0);variants[name+'_r'+str(replica)]={'combined_delta':point[j].mean(-1).tolist(),'targets_uniform_probability':point[j].tolist(),'wp_uniform_probability':scores[j].tolist(),'paired95':np.quantile(delta[:,j].mean(-1),[.025,.975],axis=0).tolist(),'standalone_uniform_wp':pooled_correlations(stand[:,j-1].sum(0)).tolist(),'error_cosine':(d[2]/np.sqrt(d[0]*d[1])).tolist(),'standalone_error_cosine':(d[6]/np.sqrt(d[0]*d[5])).tolist(),'poor_row_mse_improvement':(d[3]/d[4]).tolist(),'poor_row_mse_improvement_paired95':np.quantile(db[:,j-1,3]/db[:,j-1,4],[.025,.975],axis=0).tolist()}
        gate=gate and np.all(point[4+replica*4].mean(-1)>0)
        for control in ['gru','smooth']:
            j=4+replica*4;k=1+replica*4+FAMILIES.index(control);v=point[j]-point[k];dist=delta[:,j]-delta[:,k];contrasts['ssm_minus_'+control+'_r'+str(replica)]={'combined_delta':v.mean(-1).tolist(),'paired95':np.quantile(dist.mean(-1),[.025,.975],axis=0).tolist()};gate=gate and v.mean(-1)[1]>0
    for control in ['gru','smooth']:
        k=1+FAMILIES.index(control);d=((delta[:,4]-delta[:,k])+(delta[:,8]-delta[:,k+4]))*.5;gate=gate and np.quantile(d.mean(-1),.025,axis=0)[1]>0
    halves=[(pooled_correlations(x.sum(0))[4]-pooled_correlations(x.sum(0))[0]).mean(-1).tolist() for x in [moments[:128],moments[128:]]]
    gate=gate and np.all(np.array(halves)>0) and variants['ssm_r0']['paired95'][0][1]>0
    result={'variants':variants,'contrasts':contrasts,'primary_halves':halves,'search_gate_passed':bool(gate),'baseline_wp_uniform_probability':scores[0].tolist(),'replication_sequences':256,'boundary':'search-only; training-derived replication2 only','decision':'Qualified for one frozen search comparison' if gate else 'Ordered-memory mechanism did not pass replication; no official search.'}
    write_json(path,result);c.event('ssm_replication_completed',report_sha256=c.sha(path),gate=bool(gate),block='replication_2');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','prepare','train','replicate'],required=True);args=p.parse_args()
    with threadpool_limits(limits=1): {'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
