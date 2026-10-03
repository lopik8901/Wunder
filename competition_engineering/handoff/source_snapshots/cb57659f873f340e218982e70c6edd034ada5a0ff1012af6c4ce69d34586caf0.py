"""Controlled causal test of semantic bid/ask feature asymmetry."""
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from competition_engineering import prediction_shape_campaign as c
from competition_engineering.pipeline import cached,read_sequence,write_json
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from competition_engineering.isolation_retry import training_with_namespace_retry
from connectome.mlevolve_generated import validate_source
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library

ROOT=c.ROOT;OUT=c.OUT/'book_side';sha=c.sha

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    shape=json.loads((c.OUT/'shape_diagnostic_work/artifacts/crossfit.json').read_text());assert not shape['gate_passed']
    old=json.loads((c.OUT/'protocol.json').read_text());assert not (c.OUT/'prepared_selection.json').exists() and not (c.OUT/'prepared_development.json').exists()
    pf=pq.ParquetFile(c.prior.DATA);columns=[f'i{i}_v{j}' for i in [0,1] for j in range(22)]
    raw=np.concatenate([pf.read_row_group(g,columns=columns,use_threads=False).to_pandas().to_numpy() for g in old['roles']['fit'][:8]])
    write_json(OUT/'input_semantics_audit.json',{'groups':old['roles']['fit'][:8],'labels_read':False,'negative_volume_like_fraction':float((raw<0).mean()),'volume_like_quantiles':np.quantile(raw,[0,.01,.5,.99,1]).tolist(),'contract_sha256':sha(ROOT/'competition_engineering/assets/wnn_connectome_starterpack/docs/data_overview.md'),'constraint':'Bid/ask groups known; indices are identifiers, not depth ordering. Signed anonymized volume-like values are not physical positive queue sizes.'})
    r={'hypothesis':'Fixed bounded bid-minus-ask summaries of anonymized volume-like groups add incremental WP information beyond current features and frozen GRU states.',
       'mechanism':'Permutation-invariant within-side aggregates isolate semantic supply/demand asymmetry; matched bid-plus-ask summaries control nonlinear feature capacity. No physical queue-size, best-quote, depth ordering, time lag or new recurrent architecture is assumed.',
       'roles':old['roles'],'role_note':'384 fitting sequences reused from the failed shape diagnostic;128 selection and256 development sequences were reserved before any outcome and remain unread at this preregistration. Original untouched512 replication pool remains excluded.',
       'features':'Eight features: for each instrument normalized side-mean contrast, difference of side mean tanh, normalized difference of side second moments; plus each normalized side-mean contrast times the other instrument RMS/(1+RMS). Same eight plus-side controls. Use raw volume-like columns clipped[-8,8]; fit-only output normalization. Current115+fixed projected GRU128 are shared controls.',
       'training':'Two paired sampling seeds20261085/20261086. Same384x256=98304 fitting rows per readout, mean-one abs(clippedY) weighted residual ridge penalty0.1. Three families core,minus,plus; no feature/penalty sweep.',
       'selection':'Same128 sequence selection,1000rows each. Target strength grid[0,.05,.1,.25,.5,1] maximizes minimum uniform/proxy WP gain. Freeze all six models before development access.',
       'development_gate':'Primary minus_r0 combined gains>=0.0002 both populations, both95 lower>0, both128-sequence halves positive, targets>=-0.0002. Both paired fits must beat core and plus in BOTH populations; average paired95 lower>0 for both contrasts/populations. Only then freeze primary and consume replication_1 once under identical gate. No official search in this campaign.',
       'diagnosis':'Report all target effects, paired sequence intervals, matched-control contrasts, standalone scores omitting incumbent prediction inputs, poor-row error improvement and fitting WP. No claim of physical queue imbalance or causal price impact follows from this surrogate.',
       'failure':'No replication use on failure; no volume aggregation, feature width, temporal, weight or strength retuning.'}
    write_json(path,r)
    library=ROOT/'competition_engineering/research_cards/v16';shutil.copytree(ROOT/'competition_engineering/research_cards/v15',library)
    card=json.loads((library/'weighted_correlation_ratio.json').read_text())
    card.update(card_id='queue_imbalance_predictor',title='Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book',authors=['Martin D. Gould','Julius Bonart'],year=2015,source_url='https://arxiv.org/pdf/1512.03492',source_locator='Section3.3 Eq7; abstract and Sections5-6 out-of-sample classifiers',curator='Codex; primary equation and study-scope review',family='book_side_asymmetry',mechanism='The paper relates normalized best-bid/ask queue imbalance to the direction of the next mid-price move using logistic and local logistic models.',limitations='Connectome feature indices are not depths and volume-like values are signed/anonymized. Actual queue imbalance cannot be reconstructed from this contract. Side-aggregate surrogate is a curator hypothesis; horizons, scoring and instruments differ.',concise_relevance='Curator inference: test whether bounded semantic bid-minus-ask summaries add WP beyond raw inputs and frozen states, using matched bid-plus-ask controls before any architecture change.',diagnostic_signatures='Replicated incremental WP for side contrast beyond shared state/current and common-mode nonlinear controls.',causality='Use current inputs only; no future rows, labels, sequence identity or online target feedback.',incremental_inference='Eight fixed group summaries; no added state.',tags=['order-book','side-asymmetry','surrogate-feature'],assumptions='The known side partition may retain useful asymmetry despite anonymization. Physical magnitude or ordered-level assumptions are not made.',useful_when='Current raw features and frozen state fail to capture a small semantic nonlinear side-competition basis.',training_compute='Fixed matched linear readouts, not a new large network.',cpu_deployment='Full callback timing remains required if development and independent replication qualify.')
    write_json(library/'queue_imbalance_predictor.json',validate_card(card));cards,h=load_library(library);write_json(OUT/'research_library.json',{'version':16,'cards':len(cards),'manifest_sha256':h})
    k=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==23;k['version']=24
    k['established'].append({'lesson':'Historical seven-knot prediction calibration existed but was absent from the knowledge map; source audit now prevents claiming the mechanism is new.','source':'manual_spline_calibration.py'})
    k['weakened'].append({'hypothesis':'Nonlinear prediction-only conditional mean offers a transferable WP improvement.','reason':'Fresh two-way crossfit uniform contrasts-0.001590/-0.000777; proxy-0.000139/+0.000032, intervals cross zero. No selection/development labels or replication consumed.','source':'runs/prediction_shape_20261003/shape_diagnostic_work/artifacts/crossfit.json'})
    k['closed_branches'].append('Prediction-only spline shape tuning without new transport evidence')
    k['status']='Fixed semantic side-asymmetry experiment prospectively registered.';k['next_research_question']=r['hypothesis'];write_json(OUT/'knowledge_v24.json',k);write_json(ROOT/'competition_engineering/search_knowledge.json',k)
    c.event('book_side_experiment_preregistered',protocol_sha256=sha(path),knowledge_sha256=sha(OUT/'knowledge_v24.json'));return r

def side_features(x,plus=False):
    x=np.clip(np.asarray(x,dtype=float),-8,8);features=[];pressures=[];activities=[]
    sign=1 if plus else -1
    for offset in [0,52]:
        b=x[:,offset+22:offset+33];a=x[:,offset+33:offset+44]
        bm=b.mean(1);am=a.mean(1);b2=(b*b).mean(1);a2=(a*a).mean(1);rms=np.sqrt((b2+a2)/2)
        pressure=(bm+sign*am)/(1+rms)
        features.extend([pressure,np.tanh(b).mean(1)+sign*np.tanh(a).mean(1),(b2+sign*a2)/(1+b2+a2)])
        pressures.append(pressure);activities.append(rms/(1+rms))
    features.extend([pressures[0]*activities[1],pressures[1]*activities[0]])
    return np.column_stack(features).astype(np.float32)

def prepare(role):
    r=initialize();assert role in ['fit','selection','development']
    if role=='development':assert (OUT/'fit_work/identity.json').exists(),'Freeze candidate before development'
    directory=OUT/('development' if role=='development' else 'training');directory.mkdir(exist_ok=True)
    combo=c.prior.load_combo();pf=pq.ParquetFile(c.prior.DATA);extractor=c.LatentExtractor()
    with np.load(c.prior.OUT/'basis.npz') as q:projection=q['gru']
    with np.load(ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz') as q:trees={k:q[k] for k in q.files}
    previous={v['group']:v['source_sequence_sha256'] for v in json.loads((c.OUT/'prepared_fit.json').read_text())['records']}
    records=[]
    for count,group in enumerate(r['roles'][role]):
        path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():row=json.loads(meta.read_text());assert sha(path)==row['derived_sha256'];records.append(row);continue
        seq,need,x,y,mask=read_sequence(pf,group);source_hash=c.prior.hashlib.sha256(x.tobytes()+y.tobytes()+need.tobytes()).hexdigest()
        if role=='fit':assert source_hash==previous[group]
        p,h=extractor.predict(x);z={'x':x,'y':y,'p':p,'step':np.arange(len(x)),'need':need};base=c.prior.predict_combo(z,combo)
        idx=np.sort(np.random.default_rng(20261084+group).choice(np.flatnonzero(need),1000,replace=False));nx=np.clip((x[idx]-combo['mean'])/combo['scale'],-8,8)
        core=np.column_stack((np.ones(1000),np.clip(base[idx],-2,2),nx)).astype(np.float32)
        focus=c.probability_focus(x[idx],base[idx],z['step'][idx],combo,trees)[:,1]
        np.savez(path,core=core,gru=h[idx]@projection,minus=side_features(x[idx]),plus=side_features(x[idx],True),base=base[idx],y=y[idx],focus=focus,group=group,seq_ix=seq,indices=idx,role=0 if role=='fit' else 1 if role=='selection' else 2)
        row={'group':group,'seq_ix':seq,'role':role,'source_sequence_sha256':source_hash,'derived_sha256':sha(path)};write_json(meta,row);records.append(row)
        if (count+1)%128==0:print(role+' prepared '+str(count+1),flush=True)
    write_json(OUT/('prepared_'+role+'.json'),{'records':records});c.event('book_side_data_prepared',role=role,manifest_sha256=sha(OUT/('prepared_'+role+'.json')))
    if role=='selection':write_json(directory/'identity.json',{'groups':r['roles']['fit']+r['roles']['selection']})
    if role=='development':write_json(directory/'identity.json',{'groups':r['roles'][role]})

def design(z,family,model,standalone=False):
    current=z['core'] if not standalone else np.column_stack((z['core'][:,0],z['core'][:,3:]))
    f=np.column_stack((current,np.clip((z['gru']-model['gru_mean'])/model['gru_scale'],-8,8)))
    if family!='core':f=np.column_stack((f,np.clip((z[family]-model[family+'_mean'])/model[family+'_scale'],-8,8)))
    return f.astype(float)

def fit(train_files,output_dir):
    model={};report={};families=['core','minus','plus']
    for name,dim in [('gru',128),('minus',8),('plus',8)]:
        sx=np.zeros(dim);sxx=np.zeros(dim);n=0
        for path in train_files:
            with np.load(path) as q:
                if int(q['role'])!=0:continue
                a=q[name].astype(float);sx+=a.sum(0);sxx+=(a*a).sum(0);n+=len(a)
        mean=sx/n;model[name+'_mean']=mean;model[name+'_scale']=np.sqrt(np.maximum(sxx/n-mean*mean,.05**2))
    poor=[]
    for path in train_files:
        with np.load(path) as q:
            if int(q['role'])==1:poor.append(np.abs(np.clip(q['y'],-2,2)-np.clip(q['base'],-2,2)))
    model['poor_threshold']=np.quantile(np.concatenate(poor),.75,axis=0)
    for replica in [0,1]:
        for family in families:
            dim=243+(8 if family!='core' else 0);gram=np.zeros((2,dim,dim));rhs=np.zeros((dim,2));sg=np.zeros((2,dim-2,dim-2));sr=np.zeros((dim-2,2));mass=np.zeros(2);rows=0
            for path in train_files:
                with np.load(path) as q:z={k:q[k] for k in q.files}
                if int(z['role'])!=0:continue
                idx=np.random.default_rng(20261085+replica+int(z['group'])).permutation(len(z['base']))[:256]
                f=design(z,family,model)[idx];sf=design(z,family,model,True)[idx];y=np.clip(z['y'][idx],-2,2).astype(float);e=y-z['base'][idx]
                for t in [0,1]:
                    w=np.abs(y[:,t]);mass[t]+=w.sum();gram[t]+=f.T@(w[:,None]*f);rhs[:,t]+=f.T@(w*e[:,t]);sg[t]+=sf.T@(w[:,None]*sf);sr[:,t]+=sf.T@(w*y[:,t])
                rows+=len(idx)
            assert rows==98304
            coef=np.column_stack([np.linalg.solve(gram[t]/mass[t]+np.eye(dim)*.1,rhs[:,t]/mass[t]) for t in [0,1]])
            standalone=np.column_stack([np.linalg.solve(sg[t]/mass[t]+np.eye(dim-2)*.1,sr[:,t]/mass[t]) for t in [0,1]])
            moments=np.zeros((2,6,6,2))
            for path in train_files:
                with np.load(path) as q:z={k:q[k] for k in q.files}
                if int(z['role'])!=1:continue
                correction=design(z,family,model)@coef
                for pop,w in enumerate([np.ones(len(correction)),z['focus']]):moments[pop]+=grid_stats(z['y'],z['base'],correction,w)
            choice=choose_strengths(moments);key=family+'_r'+str(replica);model[key+'_coef']=coef;model[key+'_standalone']=standalone;model[key+'_strength']=np.array(choice['strengths'])
            report[key]={'choice':choice,'rows':rows};print('frozen '+key,flush=True)
    np.savez(output_dir+'/models.npz',**model);Path(output_dir+'/selection.json').write_text(json.dumps(report),encoding='utf-8');return {'models':6,'rows_per_model':98304}

def source():
    code='import numpy as np\nimport json\nfrom pathlib import Path\n'
    for f in [design,c.from_stats,c.focus_stats,c.grid_stats,c.choose_strengths,fit]:code+=inspect.getsource(f)+'\n'
    code+='def train(train_files,output_dir):\n    return fit(train_files,output_dir)\n';validate_source(code,training=True);return code

def train():
    initialize();work=OUT/'fit_work';work.mkdir(exist_ok=True)
    for role in ['fit','selection']:
        for row in json.loads((OUT/('prepared_'+role+'.json')).read_text())['records']:assert sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    if not (work/'artifacts/models.npz').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');c.event('book_side_fit_started',source_sha256=sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);deadline=time.monotonic()+1800
        def retry(attempt,stderr):
            (work/('namespace_attempt_'+str(attempt)+'.stderr')).write_bytes(stderr);c.event('infrastructure_retry',attempt=attempt,isolation_unchanged=True)
        process,status,stdout,stderr=training_with_namespace_retry(lambda:_launch(sandbox,work,'train',deadline,OUT/'training'),deadline,retry)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:c.event('implementation_failure',component='book side fit',error=stderr.decode(errors='replace')[-1500:]);raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={p:sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/selection.json']}
    if (work/'identity.json').exists():assert json.loads((work/'identity.json').read_text())==identity
    else:write_json(work/'identity.json',identity);c.event('book_side_models_frozen_before_development',identity=identity)

def evaluate():
    path=OUT/'development.json'
    if path.exists():print(path.read_text());return
    work=OUT/'fit_work';assert all(sha(work/p)==h for p,h in json.loads((work/'identity.json').read_text()).items())
    with np.load(work/'artifacts/models.npz') as q:model={k:q[k] for k in q.files}
    keys=[family+'_r'+str(replica) for replica in [0,1] for family in ['core','minus','plus']]
    expected={v['group']:v['derived_sha256'] for v in json.loads((OUT/'prepared_development.json').read_text())['records']};moments=[];stand=[];diagnostics=[]
    for group,z in cached(OUT/'development'):
        assert sha(OUT/'development'/f'{group:05d}.npz')==expected[group]
        predictions=[z['base']];standalone=[];diag=[]
        for key in keys:
            family=key.split('_')[0];p=z['base']+design(z,family,model)@model[key+'_coef']*model[key+'_strength'];predictions.append(p);s=design(z,family,model,True)@model[key+'_standalone'];standalone.append(s)
            y=np.clip(z['y'],-2,2);e=y-np.clip(z['base'],-2,2);ce=y-np.clip(p,-2,2);w=np.abs(y);poor=np.abs(e)>=model['poor_threshold'];diag.append(np.stack([(w*e*e).sum(0),(w*ce*ce).sum(0),(w*e*ce).sum(0),(w*poor*(e*e-ce*ce)).sum(0),(w*poor).sum(0)]))
        moments.append([[c.focus_stats(z['y'],p,w) for w in [np.ones(len(z['base'])),z['focus']]] for p in predictions]);stand.append([c.focus_stats(z['y'],p,np.ones(len(p))) for p in standalone]);diagnostics.append(diag)
    moments=np.array(moments);stand=np.array(stand);diagnostics=np.array(diagnostics);counts=c.bootstrap_counts(256,np.random.default_rng(20261087),draws=10000)
    scores=c.pooled_correlations(moments.sum(0));boot=c.pooled_correlations(c.bootstrap_stats(moments,counts));point=scores-scores[:1];delta=boot-boot[:,:1];variants={}
    for j,key in enumerate(keys,1):
        d=diagnostics[:,j-1].sum(0);variants[key]={'target_deltas':point[j].tolist(),'combined_deltas':point[j].mean(-1).tolist(),'paired95':np.quantile(delta[:,j].mean(-1),[.025,.975],axis=0).tolist(),'strength':model[key+'_strength'].tolist(),'standalone_uniform_wp':c.pooled_correlations(stand[:,j-1].sum(0)).tolist(),'error_cosine':(d[2]/np.sqrt(d[0]*d[1])).tolist(),'poor_row_weighted_mse_improvement':(d[3]/d[4]).tolist()}
    contrasts={};mechanism=True
    for control,js in [('core',[1,4]),('plus',[3,6])]:
        target_point=np.stack([point[2]-point[js[0]],point[5]-point[js[1]]]);mean_draw=((delta[:,2]-delta[:,js[0]])+(delta[:,5]-delta[:,js[1]]))/2;ci=np.quantile(mean_draw.mean(-1),[.025,.975],axis=0)
        contrasts[control]={'paired_fit_combined':target_point.mean(-1).tolist(),'mean_paired95':ci.tolist()};mechanism=mechanism and bool(np.all(target_point.mean(-1)>0) and np.all(ci[0]>0))
    halves=[]
    for m in [moments[:128],moments[128:]]:
        s=c.pooled_correlations(m.sum(0));halves.append((s[2]-s[0]).mean(-1))
    ci=np.array(variants['minus_r0']['paired95']);gate=bool(mechanism and np.all(point[2].mean(-1)>=.0002) and np.all(ci[0]>0) and np.all(np.array(halves)>0) and np.all(point[2]>=-.0002))
    np.savez(OUT/'development_moments.npz',moments=moments,standalone=stand,diagnostics=diagnostics)
    r={'variants':variants,'contrasts':contrasts,'halves_primary':np.array(halves).tolist(),'development_gate_passed':gate,'decision':'Freeze primary and use untouched replication_1 once' if gate else 'Close this fixed side-aggregate surrogate; no replication consumption.'}
    write_json(path,r);c.event('book_side_development_evaluated',gate=gate,report_sha256=sha(path));print(json.dumps(r),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initialize','prepare_fit','prepare_selection','train','prepare_development','evaluate'],required=True);a=p.parse_args()
    with threadpool_limits(limits=1):
        if a.phase.startswith('prepare_'):prepare(a.phase[8:])
        else:{'initialize':initialize,'train':train,'evaluate':evaluate}[a.phase]()
