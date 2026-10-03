"""Prospectively reserved training-only probes of different causal representations."""
import argparse
import hashlib
import inspect
import json
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from competition_engineering.manual_search_core import ROOT,COMBO,COMBO_SHA,load_combo,predict_combo
from competition_engineering.pipeline import read_sequence,write_json,cached
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.probability_readout import probability_focus
from competition_engineering.replicated_readout import grid_stats,choose_strengths
from competition_engineering.residual import from_stats
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations
from connectome.mlevolve_generated import validate_source
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library

OUT=ROOT/'competition_engineering/runs/alternative_representation_20261003'
DATA=ROOT/'competition_engineering/assets/wnn_connectome_starterpack/datasets/train.parquet'
FAMILIES=['core','instant','gru','delay','tcn']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def event(kind,**payload):
    OUT.mkdir(exist_ok=True);p=OUT/'ledger.jsonl';row={'utc':datetime.now(timezone.utc).isoformat(),'boundary':'search-only','kind':kind,'previous_ledger_sha256':sha(p) if p.exists() else None,**payload}
    with p.open('a',encoding='utf-8') as f: f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def sequence_identity(pf,group):
    stat=pf.metadata.row_group(group).column(0).statistics
    if stat is not None and stat.has_min_max:
        if stat.min!=stat.max or stat.null_count!=0: raise ValueError('Nonunique/null sequence identity')
        return int(stat.min)
    values=pf.read_row_group(group,columns=['seq_ix'],use_threads=False)['seq_ix'].to_numpy()
    if not len(values) or not np.all(values==values[0]): raise ValueError('Nonunique sequence identity')
    return int(values[0])

def reserve():
    OUT.mkdir(exist_ok=True);path=OUT/'reservation.json'
    if path.exists(): return json.loads(path.read_text())
    assert sha(COMBO)==COMBO_SHA
    excluded=set();sources={}
    for name in ['gru_train_medium_4096','gru_train_scale_phase2','gru_train_pilot','gru_sequence_smoke','gru_smoke']:
        p=ROOT/'competition_engineering/cache'/name/'identity.json'
        if p.exists():
            identity=json.loads(p.read_text());excluded.update(identity['groups']);sources[p.relative_to(ROOT).as_posix()]=sha(p)
    # Explicitly exclude training groups previously assigned to protected training evaluation,
    # using role identities only from the train split manifest; never read their outcomes/cache.
    manifest=json.loads((ROOT/'competition_engineering/dataset_manifest_phase2.json').read_text())
    for key,groups in manifest['splits'].items():
        if key.startswith('train') and key!='train_all': excluded.update(groups)
    pf=pq.ParquetFile(DATA);meta=pf.metadata;eligible=sorted(set(range(pf.num_row_groups))-excluded)
    assert len(eligible)>=1664
    groups=np.random.default_rng(20261050).permutation(eligible).tolist()
    roles={'fit':groups[:512],'development':groups[512:640]}
    for i in range(4): roles['replication_'+str(i+1)]=groups[640+i*256:896+i*256]
    identities=[]
    for role,values in roles.items():
        for group in values:
            identities.append({'row_group':group,'seq_ix':sequence_identity(pf,group),'role':role,'rows':meta.row_group(group).num_rows})
    assert len({r['seq_ix'] for r in identities})==1664
    r={'seed':20261050,'roles':roles,'identities':identities,'excluded_groups':sorted(excluded),'excluded_identity_sha256':sources,
       'training_file':{'path':DATA.relative_to(ROOT).as_posix(),'bytes':DATA.stat().st_size,'metadata_sha256':hashlib.sha256(meta.serialized_size.to_bytes(8,'little')+str(pf.schema_arrow).encode()+json.dumps(identities,sort_keys=True).encode()).hexdigest()},
       'independence':'All reserved groups are outside earlier designated research and protected training-role groups. Frozen incumbent may have seen these sequences in pretraining; independence is from new correction fit/selection, not pretrained model parameters.',
       'replication_use':'Four disjoint blocks reserved before architecture results. Each block consumed once; no adaptive tuning on a consumed block.'}
    write_json(path,r)
    k=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==14
    write_json(OUT/'starting_knowledge_v14.json',k);k['version']=15
    k['closed_branches']+=['Frozen GRU linear/nonlinear readout tuning without new mechanism evidence','Fitting-sequence count tuning without new evidence']
    k['status']='New independently reserved training-only alternative-representation campaign.'
    k['next_research_question']='Does explicit finite-lag or dilated convolutional raw-input structure provide incremental signal beyond frozen GRU features under matched readouts?'
    k['established'].append({'lesson':'Sequence-count comparisons were inconclusive, including a composition-controlled replication; the old untouched research pool is exhausted. New candidate fitting and replication roles must be reserved prospectively.','source':'runs/sequence_diversity_20261003/FINAL_REPORT.json'})
    write_json(OUT/'knowledge_v15.json',k);write_json(ROOT/'competition_engineering/search_knowledge.json',k)
    event('sequence_roles_reserved',reservation_sha256=sha(path),knowledge_sha256=sha(OUT/'knowledge_v15.json'));return r

def research():
    library=ROOT/'competition_engineering/research_cards/v13'
    if not library.exists():
        shutil.copytree(ROOT/'competition_engineering/research_cards/v12',library)
        p=library/'generic_tcn.json';card=json.loads(p.read_text());card.update(version=2,verified_on='2026-10-03',source_locator='Sections3.2-3.4: causal convolutions, dilation and residual connections',
          curator='Codex; primary full-paper architecture review',limitations='Benchmark comparisons do not establish Connectome benefit. A frozen random convolution probe is not a trained TCN and cannot reject the trained family.',
          concise_relevance='Curator inference: matched raw-delay, current-only and GRU probes can isolate finite-lag signal before end-to-end convolution training.')
        write_json(p,validate_card(card))
        card=json.loads((library/'online_polynomial_memory.json').read_text());card.update(card_id='legendre_memory_unit',title='Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks',authors=['Aaron Voelker','Ivana Kajić','Chris Eliasmith'],year=2019,version=1,source_url='https://compneuro.uwaterloo.ca/publications/voelker2019lmu.html',source_locator='Primary author publication abstract; NeurIPS2019 paper linked therein',verified_on='2026-10-03',mechanism='A structured linear dynamical memory approximates a sliding-window signal with orthogonal polynomial coefficients.',limitations='Orthogonal memory differs from independent EMA summaries; fixed-window and discretization assumptions need checking. No Connectome benefit is claimed.',concise_relevance='Curator inference: test ordered history information with a compact stable state update if finite-lag diagnostics justify a longer-window representation.')
        write_json(library/'legendre_memory_unit.json',validate_card(card))
    cards,h=load_library(library);write_json(OUT/'research_library.json',{'version':13,'cards':len(cards),'manifest_sha256':h})

def initialize():
    reservation=reserve();p=OUT/'probe_protocol.json'
    if p.exists(): return json.loads(p.read_text())
    research()
    r={'observation':'GRU-state probes and extra nonlinear readout capacity fit training better without credible official-search gains. New disjoint data are available.',
       'hypothesis':'Explicit bounded history of raw observations carries incremental correction information missed by the GRU representation.',
       'mechanism':'Exact delays preserve observations; fixed dilated causal convolutions mix them nonlinearly without recurrent compression or additional optimization.',
       'experiment':'Same512 fitting groups,128 sampled rows each, two paired sampling seeds; same128 development groups,500rows each. Mean-one abs(clipped target) residual ridge, penalty0.1 per row, same per-target WP strength grid. All augmented families128 features plus115 current core.',
       'families':FAMILIES,'primary':'tcn replica0; replica1 repeatability, no best-seed selection',
       'basis':'Fixed seed20261051. Input projection112x32;4 kernel2 tanh causal convolutions width32, dilations1,4,16,64; concatenate4 layer states128, receptive field86rows. Delay uses projected input at1,4,16,64. Instant uses128 random tanh current features; GRU projected256to128. Fit-only normalization.',
       'success':'TCN pooled gains positive in both uniform/probability populations and both128-sequence replication halves; probability95 lower>0; TCN-minus-instant and TCN-minus-GRU probability contrasts positive in both paired fits and average95 lower>0. Delay contrast is diagnostic, not mandatory.',
       'failure':'Failed gate blocks official-search access; no strengths/architecture tuning on replication. Random-probe failure is not rejection of learned convolution.',
       'replication_block':'replication_1','complementarity':'Freeze poor-row thresholds at development 75th percentile absolute incumbent error per target. Report error correlation, weighted residual improvement and correction/error covariance on fresh rows; simple residual correction already forms the frozen ensemble.',
       'standalone':'Report direct ridge target prediction from normalized raw current inputs plus each alternative128 representation, omitting incumbent prediction features. This is a frozen-feature standalone probe, not a trained full architecture.',
       'search':'After gate only, freeze one primary and run one predeclared comparison; no search-based tuning. Qualification delta>=0.0002, paired99 lower>0, each target delta>=-0.0002, callback<=76.9275378083us/row.'}
    write_json(p,r)
    rng=np.random.default_rng(20261051)
    np.savez(OUT/'basis.npz',projection=(rng.normal(size=(112,32))/np.sqrt(112)).astype(np.float32),instant=(rng.normal(size=(112,128))/np.sqrt(112)).astype(np.float32),gru=(rng.normal(size=(256,128))/np.sqrt(256)).astype(np.float32),weights=(rng.normal(size=(4,2,32,32))/8).astype(np.float32),bias=rng.uniform(-.5,.5,size=(4,32)).astype(np.float32))
    event('experiment_preregistered',protocol_sha256=sha(p),basis_sha256=sha(OUT/'basis.npz'));return r

def lag(a,d):
    b=np.zeros_like(a)
    if d<len(a): b[d:]=a[:-d]
    return b

def temporal_features(x,basis):
    projected=x@basis['projection'];delay=np.concatenate([lag(projected,d) for d in [1,4,16,64]],axis=1)
    h=projected;layers=[]
    for i,d in enumerate([1,4,16,64]):
        h=np.tanh(h@basis['weights'][i,0]+lag(h,d)@basis['weights'][i,1]+basis['bias'][i]);layers.append(h)
    return delay,np.concatenate(layers,axis=1)

def prepare(block='replication_1'):
    initialize();r=reserve();combo=load_combo();extractor=LatentExtractor();pf=pq.ParquetFile(DATA)
    with np.load(OUT/'basis.npz') as q: basis={k:q[k] for k in q.files}
    teacher=ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q: trees={k:q[k] for k in q.files}
    allrecords=[]
    for role in ['fit','development',block]:
        directory=OUT/('training' if role in ['fit','development'] else role);directory.mkdir(exist_ok=True)
        for count,group in enumerate(r['roles'][role]):
            path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                row=json.loads(meta.read_text());assert sha(path)==row['derived_sha256'];allrecords.append(row);continue
            seq,need,x,y,mask=read_sequence(pf,group);p,hidden=extractor.predict(x)
            z={'x':x,'y':y,'p':p,'step':np.arange(len(x)),'need':need};base_full=predict_combo(z,combo)
            idx=np.sort(np.random.default_rng(20261052+group).choice(np.flatnonzero(need),500,replace=False))
            nx=np.clip((x-combo['mean'])/combo['scale'],-8,8).astype(np.float32)
            delay,tcn=temporal_features(nx,basis);base=base_full[idx]
            core=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),nx[idx])).astype(np.float32)
            focus=probability_focus(x[idx],base,z['step'][idx],combo,trees)[:,1]
            np.savez(path,core=core,instant=np.tanh(nx[idx]@basis['instant']),gru=hidden[idx]@basis['gru'],delay=delay[idx],tcn=tcn[idx],base=base,y=y[idx],focus=focus,indices=idx,group=group,seq_ix=seq,role=0 if role=='fit' else 1 if role=='development' else 2)
            # Hash sequence content independently of Parquet compression and record exact identity.
            content=hashlib.sha256(x.tobytes()+y.tobytes()+need.tobytes()).hexdigest()
            row={'row_group':group,'seq_ix':seq,'role':role,'source_sequence_sha256':content,'derived_sha256':sha(path)};write_json(meta,row);allrecords.append(row)
            if (count+1)%64==0: print(role+' prepared '+str(count+1),flush=True)
        write_json(directory/'identity.json',{'groups':r['roles']['fit']+r['roles']['development'] if role in ['fit','development'] else r['roles'][role]})
    manifest=OUT/('prepared_'+block+'.json')
    if not manifest.exists(): write_json(manifest,{'records':allrecords});event('data_prepared',manifest_sha256=sha(manifest),block=block)

def design(z,name,mean,scale,standalone=False):
    core=z['core'] if not standalone else np.column_stack((z['core'][:,0],z['core'][:,3:]))
    if name=='core': return core
    return np.column_stack((core,np.clip((z[name]-mean)/scale,-8,8))).astype(np.float32)

def fit(train_files,output_dir):
    families=['core','instant','gru','delay','tcn'];models={};report={}
    for name in families:
        if name=='core': mean=np.zeros(0);scale=np.ones(0)
        else:
            sx=np.zeros(128);sxx=np.zeros(128);n=0
            for path in train_files:
                with np.load(path) as z:
                    if int(z['role'])!=0: continue
                    a=z[name].astype(float);sx+=a.sum(0);sxx+=(a*a).sum(0);n+=len(a)
            mean=sx/n;scale=np.sqrt(np.maximum(sxx/n-mean*mean,.05**2))
        models[name+'_mean']=mean;models[name+'_scale']=scale
        for replica in [0,1]:
            dim=115+(0 if name=='core' else 128);sdim=dim-2
            grams=np.zeros((2,dim,dim));rhs=np.zeros((dim,2));sg=np.zeros((2,sdim,sdim));sr=np.zeros((sdim,2));mass=np.zeros(2);rows=0
            for path in train_files:
                with np.load(path) as q: z={k:q[k] for k in q.files}
                if int(z['role'])!=0: continue
                idx=np.random.default_rng(20261053+int(z['group'])+replica*100000).permutation(len(z['base']))[:128]
                f=design(z,name,mean,scale)[idx].astype(float);sf=design(z,name,mean,scale,True)[idx].astype(float)
                y=np.clip(z['y'][idx],-2,2).astype(float);weight=np.abs(y);residual=y-z['base'][idx]
                for t in [0,1]:
                    w=weight[:,t];grams[t]+=f.T@(w[:,None]*f);rhs[:,t]+=f.T@(w*residual[:,t]);sg[t]+=sf.T@(w[:,None]*sf);sr[:,t]+=sf.T@(w*y[:,t]);mass[t]+=w.sum()
                rows+=len(idx)
            assert rows==65536
            coef=np.column_stack([np.linalg.solve(grams[t]/mass[t]*rows+np.eye(dim)*rows*.1,rhs[:,t]/mass[t]*rows) for t in [0,1]])
            standalone=np.column_stack([np.linalg.solve(sg[t]/mass[t]*rows+np.eye(sdim)*rows*.1,sr[:,t]/mass[t]*rows) for t in [0,1]])
            moments=np.zeros((2,6,6,2));poor=[]
            for path in train_files:
                with np.load(path) as q: z={k:q[k] for k in q.files}
                if int(z['role'])!=1: continue
                f=design(z,name,mean,scale);correction=f@coef
                for pop,population in enumerate([np.ones(len(f)),z['focus']]): moments[pop]+=grid_stats(z['y'],z['base'],correction,population)
                poor.append(np.abs(np.clip(z['y'],-2,2)-np.clip(z['base'],-2,2)))
            choice=choose_strengths(moments);key=name+'_r'+str(replica)
            models[key+'_coef']=coef.astype(np.float32);models[key+'_strength']=np.array(choice['strengths']);models[key+'_standalone']=standalone.astype(np.float32)
            report[key]={'choice':choice,'rows':rows}
            models['poor_threshold']=np.quantile(np.concatenate(poor),.75,axis=0)
            print('frozen '+key,flush=True)
    np.savez(output_dir+'/models.npz',**models);Path(output_dir+'/selection.json').write_text(json.dumps(report),encoding='utf-8')
    return {'models':10,'rows_per_model':65536}

def source():
    code='import numpy as np\nimport json\nfrom pathlib import Path\n'
    for f in [from_stats,focus_stats,grid_stats,choose_strengths,design,fit]: code+=inspect.getsource(f)+'\n'
    code+='def train(train_files,output_dir):\n    return fit(train_files,output_dir)\n'
    validate_source(code,training=True);return code

def train():
    initialize();work=OUT/'probe_fit';work.mkdir(exist_ok=True)
    for row in json.loads((OUT/'prepared_replication_1.json').read_text())['records']:
        if row['role'] in ['fit','development']: assert sha(OUT/'training'/f"{row['row_group']:05d}.npz")==row['derived_sha256']
    if not (work/'artifacts/models.npz').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');event('fit_started',source_sha256=sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);process,status,watcher=_launch(sandbox,work,'train',time.monotonic()+1800,OUT/'training')
        stdout,stderr=process.communicate(timeout=1810);watcher.join(timeout=1)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:
            event('implementation_failure',status=status,error=stderr.decode(errors='replace')[-1500:]);raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={p:sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/selection.json']}
    if (work/'identity.json').exists(): assert json.loads((work/'identity.json').read_text())==identity
    else: write_json(work/'identity.json',identity);event('models_frozen',identity=identity)

def replicate():
    path=OUT/'probe_replication.json'
    if path.exists(): print(path.read_text());return
    work=OUT/'probe_fit';assert all(sha(work/p)==h for p,h in json.loads((work/'identity.json').read_text()).items())
    with np.load(work/'artifacts/models.npz') as q: model={k:q[k] for k in q.files}
    expected={r['row_group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_replication_1.json').read_text())['records'] if r['role']=='replication_1'}
    moments=[];stand=[];diagnostics=[]
    for group,z in cached(OUT/'replication_1'):
        assert sha(OUT/'replication_1'/f"{int(z['group']):05d}.npz")==expected[int(z['group'])]
        predictions=[z['base']];standalone=[];diag=[]
        for r in [0,1]:
            for name in FAMILIES:
                key=name+'_r'+str(r);f=design(z,name,model[name+'_mean'],model[name+'_scale']);p=z['base']+f@model[key+'_coef']*model[key+'_strength'];predictions.append(p)
                s=design(z,name,model[name+'_mean'],model[name+'_scale'],True)@model[key+'_standalone'];standalone.append(s)
                y=np.clip(z['y'],-2,2);e=y-np.clip(z['base'],-2,2);ce=y-np.clip(p,-2,2);w=np.abs(y);poor=np.abs(e)>=model['poor_threshold'];correction=p-z['base']
                se=y-np.clip(s,-2,2)
                diag.append(np.stack([(w*e*e).sum(0),(w*ce*ce).sum(0),(w*e*ce).sum(0),(w*e*correction).sum(0),(w*poor*(e*e-ce*ce)).sum(0),(w*poor).sum(0),(w*se*se).sum(0),(w*e*se).sum(0),(w*poor*(e*e-se*se)).sum(0)]))
        moments.append([[focus_stats(z['y'],p,pop) for pop in [np.ones(len(z['base'])),z['focus']]] for p in predictions])
        stand.append([focus_stats(z['y'],p,np.ones(len(z['base']))) for p in standalone]);diagnostics.append(diag)
    moments=np.array(moments);stand=np.array(stand);diag=np.array(diagnostics)
    np.savez(OUT/'probe_replication_moments.npz',moments=moments,standalone=stand,diagnostics=diag)
    counts=bootstrap_counts(256,np.random.default_rng(20261054),draws=10000);boot=pooled_correlations(bootstrap_stats(moments,counts));scores=pooled_correlations(moments.sum(0));delta=boot-boot[:,:1];point=scores-scores[:1];diag_boot=bootstrap_stats(diag,counts)
    variants={};contrasts={}
    for r in [0,1]:
        for i,name in enumerate(FAMILIES):
            j=1+r*5+i;d=diag[:,j-1].sum(0)
            variants[name+'_r'+str(r)]={'wp_uniform_probability':scores[j].tolist(),'delta_uniform_probability':point[j].tolist(),'combined_delta':point[j].mean(-1).tolist(),'paired95':np.quantile(delta[:,j].mean(-1),[.025,.975],axis=0).tolist(),'standalone_uniform_wp':pooled_correlations(stand[:,j-1].sum(0)).tolist(),'error_cosine':(d[2]/np.sqrt(d[0]*d[1])).tolist(),'standalone_error_cosine':(d[7]/np.sqrt(d[0]*d[6])).tolist(),'standalone_poor_row_weighted_mse_improvement':(d[8]/d[5]).tolist(),'poor_row_weighted_mse_improvement':(d[4]/d[5]).tolist(),'error_correction_inner_product':d[3].tolist()}
            variants[name+'_r'+str(r)]['poor_row_mse_improvement_paired95']=np.quantile(diag_boot[:,j-1,4]/diag_boot[:,j-1,5],[.025,.975],axis=0).tolist()
            variants[name+'_r'+str(r)]['standalone_poor_row_mse_improvement_paired95']=np.quantile(diag_boot[:,j-1,8]/diag_boot[:,j-1,5],[.025,.975],axis=0).tolist()
        for control in ['instant','gru','delay']:
            j=5+r*5;k=1+r*5+FAMILIES.index(control);v=point[j]-point[k];dist=delta[:,j]-delta[:,k]
            contrasts['tcn_minus_'+control+'_r'+str(r)]={'combined_delta':v.mean(-1).tolist(),'paired95':np.quantile(dist.mean(-1),[.025,.975],axis=0).tolist()}
    halves=[(pooled_correlations(x.sum(0))[5]-pooled_correlations(x.sum(0))[0]).mean(-1).tolist() for x in [moments[:128],moments[128:]]]
    gate=bool(np.all(point[5].mean(-1)>0) and np.all(np.array(halves)>0) and variants['tcn_r0']['paired95'][0][1]>0)
    for control in ['instant','gru']:
        k=1+FAMILIES.index(control);d=((delta[:,5]-delta[:,k])+(delta[:,10]-delta[:,k+5]))*.5
        gate=gate and all(contrasts['tcn_minus_'+control+'_r'+str(r)]['combined_delta'][1]>0 for r in [0,1]) and np.quantile(d.mean(-1),.025,axis=0)[1]>0
    result={'variants':variants,'contrasts':contrasts,'primary_halves':halves,'search_gate_passed':bool(gate),'baseline_wp_uniform_probability':scores[0].tolist(),'replication_sequences':256,'boundary':'training-only; no official search accessed','interpretation':'Fixed random feature probes; these results do not test a learned end-to-end TCN.'}
    write_json(path,result);event('replication_completed',report_sha256=sha(path),gate=bool(gate),block='replication_1');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reserve','initialize','prepare','train','replicate'],required=True);args=p.parse_args()
    with threadpool_limits(limits=1): {'reserve':reserve,'initialize':initialize,'prepare':prepare,'train':train,'replicate':replicate}[args.phase]()
