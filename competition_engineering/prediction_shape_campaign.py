"""Prospective search-only test of weighted conditional-mean prediction shape."""
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from competition_engineering import alternative_representation as prior
from competition_engineering import causal_context_diagnostic as context
from competition_engineering.pipeline import read_sequence,write_json,cached
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.probability_readout import probability_focus
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.replicated_readout import grid_stats,choose_strengths
from competition_engineering.residual import from_stats
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.representation_evaluation import bootstrap_stats
from competition_engineering.search_population_audit import pooled_correlations
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_launch
from competition_engineering.isolation_retry import training_with_namespace_retry
from connectome.mlevolve_generated import validate_source
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library

ROOT=prior.ROOT;OUT=ROOT/'competition_engineering/runs/prediction_shape_20261003';sha=prior.sha

def event(kind,**payload):
    OUT.mkdir(exist_ok=True)
    from datetime import datetime,timezone
    p=OUT/'ledger.jsonl';row={'utc':datetime.now(timezone.utc).isoformat(),'boundary':'search-only','kind':kind,'previous_ledger_sha256':sha(p) if p.exists() else None,**payload}
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def reserve():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    assert sha(prior.COMBO)==prior.COMBO_SHA
    old=json.loads((context.OUT/'protocol.json').read_text())
    excluded=set(old['excluded_groups'])|set(sum(old['roles'].values(),[]))
    pf=pq.ParquetFile(prior.DATA);groups=np.random.default_rng(20261080).permutation(sorted(set(range(pf.num_row_groups))-excluded)).tolist();assert len(groups)>=768
    roles={'fit':groups[:384],'selection':groups[384:512],'development':groups[512:768]}
    identities=[{'group':g,'seq_ix':prior.sequence_identity(pf,g),'role':role} for role,gs in roles.items() for g in gs];assert len({r['seq_ix'] for r in identities})==768
    r={'hypothesis':'The frozen incumbent prediction has a transferable nonlinear target-weighted conditional-mean shape; correcting it improves WP without new input information or temporal capacity.',
       'reason':'Prior corrections overfit or transfer weakly. A bounded one-dimensional shape test isolates information encoding from representation acquisition. Previously tested affine scales and broad residual objectives do not test this fixed nonlinear prediction-only mechanism.',
       'roles':roles,'identities':identities,'excluded_groups':sorted(excluded),
       'mechanism':'Fit a fixed17-knot linear interpolant on clipped incumbent predictions with abs(clippedY) times frozen probability-proxy weights. Knot values bounded[-1.9,1.9], fixed second-difference penalty1e-4 of fitting weight mass. Fit a positive affine control; globally rescale it inside clipping limits, preserving correlation exactly.',
       'selection':'1000 sampled required rows per group. Fit384 groups. On128 selection groups, choose target strengths from[0,.05,.1,.25,.5,1] maximizing minimum uniform/proxy WP gain. Freeze all curves and strengths before any development labels are read. No knot/penalty/seed/weighting sweep.',
       'gate':'On256 fresh development groups, combined gain>=0.0002 in both populations, paired95 lower>0 in both, both128-sequence halves positive, each target gain>=-0.0002. Additionally nonlinear-minus-affine combined paired95 lower>0 in both populations. Only then freeze a candidate and use at most replication_1 from the untouched512 pool under the same gate. Replication_2 remains untouched.',
       'independence':'New fitting/selection/development groups exclude all earlier reserved groups and protected train roles. Incumbent pretraining independence is not claimed. Probability population is a proxy, not official mask.',
       'boundaries':'No official search, protected evaluation, promotion, leaderboard or submission feedback. Failed development gate closes this fixed shape family without replication.',
       'next':'If gate fails, diagnose whether shape is absent, seed-independent but nontransferable, or population-conflicted; do not tune the fixed curve.'}
    write_json(path,r)
    library=ROOT/'competition_engineering/research_cards/v15';shutil.copytree(ROOT/'competition_engineering/research_cards/v14',library)
    card=json.loads((library/'classification_temperature_scaling.json').read_text())
    card.update(card_id='weighted_correlation_ratio',title='On Conditional Correlations',authors=['Lei Yu'],year=2019,version=1,source_url='https://arxiv.org/pdf/1811.03918',source_locator='Introduction correlation-ratio definition; Section3.1 Theorem2 and Remark3, pages1 and4',verified_on='2026-10-03',curator='Codex; primary full-paper review',family='prediction_shape',mechanism='The correlation ratio characterizes the largest Pearson correlation obtainable by transforming only one variable and connects it to conditional expectation and MMSE.',limitations='The paper does not study Connectome, target-dependent sample weights, finite-sample spline transfer or clipped prediction deployment. Weighted-measure adaptation and finite-basis experiment are curator derivations; no gain is guaranteed.',concise_relevance='Curator derivation: under dQ proportional to abs(clippedY) times population weight dP, E_Q[clippedY|prediction] maximizes correlation over functions of prediction. A bounded spline tests approximation and transfer while an affine control isolates shape.',diagnostic_signatures='Held-out nonlinear-minus-affine WP gain under both uniform and proxy populations.',causality='Curve uses only current frozen prediction; no sequence identity, labels or future at inference.',incremental_inference='Two fixed17-knot interpolations per row; no new recurrent state.',tags=['correlation-ratio','weighted-measure','prediction-shape'],claim_confidence='limited')
    write_json(library/'weighted_correlation_ratio.json',validate_card(card));cards,h=load_library(library)
    write_json(OUT/'research_library.json',{'version':15,'cards':len(cards),'manifest_sha256':h,'reviewed_sections':'Primary introduction and Theorem2/Remark3','weighted_derivation':'Q has density w/E_P[w]. Cauchy-Schwarz: Corr_Q(Y,g(P)) <= sqrt(Var_Q(E_Q[Y|P])/Var_Q(Y)), attained by positive affine conditional mean. Approximation and transport require independent evidence.'})
    k=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text());assert k['version']==22;k['version']=23;k['status']='Prediction-only shape mechanism prospectively reserved; all earlier checkpoints closed.';k['next_research_question']=r['hypothesis']
    write_json(OUT/'knowledge_v23.json',k);write_json(ROOT/'competition_engineering/search_knowledge.json',k);event('experiment_preregistered',protocol_sha256=sha(path),knowledge_sha256=sha(OUT/'knowledge_v23.json'))
    return r

def prepare(role):
    r=reserve();assert role in ['fit','selection','development']
    if role=='selection':assert json.loads((OUT/'shape_diagnostic_work/artifacts/crossfit.json').read_text())['gate_passed'],'No selection access before independent shape evidence'
    if role=='development':assert (OUT/'fit_work/identity.json').exists(),'Freeze selection before development access'
    directory=OUT/('training' if role!='development' else 'development');directory.mkdir(exist_ok=True)
    pf=pq.ParquetFile(prior.DATA);combo=prior.load_combo();extractor=LatentExtractor()
    with np.load(ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz') as q:trees={k:q[k] for k in q.files}
    records=[]
    for count,group in enumerate(r['roles'][role]):
        path=directory/f'{group:05d}.npz';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():row=json.loads(meta.read_text());assert sha(path)==row['derived_sha256'];records.append(row);continue
        seq,need,x,y,mask=read_sequence(pf,group);p,_=extractor.predict(x);z={'x':x,'y':y,'p':p,'step':np.arange(len(x)),'need':need};base=prior.predict_combo(z,combo)
        eligible=np.flatnonzero(need);idx=np.sort(np.random.default_rng(20261081+group).choice(eligible,1000,replace=False))
        focus=probability_focus(x[idx],base[idx],z['step'][idx],combo,trees)[:,1]
        np.savez(path,base=base[idx],y=y[idx],focus=focus,group=group,seq_ix=seq,indices=idx,role=0 if role=='fit' else 1 if role=='selection' else 2)
        row={'group':group,'seq_ix':seq,'role':role,'source_sequence_sha256':prior.hashlib.sha256(x.tobytes()+y.tobytes()+need.tobytes()).hexdigest(),'derived_sha256':sha(path)};write_json(meta,row);records.append(row)
        if (count+1)%128==0:print(role+' prepared '+str(count+1),flush=True)
    write_json(OUT/('prepared_'+role+'.json'),{'records':records});event('data_prepared',role=role,manifest_sha256=sha(OUT/('prepared_'+role+'.json')))
    if role=='selection':write_json(directory/'identity.json',{'groups':r['roles']['fit']+r['roles']['selection']})
    if role=='development':write_json(directory/'identity.json',{'groups':r['roles'][role]})

def interpolation_basis(p,knots):
    p=np.clip(np.asarray(p,dtype=float),knots[0],knots[-1]);j=np.clip(np.searchsorted(knots,p,side='right')-1,0,len(knots)-2)
    fraction=(p-knots[j])/(knots[j+1]-knots[j]);f=np.zeros((len(p),len(knots)))
    f[np.arange(len(p)),j]=1-fraction;f[np.arange(len(p)),j+1]=fraction
    return f

def curve_predict(base,knots,values,strength):
    p=np.clip(base,-2,2).astype(float)
    transformed=np.column_stack([np.interp(p[:,t],knots,values[:,t]) for t in [0,1]])
    return p+np.asarray(strength)*(transformed-p)

def fit(train_files,output_dir):
    from scipy.optimize import lsq_linear
    knots=np.linspace(-2,2,17);grams=np.zeros((2,17,17));rhs=np.zeros((17,2));ag=np.zeros((2,2,2));ar=np.zeros((2,2));mass=np.zeros(2)
    for path in train_files:
        with np.load(path) as q:z={k:q[k] for k in q.files}
        if int(z['role'])!=0:continue
        y=np.clip(z['y'],-2,2).astype(float);p=np.clip(z['base'],-2,2)
        for t in [0,1]:
            f=interpolation_basis(p[:,t],knots);a=np.column_stack((np.ones(len(p)),p[:,t]));w=np.abs(y[:,t])*z['focus'];mass[t]+=w.sum()
            grams[t]+=f.T@(w[:,None]*f);rhs[:,t]+=f.T@(w*y[:,t]);ag[t]+=a.T@(w[:,None]*a);ar[:,t]+=a.T@(w*y[:,t])
    second=np.diff(np.eye(17),n=2,axis=0);values=[];affine=[];diagnostics=[]
    for t in [0,1]:
        gram=grams[t]+mass[t]*1e-4*(second.T@second)+np.eye(17)*mass[t]*1e-10
        lower=np.linalg.cholesky(gram);result=lsq_linear(lower.T,np.linalg.solve(lower,rhs[:,t]),bounds=(-1.9,1.9),tol=1e-12)
        if not result.success:raise RuntimeError('Bounded curve optimizer failed')
        values.append(result.x)
        a=np.linalg.solve(ag[t],ar[:,t]);assert a[1]>0,'Negative affine slope invalidates mechanism control'
        v=a[0]+a[1]*knots;v*=min(1.,1.9/np.max(np.abs(v)));affine.append(v)
        diagnostics.append({'weighted_mass':float(mass[t]),'nonmonotone_segments':int((np.diff(result.x)<0).sum()),'slope_change_norm':float(np.linalg.norm(second@result.x)),'affine_coef':a.tolist(),'optimizer_optimality':float(result.optimality)})
    values=np.array(values).T;affine=np.array(affine).T;moments=np.zeros((2,2,6,6,2))
    for path in train_files:
        with np.load(path) as q:z={k:q[k] for k in q.files}
        if int(z['role'])!=1:continue
        base=np.clip(z['base'],-2,2)
        for m,v in enumerate([values,affine]):
            correction=curve_predict(base,knots,v,[1,1])-base
            for pop,w in enumerate([np.ones(len(base)),z['focus']]):moments[m,pop]+=grid_stats(z['y'],base,correction,w)
    choices=[choose_strengths(m) for m in moments]
    np.savez(output_dir+'/models.npz',knots=knots,values=values,affine=affine,strength=choices[0]['strengths'],affine_strength=choices[1]['strengths'])
    Path(output_dir+'/selection.json').write_text(json.dumps({'choices':choices,'fit_diagnostics':diagnostics,'selection_moments':moments.tolist()}),encoding='utf-8')
    return {'fits':2,'rows':384000,'selected_strength':choices[0]['strengths']}

def source():
    code='import numpy as np\nimport json\nfrom pathlib import Path\n'
    for f in [interpolation_basis,curve_predict,focus_stats,from_stats,grid_stats,choose_strengths,fit]:code+=inspect.getsource(f)+'\n'
    code+='def train(train_files,output_dir):\n    return fit(train_files,output_dir)\n';validate_source(code,training=True);return code

def train():
    reserve();assert json.loads((OUT/'shape_diagnostic_work/artifacts/crossfit.json').read_text())['gate_passed'],'Fresh shape evidence required to reopen historical spline family'
    work=OUT/'fit_work';work.mkdir(exist_ok=True)
    for role in ['fit','selection']:
        for row in json.loads((OUT/('prepared_'+role+'.json')).read_text())['records']:assert sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    if not (work/'artifacts/models.npz').exists():
        (work/'train.py').write_text(source(),encoding='utf-8');event('fit_started',source_sha256=sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);deadline=time.monotonic()+1800
        def retry(attempt,stderr):
            (work/('namespace_attempt_'+str(attempt)+'.stderr')).write_bytes(stderr);event('infrastructure_retry',attempt=attempt,isolation_unchanged=True)
        process,status,stdout,stderr=training_with_namespace_retry(lambda:_launch(sandbox,work,'train',deadline,OUT/'training'),deadline,retry)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:event('implementation_failure',error=stderr.decode(errors='replace')[-1500:]);raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    identity={p:sha(work/p) for p in ['train.py','artifacts/models.npz','artifacts/selection.json']}
    if (work/'identity.json').exists():assert json.loads((work/'identity.json').read_text())==identity
    else:write_json(work/'identity.json',identity);event('candidate_frozen_before_development',identity=identity)

def crossfit(train_files,output_dir):
    files=sorted(train_files);assert len(files)==384
    report=[]
    for fold in [0,1]:
        fitting=files[fold*192:(fold+1)*192];testing=files[(1-fold)*192:(2-fold)*192]
        fit(fitting,output_dir)
        model_path=Path(output_dir)/'models.npz';selection_path=Path(output_dir)/'selection.json'
        with np.load(model_path) as q:model={k:q[k] for k in q.files}
        model_path.replace(Path(output_dir)/('crossfit_'+str(fold)+'.npz'));selection_path.replace(Path(output_dir)/('crossfit_'+str(fold)+'.json'))
        moments=[]
        for path in testing:
            with np.load(path) as q:z={k:q[k] for k in q.files}
            curve=curve_predict(z['base'],model['knots'],model['values'],[1,1]);affine=curve_predict(z['base'],model['knots'],model['affine'],[1,1])
            moments.append([[focus_stats(z['y'],p,w) for w in [np.ones(len(curve)),z['focus']]] for p in [z['base'],curve,affine]])
        moments=np.array(moments);counts=bootstrap_counts(192,np.random.default_rng(20261083+fold),draws=10000)
        scores=pooled_correlations(moments.sum(0));gain=(scores[1]-scores[2]).mean(-1);boot=pooled_correlations(bootstrap_stats(moments,counts));ci=np.quantile((boot[:,1]-boot[:,2]).mean(-1),[.025,.975],axis=0)
        np.savez(Path(output_dir)/('moments_'+str(fold)+'.npz'),moments=moments)
        report.append({'nonlinear_minus_affine_combined':gain.tolist(),'paired95':ci.tolist(),'target_contrast':(scores[1]-scores[2]).tolist(),'curve_minus_incumbent':(scores[1]-scores[0]).tolist(),'gate_passed':bool(np.all(gain>=.0002) and np.all(ci[0]>0))})
    r={'folds':report,'gate_passed':all(row['gate_passed'] for row in report),'interpretation':'Independent-sequence crossfit mechanism diagnostic, not candidate selection or official-search evidence.'}
    Path(output_dir+'/crossfit.json').write_text(json.dumps(r),encoding='utf-8');return r

def diagnose_fit():
    reserve();amendment=OUT/'historical_audit_amendment.json'
    if not amendment.exists():
        write_json(amendment,{'historical_source':'competition_engineering/manual_spline_calibration.py','source_sha256':sha(ROOT/'competition_engineering/manual_spline_calibration.py'),'discovery':'Historical seven-knot residual spline was not indexed in accumulated knowledge. Do not claim nonlinear calibration is untested. Historical search outcomes are not accessed for tuning.','new_gate':'Before selection/development access, two complementary192-sequence crossfits on the fresh384 fitting groups must EACH show nonlinear-minus-affine combined WP>=0.0002 and paired95 lower>0 for both uniform and probability populations. Fixed curves and no strength selection in this diagnostic. Only passing this signal gate justifies reopening the bounded conditional-mean hypothesis.','no_tuning':'Failure closes the branch; no changes to knots, penalty, seed or weighting.'})
        event('historical_audit_protocol_amended_before_fit',amendment_sha256=sha(amendment))
    work=OUT/'shape_diagnostic_work';work.mkdir(exist_ok=True)
    for row in json.loads((OUT/'prepared_fit.json').read_text())['records']:assert sha(OUT/'training'/f"{row['group']:05d}.npz")==row['derived_sha256']
    assert len(list((OUT/'training').glob('*.npz')))==384,'Diagnostic must not mount selection/development data'
    write_json(OUT/'training/identity.json',{'groups':json.loads((OUT/'protocol.json').read_text())['roles']['fit']})
    if not (work/'artifacts/crossfit.json').exists():
        code=source().replace('return fit(train_files,output_dir)','return crossfit(train_files,output_dir)')
        extra=''
        for f in [bootstrap_counts,bootstrap_stats,pooled_correlations,crossfit]:extra+=inspect.getsource(f)+'\n'
        code=code[:code.index('def train(train_files,output_dir):')]+extra+'def train(train_files,output_dir):\n    return crossfit(train_files,output_dir)\n'
        validate_source(code,training=True);(work/'train.py').write_text(code,encoding='utf-8');event('shape_diagnostic_started',source_sha256=sha(work/'train.py'))
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,work);deadline=time.monotonic()+1800
        def retry(attempt,stderr):
            (work/('namespace_attempt_'+str(attempt)+'.stderr')).write_bytes(stderr);event('infrastructure_retry',attempt=attempt,isolation_unchanged=True)
        process,status,stdout,stderr=training_with_namespace_retry(lambda:_launch(sandbox,work,'train',deadline,OUT/'training'),deadline,retry)
        (work/'stdout.log').write_bytes(stdout);(work/'stderr.log').write_bytes(stderr)
        if process.returncode or status:event('implementation_failure',component='shape diagnostic',error=stderr.decode(errors='replace')[-1500:]);raise RuntimeError(stderr.decode(errors='replace')[-1500:])
    r=json.loads((work/'artifacts/crossfit.json').read_text());event('shape_diagnostic_completed',gate=r['gate_passed'],report_sha256=sha(work/'artifacts/crossfit.json'));print(json.dumps(r),flush=True)

def evaluate():
    path=OUT/'development.json'
    if path.exists():print(path.read_text());return
    work=OUT/'fit_work';assert all(sha(work/p)==h for p,h in json.loads((work/'identity.json').read_text()).items())
    with np.load(work/'artifacts/models.npz') as q:model={k:q[k] for k in q.files}
    expected={r['group']:r['derived_sha256'] for r in json.loads((OUT/'prepared_development.json').read_text())['records']};moments=[]
    for group,z in cached(OUT/'development'):
        assert sha(OUT/'development'/f'{group:05d}.npz')==expected[group]
        p=curve_predict(z['base'],model['knots'],model['values'],model['strength']);a=curve_predict(z['base'],model['knots'],model['affine'],model['affine_strength'])
        moments.append([[focus_stats(z['y'],v,w) for w in [np.ones(len(p)),z['focus']]] for v in [z['base'],p,a]])
    moments=np.array(moments);counts=bootstrap_counts(256,np.random.default_rng(20261082),draws=10000)
    scores=pooled_correlations(moments.sum(0));delta=scores[1]-scores[0];contrast=scores[1]-scores[2]
    boot=pooled_correlations(bootstrap_stats(moments,counts));ci=np.quantile((boot[:,1]-boot[:,0]).mean(-1),[.025,.975],axis=0);cci=np.quantile((boot[:,1]-boot[:,2]).mean(-1),[.025,.975],axis=0)
    halves=[]
    for block in [moments[:128],moments[128:]]:
        s=pooled_correlations(block.sum(0));halves.append((s[1]-s[0]).mean(-1))
    gate=bool(np.all(delta.mean(-1)>=.0002) and np.all(ci[0]>0) and np.all(np.array(halves)>0) and np.all(delta>=-.0002) and np.all(cci[0]>0))
    np.savez(OUT/'development_moments.npz',moments=moments)
    r={'scores_uniform_probability_target':scores.tolist(),'target_deltas':delta.tolist(),'combined_deltas':delta.mean(-1).tolist(),'paired95':ci.tolist(),'nonlinear_minus_affine':contrast.mean(-1).tolist(),'nonlinear_minus_affine_paired95':cci.tolist(),'halves':np.array(halves).tolist(),'strength':model['strength'].tolist(),'development_gate_passed':gate,'decision':'Frozen candidate qualifies for untouched replication' if gate else 'Close this fixed prediction-shape model; preserve all untouched replication sequences.'}
    write_json(path,r);event('development_evaluated',gate=gate,report_sha256=sha(path));print(json.dumps(r),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reserve','prepare_fit','diagnose_fit','prepare_selection','train','prepare_development','evaluate'],required=True);a=p.parse_args()
    with threadpool_limits(limits=1):
        if a.phase.startswith('prepare_'):prepare(a.phase[8:])
        else:{'reserve':reserve,'train':train,'evaluate':evaluate,'diagnose_fit':diagnose_fit}[a.phase]()
