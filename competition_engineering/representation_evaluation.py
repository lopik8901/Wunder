"""Frozen training replication and one-shot search comparisons for state probes."""
from competition_engineering import representation_campaign as campaign
import argparse
import json
import shutil
import time
import numpy as np
from competition_engineering.replicated_readout import OUT,readout_design
from competition_engineering.pipeline import cached,write_json
from competition_engineering.manual_search_core import TRAIN_1024,SEARCH,SEARCH_ROOT_WP,load_combo,predict_combo,COMBO_SHA
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.residual import sufficient
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.search_population_audit import pooled_correlations
from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_check_artifacts,_package,validate_callback,replay
from competition_engineering import frozen_representation_onnx,campaign_onnx
from wnn_connectome_starterpack.utils import GlobalAccumulator

def load_frozen():
    work=OUT/'isolated_training';identity=json.loads((work/'fit_identity.json').read_text())
    assert all(campaign.sha(work/key)==value for key,value in identity.items())
    training=json.loads((work/'artifacts/training.json').read_text())
    with np.load(work/'artifacts/models.npz') as q:
        models={k:q[k] for k in q.files}
    variants={name+'_selected':{'family':name,'strengths':training['selections'][name]['strengths']} for name in ['current','latent']}
    variants.update({name+'_fixed025':{'family':name,'strengths':[.25,.25]} for name in ['current','latent']})
    return models,variants,identity

def bootstrap_stats(moments,counts):
    return (counts@moments.reshape(len(moments),-1)).reshape(len(counts),*moments.shape[1:])

def replication():
    destination=OUT/'replication.json'
    if destination.exists():
        print(destination.read_text());return
    models,variants,identity=load_frozen();records=[];baselines=[]
    groups=json.loads((campaign.REPLICATION/'identity.json').read_text())['groups']
    manifest=json.loads((campaign.OUT/'prepared_manifest.json').read_text())
    expected={row['group']:row['derived_sha256'] for row in manifest['sequences'] if row['role']=='replication'}
    for group in groups:
        path=campaign.REPLICATION/f'{group:05d}.npz';assert campaign.sha(path)==expected[group]
        with np.load(path) as q:
            z={k:q[k] for k in q.files}
        assert int(z['role'])==3
        f=readout_design(z['current'],z['hidden'],models['mean'],models['scale'])
        focuses=[np.ones(len(f)),z['focus']]
        baselines.append([focus_stats(z['y'],z['base'],focus) for focus in focuses]);row=[]
        for variant in variants.values():
            replicas=[]
            for suffix in ['a','b','pooled']:
                coef=models[variant['family']+'_'+suffix]*np.array(variant['strengths'],np.float32)
                p=z['base']+f@coef
                replicas.append([focus_stats(z['y'],p,focus) for focus in focuses])
            row.append(replicas)
        records.append(row)
    records=np.array(records);baselines=np.array(baselines)
    np.savez(OUT/'replication_moments.npz',moments=records,baseline=baselines,groups=groups)
    counts=bootstrap_counts(len(groups),np.random.default_rng(20261014),draws=10000)
    bwp=pooled_correlations(baselines.sum(0));scores=pooled_correlations(records.sum(0))
    boot=pooled_correlations(bootstrap_stats(records,counts))-pooled_correlations(bootstrap_stats(baselines,counts))[:,None,None]
    point=scores-bwp[None,None]
    report={}
    for i,(name,variant) in enumerate(variants.items()):
        report[name]={'choice':variant,'replicas':{suffix:{'delta_per_target_uniform_probability':point[i,j].tolist(),
            'combined_delta_uniform_probability':point[i,j].mean(-1).tolist(),
            'paired99_combined_uniform_probability':np.quantile(boot[:,i,j].mean(-1),[.005,.995],axis=0).tolist(),
            'paired99_per_target_uniform_probability':np.quantile(boot[:,i,j],[.005,.995],axis=0).tolist()}
            for j,suffix in enumerate(['a','b','pooled'])}}
    names=list(variants);matched={}
    for mode in ['selected','fixed025']:
        a=names.index('latent_'+mode);b=names.index('current_'+mode)
        difference=boot[:,a,2]-boot[:,b,2]
        matched[mode]={'delta_per_target_uniform_probability':(point[a,2]-point[b,2]).tolist(),
            'paired99_combined_uniform_probability':np.quantile(difference.mean(-1),[.005,.995],axis=0).tolist()}
    result={'variants':report,'latent_minus_current':matched,'baseline_wp_uniform_probability':bwp.tolist(),
        'replication_sequences':len(groups),'model_identity':identity,
        'scope':'Frozen choices on training-derived replication groups absent from fitting sandbox; not independent of pretrained GRU/incumbent provenance.',
        'decision':'Proceed to the four predeclared search comparisons; training replication informs mechanism interpretation, not post-hoc choice.',
        'moments_sha256':campaign.sha(OUT/'replication_moments.npz')}
    write_json(destination,result);campaign.event('replication_completed',report_sha256=campaign.sha(destination))
    print(json.dumps({'variants':{k:v['replicas']['pooled'] for k,v in report.items()},'matched':matched}),flush=True)

def full_features(z,base,models,extractor,combo):
    _,hidden=extractor.predict(z['x'])
    current=np.column_stack((np.ones(len(base)),np.clip(base,-2,2),np.clip((z['x']-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
    return readout_design(current,hidden,models['mean'],models['scale'])

def check():
    if not (OUT/'replication.json').exists():
        raise ValueError('Complete frozen training replication before consulting search')
    if (OUT/'candidate_checks.json').exists():
        print((OUT/'candidate_checks.json').read_text());return
    models,variants,identity=load_frozen();combo=load_combo();extractor=LatentExtractor()
    coefficients={name:(models[v['family']+'_pooled']*np.array(v['strengths'],np.float32)).astype(np.float32) for name,v in variants.items()}
    validations={};identities={};_,training_probe=next(cached(TRAIN_1024))
    probe_base=predict_combo(training_probe,combo);probe_features=full_features(training_probe,probe_base,models,extractor,combo)
    for name,coef in coefficients.items():
        directory=OUT/('check_'+name);directory.mkdir(exist_ok=True)
        if not (directory/'export_identity.json').exists():
            (directory/'artifacts').mkdir(exist_ok=True)
            np.savez(directory/'artifacts/model.npz',coef=coef,mean=models['mean'],scale=models['scale'])
            frozen_representation_onnx.export(directory/'artifacts/model.npz',directory/'artifacts/fused.onnx',directory)
            (directory/'callback.py').write_text(campaign_onnx.callback_source('current'),encoding='utf-8')
            artifacts=_check_artifacts(directory,{'callback.py':campaign.sha(directory/'callback.py')});archive,deployment=_package(directory)
            write_json(directory/'export_identity.json',{'artifacts':artifacts,'deployment':deployment,'archive_sha256':campaign.sha(archive),'archive_bytes':archive.stat().st_size})
        identities[name]=json.loads((directory/'export_identity.json').read_text())
        assert all(campaign.sha(directory/'deploy'/key)==value for key,value in identities[name]['deployment'].items())
        if (directory/'validation.json').exists():
            validations[name]=json.loads((directory/'validation.json').read_text());continue
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,directory);deadline=time.monotonic()+1800
        validation=validate_callback(sandbox,directory,TRAIN_1024,deadline)
        actual,_=replay(sandbox,directory,[training_probe],deadline)
        expected=probe_base+probe_features@coef;need=training_probe['need']
        np.testing.assert_allclose(actual[0][need],expected[need],atol=3e-5,rtol=3e-5)
        validation.update(stream_batch_parity=True,label_mask_independence=True,finite_outputs=True,
            max_abs_parity=float(np.max(np.abs(actual[0][need]-expected[need]))))
        write_json(directory/'validation.json',validation);validations[name]=validation
        campaign.event('callback_verified_before_search',candidate=name,cpu_us=validation['callback_us_per_row'])
        print(json.dumps({'callback':name,'cpu_us':validation['callback_us_per_row']}),flush=True)
    names=list(variants);baseline=GlobalAccumulator();accs={name:GlobalAccumulator() for name in names}
    atlases={name:SearchErrorDiagnostics(root_predictor=lambda z:predict_combo(z,combo)) for name in names};moments=[]
    packed=np.column_stack([coefficients[name] for name in names])
    for count,(_,z) in enumerate(cached(SEARCH)):
        base=predict_combo(z,combo);f=full_features(z,base,models,extractor,combo);corrections=f@packed
        baseline.add(z['y'],base,z['mask']);mask=z['mask'];row=[sufficient(z['y'][mask],base[mask])]
        for j,name in enumerate(names):
            p=base+corrections[:,j*2:j*2+2]
            if not np.isfinite(p[z['need']]).all():
                raise ValueError('Nonfinite required search predictions')
            accs[name].add(z['y'],p,mask);atlases[name].add(z,p);row.append(sufficient(z['y'][mask],p[mask]))
        moments.append(row)
        if (count+1)%16==0:
            print('SEARCH_SEQUENCES='+str(count+1),flush=True)
    moments=np.array(moments);np.savez(OUT/'search_moments.npz',moments=moments)
    root=baseline.result();assert abs(root['weighted_pearson']-SEARCH_ROOT_WP)<2e-6
    counts=bootstrap_counts(64,np.random.default_rng(20261015),draws=10000)
    boot=pooled_correlations(bootstrap_stats(moments,counts));delta=boot[:,1:]-boot[:,:1]
    reports={}
    for j,name in enumerate(names):
        score=accs[name].result();target_delta=[score[k]-root[k] for k in ['t0','t1']]
        difference=score['weighted_pearson']-root['weighted_pearson'];ci=np.quantile(delta[:,j].mean(-1),[.005,.995]).tolist()
        validation=validations[name]
        eligible=difference>=.0002 and ci[0]>0 and min(target_delta)>=-.0002 and validation['callback_us_per_row']<=76.92753780833335
        result={'selection':variants[name],'search':{'reference':{k:root[k] for k in ['t0','t1','weighted_pearson']},
            'candidate':{k:score[k] for k in ['t0','t1','weighted_pearson']},'delta_combined':difference,'delta_per_target':target_delta,
            'scored_rows':root['selected_rows'],'sequences':root['blocks'],'search_error_diagnostics':atlases[name].result()},
            'paired99':ci,'paired99_per_target':np.quantile(delta[:,j],[.005,.995],axis=0).tolist(),
            'callback_checks':validation,'identity':{'fit':identity,'export':identities[name]},
            'status':'AWAITING_PROMOTION' if eligible else 'scientific_negative_or_insufficient_search_evidence',
            'evidence':'Reused fixed64 search; sequence intervals do not correct adaptive research.'}
        directory=OUT/('check_'+name);write_json(directory/'report.json',result);reports[name]=result
        if eligible:
            awaiting=campaign.OUT/'awaiting_promotion'/name;awaiting.parent.mkdir(exist_ok=True)
            shutil.copytree(directory/'deploy',awaiting)
            write_json(awaiting/'status.json',{'status':'AWAITING_PROMOTION','incumbent_sha256':COMBO_SHA,'report_sha256':campaign.sha(directory/'report.json')})
        campaign.event('candidate_search_completed',candidate=name,wp=score['weighted_pearson'],paired99=ci,status=result['status'])
        print(json.dumps({'candidate':name,'wp':score['weighted_pearson'],'delta_per_target':target_delta,'paired99':ci,'status':result['status']}),flush=True)
    comparisons={}
    for mode in ['selected','fixed025']:
        a=names.index('latent_'+mode);b=names.index('current_'+mode);d=delta[:,a]-delta[:,b]
        comparisons[mode]={'paired99_combined':np.quantile(d.mean(-1),[.005,.995]).tolist(),
            'paired99_per_target':np.quantile(d,[.005,.995],axis=0).tolist(),
            'delta_combined':reports['latent_'+mode]['search']['delta_combined']-reports['current_'+mode]['search']['delta_combined']}
    write_json(OUT/'matched_search_comparisons.json',comparisons)
    write_json(OUT/'candidate_checks.json',reports)
    campaign.event('model_experiment_completed',id='frozen_state_readout',report_sha256=campaign.sha(OUT/'candidate_checks.json'),matched=comparisons)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['replication','search'],required=True)
    args=parser.parse_args();replication() if args.phase=='replication' else check()
