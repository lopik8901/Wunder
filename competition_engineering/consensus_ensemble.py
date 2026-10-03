"""One conditional ensemble mechanism with a matched-amplitude control."""
from competition_engineering import representation_campaign as campaign
import argparse
import json
import shutil
import time
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from competition_engineering.representation_evaluation import load_frozen,full_features,bootstrap_stats
from competition_engineering.replicated_readout import readout_design
from competition_engineering.pipeline import cached,write_json
from competition_engineering.manual_search_core import TRAIN_4096,TRAIN_1024,SEARCH,COMBO_SHA,SEARCH_ROOT_WP,load_combo,predict_combo
from competition_engineering.frozen_latent_readout import LatentExtractor
from competition_engineering.probability_readout import probability_focus
from competition_engineering.prospective_selection_diagnostic import focus_stats
from competition_engineering.residual import sufficient
from competition_engineering.gradient_uncertainty import bootstrap_counts
from competition_engineering.search_population_audit import pooled_correlations
from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
from competition_engineering.generated_runner import GeneratedSandbox,WORKER,_check_artifacts,_package,validate_callback,replay
from competition_engineering import frozen_representation_onnx,campaign_onnx
from wnn_connectome_starterpack.utils import GlobalAccumulator

OUT=campaign.OUT/'consensus_ensemble'
NAMES=['mean','matched_amplitude','consensus']

def consensus(a,b):
    mean=(a+b)*.5
    return np.sign(mean)*np.maximum(np.abs(mean)-np.abs(a-b)*.5,0)

def corrections(features,models,factors):
    a=.25*(features@models['latent_a']);b=.25*(features@models['latent_b'])
    mean=(a+b)*.5
    return [mean,mean*np.asarray(factors,np.float32),consensus(a,b)]

def initialize():
    OUT.mkdir(exist_ok=True);path=OUT/'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    _,_,identity=load_frozen();original=campaign.initialize()
    pool=json.loads((TRAIN_4096/'identity.json').read_text())['groups']
    excluded=set(original['excluded_recent_tree_selection_groups'])
    for values in original['groups'].values():
        excluded.update(values)
    fresh=np.random.default_rng(20261017).permutation(sorted(set(pool)-excluded))[:256].tolist()
    assert len(fresh)==256 and not set(fresh)&excluded
    protocol={'observation':'Frozen-state fits recover development signal, but selected correction sizes and replica uncertainty vary; official transfer remains weak. Position gating lacked replicated support.',
        'hypothesis':'Per-row disagreement between disjointly fitted readouts identifies unstable corrections better than uniform shrinkage.',
        'models':'Frozen latent A/B fits, no refit. Fixed strength0.25 for all comparisons.',
        'variants':{'mean':'Arithmetic average of A/B corrections.',
            'matched_amplitude':'Mean multiplied by per-target RMS(consensus)/RMS(mean), estimated from fitting inputs only; no target or WP optimization.',
            'consensus':'Keep the smaller absolute correction when signs agree, otherwise zero; equivalent soft-thresholded mean.'},
        'mechanism_rule':'Consensus must outperform the matched-amplitude control, not merely the larger mean, to support useful disagreement information.',
        'replication_groups':fresh,'source_identity_sha256':campaign.sha(TRAIN_4096/'identity.json'),
        'fit_identity':identity,'qualification':original['qualification'],
        'replication':'New256 training groups excluded from this campaign fit/selection/replication and the prior tree-selection group. Whole-system independence still not claimed.',
        'scope':'No model fits or thresholds selected from search; three frozen alternatives receive paired comparisons.'}
    write_json(path,protocol);campaign.event('model_experiment_planned',id='consensus_ensemble',protocol_sha256=campaign.sha(path))
    return protocol

def prepare():
    protocol=initialize();models,_,_=load_frozen();combo=load_combo();extractor=LatentExtractor()
    factors_path=OUT/'amplitude_control.json'
    if not factors_path.exists():
        squares=np.zeros((2,2));agree=np.zeros(2);rows=0
        for _,z in cached(campaign.TRAIN):
            if int(z['role'])>=2:
                continue
            f=readout_design(z['current'],z['hidden'],models['mean'],models['scale'])
            a=.25*(f@models['latent_a']);b=.25*(f@models['latent_b']);mean=(a+b)*.5;guarded=consensus(a,b)
            squares[0]+=(mean.astype(float)**2).sum(0);squares[1]+=(guarded.astype(float)**2).sum(0)
            agree+=(a*b>0).sum(0);rows+=len(f)
        factors=np.sqrt(squares[1]/np.maximum(squares[0],1e-30))
        write_json(factors_path,{'factors':factors.tolist(),'agreement_fraction':(agree/rows).tolist(),'rows':rows,
            'squared_correction_sums_mean_consensus':squares.tolist(),'scope':'Fitting inputs only, no labels used in amplitude-control construction.'})
        campaign.event('ensemble_rule_frozen',amplitude_control_sha256=campaign.sha(factors_path))
    teacher=campaign.ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz'
    with np.load(teacher) as q:
        trees={k:q[k] for k in q.files}
    directory=OUT/'prepared_replication';directory.mkdir(exist_ok=True);records=[]
    for count,group in enumerate(protocol['replication_groups']):
        target=directory/f'{group:05d}.npz';meta=directory/f'{group:05d}.json';source=TRAIN_4096/f'{group:05d}.npz'
        if target.exists() and meta.exists():
            row=json.loads(meta.read_text());assert campaign.sha(target)==row['derived_sha256'];records.append(row);continue
        with np.load(source) as q:
            z={k:q[k] for k in q.files}
        p,hidden=extractor.predict(z['x'])
        np.testing.assert_allclose(p[z['need']],z['p'][z['need']],atol=3e-5,rtol=3e-5)
        idx=np.sort(np.random.default_rng(20261017+group).choice(np.flatnonzero(z['need']),500,replace=False))
        base=predict_combo(z,combo)[idx]
        current=np.column_stack((np.ones(len(idx)),np.clip(base,-2,2),np.clip((z['x'][idx]-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
        focus=probability_focus(z['x'][idx],base,z['step'][idx],combo,trees)[:,1]
        np.savez(target,current=current,hidden=hidden[idx],base=base,y=z['y'][idx],focus=focus,indices=idx,group=group)
        row={'group':group,'source_sha256':campaign.sha(source),'derived_sha256':campaign.sha(target)}
        write_json(meta,row);records.append(row)
        if (count+1)%64==0:
            print('ENSEMBLE_REPLICATION_PREPARED='+str(count+1),flush=True)
    write_json(directory/'identity.json',{'groups':protocol['replication_groups'],'scope':'Designated training-derived replication only'})
    if not (OUT/'prepared_manifest.json').exists():
        write_json(OUT/'prepared_manifest.json',{'records':records,'protocol_sha256':campaign.sha(OUT/'protocol.json'),'teacher_sha256':campaign.sha(teacher)})
        campaign.event('ensemble_replication_prepared',manifest_sha256=campaign.sha(OUT/'prepared_manifest.json'))

def replication():
    destination=OUT/'replication.json'
    if destination.exists():
        print(destination.read_text());return
    models,_,_=load_frozen();factors=json.loads((OUT/'amplitude_control.json').read_text())['factors'];moments=[]
    for _,z in cached(OUT/'prepared_replication'):
        f=readout_design(z['current'],z['hidden'],models['mean'],models['scale'])
        predictions=[z['base']]+[z['base']+c for c in corrections(f,models,factors)]
        moments.append([[focus_stats(z['y'],p,focus) for focus in [np.ones(len(f)),z['focus']]] for p in predictions])
    moments=np.array(moments);np.savez(OUT/'replication_moments.npz',moments=moments)
    counts=bootstrap_counts(256,np.random.default_rng(20261018),draws=10000)
    scores=pooled_correlations(moments.sum(0));boot=pooled_correlations(bootstrap_stats(moments,counts))
    delta=boot[:,1:]-boot[:,:1];point=scores[1:]-scores[:1]
    result={'variants':{name:{'combined_delta_uniform_probability':point[i].mean(-1).tolist(),
        'delta_per_target_uniform_probability':point[i].tolist(),
        'paired99_combined_uniform_probability':np.quantile(delta[:,i].mean(-1),[.005,.995],axis=0).tolist()}
        for i,name in enumerate(NAMES)},
        'consensus_minus_amplitude_control':{'combined_delta_uniform_probability':(point[2]-point[1]).mean(-1).tolist(),
            'paired99_combined_uniform_probability':np.quantile((delta[:,2]-delta[:,1]).mean(-1),[.005,.995],axis=0).tolist()},
        'scope':'Fresh correction-disjoint training group; no claim of independent pretrained-model evaluation.',
        'moments_sha256':campaign.sha(OUT/'replication_moments.npz')}
    write_json(destination,result);campaign.event('ensemble_replication_completed',report_sha256=campaign.sha(destination))
    print(json.dumps(result),flush=True)

def export_consensus(model_path,destination,work,models):
    frozen_representation_onnx.export(model_path,destination,work,force_latent=True)
    model=onnx.load(destination);graph=model.graph;last=graph.node[-1]
    assert last.op_type=='Add'
    base=last.input[0];correction=next(node for node in graph.node if last.input[1] in node.output)
    assert correction.op_type=='MatMul'
    features=correction.input[0];counter=0
    def const(value,dtype=np.float32):
        nonlocal counter
        counter+=1;name='consensus_constant_'+str(counter)
        graph.initializer.append(nh.from_array(np.asarray(value,dtype=dtype),name));return name
    def op(kind,*inputs,**attrs):
        nonlocal counter
        counter+=1;name='consensus_value_'+str(counter);graph.node.append(h.make_node(kind,list(inputs),[name],**attrs));return name
    values=op('MatMul',features,const(np.column_stack((models['latent_a'],models['latent_b']))*.25))
    a=op('Gather',values,const([0,1],np.int64),axis=0);b=op('Gather',values,const([2,3],np.int64),axis=0)
    mean=op('Mul',op('Add',a,b),const(.5));radius=op('Mul',op('Abs',op('Sub',a,b)),const(.5))
    prediction=op('Mul',op('Sign',mean),op('Max',op('Sub',op('Abs',mean),radius),const(0)))
    graph.output[0].name=op('Add',base,prediction);onnx.checker.check_model(model);onnx.save(model,destination)

def search():
    if not (OUT/'replication.json').exists():
        raise ValueError('Fresh replication must precede search')
    if (OUT/'candidate_checks.json').exists():
        print((OUT/'candidate_checks.json').read_text());return
    models,_,fit_identity=load_frozen();factors=json.loads((OUT/'amplitude_control.json').read_text())['factors']
    combo=load_combo();extractor=LatentExtractor();validations={};identities={}
    _,probe=next(cached(TRAIN_1024));base=predict_combo(probe,combo);f=full_features(probe,base,models,extractor,combo)
    probe_corrections=corrections(f,models,factors)
    for index,name in enumerate(NAMES):
        directory=OUT/('check_'+name);directory.mkdir(exist_ok=True)
        if not (directory/'export_identity.json').exists():
            (directory/'artifacts').mkdir(exist_ok=True)
            coef=(models['latent_a']+models['latent_b'])*.125
            if name=='matched_amplitude':
                coef=coef*np.array(factors,np.float32)
            path=directory/'artifacts/model.npz';np.savez(path,coef=coef,mean=models['mean'],scale=models['scale'])
            if name=='consensus':
                export_consensus(path,directory/'artifacts/fused.onnx',directory,models)
            else:
                frozen_representation_onnx.export(path,directory/'artifacts/fused.onnx',directory)
            (directory/'callback.py').write_text(campaign_onnx.callback_source('current'),encoding='utf-8')
            artifacts=_check_artifacts(directory,{'callback.py':campaign.sha(directory/'callback.py')});archive,deployment=_package(directory)
            write_json(directory/'export_identity.json',{'artifacts':artifacts,'deployment':deployment,'archive_sha256':campaign.sha(archive)})
        identities[name]=json.loads((directory/'export_identity.json').read_text())
        assert all(campaign.sha(directory/'deploy'/key)==value for key,value in identities[name]['deployment'].items())
        if (directory/'validation.json').exists():
            validations[name]=json.loads((directory/'validation.json').read_text());continue
        sandbox=GeneratedSandbox();sandbox.preflight(WORKER,directory);deadline=time.monotonic()+1800
        validation=validate_callback(sandbox,directory,TRAIN_1024,deadline)
        actual,_=replay(sandbox,directory,[probe],deadline);expected=base+probe_corrections[index];need=probe['need']
        np.testing.assert_allclose(actual[0][need],expected[need],atol=3e-5,rtol=3e-5)
        validation.update(stream_batch_parity=True,label_mask_independence=True,finite_outputs=True,
            max_abs_parity=float(np.max(np.abs(actual[0][need]-expected[need]))))
        write_json(directory/'validation.json',validation);validations[name]=validation
        campaign.event('ensemble_callback_verified',candidate=name,cpu_us=validation['callback_us_per_row'])
        print(json.dumps({'callback':name,'cpu_us':validation['callback_us_per_row']}),flush=True)
    baseline=GlobalAccumulator();accs={name:GlobalAccumulator() for name in NAMES}
    atlases={name:SearchErrorDiagnostics(root_predictor=lambda z:predict_combo(z,combo)) for name in NAMES};moments=[]
    for _,z in cached(SEARCH):
        base=predict_combo(z,combo);f=full_features(z,base,models,extractor,combo);values=corrections(f,models,factors)
        m=z['mask'];baseline.add(z['y'],base,m);row=[sufficient(z['y'][m],base[m])]
        for name,value in zip(NAMES,values):
            p=base+value
            if not np.isfinite(p[z['need']]).all():
                raise ValueError('Nonfinite required output')
            accs[name].add(z['y'],p,m);atlases[name].add(z,p);row.append(sufficient(z['y'][m],p[m]))
        moments.append(row)
    moments=np.array(moments);np.savez(OUT/'search_moments.npz',moments=moments)
    root=baseline.result();assert abs(root['weighted_pearson']-SEARCH_ROOT_WP)<2e-6
    counts=bootstrap_counts(64,np.random.default_rng(20261018),draws=10000)
    boot=pooled_correlations(bootstrap_stats(moments,counts));delta=boot[:,1:]-boot[:,:1];reports={}
    for i,name in enumerate(NAMES):
        score=accs[name].result();d=score['weighted_pearson']-root['weighted_pearson'];td=[score[k]-root[k] for k in ['t0','t1']]
        ci=np.quantile(delta[:,i].mean(-1),[.005,.995]).tolist()
        eligible=d>=.0002 and ci[0]>0 and min(td)>=-.0002 and validations[name]['callback_us_per_row']<=76.92753780833335
        result={'search':{'reference':{k:root[k] for k in ['t0','t1','weighted_pearson']},
            'candidate':{k:score[k] for k in ['t0','t1','weighted_pearson']},'delta_combined':d,'delta_per_target':td,
            'search_error_diagnostics':atlases[name].result()},'paired99':ci,
            'paired99_per_target':np.quantile(delta[:,i],[.005,.995],axis=0).tolist(),
            'callback_checks':validations[name],'identity':{'fit':fit_identity,'export':identities[name]},
            'status':'AWAITING_PROMOTION' if eligible else 'scientific_negative_or_insufficient_search_evidence',
            'evidence':'Descriptive adaptive-search comparison; no corrected error guarantee.'}
        directory=OUT/('check_'+name);write_json(directory/'report.json',result);reports[name]=result
        if eligible:
            awaiting=campaign.OUT/'awaiting_promotion'/('ensemble_'+name);awaiting.parent.mkdir(exist_ok=True)
            shutil.copytree(directory/'deploy',awaiting)
            write_json(awaiting/'status.json',{'status':'AWAITING_PROMOTION','report_sha256':campaign.sha(directory/'report.json'),'incumbent_sha256':COMBO_SHA})
        campaign.event('ensemble_search_completed',candidate=name,wp=score['weighted_pearson'],paired99=ci,status=result['status'])
        print(json.dumps({'candidate':name,'wp':score['weighted_pearson'],'delta_per_target':td,'paired99':ci,'status':result['status']}),flush=True)
    contrast={'combined_delta':reports['consensus']['search']['delta_combined']-reports['matched_amplitude']['search']['delta_combined'],
        'paired99':np.quantile((delta[:,2]-delta[:,1]).mean(-1),[.005,.995]).tolist()}
    write_json(OUT/'mechanism_comparison.json',contrast);write_json(OUT/'candidate_checks.json',reports)
    campaign.event('model_experiment_completed',id='consensus_ensemble',contrast=contrast,report_sha256=campaign.sha(OUT/'candidate_checks.json'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','replication','search'],required=True)
    args=parser.parse_args();{'prepare':prepare,'replication':replication,'search':search}[args.phase]()
