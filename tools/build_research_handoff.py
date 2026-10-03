"""Export allowlisted research metadata; never run models or read mixed ledgers.

Large-file hashes are opaque integrity checks, not outcome observation.
This is an end-of-session preservation tool, not an experiment runner.
"""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'competition_engineering/handoff'
CAMPAIGNS=['autonomous_20261001','objective_alignment_20261002','reassessment_20261002','root_cause_20261002','representation_20261002','nonlinear_state_20261003','sequence_diversity_20261003','alternative_representation_20261003','causal_context_20261003','prediction_shape_20261003']
OMIT_KEYS={'evidence_sha256','artifact_sha256s','input_hashes','raw_results','protected_results','holdout_results','confirm_scores','validation_scores','submission_feedback','leaderboard_results'}
TEXT_LIMIT=3_000_000

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def portable(value):
    if isinstance(value,dict):return {portable(str(k)):portable(v) for k,v in value.items() if k not in OMIT_KEYS}
    if isinstance(value,list):return [portable(v) for v in value]
    if isinstance(value,str):
        value=value.replace(str(ROOT),'<REPO>').replace(ROOT.as_posix(),'<REPO>').replace('/mnt/'+ROOT.drive.rstrip(':').lower()+ROOT.as_posix().split(':',1)[-1],'<REPO>')
        value=re.sub(r'[A-Za-z]:[\\/][^\s\"\n]*','<LOCAL_PATH>',value)
        value=re.sub(r'/(?:home|Users)/[^\s\"\n]*','<LOCAL_PATH>',value)
    return value

def metadata():
    OUT.mkdir(exist_ok=True);inventory=[];trajectory=[]
    for campaign in CAMPAIGNS:
        directory=ROOT/'competition_engineering/runs'/campaign
        for p in sorted(directory.rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(directory)
            if any(s in {'training','fit','development','selection','replication_1','replication_2','replication_3','replication_4','replication_5','fit_pool','readout_fit','source_snapshot','source_snapshots','__pycache__','failed_fit_001'} for s in relative.parts[:-1]):continue
            if p.stem.isdigit() or p.stat().st_size>TEXT_LIMIT:continue
            if p.suffix=='.json' and p.name!='mlevolve_search_history.json':
                value=json.loads(p.read_text(encoding='utf-8'));destination=OUT/'evidence'/campaign/relative
                write(destination,portable(value));inventory.append({'source':p.relative_to(ROOT).as_posix(),'source_sha256':sha(p),'export':destination.relative_to(ROOT).as_posix(),'export_sha256':sha(destination),'transformation':'Portable paths and explicit bulky/private field exclusions; original file remains immutable locally.'})
            elif p.name=='FINAL_REPORT.md':
                destination=OUT/'evidence'/campaign/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(portable(p.read_text(encoding='utf-8')),encoding='utf-8',newline='\n');inventory.append({'source':p.relative_to(ROOT).as_posix(),'source_sha256':sha(p),'export':destination.relative_to(ROOT).as_posix(),'export_sha256':sha(destination)})
            elif p.name.endswith('.jsonl'):
                for number,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
                    row=json.loads(line);trajectory.append({'campaign':campaign,'source':p.relative_to(ROOT).as_posix(),'line':number,'original_event_sha256':hashlib.sha256(line.encode()).hexdigest(),'event':portable(row)})
        for folder in ['source_snapshot','source_snapshots']:
            for p in directory.glob(folder+'/*.py'):
                digest=sha(p);destination=OUT/'source_snapshots'/(digest+'.py');destination.parent.mkdir(exist_ok=True)
                if not destination.exists():shutil.copyfile(p,destination)
                inventory.append({'source':p.relative_to(ROOT).as_posix(),'source_sha256':digest,'export':destination.relative_to(ROOT).as_posix(),'export_sha256':digest,'transformation':'Exact frozen source bytes, deduplicated by content hash.'})
    write(OUT/'export_provenance.json',inventory)
    p=OUT/'experiment_ledger.jsonl';previous=None
    with p.open('w',encoding='utf-8',newline='\n') as f:
        for row in trajectory:
            row['previous_export_event_sha256']=previous;encoded=json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write(encoded+'\n');previous=hashlib.sha256(encoded.encode()).hexdigest()
    history=json.loads((ROOT/'competition_engineering/runs/reassessment_20261002/mlevolve_search_history.json').read_text())
    keep=['node_id','status','mechanism','hypothesis','search_scores','settings','signature','training_diagnostics','cpu_us']
    write(OUT/'mlevolve_attempts.json',{'count':len(history),'domain':'Dedicated search-only retrospective history; adaptively reused64-sequence search, not independent validation. Early contaminated pilot is not an uncontaminated experiment; supervisor feedback remains excluded.','records':[portable({k:r.get(k) for k in keep}) for r in history]})
    reservations={
      'alternative.json':'alternative_representation_20261003/reservation.json',
      'independent_decoder.json':'alternative_representation_20261003/disjoint_readout/protocol.json',
      'raw_target.json':'alternative_representation_20261003/raw_target/protocol.json',
      'untouched_512.json':'causal_context_20261003/protocol.json',
      'prediction_mechanisms.json':'prediction_shape_20261003/protocol.json'}
    originals={}
    for name,source in reservations.items():
        p=ROOT/'competition_engineering/runs'/source;destination=OUT/'reservations'/name;destination.parent.mkdir(exist_ok=True)
        # Frozen specifications are copied byte-for-byte, never normalized.
        text=p.read_text();assert not re.search(r'[A-Za-z]:\\|/home/[^/]+/',text),'Private path in frozen reservation'
        shutil.copyfile(p,destination);originals[name]={'source':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}
    legacy=json.loads((ROOT/'competition_engineering/dataset_manifest_phase2.json').read_text());write(OUT/'reservations/legacy_roles.json',{'seed':legacy['seed'],'splits':legacy['splits'],'source_sha256':sha(ROOT/'competition_engineering/dataset_manifest_phase2.json'),'no_outcomes':True})
    alternative=json.loads((OUT/'reservations/alternative.json').read_text());context=json.loads((OUT/'reservations/untouched_512.json').read_text());recent=json.loads((OUT/'reservations/prediction_mechanisms.json').read_text());raw=json.loads((OUT/'reservations/raw_target.json').read_text())
    reserved=set(context['excluded_groups'])|set(sum(context['roles'].values(),[]))|set(sum(recent['roles'].values(),[]))
    pool=context['roles']['replication_1']+context['roles']['replication_2'];assert len(pool)==len(set(pool))==512;assert not set(pool)&set(sum(recent['roles'].values(),[]))
    write(OUT/'data_roles.json',{'schema_version':1,'untouched_512':{'specification':'reservations/untouched_512.json','sha256':originals['untouched_512.json']['sha256'],'roles':['replication_1','replication_2'],'groups':pool,'count':512,'status':'No feature/target preparation or evaluation. Identity metadata and opaque hashes do not consume a replication population.','access_rule':'Candidate must pass a prospectively justified development gate; freeze candidate and controls before consuming a block once.'},'additional_prior_reserved_256':{'groups':raw['groups']['future_reserved'],'status':'Also preserved; not part of the512 pool, not automatically released for reuse.'},'all_assigned_or_excluded_train_groups':sorted(reserved),'unassigned_train_groups':sorted(set(range(10607))-reserved),'unassigned_count':10607-len(reserved),'legacy_excluded_train_groups':alternative['excluded_groups'],'protected_training_groups':legacy['splits']['train_holdout'],'official_search':{'split':'valid tune','groups':legacy['splits']['tune'],'count':64,'status':'Heavily adaptively reused; descriptive comparisons only; not independent validation or a target for further adaptation.'},'protected_validation':{'confirm_groups':legacy['splits']['confirm'],'other_validation_groups':sorted(set(range(1873))-set(legacy['splits']['tune'])-set(legacy['splits']['confirm'])),'access':'Supervisor-only, explicit separate authorization. No protected outcomes in this handoff. Historical exposure status not audited; never assume protected pools are fresh or reset evaluation counters.'},'consumed_latest':{'context_fit_and_development':context['roles']['fit']+context['roles']['development'],'prediction_mechanisms_fit_selection_development':recent['roles']},'pretraining_caveat':'The supplied GRU training provenance is undisclosed. Independence is from new correction fitting/selection, not pretrained model parameters.','reservation_provenance':originals})
    # Preserve required legacy identity metadata, without outcomes or caches.
    write(OUT/'dataset_manifest_portable.json',{'seed':legacy['seed'],'datasets':{k:{key:v[key] for key in ['bytes','rows','groups','columns','unique_sequence_ids']} for k,v in legacy['datasets'].items()},'splits':legacy['splits'],'source_sha256':sha(ROOT/'competition_engineering/dataset_manifest_phase2.json')})
    print(json.dumps({'metadata_exports':len(inventory),'trajectory_events':len(trajectory),'mlevolve_attempts':len(history),'untouched_replication':512,'unassigned_training':10607-len(reserved)}))

def external():
    required=[ROOT/'competition_engineering/deployment/manual_targetwise_combo_v1'/name for name in ['combo.npz','baseline.onnx','gru.py','solution.py','manifest.json']]
    required+=[ROOT/'competition_engineering/submissions/manual_targetwise_combo_v1.zip',ROOT/'competition_engineering/runs/reassessment_20261002/nonlinear_weighted_t0/isolated_training/mask_trees.npz',ROOT/'competition_engineering/runs/alternative_representation_20261003/basis.npz']
    required+=[ROOT/'competition_engineering/dataset_manifest_phase2.json']
    directory=ROOT/'competition_engineering/handoff_external';directory.mkdir(exist_ok=True);archive=directory/'connectome_required_artifacts.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in required:z.write(p,p.relative_to(ROOT).as_posix())
    items=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'category':'frozen_incumbent' if 'manual_targetwise' in p.as_posix() else 'required_local_artifact','tracked':False,'obtain':'Transfer the required-artifacts ZIP; restore exact relative path, verify hash. Do not retrain or substitute.'} for p in required]
    datasets=[]
    for name in ['train.parquet','valid.parquet','valid_mask.parquet']:
        p=ROOT/'competition_engineering/assets/wnn_connectome_starterpack/datasets'/name
        print('Opaque integrity hash: '+name,flush=True)
        datasets.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'category':'large_dataset' if name=='train.parquet' else 'protected_dataset','tracked':False,'obtain':'Obtain the official starter pack through your authorized competition access or copy the exact existing file privately; place under this relative path. No credentials or signed download URLs are stored.'})
    optional=[]
    for campaign in CAMPAIGNS:
        d=ROOT/'competition_engineering/runs'/campaign
        for p in d.rglob('*'):
            if p.is_file() and p.suffix in {'.npz','.onnx','.pt','.pth'} and ('artifacts' in p.parts or p.parent==d) and not any(s in p.parts for s in ['failed_fit_001','training','development','selection','fit_pool']):
                optional.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'category':'optional_historical_artifact','tracked':False,'obtain':'Optional private copy of this exact relative path. Aggregate evidence and source/config/reservation are in Git; exact historical replay needs original artifacts and caches. Do not reexecute on consumed pools as independent evidence.'})
    write(OUT/'external_artifacts.json',{'schema_version':1,'required_transfer_archive':{'path':archive.relative_to(ROOT).as_posix(),'bytes':archive.stat().st_size,'sha256':sha(archive),'tracked':False,'extract_into':'Repository root; contains only the explicit required artifact allowlist.'},'required':items,'datasets':datasets,'optional_historical':optional,'excluded_local_state':['Bulk derived row caches; original run directories and logs; mixed supervisor ledger/protected outcomes; virtual environments; GPU caches; credentials. Preserve locally, not in Git.'],'reproducible':'Derived training features/moments can be reconstructed from frozen data identities, saved sources, protocols and original artifacts. This requires computation and separately authorized experiment work; bootstrap performs no reconstruction.'})
    print(json.dumps({'external_archive':archive.relative_to(ROOT).as_posix(),'bytes':archive.stat().st_size,'optional_artifacts':len(optional)}))

def state():
    from connectome.research_retrieval import load_library
    knowledge=json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text());assert knowledge['version']==25
    cards,h=load_library(ROOT/'competition_engineering/research_cards/v16')
    families=[
      ('Historical manual/MLEvolve','adaptive_search','Incumbent formed by target-specific magnitude/gated ridge;82 attempts plus manual trajectory.','autonomous_20261001/FINAL_REPORT.json'),
      ('Additive/diagonal bilinear temporal','weakened','Two tested memory mechanisms do not establish gain; repaired latency/worker errors are distinct.','autonomous_20261001/FINAL_REPORT.json'),
      ('Objective alignment','weakened','60 fits/8 suites; WP selection justified, objective mismatch alone not sufficient.','objective_alignment_20261002/final_review.json'),
      ('Population and gradient transport','established_diagnostic','t1 population contrast survives counterfactuals; point gradients uncertain; mask prediction is not transport proof.','reassessment_20261002/FINAL_REPORT.json'),
      ('Weight normalization/tree execution','established_infrastructure','Mean-one penalty invariance and exact ONNX execution verified; no stronger scorer.','root_cause_20261002/FINAL_REPORT.json'),
      ('Frozen GRU linear information','replicated_development','Training-derived signal replicated beyond current features; official-score improvement not established.','representation_20261002/FINAL_REPORT.json'),
      ('Consensus/disagreement/position','weakened','Matched-amplitude comparison and latency reject claimed mechanism; exploratory search point uncertain.','representation_20261002/FINAL_REPORT.json'),
      ('Nonlinear state/readout/innovation','weakened','Fixed/learned/treshold nonlinear readouts fail incremental mechanism/official transfer.','nonlinear_state_20261003/FINAL_REPORT.json'),
      ('Fitting-sequence diversity','inconclusive','Both original-mixture and composition-balanced paired comparisons span zero.','sequence_diversity_20261003/FINAL_REPORT.json'),
      ('Alternative causal architectures/rescues','weakened_specific_tests','Five stages,1280 fresh replication sequences; all prospective gates fail. Not rejection of all architectures.','alternative_representation_20261003/FINAL_REPORT.json'),
      ('Prefix context gradient transport','weakened_specific_test','Four development error reductions negative; no replication; not noise proof.','causal_context_20261003/FINAL_REPORT.json'),
      ('Prediction-only calibration','weakened_specific_test','Historical source audit plus fresh crossfit fitting/transfer gap; no selection/development read for this branch.','prediction_shape_20261003/FINAL_REPORT.json'),
      ('Semantic bid/ask features','weakened_specific_test','Primary gate fails; apparent repeat advantage disappears after matching strengths; no replication.','prediction_shape_20261003/book_side/development.json')]
    write(OUT/'research_state.json',{'schema_version':1,'as_of':'2026-10-03','session':'Ended; bootstrap must summarize and wait for user approval before experiments.','canonical_handoff':'../RESEARCH_HANDOFF.md','incumbent':{**knowledge['incumbent'],'t0_wp':.6479917119604004,'t1_wp':.6696789327029276,'metric_domain':'Historical adaptively reused64-sequence official search','archive_sha256':'c8acc7153f60b7540c3b988877f30af8a1ec799d87026d09b46a9c3a37521076'},'awaiting_promotion':[],'knowledge_version':25,'library':{'version':16,'cards':len(cards),'manifest_sha256':h},'data_registry':'data_roles.json','evidence_categories':['established_infrastructure','established_diagnostic','replicated_development','adaptive_search','weakened_specific_test','inconclusive','curator_inference','still_open'],'families':[{'family':name,'evidence_category':category,'conclusion':conclusion,'evidence':'evidence/'+source} for name,category,conclusion,source in families],'closed_without_new_evidence':knowledge['closed_branches'],'exploratory_only':knowledge['exploratory_only'],'saturation_judgment':{'category':'curator_inference','statement':'Recent local-correction neighborhood appears increasingly saturated. Not a proven ceiling or exhaustion of resources; materially different causal representations remain open with measurable mechanism evidence.'},'still_open':['Observable incremental WP signal beyond current inputs and frozen state with matched controls','Causal representation/feature mapping supported by semantics and measured transfer','Population-dependent utility versus selection noise; no reused-search adaptation','Prospective stability of selection versus strength/seed fluctuations'],'constraints':{'callback_practical_us_per_row':76.92753780833335,'infrastructure_cap_us_per_row':90,'cpu_vcpus':1,'ram_bytes':16000000000,'artifact_zip_bytes_max':20000000,'whole_test_seconds_max':3600,'candidate_fit_isolation':'Fail-closed Linux bubblewrap, no network, only fitting/selection data mounted'},'boundaries':{'replication':'512 untouched, candidate development gate and freeze first; each block consumed once','protected':'No protected outcomes in research memory; separate supervisor/user authorization needed; historical exposure not assumed fresh','no_experiments_in_handoff':True,'no_protected_evaluation_or_submission_in_handoff':True},'outstanding_technical_issues':['External dataset/weights are absent from Git and must be restored/verified','Old MLEvolve root/config differs from practical incumbent; no verified in-place journal resume','Full historical cache verification requires optional private bulk run transfer','GPU setup is optional/device-specific; all latency must be remeasured on deployment host']})

def changed_paths(root):
    import subprocess
    return subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=root,text=True).splitlines()

def manifest():
    modified=changed_paths(ROOT)
    paths=set(modified)|{'CONTINUE_RESEARCH.md','requirements-handoff.txt','.gitattributes','.gitignore','competition_engineering/RESEARCH_HANDOFF.md','competition_engineering/search_knowledge.json','connectome/research_planning.py'}
    for folder in ['competition_engineering','tools','tests']:
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).glob('*.py'))
    paths.add('competition_engineering/mlevolve_task/description.md')
    paths.add('competition_engineering/protocols/mask_aware_v1.json')
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'competition_engineering/research_cards').rglob('*.json'))
    paths.update(p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json')
    paths.discard('competition_engineering/handoff/artifact_manifest.json')
    entries=[]
    for name in sorted(paths):
        p=ROOT/name;assert p.is_file(),name
        raw=p.read_bytes();exact=name.startswith(('competition_engineering/handoff/evidence/','competition_engineering/handoff/reservations/','competition_engineering/handoff/source_snapshots/')) or name=='competition_engineering/search_knowledge.json'
        exact=exact or name=='competition_engineering/protocols/mask_aware_v1.json'
        content=raw if exact else raw.replace(b'\r\n',b'\n')
        entries.append({'path':name,'sha256':hashlib.sha256(content).hexdigest(),'bytes_canonical':len(content),'hash_mode':'exact_bytes' if exact else 'git_lf_text','category':'tracked_source_or_metadata','tracked':True})
    write(OUT/'artifact_manifest.json',{'schema_version':1,'tracked':entries,'self_integrity':'This manifest is anchored by its Git commit; its own digest is intentionally not recursively embedded. git_lf_text hashes normalizeCRLF toLF, matching .gitattributes. Frozen evidence/specification/source snapshots use exact bytes.','external_manifest':'external_artifacts.json','external_artifact_categories':['frozen_incumbent','required_local_artifact','large_dataset','protected_dataset','optional_historical_artifact'],'excluded':'No datasets, caches, model/checkpoint binaries, environments, raw supervisor records or unnecessary logs are staged.'})
    write(OUT/'staging_paths.json',sorted(paths|{'competition_engineering/handoff/artifact_manifest.json','competition_engineering/handoff/staging_paths.json'}))
    # staging_paths is itself metadata; include its final bytes in the manifest.
    entries.append({'path':'competition_engineering/handoff/staging_paths.json','sha256':sha(OUT/'staging_paths.json'),'bytes_canonical':(OUT/'staging_paths.json').stat().st_size,'hash_mode':'git_lf_text','category':'tracked_source_or_metadata','tracked':True}) if not any(e['path']=='competition_engineering/handoff/staging_paths.json' for e in entries) else None
    for e in entries:
        if e['path']=='competition_engineering/handoff/staging_paths.json':e['sha256']=sha(OUT/'staging_paths.json');e['bytes_canonical']=(OUT/'staging_paths.json').stat().st_size
    value=json.loads((OUT/'artifact_manifest.json').read_text());value['tracked']=entries;write(OUT/'artifact_manifest.json',value)
    print(json.dumps({'staging_paths':len(paths),'manifest_entries':len(entries)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['metadata','external','state','manifest'],required=True)
    {'metadata':metadata,'external':external,'state':state,'manifest':manifest}[p.parse_args().phase]()
