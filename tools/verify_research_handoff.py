"""Read-only portable integrity, synthetic regression and isolation checks."""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'competition_engineering/handoff'

def sha(path,mode='exact_bytes'):
    if mode=='git_lf_text':return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe_target(root,name):
    if '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe transfer member path')
    target=(root/name).resolve()
    if not target.is_relative_to(root.resolve()):raise ValueError('Transfer target escapes repository')
    return target

def restore_bundle(archive,root=ROOT,manifest=None):
    m=manifest or json.loads((OUT/'external_artifacts.json').read_text())
    assert sha(archive)==m['required_transfer_archive']['sha256'],'Transfer archive hash mismatch'
    expected={r['path']:r for r in m['required']}
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(expected) and set(z.namelist())==set(expected),'Unexpected/duplicate archive member'
        targets={name:safe_target(root,name) for name in expected}
        for name,row in expected.items():
            target=targets[name]
            if target.exists():assert sha(target)==row['sha256'],'Refusing to overwrite a different existing artifact: '+name
            assert z.getinfo(name).file_size==row['bytes'],'Unexpected uncompressed member size'
            assert hashlib.sha256(z.read(name)).hexdigest()==row['sha256'],'Member hash mismatch'
        for name,row in expected.items():
            target=targets[name]
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive creation, after validating every member and target.
            with target.open('xb') as f:f.write(z.read(name))
    print('Restored/verified',len(expected),'exact external artifacts; no model execution')

def metadata():
    m=json.loads((OUT/'artifact_manifest.json').read_text())
    for row in m['tracked']:
        p=safe_target(ROOT,row['path']);assert p.is_file(),row['path'];assert sha(p,row['hash_mode'])==row['sha256'],row['path']
    roles=json.loads((OUT/'data_roles.json').read_text());pool=roles['untouched_512'];assert len(set(pool['groups']))==pool['count']==512
    p=OUT/pool['specification'];assert sha(p)==pool['sha256'];spec=json.loads(p.read_text());assert spec['roles']['replication_1']+spec['roles']['replication_2']==pool['groups']
    assert not set(pool['groups'])&set(roles['protected_training_groups']);assert not set(pool['groups'])&set(sum(roles['consumed_latest']['prediction_mechanisms_fit_selection_development'].values(),[]))
    assert not set(pool['groups'])&set(roles['unassigned_train_groups'])
    previous=None;events=0
    for line in (OUT/'experiment_ledger.jsonl').read_text().splitlines():
        row=json.loads(line);assert row['previous_export_event_sha256']==previous;previous=hashlib.sha256(line.encode()).hexdigest();events+=1
    from connectome.research_retrieval import load_library
    state=json.loads((OUT/'research_state.json').read_text());cards,h=load_library(ROOT/'competition_engineering/research_cards/v16');assert h==state['library']['manifest_sha256'] and len(cards)==state['library']['cards']
    assert json.loads((ROOT/'competition_engineering/search_knowledge.json').read_text())['version']==state['knowledge_version']==25
    print(json.dumps({'tracked_hashes':len(m['tracked']),'events':events,'library_cards':len(cards),'untouched_replication':512,'unassigned_training':roles['unassigned_count']}))

def external():
    m=json.loads((OUT/'external_artifacts.json').read_text())
    for row in m['required']:
        p=safe_target(ROOT,row['path']);assert p.is_file(),'Required external artifact missing: '+row['path'];assert sha(p)==row['sha256'],row['path']
    print('Verified exact incumbent, archive and required local dependencies; no scoring')

def datasets():
    import pyarrow.parquet as pq
    m=json.loads((OUT/'external_artifacts.json').read_text())
    for row in m['datasets']:
        p=safe_target(ROOT,row['path']);assert p.stat().st_size==row['bytes'];assert sha(p)==row['sha256'],row['path'];print('Opaque dataset hash verified: '+p.name,flush=True)
    train=ROOT/'competition_engineering/assets/wnn_connectome_starterpack/datasets/train.parquet';pf=pq.ParquetFile(train);assert pf.num_row_groups==10607 and pf.metadata.num_rows==212140000
    spec=json.loads((OUT/'reservations/untouched_512.json').read_text())
    for row in spec['identities']:
        if not row['role'].startswith('replication_'):continue
        group=row['group'];stat=pf.metadata.row_group(group).column(0).statistics
        if stat is not None and stat.has_min_max:assert stat.min==stat.max==row['seq_ix'] and stat.null_count==0
        else:
            values=pf.read_row_group(group,columns=['seq_ix'],use_threads=False)['seq_ix'].to_pylist();assert values and all(v==row['seq_ix'] for v in values)
    print('Verified512 reserved sequence identities using ID column only; no target/feature analysis')

def tests():
    p=subprocess.run([sys.executable,'-m','pytest','tests','-q','-p','no:cacheprovider'],cwd=ROOT,capture_output=True,text=True)
    output=p.stdout+'\n'+p.stderr;directory=ROOT/'competition_engineering/handoff_external';directory.mkdir(exist_ok=True);system=platform.system().lower();(directory/('regression_'+system+'.log')).write_text(output,encoding='utf-8')
    print(output[-3000:]);assert p.returncode==0,'Synthetic regression failure; see local external regression log'
    result={'mode':'Synthetic regression/model parity only; no ML competition experiment/evaluation','python':sys.version.split()[0],'platform':platform.system(),'passed':int(re.search(r'(\d+) passed',output)[1]),'skipped':int(re.search(r'(\d+) skipped',output)[1]) if re.search(r'(\d+) skipped',output) else 0,'returncode':p.returncode}
    (OUT/('verification_'+system+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

def isolation():
    assert sys.platform=='linux','Production isolation requires Linux/WSL; no fallback permitted'
    from competition_engineering.generated_sandbox import GeneratedSandbox
    from competition_engineering.generated_runner import WORKER
    directory=ROOT/'competition_engineering/handoff_external';directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='isolation_probe_',dir=directory) as work:GeneratedSandbox().preflight(WORKER,Path(work))
    (OUT/'isolation_verification.json').write_text(json.dumps({'production_isolation':'verified','probe':'Trusted worker intentionally rejects probe; no candidate or data mounted','network':'disabled','platform':platform.system()},indent=2)+'\n',encoding='utf-8')
    print('Production Linux bubblewrap preflight passed; no fitting or inference candidate ran')

def git_check():
    m=json.loads((OUT/'artifact_manifest.json').read_text());tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    for row in m['tracked']:assert row['path'] in tracked,'Not tracked: '+row['path']
    for row in json.loads((OUT/'external_artifacts.json').read_text())['required']:assert row['path'] not in tracked,'External artifact incorrectly tracked: '+row['path']
    print('Git coverage verified; required binaries remain intentionally external')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata-only',action='store_true');p.add_argument('--external',action='store_true');p.add_argument('--datasets',action='store_true');p.add_argument('--tests',action='store_true');p.add_argument('--isolation',action='store_true');p.add_argument('--check-git',action='store_true');p.add_argument('--restore-bundle',type=Path);a=p.parse_args()
    if a.restore_bundle:restore_bundle(a.restore_bundle)
    for requested,function in [(a.metadata_only,metadata),(a.external,external),(a.datasets,datasets),(a.tests,tests),(a.isolation,isolation),(a.check_git,git_check)]:
        if requested:function()
