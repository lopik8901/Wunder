import hashlib
import json
import zipfile
import pytest
from tools.verify_research_handoff import restore_bundle,safe_target,sha
from tools.build_research_handoff import portable

def test_lf_hash_portable_without_changing_frozen_hash(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'one\r\ntwo\r\n');b.write_bytes(b'one\ntwo\n')
    assert sha(a,'git_lf_text')==sha(b,'git_lf_text');assert sha(a)!=sha(b)

def test_restore_rejects_traversal_and_refuses_overwrite(tmp_path):
    for name in ['../escape','/absolute','bad\\path']:
        with pytest.raises(ValueError):safe_target(tmp_path,name)
    root=tmp_path/'repo';root.mkdir();archive=tmp_path/'bundle.zip';raw=b'frozen exact artifact'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('model/artifact.npz',raw)
    m={'required_transfer_archive':{'sha256':sha(archive)},'required':[{'path':'model/artifact.npz','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}
    restore_bundle(archive,root,m);assert (root/'model/artifact.npz').read_bytes()==raw
    (root/'model/artifact.npz').write_bytes(b'different')
    with pytest.raises(AssertionError,match='Refusing'):restore_bundle(archive,root,m)
    assert (root/'model/artifact.npz').read_bytes()==b'different'

def test_restore_rejects_extra_members_before_writing(tmp_path):
    archive=tmp_path/'bundle.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('extra',b'bad')
    m={'required_transfer_archive':{'sha256':sha(archive)},'required':[]}
    with pytest.raises(AssertionError,match='Unexpected'):restore_bundle(archive,tmp_path,m)
    assert not (tmp_path/'extra').exists()

def test_portable_export_preserves_scientific_uncertainty_and_omits_protected_fields():
    raw={'paired95':[-.001,.002],'status':'inconclusive','private_path':r'C:\Users\someone\data','holdout_results':{'score':.9},'evidence_sha256':{'large':'digest'}}
    clean=portable(raw);assert clean['paired95']==raw['paired95'] and clean['status']=='inconclusive'
    assert clean['private_path']=='<LOCAL_PATH>' and 'holdout_results' not in clean and 'evidence_sha256' not in clean
def test_changed_paths_preserves_staged_and_unstaged_changes(tmp_path):
    import subprocess
    from tools.build_research_handoff import changed_paths
    def git(*args):
        subprocess.run(['git',*args],cwd=tmp_path,check=True,capture_output=True)
    git('init')
    for name in ['staged.py','unstaged.py']:
        (tmp_path/name).write_text('original\n')
    git('add','.')
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','fixture')
    for name in ['staged.py','unstaged.py']:
        (tmp_path/name).write_text('changed\n')
    git('add','staged.py')
    assert set(changed_paths(tmp_path)) == {'staged.py','unstaged.py'}

def test_manifest_is_nonrecursive_and_paths_are_unique():
    from tools.build_research_handoff import OUT
    manifest=json.loads((OUT/'artifact_manifest.json').read_text())
    paths=[entry['path'] for entry in manifest['tracked']]
    assert 'competition_engineering/handoff/artifact_manifest.json' not in paths
    assert len(paths)==len(set(paths))
