"""Hash only the fixed train/search caches and frozen model dependencies."""
import hashlib
import json
from pathlib import Path

from competition_engineering.autonomous_campaign import CAMPAIGN, ROOT, event, sha
from competition_engineering.manual_search_core import TRAIN_1024, SEARCH
from competition_engineering.pipeline import write_json


def main():
    out = CAMPAIGN / 'input_hashes.json'
    if out.exists():
        raise ValueError('Existing input manifest is immutable')
    result = {}
    for directory in (TRAIN_1024, SEARCH):
        identity = json.loads((directory / 'identity.json').read_text())
        files = {}
        for group in identity['groups']:
            path = directory / f'{group:05d}.npz'
            files[path.name] = sha(path)
        result[str(directory.relative_to(ROOT))] = {
            'identity_sha256': sha(directory / 'identity.json'), 'files': files,
            'manifest_sha256': hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        }
        print(json.dumps({'directory': directory.name, 'files': len(files)}), flush=True)
    dependencies = [
        'competition_engineering/manual_search_core.py',
        'competition_engineering/residual.py',
        'competition_engineering/search_error_diagnostics.py',
        'competition_engineering/generated_runner.py',
        'competition_engineering/generated_sandbox.py',
        'competition_engineering/generated_worker.py',
        'wnn_connectome_starterpack/utils.py',
        'competition_engineering/deployment/manual_targetwise_combo_v1/combo.npz',
        'competition_engineering/deployment/manual_targetwise_combo_v1/baseline.onnx',
        'competition_engineering/deployment/manual_targetwise_combo_v1/solution.py',
        'competition_engineering/deployment/manual_targetwise_combo_v1/gru.py',
    ]
    result['dependencies'] = {name: sha(ROOT / name) for name in dependencies}
    write_json(out, result)
    event('input_manifest_frozen', sha256=sha(out), path=str(out.relative_to(ROOT)))


if __name__ == '__main__':
    main()
