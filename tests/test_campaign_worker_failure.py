"""Worker EOF must preserve a useful sanitized infrastructure diagnosis."""
import json
import numpy as np

from competition_engineering.generated_runner import run_generated
from competition_engineering.generated_sandbox import GeneratedSandbox


def test_callback_exception_survives_eof_boundary(tmp_path):
    train=tmp_path/'train'
    train.mkdir()
    (train/'identity.json').write_text(json.dumps({'groups':[0]}))
    np.savez(train/'00000.npz',seq=1,x=np.zeros((120,112),np.float32),
             y=np.zeros((120,2),np.float32),need=np.arange(120)>=99,
             mask=np.arange(120)>=99)
    result=run_generated({
        'attempt_dir':str(tmp_path/'attempt'),'train_dir':str(train),'search_dir':str(train),
        'hypothesis':'Synthetic failure propagation, without evaluation evidence.',
        'train_tier':256,'timeout_seconds':30,
        'train_source':"import numpy as np\ndef train(train_files, output_dir):\n    np.savez(output_dir + '/model.npz', value=np.zeros(1))\n    return {}\n",
        'callback_source':"class PredictionModel:\n    def predict(self, point):\n        raise RuntimeError('deliberate inference failure')\n",
    },sandbox=GeneratedSandbox(synthetic_only=True),synthetic=True)
    assert result['status']=='error'
    assert result['stage']=='validation'
    assert 'deliberate inference failure' in result['failure_reason']
    assert 'metrics' not in result
