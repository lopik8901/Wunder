"""Package frozen tune-selected calibration and verify real callback parity."""
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from competition_engineering.pipeline import cached, load_model, write_json
from competition_engineering.residual import sufficient, from_stats
from wnn_connectome_starterpack.utils import DataPoint

root=Path(__file__).resolve().parents[1]/"competition_engineering"
config=json.loads((root/"reports/gru_diagnostics.json").read_text())["tune"]["calibration_scale_bias"]
scale=[v[0] for v in config]; bias=[v[1] for v in config]
output=root/"submissions/calibrated_gru"
output.mkdir(parents=True,exist_ok=True)
source=root/"assets/wnn_connectome_starterpack/baseline"
shutil.copy2(source/"baseline.onnx",output/"baseline.onnx")
shutil.copy2(source/"solution.py",output/"gru.py")
(output/"solution.py").write_text('''"""Frozen tune-selected output calibration; GRU inputs/state unchanged."""
import numpy as np
from gru import PredictionModel as GRU

class PredictionModel:
    def __init__(self):
        self.gru = GRU()
        self.scale = np.array(SCALE, dtype=np.float32)
        self.bias = np.array(BIAS, dtype=np.float32)

    def predict(self, data_point):
        prediction = self.gru.predict(data_point)
        if prediction is None:
            return None
        return prediction * self.scale + self.bias
'''.replace('SCALE',repr(scale)).replace('BIAS',repr(bias)),encoding="utf-8")
base=[]; candidate=[]; replay_times=[]
model=load_model(output/"solution.py")
for index,(_,z) in enumerate(cached(root/"cache/gru_confirm")):
    pred=(z["p"]*np.array(scale,np.float32)+np.array(bias,np.float32)).astype(np.float32)
    base.append(sufficient(z["y"][z["mask"]],z["p"][z["mask"]]))
    candidate.append(sufficient(z["y"][z["mask"]],pred[z["mask"]]))
    if index<2:
        actual=np.full_like(pred,np.nan); started=time.perf_counter()
        for step,row in enumerate(z["x"]):
            p=model.predict(DataPoint(int(z["seq"]),step,bool(z["need"][step]),row))
            if z["need"][step]: actual[step]=p
            else: assert p is None
        replay_times.append(time.perf_counter()-started)
        np.testing.assert_array_equal(actual[99:],pred[99:])
base=np.stack(base); candidate=np.stack(candidate)
rng=np.random.default_rng(20260928); deltas=[]
for _ in range(1000):
    indices=rng.integers(len(base),size=len(base))
    deltas.append(from_stats(candidate[indices].sum(0))-from_stats(base[indices].sum(0)))
deltas=np.array(deltas)
archive=output.parent/"calibrated_gru.zip"
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ["solution.py","gru.py","baseline.onnx"]: z.write(output/name,name)
report={"scale":scale,"bias":bias,"selection":"tune only, frozen before confirmation",
        "confirm_gru_wp":from_stats(base.sum(0)).tolist(),
        "confirm_calibrated_wp":from_stats(candidate.sum(0)).tolist(),
        "bootstrap_95ci_per_target":np.quantile(deltas,[.025,.975],axis=0).tolist(),
        "bootstrap_95ci_combined":np.quantile(deltas.mean(1),[.025,.975]).tolist(),
        "callback_two_sequences_seconds":replay_times,"callback_parity":"exact float32 match, including sequence reset",
        "zip_bytes":archive.stat().st_size,"submission_path":str(archive),
        "status":"development baseline; final gate and Linux runtime validation pending"}
write_json(root/"reports/calibration_preserved.json",report)
print(json.dumps(report,indent=2))
