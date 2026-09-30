"""Package the frozen ridge winner; benchmark and verify callback outputs."""
import json
import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from competition_engineering.pipeline import cached, load_model, write_json
from competition_engineering.residual import features, calibrated_cached
from wnn_connectome_starterpack.utils import DataPoint

root=Path(__file__).resolve().parents[1]/"competition_engineering"
parser=argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint",default=str(root/"checkpoints/ridge256_v1/ridge.npz"))
parser.add_argument("--name",default="ridge_gru")
args=parser.parse_args()
if Path(args.name).name!=args.name: raise ValueError("name must be a single directory name")
output=root/"submissions"/args.name
if args.name!="ridge_gru" and output.exists(): raise FileExistsError(output)
output.mkdir(parents=True,exist_ok=True)
source=root/"assets/wnn_connectome_starterpack/baseline"
shutil.copy2(source/"baseline.onnx",output/"baseline.onnx")
shutil.copy2(source/"solution.py",output/"gru.py")
shutil.copy2(args.checkpoint,output/"ridge.npz")
(output/"solution.py").write_text('''"""Train-only ridge correction of frozen calibrated official GRU."""
from pathlib import Path
import numpy as np
from gru import PredictionModel as GRU

class PredictionModel:
    def __init__(self):
        self.gru = GRU()
        with np.load(Path(__file__).with_name("ridge.npz")) as z:
            self.mean = z["mean"]
            self.scale = z["scale"]
            self.coef = z["coef"]
            self.strengths = z["strengths"]
            self.base_scale = z["base_scale"].astype(np.float32)
            self.base_bias = z["base_bias"].astype(np.float32)

    def predict(self, data_point):
        prediction = self.gru.predict(data_point)
        if prediction is None:
            return None
        prediction = prediction * self.base_scale + self.base_bias
        normalized = np.clip((data_point.state-self.mean)/self.scale, -8, 8)
        features = np.concatenate(([1.], normalized, prediction)).astype(np.float32)
        return (prediction + self.strengths * (features @ self.coef)).astype(np.float32)
''',encoding="utf-8")
with np.load(output/"ridge.npz") as z: config={k:z[k] for k in z.files}
model=load_model(output/"solution.py"); times=[]; maxdiff=0
with threadpool_limits(limits=1):
    for index,(_,z) in enumerate(calibrated_cached(root/"cache/gru_confirm",config["base_scale"],config["base_bias"])):
        if index>=2: break
        expected=(z["p"]+config["strengths"]*(features(z,config["mean"],config["scale"])@config["coef"])).astype(np.float32)
        actual=np.full_like(expected,np.nan); started=time.perf_counter()
        for step,row in enumerate(z["x"]):
            value=model.predict(DataPoint(int(z["seq"]),step,bool(z["need"][step]),row))
            if z["need"][step]: actual[step]=value
            else: assert value is None
        times.append(time.perf_counter()-started)
        np.testing.assert_allclose(actual[99:],expected[99:],atol=1e-6,rtol=1e-6)
        maxdiff=max(maxdiff,float(np.max(np.abs(actual[99:]-expected[99:]))))
archive=output.parent/f"{args.name}.zip"
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name in ["solution.py","gru.py","baseline.onnx","ridge.npz"]: z.write(output/name,name)
report_path=root/"reports"/("ridge_preserved.json" if args.name=="ridge_gru" else f"{args.name}_preserved.json")
write_json(report_path,{
    "zip_bytes":archive.stat().st_size,"callback_two_sequences_seconds":times,
    "max_abs_parity_difference":maxdiff,"microseconds_per_row":sum(times)/40000*1e6,
    "estimated_max_rows_60min":int(3600/(sum(times)/40000)),
    "status":"frozen primary candidate before final gate; Linux runtime validation pending"})
print(report_path.read_text())
