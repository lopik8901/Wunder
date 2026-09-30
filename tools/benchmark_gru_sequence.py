"""Test an offline dynamic-shape GRU cache accelerator against official replay.

Only shape metadata changes in memory. Original weights, operations, feature
values, submission artifact and official row-wise evaluator remain untouched.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from competition_engineering.pipeline import cached, write_json

root=Path(__file__).resolve().parents[1]/"competition_engineering"
graph=onnx.load(root/"assets/wnn_connectome_starterpack/baseline/baseline.onnx")
for node in graph.graph.node:
    if node.op_type=="GRU":
        for attr in node.attribute:
            if attr.name=="direction": assert attr.s==b"forward"
for value in graph.graph.input:
    shape=value.type.tensor_type.shape.dim
    if value.name=="features": shape[0].dim_param="batch"; shape[1].dim_param="steps"
    else: shape[1].dim_param="batch"
for value in graph.graph.output:
    shape=value.type.tensor_type.shape.dim
    if value.name=="prediction": shape[0].dim_param="batch"; shape[1].dim_param="steps"
    else: shape[1].dim_param="batch"
del graph.graph.value_info[:]
onnx.checker.check_model(graph)
options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
session=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=["CPUExecutionProvider"])
blocks=[]
for _,z in cached(root/"cache/gru_confirm"):
    blocks.append(z)
    if len(blocks)==2: break
inputs=np.stack([z["x"] for z in blocks])
def predict(x):
    state=np.zeros((1,len(x),128),np.float32)
    return session.run(None,{"features":x,"hidden_0":state,"hidden_1":state.copy()})[0]
started=time.perf_counter(); predictions=predict(inputs); elapsed=time.perf_counter()-started
reference=np.stack([z["p"] for z in blocks])
np.testing.assert_allclose(predictions[:,99:],reference[:,99:],atol=3e-5,rtol=3e-5)
one=predict(inputs[:1])
np.testing.assert_allclose(one,predictions[:1],atol=3e-5,rtol=3e-5)
changed=inputs.copy(); changed[:,1000:]+=1
causal=predict(changed)
np.testing.assert_array_equal(predictions[:,:1000],causal[:,:1000])
report={"sequences":2,"rows":40000,"seconds":elapsed,"rows_per_second":40000/elapsed,
        "max_abs_difference_vs_official":float(np.max(np.abs(predictions[:,99:]-reference[:,99:]))),
        "causal_prefix_test":"exact match","independent_sequence_test":"passed",
        "extrapolated_full_train_seconds":212140000/(40000/elapsed),
        "status":"offline-cache optimization preflight only; final gate still uses official callback"}
write_json(root/"reports/gru_sequence_benchmark.json",report)
print(json.dumps(report,indent=2))
