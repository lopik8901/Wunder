"""Record downloaded baseline interfaces and independent mask consistency."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import pyarrow.parquet as pq

root=Path(__file__).resolve().parents[1]
assets=root/"competition_engineering/assets/wnn_connectome_starterpack"
model=onnx.load(assets/"baseline/baseline.onnx")
def interface(values):
    return [{"name":v.name,"shape":[d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]} for v in values]
report={"onnx_sha256":hashlib.sha256((assets/"baseline/baseline.onnx").read_bytes()).hexdigest(),
        "inputs":interface(model.graph.input),"outputs":interface(model.graph.output),
        "operations":dict(Counter(n.op_type for n in model.graph.node)),
        "normalized_document_matches":{}}
for name in ["utils.py","METRIC.md","README.md","docs/data_overview.md","docs/submission_guide.md"]:
    report["normalized_document_matches"][name]=(assets/name).read_text(encoding="utf-8")==(
        root/"wnn_connectome_starterpack"/name).read_text(encoding="utf-8")
valid=pq.ParquetFile(assets/"datasets/valid.parquet")
mask=pq.ParquetFile(assets/"datasets/valid_mask.parquet")
assert mask.metadata.num_rows==valid.metadata.num_rows
assert mask.num_row_groups==valid.num_row_groups
columns=["seq_ix","step_in_seq","is_scored"]
assert mask.schema_arrow.names==columns
for group in [0,valid.num_row_groups//2,valid.num_row_groups-1]:
    a=valid.read_row_group(group,columns=columns); b=mask.read_row_group(group)
    for name in columns: np.testing.assert_array_equal(a[name].to_numpy(),b[name].to_numpy())
report["separate_mask"]={"rows":mask.metadata.num_rows,"groups":mask.num_row_groups,
                         "sampled_groups_match":[0,valid.num_row_groups//2,valid.num_row_groups-1]}
destination=root/"competition_engineering/environment/official_assets.json"
destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
