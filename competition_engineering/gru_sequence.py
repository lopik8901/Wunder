"""Optional offline GRU caching accelerator; preserves graph operations/weights."""
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort


class SequenceGRU:
    def __init__(self, path):
        graph=onnx.load(Path(path))
        if [v.name for v in graph.graph.input] != ["features","hidden_0","hidden_1"]:
            raise ValueError("unsupported official ONNX interface")
        if [v.name for v in graph.graph.output] != ["prediction","next_hidden_0","next_hidden_1"]:
            raise ValueError("unsupported official ONNX outputs")
        for node in graph.graph.node:
            if node.op_type=="GRU":
                for attr in node.attribute:
                    if attr.name=="direction" and attr.s!=b"forward":
                        raise ValueError("only forward causal GRUs are supported")
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
        self.session=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=["CPUExecutionProvider"])

    def predict(self, sequences):
        x=np.asarray(sequences,dtype=np.float32)
        if x.ndim!=3 or x.shape[-1]!=112 or not np.isfinite(x).all():
            raise ValueError("expected finite (batch, steps, 112) features")
        state=np.zeros((1,len(x),128),np.float32)
        prediction=self.session.run(None,{"features":x,"hidden_0":state,"hidden_1":state.copy()})[0]
        if prediction.shape!=(*x.shape[:2],2) or not np.isfinite(prediction).all():
            raise ValueError("invalid sequence outputs")
        return prediction
