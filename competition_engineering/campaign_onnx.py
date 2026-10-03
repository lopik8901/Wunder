"""Exact algebraic export of frozen combo plus causal residual into one graph.

This changes execution representation only, not fitting, features or scoring.
All learned/frozen evidence remains in its original immutable directory.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T

from competition_engineering.manual_search_core import COMBO, load_combo


def export(model_path, destination, mode, late_only=False, strength=1., target=0):
    frozen = COMBO.parent
    graph_model = onnx.load(frozen / 'baseline.onnx')
    graph = graph_model.graph
    combo = load_combo()
    with np.load(model_path) as q:
        coef = q['coef'].copy()
    counter = 0
    def const(value, dtype=np.float64):
        nonlocal counter
        counter += 1
        name = f'campaign_const_{counter}'
        graph.initializer.append(nh.from_array(np.asarray(value, dtype=dtype), name))
        return name
    def op(kind, *inputs, **attrs):
        nonlocal counter
        counter += 1
        name = f'campaign_value_{counter}'
        graph.node.append(h.make_node(kind, list(inputs), [name], **attrs))
        return name
    def cast(value, dtype):
        return op('Cast', value, to=dtype)
    def clip(value, bound, dtype=np.float64):
        return op('Clip', value, const(-bound, dtype), const(bound, dtype))
    def gather(value, index):
        return op('Gather', value, const([index], np.int64), axis=0)
    x = op('Reshape', 'features', const([112], np.int64))
    norm64 = clip(op('Div', op('Sub', cast(x, T.DOUBLE), const(combo['mean'])), const(combo['scale'])), 8)
    norm32 = cast(norm64, T.FLOAT)
    raw = op('Reshape', 'prediction', const([2], np.int64))
    base = op('Add', op('Mul', raw, const(combo['base_scale'], np.float32)), const(combo['base_bias'], np.float32))
    feat = cast(op('Concat', const([1.], np.float32), norm32, base, axis=0), T.DOUBLE)
    base64 = cast(base, T.DOUBLE)
    # Coalesce the five frozen linear outputs into one matrix-vector kernel.
    # Coefficient folding changes double-roundoff only; parity is explicitly
    # checked against the unchanged NumPy reference before search scoring.
    weights = np.column_stack((combo['root_coef'] * combo['root_strengths'],
                               combo['magnitude_coef'][:, 0] * combo['magnitude_strengths'][0],
                               combo['gated_coef'][0, :, 1] * combo['gated_strengths'][1],
                               combo['gated_coef'][1, :, 1] * combo['gated_strengths'][1]))
    outputs = op('Add', op('MatMul', feat, const(weights)),
                 op('Gather', base64, const([0,1,0,1,1], np.int64), axis=0))
    root = cast(op('Gather', outputs, const([0,1], np.int64), axis=0), T.FLOAT)
    high = op('GreaterOrEqual', op('ReduceMax', op('Abs', root), axes=[0], keepdims=0), const(1, np.float32))
    frozen_output = cast(op('Concat', gather(outputs, 2),
                            op('Where', high, gather(outputs, 4), gather(outputs, 3)), axis=0), T.FLOAT)
    if target not in (0, 1):
        raise ValueError('target must be zero or one')
    graph.input.append(h.make_tensor_value_info('campaign_step', T.INT64, []))
    features = [const([1.], np.float32), clip(frozen_output, 2, np.float32), norm32]
    state_outputs = []
    if mode != 'current':
        graph.input.extend([h.make_tensor_value_info('campaign_fast', T.DOUBLE, [112]),
                        h.make_tensor_value_info('campaign_slow', T.DOUBLE, [112]),
                        ])
        current = cast(norm32, T.DOUBLE)
        fast = op('Add', op('Mul', current, const(.01)), op('Mul', 'campaign_fast', const(.99)))
        slow = op('Add', op('Mul', current, const(.001)), op('Mul', 'campaign_slow', const(.999)))
        features += [cast(fast, T.FLOAT), cast(slow, T.FLOAT)]
        state_outputs = [h.make_tensor_value_info(fast, T.DOUBLE, [112]), h.make_tensor_value_info(slow, T.DOUBLE, [112])]
    else:
        if len(coef) not in (115, 339):
            raise ValueError('current export requires115 current coefficients or339 zero-padded coefficients')
        if len(coef) == 339:
            if np.count_nonzero(coef[115:]):
                raise ValueError('current export cannot discard learned temporal coefficients')
            coef = coef[:115]
    extra_outputs = []
    if mode == 'bilinear':
        features += [cast(op('Mul', current, fast), T.FLOAT), cast(op('Mul', current, slow), T.FLOAT)]
    elif mode == 'normalize':
        graph.input.append(h.make_tensor_value_info('campaign_second', T.DOUBLE, [112]))
        second = op('Add', op('Mul', op('Mul', current, current), const(.001)), op('Mul', 'campaign_second', const(.999)))
        variance = op('Max', op('Sub', second, op('Mul', slow, slow)), const(.1))
        features += [cast(clip(op('Div', op('Sub', current, slow), op('Sqrt', variance)), 8), T.FLOAT)]
        extra_outputs.append(h.make_tensor_value_info(second, T.DOUBLE, [112]))
    elif mode not in ('linear', 'current'):
        raise ValueError('unsupported mode')
    residual = op('MatMul', op('Concat', *features, axis=0), const(coef.reshape(-1, 1), np.float32))
    residual = op('Mul', residual, const(strength, np.float32))
    if late_only:
        active = op('GreaterOrEqual', 'campaign_step', const(13333, np.int64))
        residual = op('Where', active, residual, const([0.], np.float32))
    corrected = [gather(frozen_output, 0), gather(frozen_output, 1)]
    corrected[target] = op('Add', corrected[target], residual)
    output = op('Concat', *corrected, axis=0)
    del graph.output[:]
    graph.output.extend([h.make_tensor_value_info(output, T.FLOAT, [2]),
                         h.make_tensor_value_info('next_hidden_0', T.FLOAT, [1, 1, 128]),
                         h.make_tensor_value_info('next_hidden_1', T.FLOAT, [1, 1, 128]),
                         *state_outputs, *extra_outputs])
    onnx.checker.check_model(graph_model)
    onnx.save(graph_model, destination)


def callback_source(mode):
    return f'''from pathlib import Path
import numpy as np
import onnxruntime as ort

class PredictionModel:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(Path(__file__).with_name('fused.onnx')), sess_options=options, providers=['CPUExecutionProvider'])
        self.sequence = None
        self.step_array = np.zeros((), np.int64)
        self.features = np.zeros((1,1,112), np.float32)
        names = ['hidden_0', 'hidden_1', 'campaign_fast', 'campaign_slow']
        shapes = [(1,1,128), (1,1,128), (112,), (112,)]
        types = [np.float32, np.float32, np.float64, np.float64]
        if {mode!r} == 'current':
            names, shapes, types = names[:2], shapes[:2], types[:2]
        if {mode!r} == 'normalize':
            names.append('campaign_second')
            shapes.append((112,))
            types.append(np.float64)
        self.states = [[np.zeros(shape, dtype) for shape,dtype in zip(shapes,types)] for _ in range(2)]
        self.outputs = [np.zeros(2,np.float32), np.zeros(2,np.float32)]
        output_names = [value.name for value in self.session.get_outputs()]
        self.bindings = []
        for side in range(2):
            binding = self.session.io_binding()
            binding.bind_cpu_input('features',self.features)
            binding.bind_cpu_input('campaign_step',self.step_array)
            for name,value in zip(names,self.states[side]):
                binding.bind_cpu_input(name,value)
            binding.bind_ortvalue_output(output_names[0],ort.OrtValue.ortvalue_from_numpy(self.outputs[side]))
            for name,value in zip(output_names[1:],self.states[1-side]):
                binding.bind_ortvalue_output(name,ort.OrtValue.ortvalue_from_numpy(value))
            self.bindings.append(binding)

    def predict(self, point):
        if point.seq_ix != self.sequence or point.step_in_seq == 0:
            if point.step_in_seq != 0:
                raise ValueError('A sequence must start at step zero')
            self.sequence = point.seq_ix
            self.previous = -1
            self.side = 0
            for states in self.states:
                for state in states:
                    state.fill(0)
        if point.step_in_seq != self.previous+1:
            raise ValueError('Rows must arrive in sequence order')
        self.previous = point.step_in_seq
        state = np.asarray(point.state, np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError('Expected finite feature row')
        self.step_array[...] = point.step_in_seq
        self.features[0,0] = state
        side = self.side
        self.session.run_with_iobinding(self.bindings[side])
        self.side = 1-side
        if not point.need_prediction:
            return None
        return self.outputs[side].copy()
'''
