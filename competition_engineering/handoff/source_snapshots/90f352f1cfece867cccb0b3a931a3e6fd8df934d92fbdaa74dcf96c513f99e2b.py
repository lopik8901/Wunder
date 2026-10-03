"""Compile numeric histogram trees into a single CPU ONNX ensemble operator.

No estimator runtime is deployed. Float64 split thresholds are rounded downward
to preserve <= routing for float32 inputs, including values at split boundaries.
"""
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from competition_engineering import campaign_onnx

def float32_floor(value):
    rounded=np.float32(value)
    return np.nextafter(rounded,np.float32(-np.inf)) if float(rounded)>value else rounded

def attributes(trees,baseline):
    attrs={key:[] for key in ('nodes_treeids','nodes_nodeids','nodes_featureids','nodes_modes',
        'nodes_values','nodes_truenodeids','nodes_falsenodeids','nodes_missing_value_tracks_true',
        'target_treeids','target_nodeids','target_ids','target_weights')}
    for tree_id,nodes in enumerate(trees):
        if np.any(nodes['is_categorical']):
            raise ValueError('Only numeric trees are supported')
        for node_id,node in enumerate(nodes):
            leaf=bool(node['is_leaf'])
            for key,value in [('nodes_treeids',tree_id),('nodes_nodeids',node_id),
                ('nodes_featureids',0 if leaf else int(node['feature_idx'])),
                ('nodes_modes','LEAF' if leaf else 'BRANCH_LEQ'),
                ('nodes_values',0. if leaf else float(float32_floor(node['num_threshold']))),
                ('nodes_truenodeids',0 if leaf else int(node['left'])),
                ('nodes_falsenodeids',0 if leaf else int(node['right'])),
                ('nodes_missing_value_tracks_true',int(node['missing_go_to_left']))]:
                attrs[key].append(value)
            if leaf:
                attrs['target_treeids'].append(tree_id);attrs['target_nodeids'].append(node_id)
                attrs['target_ids'].append(0);attrs['target_weights'].append(float(node['value']))
    attrs.update(n_targets=1,aggregate_function='SUM',post_transform='NONE',base_values=[float(baseline)])
    return attrs

def ensemble_model(trees,baseline,width):
    node=h.make_node('TreeEnsembleRegressor',['x'],['y'],domain='ai.onnx.ml',**attributes(trees,baseline))
    graph=h.make_graph([node],'compact_numeric_ensemble',
        [h.make_tensor_value_info('x',T.FLOAT,[None,width])],
        [h.make_tensor_value_info('y',T.FLOAT,[None,1])])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17),h.make_opsetid('ai.onnx.ml',1)])
    model.ir_version=8;onnx.checker.check_model(model)
    return model

def export(trees,baseline,destination,work,strength=.25,target=0):
    if target not in (0,1):
        raise ValueError('Target must be 0 or 1')
    zero=work/'zero_readout.npz';np.savez(zero,coef=np.zeros(115,np.float32))
    campaign_onnx.export(zero,destination,'current')
    model=onnx.load(destination);graph=model.graph;frozen=graph.output[0].name
    combo=campaign_onnx.load_combo();counter=0
    def constant(value,dtype):
        nonlocal counter
        counter+=1;name='compact_constant_'+str(counter)
        graph.initializer.append(nh.from_array(np.asarray(value,dtype=dtype),name));return name
    def op(kind,*inputs,**attrs):
        nonlocal counter
        counter+=1;name='compact_value_'+str(counter)
        graph.node.append(h.make_node(kind,list(inputs),[name],**attrs));return name
    x=op('Reshape','features',constant([112],np.int64))
    norm=op('Div',op('Sub',op('Cast',x,to=T.DOUBLE),constant(combo['mean'],np.float64)),constant(combo['scale'],np.float64))
    norm=op('Cast',op('Clip',norm,constant(-8,np.float64),constant(8,np.float64)),to=T.FLOAT)
    p=op('Clip',frozen,constant(-2,np.float32),constant(2,np.float32))
    features=op('Reshape',op('Concat',norm,p,axis=0),constant([1,114],np.int64))
    graph.node.append(h.make_node('TreeEnsembleRegressor',[features],['compact_tree_prediction'],domain='ai.onnx.ml',**attributes(trees,baseline)))
    residual=op('Reshape','compact_tree_prediction',constant([1],np.int64))
    direction=np.zeros((1,2),np.float32);direction[0,target]=strength
    output=op('Add',frozen,op('MatMul',residual,constant(direction,np.float32)))
    graph.output[0].name=output
    model.opset_import.append(h.make_opsetid('ai.onnx.ml',1))
    onnx.checker.check_model(model);onnx.save(model,destination)
