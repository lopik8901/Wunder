import numpy as np
import pytest

def test_weight_normalization_makes_regularized_fit_scale_invariant():
    from competition_engineering.probability_readout import normalize_focus
    rng=np.random.default_rng(43);x=rng.normal(size=(100,6));y=rng.normal(size=100)
    weights=rng.uniform(.001,1,100)
    def fit(w):
        w=normalize_focus(w)
        return np.linalg.solve(x.T@(w[:,None]*x)+np.eye(6)*100,x.T@(w*y))
    np.testing.assert_allclose(fit(weights),fit(weights*321.7),atol=1e-14)
    for bad in ([0,0],[-1,1],[float('nan'),1]):
        with pytest.raises(ValueError):
            normalize_focus(bad)

def test_probability_readout_training_source_passes_unchanged_guard():
    from competition_engineering.probability_readout import source
    from connectome.mlevolve_generated import validate_source
    validate_source(source(),training=True)

def test_paired_moment_bootstrap_preserves_identical_populations():
    from competition_engineering.root_cause_campaign import bootstrap_moments
    rng=np.random.default_rng(4)
    a=rng.normal(size=(10,3,6,2));moments=np.stack([a,a],axis=2)
    counts=rng.multinomial(10,np.ones(10)/10,100)
    result=bootstrap_moments(moments,counts)
    np.testing.assert_array_equal(result[:,:,0],result[:,:,1])

def test_tree_export_matches_sklearn_at_split_boundaries():
    from sklearn.ensemble import HistGradientBoostingRegressor
    import onnxruntime as ort
    from competition_engineering.compact_tree_onnx import ensemble_model
    rng=np.random.default_rng(17);x=rng.normal(size=(900,5)).astype(np.float32)
    y=np.sin(x[:,0]*2)+x[:,1]*x[:,2]
    m=HistGradientBoostingRegressor(max_iter=32,max_leaf_nodes=7,min_samples_leaf=15,
        early_stopping=False,random_state=11).fit(x,y)
    trees=[tree[0].nodes for tree in m._predictors]
    probes=[x]
    for nodes in trees:
        for node in nodes:
            if node['is_leaf']:
                continue
            j=int(node['feature_idx']);t=np.float32(node['num_threshold'])
            rows=np.zeros((3,5),np.float32)
            rows[:,j]=[np.nextafter(t,np.float32(-np.inf)),t,np.nextafter(t,np.float32(np.inf))]
            probes.append(rows)
    probe=np.concatenate(probes)
    model=ensemble_model(trees,float(m._baseline_prediction.ravel()[0]),5)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    actual=session.run(None,{'x':probe})[0].ravel()
    np.testing.assert_allclose(actual,m.predict(probe),atol=8e-7,rtol=8e-7)
    # A future observation cannot alter an earlier tree prediction.
    changed=probe.copy();changed[100:]+=2
    np.testing.assert_array_equal(actual[:100],session.run(None,{'x':changed})[0].ravel()[:100])

def test_tree_threshold_rounding_preserves_float32_routing():
    from competition_engineering.compact_tree_onnx import float32_floor
    lo=np.float32(1);hi=np.nextafter(lo,np.float32(np.inf))
    threshold=float(lo)*.25+float(hi)*.75
    assert np.float32(threshold)==hi
    assert float32_floor(threshold)==lo

def test_library_revision_rejects_duplicate_before_publishing(tmp_path):
    import json
    from pathlib import Path
    from tools.expand_root_cause_library import publish_revision
    from connectome.research_retrieval import load_library
    parent=tmp_path/'v1';parent.mkdir();destination=tmp_path/'v2'
    original=Path(__file__).resolve().parents[1]/'competition_engineering/research_cards/v8/importance_weighted_model_selection.json'
    card=json.loads(original.read_text())
    (parent/original.name).write_bytes(original.read_bytes())
    duplicate={**card,'card_id':'duplicate_source'}
    with pytest.raises(ValueError,match='Duplicate source'):
        publish_revision(parent,destination,duplicate)
    assert not destination.exists()
    assert (parent/original.name).read_bytes()==original.read_bytes()
    revised={**card,'version':2,'limitations':card['limitations']+' Revised with further diagnostic limits.'}
    cards,manifest=publish_revision(parent,destination,revised)
    assert load_library(destination)[1]==manifest
    assert cards[0]['version']==2
    assert not destination.with_name('v2.staging').exists()

def test_tree_selection_pools_moments_and_passes_source_guard():
    from competition_engineering.tree_training_selection import selection_moments,source
    from competition_engineering.residual import from_stats,sufficient
    from connectome.mlevolve_generated import validate_source
    rng=np.random.default_rng(117)
    y=rng.normal(size=500)*2;base=rng.normal(size=500);correction=rng.normal(size=500)*.1
    total=selection_moments(y[:30],base[:30],correction[:30],np.ones(30))
    total+=selection_moments(y[30:],base[30:],correction[30:],np.ones(470))
    for i,strength in enumerate([0.,.05,.1,.25,.5,1.]):
        expected=from_stats(sufficient(np.column_stack([y,y]),np.column_stack([base+strength*correction]*2)))
        np.testing.assert_allclose(from_stats(np.repeat(total[i,:,None],2,axis=1)),expected,atol=1e-14)
    validate_source(source(),training=True)

def test_closed_campaign_knowledge_is_independent_of_live_pointer(tmp_path):
    import json
    from tools.verify_reassessment import verify_frozen_knowledge
    frozen=tmp_path/'knowledge_v3.json';live=tmp_path/'search_knowledge.json'
    knowledge={'version':3,'incumbent':{'sha256':'abc','wp':.6588}}
    frozen.write_text(json.dumps(knowledge))
    live.write_text(json.dumps({**knowledge,'version':4,'new_lesson':'later research'}))
    verify_frozen_knowledge(frozen,'abc',.6588)
    frozen.write_text(json.dumps({**knowledge,'incumbent':{'sha256':'tampered','wp':.6588}}))
    with pytest.raises(AssertionError):
        verify_frozen_knowledge(frozen,'abc',.6588)
