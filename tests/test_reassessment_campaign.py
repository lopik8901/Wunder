import pytest
from competition_engineering.reassessment_campaign import record_mapping,search_scores
from connectome.mlevolve_history import SEARCH_METRIC_DOMAIN

@pytest.mark.parametrize('value',[None,[],0,'invalid'])
def test_legacy_history_optional_fields(value):
    assert record_mapping({'spec':value},'spec')=={}

def test_history_mapping_preserved():
    assert record_mapping({'spec':{'kind':'ridge'}},'spec')=={'kind':'ridge'}

def test_scores_require_search_domain_and_finite_values():
    assert search_scores({'metric_domain':'protected','metrics':{'combined_WP':1}})=={}
    assert search_scores({'metric_domain':SEARCH_METRIC_DOMAIN,'metrics':{'combined_WP':None,'WP_t0':float('nan'),'WP_t1':.5}})=={'WP_t1':.5}

def test_gradient_primitives_equal_numeric_official_derivative():
    import numpy as np
    from competition_engineering.gradient_uncertainty import primitives,gradient
    from competition_engineering.residual import sufficient,from_stats
    rng=np.random.default_rng(4)
    f=rng.normal(size=(400,8));y=rng.normal(size=(400,2))*2
    base=rng.normal(size=(400,2))*1.3
    stats,cross=primitives(f,y,base);g=gradient(stats,cross)
    for j in range(8):
        h=1e-6
        expected=(from_stats(sufficient(y,base+h*f[:,j,None]))-from_stats(sufficient(y,base-h*f[:,j,None])))/(2*h)
        np.testing.assert_allclose(g[j],expected,rtol=2e-7,atol=1e-8)
    np.testing.assert_allclose(gradient(np.stack([stats,stats]),np.stack([cross,cross])),np.stack([g,g]))

def test_matched_population_uses_common_counts_not_sample_frequency():
    import numpy as np
    from competition_engineering.gradient_uncertainty import matched_contrast
    from competition_engineering.residual import sufficient
    y=np.array([[.3,.4],[.9,1.1],[-.8,-.7]])
    p=y*.8
    cells=np.zeros((2,2,7,2))
    for flag in (0,1):
        for bucket in (0,1):
            factor=100 if flag==bucket else 1
            cells[flag,bucket,:6]=sufficient(y,p)*factor
            cells[flag,bucket,6]=len(y)*factor
    q=matched_contrast(cells,np.ones((2,2),bool))
    np.testing.assert_allclose(q['mse_ratio'],1)
    np.testing.assert_allclose(q['wp'][0],q['wp'][1])
    np.testing.assert_allclose(q['matched_rows_per_population'],6)

def test_shared_sequence_bootstrap_preserves_identical_population_contrast():
    import numpy as np
    from competition_engineering.gradient_uncertainty import primitives,resample,bootstrap_counts,cosine
    rng=np.random.default_rng(42)
    pairs=[primitives(rng.normal(size=(30,5)),rng.normal(size=(30,2)),rng.normal(size=(30,2))) for _ in range(10)]
    stats=np.stack([p[0] for p in pairs]);cross=np.stack([p[1] for p in pairs])
    counts=bootstrap_counts(10,rng,draws=100)
    a=resample(stats,cross,rng,counts=counts);b=resample(stats,cross,rng,counts=counts)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(cosine(a,b),1)

def test_manual_knowledge_context_is_allowlisted_and_not_a_tree_metric(tmp_path):
    import json
    from connectome.mlevolve_history import persistent_search_lessons
    path=tmp_path/'knowledge.json'
    q={'schema_version':1,'incumbent':{'wp':.658835322,'sha256':'a'*64},
       'established':[{'lesson':'Use official WP.'},{'lesson':'protected holdout WP1.0'}],
       'weakened':[{'hypothesis':'Loss choice alone is the bottleneck.'}],
       'closed_branches':['Broad loss grid'], 'protected_metrics':{'combined_WP':1.0}}
    path.write_text(json.dumps(q))
    text=persistent_search_lessons(path)
    assert 'WP=0.658835322' in text and 'Use official WP.' in text
    assert 'holdout' not in text and 'WP1.0' not in text and 'protected_metrics' not in text
    assert 'does not rescore' in text
    q['incumbent']['wp']=float('nan');path.write_text(json.dumps(q))
    assert persistent_search_lessons(path)==''

def test_fresh_training_group_partition_is_deterministic_and_disjoint():
    from competition_engineering.prospective_selection_diagnostic import new_groups
    a=new_groups(100,range(20),size=40);b=new_groups(100,range(20),size=40)
    assert a==b and len(set(a))==40 and not set(a)&set(range(20))
    with pytest.raises(ValueError):
        new_groups(10,range(9),size=2)

def test_weighted_diagnostic_moments_reproduce_subsetting_and_gradient():
    import numpy as np
    from competition_engineering.gradient_uncertainty import primitives,gradient
    from competition_engineering.nonlinear_selection_diagnostic import weighted_primitives
    rng=np.random.default_rng(20)
    f=rng.normal(size=(100,6));y=rng.normal(size=(100,2));p=rng.normal(size=(100,2))
    mask=rng.random(100)<.3
    a=weighted_primitives(f,y,p,mask.astype(float));b=primitives(f[mask],y[mask],p[mask])
    np.testing.assert_allclose(a[0],b[0]);np.testing.assert_allclose(a[1],b[1])
    np.testing.assert_allclose(gradient(*a),gradient(*b))

def test_within_correlation_equals_explicit_weighted_sequence_centering():
    import numpy as np
    from competition_engineering.population_mechanism_diagnostic import within_correlation
    from competition_engineering.residual import sufficient
    rng=np.random.default_rng(12);blocks=[];vy=0;vp=0;cov=0
    for shift in [-2,0,2]:
        y=np.clip(rng.normal(size=(100,2))+shift,-2,2);p=np.clip(y+rng.normal(size=(100,2))*.4,-2,2)
        w=np.abs(y);yc=y-(w*y).sum(0)/w.sum(0);pc=p-(w*p).sum(0)/w.sum(0)
        vy=vy+(w*yc*yc).sum(0);vp=vp+(w*pc*pc).sum(0);cov=cov+(w*yc*pc).sum(0)
        blocks.append(sufficient(y,p))
    np.testing.assert_allclose(within_correlation(np.array(blocks)),cov/np.sqrt(vy*vp))

def test_mask_tree_serialization_parity_and_causal_prefix():
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from tools.diagnose_causal_scoring_population import classifier_features
    from competition_engineering.nonlinear_weighted_readout import mask_focus
    rng=np.random.default_rng(40);x=rng.normal(size=(300,112));p=rng.normal(size=(300,2));step=np.arange(300)
    combo={'mean':np.zeros(112),'scale':np.ones(112)}
    f=classifier_features(x,p,step,combo['mean'],combo['scale'])[:,1:]
    m=HistGradientBoostingClassifier(max_iter=64,max_leaf_nodes=7,min_samples_leaf=10,early_stopping=False,random_state=1)
    m.fit(f,x[:,0]>0)
    trees={'rates':np.full(4,.5)}
    for fold in range(4):
        trees[f'fold{fold}_baseline']=m._baseline_prediction
        for j,t in enumerate(m._predictors):
            trees[f'fold{fold}_tree{j}']=t[0].nodes
    actual=mask_focus(x,p,step,combo,trees)
    expected=np.clip(m.predict_proba(f)[:,1]/.5,.2,5)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(actual[:100],mask_focus(x[:100],p[:100],step[:100],combo,trees))

def test_weighted_readout_source_retains_production_guard():
    from competition_engineering.nonlinear_weighted_readout import source
    from connectome.mlevolve_generated import validate_source
    validate_source(source(),training=True)
