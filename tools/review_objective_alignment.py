"""Checkpoint scientific decisions and source-reviewed library additions."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from competition_engineering import objective_alignment as oa
from competition_engineering.manual_search_core import ROOT
from competition_engineering.pipeline import write_json
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library


def summarize(tier, target=0):
    directory=oa.output_directory(tier,target)
    oa.OUT=directory
    oa.verify_fit_identity(directory/'isolated_training')
    q=json.loads((directory/'diagnosis.json').read_text())
    baseline=q['search']['incumbent']['wp']
    rows=[]
    for family in ['raw_residual','pearson_tangent','direct_wp']:
        wp=q['search'][family+'_select_wp']; mse=q['search'][family+'_select_weighted_raw_mse']
        rows.append({'family':family,'wp_selected_search':wp['wp'],'mse_selected_search':mse['wp'],
                     'score_selection_advantage':wp['wp']-mse['wp'],'delta_vs_incumbent':wp['wp']-baseline})
    best=max(q['search'],key=lambda k:q['search'][k]['wp'])
    fit_gaps={name:{'fit_wp_gain':max(r['wp'] for r in m['fit'])-m['fit'][0]['wp'],
                    'internal_selection_wp_gain':max(r['wp'] for r in m['internal_holdout'])-m['internal_holdout'][0]['wp']}
              for name,m in q['training']['models'].items()}
    comparisons=[]
    for name,info in q['training']['models'].items():
        for fit_row in info['fit'][1:]:
            key=name+'@'+str(fit_row['strength'])
            search=q['search'][key]
            comparisons.append({'model':name,'strength':fit_row['strength'],
                 'fit_mse_delta':fit_row['weighted_raw_mse']-info['fit'][0]['weighted_raw_mse'],
                 'fit_wp_delta':fit_row['wp']-info['fit'][0]['wp'],
                 'search_wp_delta':search['wp']-baseline})
    relationship={'candidate_points':len(comparisons),
                  'fit_mse_improved_search_wp_worsened':sum(c['fit_mse_delta']<0 and c['search_wp_delta']<0 for c in comparisons),
                  'fit_wp_improved_search_wp_worsened':sum(c['fit_wp_delta']>0 and c['search_wp_delta']<0 for c in comparisons),
                  'spearman_fit_mse_improvement_vs_search_wp':float(spearmanr([-c['fit_mse_delta'] for c in comparisons],[c['search_wp_delta'] for c in comparisons]).statistic),
                  'caveat':'Descriptive correlations across dependent grid candidates; no independent statistical claim'}
    result={'tier':tier,'target':target,'rows':rows,'best_diagnostic_grid':best,'best_grid_delta':q['search'][best]['wp']-baseline,
            'generalization_gaps':fit_gaps,'loss_score_relationship':relationship,
            'answer':'Lower residual MSE can yield worse official WP; score-based selection mitigates the mismatch but no practically stronger correction has yet been established.',
            'caveat':'Internal selection excludes correction-fit sequences only. The frozen incumbent was already trained and selected; this is not independent validation.',
            'decision':'Replicate once at fixed4096 training tier because direct-WP fitting gains did not transfer to training-sequence selection.' if tier==1024 else
                       'Abandon this static objective correction if no candidate satisfies the frozen rule; review distribution and sequence generalization next.'}
    destination=directory/'scientific_review_v2.json'
    if not destination.exists():
        write_json(destination,result)
        oa.event('scientific_interpretation',**result)
    print(json.dumps(result),flush=True)


def expand_library():
    library=ROOT/'competition_engineering/research_cards/v4'
    if library.exists():
        return
    parent=ROOT/'competition_engineering/research_cards/v3'
    shutil.copytree(parent,library)
    template=json.loads((parent/'functional_gradient_fitting.json').read_text())
    common={'version':1,'verified_on':'2026-10-02','curator':'Codex, primary publisher abstract review',
            'claim_confidence':'limited','dependencies':'Proposed task adaptations use NumPy/SciPy; no paper implementation is copied.',
            'licensing':'Original curator paraphrase and bibliographic metadata only.',
            'causality':'Curator adaptation: training targets are offline only; deployed corrections consume current causal inputs.',
            'incremental_inference':'Curator adaptation: training objective changes can retain a stateless readout.',
            'cpu_deployment':'Curator inference: retain bounded current-row operations and benchmark the callback.'}
    new_cards=[dict(common,card_id='nondecomposable_objective_linearization',family='objective',
        tags=['objective','gradient','pooled','linearization','correlation'],
        title='Optimizing Non-decomposable Performance Measures: A Tale of Two Classes',
        authors=['Harikrishna Narasimhan','Purushottam Kar','Prateek Jain'],year=2015,
        source_url='https://proceedings.mlr.press/v37/narasimhana15.html',source_locator='Publisher abstract, PMLR 37:199-208',
        mechanism='Adaptive linearization enables stochastic updates for specific nondecomposable classification metrics.',
        assumptions='The analyzed metrics are concave or pseudo-linear functions of true positive and negative rates.',
        limitations='Its classification convergence guarantees do not apply to clipped weighted Pearson. The task derivative needs separate verification.',
        useful_when='Curator inference: diagnose pooled-objective geometry before choosing a per-row proxy.',
        diagnostic_signatures='Curator inference: residual-error and official-score rankings disagree.',
        training_compute='Curator adaptation: global sufficient moments and a checked objective derivative replace raw residual targets.',
        concise_relevance='Motivates objective-aware fitting without claiming that task score gains must follow.'),
        dict(common,card_id='importance_weighted_model_selection',family='distribution_shift',
        tags=['covariate','shift','selection','distribution','generalization'],
        title='Covariate Shift Adaptation by Importance Weighted Cross Validation',
        authors=['Masashi Sugiyama','Matthias Krauledat','Klaus-Robert Müller'],year=2007,
        source_url='https://www.jmlr.org/papers/v8/sugiyama07a.html',source_locator='Publisher abstract, JMLR 8(35):985-1005',
        mechanism='Importance weighting adjusts cross-validation risk when input distributions differ.',
        assumptions='Covariate shift keeps the conditional target distribution given inputs unchanged.',
        limitations='Input drift alone does not establish this assumption. The paper does not provide a clipped Pearson guarantee.',
        useful_when='Curator inference: only after diagnosing input-distribution differences and uncertainty in selection transfer.',
        diagnostic_signatures='Curator inference: objective-aligned fitting improves training yet fails across sequences.',
        training_compute='Density-ratio estimation adds statistical and computational uncertainty.',
        concise_relevance='Provides a falsifiable next question about selection transfer; adaptation remains conditional on evidence.')]
    for changes in new_cards:
        card={**template,**changes}
        write_json(library/(card['card_id']+'.json'),validate_card(card))
    cards,manifest=load_library(library)
    write_json(oa.OUT/'library_v4_manifest.json',{'parent_sha256':load_library(parent)[1],
               'manifest_sha256':manifest,'count':len(cards),'added':[c['card_id'] for c in new_cards]})
    oa.event('library_expanded',manifest_sha256=manifest,count=len(cards),added=[c['card_id'] for c in new_cards],
             diagnosis_before_retrieval='Objective-aligned fit WP gains mostly vanished in training-sequence selection and fixed search.')


def expand_group_library():
    library=ROOT/'competition_engineering/research_cards/v5'
    if library.exists():
        return
    parent=ROOT/'competition_engineering/research_cards/v4'
    shutil.copytree(parent,library)
    card=json.loads((parent/'functional_gradient_fitting.json').read_text())
    card.update(card_id='risk_variance_across_sequence_groups',version=1,family='distribution_shift',
        tags=['objective','groups','risk','variance','generalization'],
        title='Out-of-Distribution Generalization via Risk Extrapolation (REx)',
        authors=['David Krueger','Ethan Caballero','Joern-Henrik Jacobsen','Amy Zhang','Jonathan Binas','Dinghuai Zhang','Remi Le Priol','Aaron Courville'],
        year=2021,source_url='https://proceedings.mlr.press/v139/krueger21a.html',
        source_locator='Publisher abstract, PMLR139:5815-5826',verified_on='2026-10-02',
        mechanism='V-REx penalizes variance of training-domain risks to reduce sensitivity to distribution shifts.',
        assumptions='Training-domain variation must represent relevant future variation.',
        limitations='Prediction-mean quartiles may not be actual domains. Penalizing WP improvement variance is a curator adaptation, not a reproduction or a task guarantee.',
        useful_when='Curator inference: only after training-group objective gradients exhibit disagreement.',
        diagnostic_signatures='Curator inference: aligned fitting gains fail across training sequences and official scoring populations.',
        training_compute='Curator adaptation: compute exact clipped WP derivatives in four training sequence groups and jointly optimize one regularized readout.',
        causality='Curator adaptation: training groups may use offline sequence summaries; the deployed readout never receives group identity or future rows.',
        incremental_inference='Curator adaptation: same current-row correction and fixed coefficients as the pooled control.',
        cpu_deployment='Curator adaptation: reuse the verified current-row ONNX export and benchmark the actual callback.',
        dependencies='Proposed adaptation uses NumPy/SciPy; no paper implementation is copied.',
        curator='Codex, primary publisher abstract review',claim_confidence='limited',
        concise_relevance='Tests cross-sequence objective stability after pooling and coarse scoring propensity alignment fail.')
    write_json(library/(card['card_id']+'.json'),validate_card(card))
    cards,manifest=load_library(library)
    write_json(oa.OUT/'library_v5_manifest.json',{'parent_sha256':load_library(parent)[1],
               'manifest_sha256':manifest,'count':len(cards),'added':[card['card_id']]})
    oa.event('library_expanded',manifest_sha256=manifest,count=len(cards),added=[card['card_id']],
             diagnosis_before_retrieval='Clipped WP training gains do not transfer; coarse propensity further reduces within-search gradient agreement.')


def expand_frozen_feature_library():
    parent=ROOT/'competition_engineering/research_cards/v5';library=ROOT/'competition_engineering/research_cards/v6'
    if library.exists():
        return
    shutil.copytree(parent,library)
    card=json.loads((parent/'functional_gradient_fitting.json').read_text())
    card.update(card_id='frozen_recurrent_feature_readout',version=1,family='frozen_representation',
        tags=['objective','latent','fixed','readout','causal'],
        title='The echo state approach to analysing and training recurrent neural networks, with an Erratum note',
        authors=['Herbert Jaeger'],year=2001,source_url='https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf',
        source_locator='Corrected2010 technical report; abstract, introduction and Section3, training readout functions',
        verified_on='2026-10-02',curator='Codex, primary full-text mechanism review',claim_confidence='limited',
        mechanism='Output weights are trained as readout functions of fixed recurrent activations using linear regression.',
        assumptions='The paper analyzes echo-state networks and their state properties.',
        limitations='A pretrained GRU is not the paper reservoir. No echo-state theorem or improvement guarantee transfers; exact causal-prefix and state-parity checks are required.',
        useful_when='Curator inference: existing frozen latent activations may retain task information compressed out of prediction values.',
        diagnostic_signatures='Curator inference: output/current-feature objective corrections fail; diagnose latent objective gradients before fitting.',
        training_compute='Curator adaptation: forward the frozen graph on training rows and fit a bounded objective-aware readout.',
        causality='Curator adaptation: use only the current forward hidden activations; recurrent weights and state updates stay frozen.',
        incremental_inference='Curator adaptation: reuse hidden states already computed by the incumbent; no additional recurrent state.',
        cpu_deployment='Curator adaptation: fuse the readout into the existing rowwise graph and benchmark actual callbacks.',
        dependencies='Proposed adaptation uses ONNX/ONNX Runtime and NumPy; no paper implementation is copied.',
        concise_relevance='A distinct fixed-feature probe, gated on measured objective alignment, rather than new recurrent dynamics.')
    write_json(library/(card['card_id']+'.json'),validate_card(card))
    cards,manifest=load_library(library)
    write_json(oa.OUT/'library_v6_manifest.json',{'parent_sha256':load_library(parent)[1],
               'manifest_sha256':manifest,'count':len(cards),'added':[card['card_id']]})
    oa.event('library_expanded',manifest_sha256=manifest,count=len(cards),added=[card['card_id']],
             diagnosis_before_retrieval='Pooled/current-row objective corrections, group-risk variance and exact moment replication fail; existing GRU state has not been probed directly.')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--tier',type=int,choices=[1024,4096],default=1024)
    p.add_argument('--target',type=int,choices=[0,1],default=0)
    p.add_argument('--library',action='store_true'); args=p.parse_args()
    summarize(args.tier,args.target)
    if args.library:
        expand_library()
