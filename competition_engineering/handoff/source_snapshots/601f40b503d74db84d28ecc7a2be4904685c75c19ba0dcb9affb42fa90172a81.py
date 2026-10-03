"""Add primary-source methodology cards without rewriting earlier libraries."""
import shutil
from competition_engineering import reassessment_campaign as campaign
from competition_engineering.pipeline import write_json
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library

def main():
    parent=campaign.ROOT/'competition_engineering/research_cards/v6'
    destination=campaign.ROOT/'competition_engineering/research_cards/v7'
    if destination.exists():
        raise ValueError('Research library versions are immutable')
    shutil.copytree(parent,destination)
    definitions=[
        ('model_selection_variance','On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation',
         ['Gavin C. Cawley','Nicola L. C. Talbot'],2010,'https://www.jmlr.org/papers/v11/cawley10a.html','Abstract; introduction; sections4-5',
         'Selection-criterion variance can cause overfitting and optimistic evaluation bias.',
         'Curator inference: prospective correction-disjoint selection and audit can diagnose the search process.',
         'Their demonstrations use classification and cross-validation; task-specific correlation behavior must be measured.'),
        ('cross_validation_uncertainty','No Unbiased Estimator of the Variance of K-Fold Cross-Validation',
         ['Yoshua Bengio','Yves Grandvalet'],2004,'https://www.jmlr.org/papers/v5/grandvalet04a.html','Abstract; theorem and conclusion in primary PDF',
         'There is no universally unbiased variance estimator for K-fold cross-validation under all distributions.',
         'Curator inference: overlapping-fold score variation is not an independent uncertainty estimate.',
         'The theorem concerns K-fold estimates, not the validity of our sequence bootstrap.'),
        ('adaptive_analysis_validity','Preserving Statistical Validity in Adaptive Data Analysis',
         ['Cynthia Dwork','Vitaly Feldman','Moritz Hardt','Toniann Pitassi','Omer Reingold','Aaron Roth'],2014,
         'https://arxiv.org/abs/1411.2664','Primary abstract, preprint v3; original preprint2014',
         'Coordinated privacy-based perturbation can control error for adaptively chosen expectation queries.',
         'Curator inference: repartitioning already-exposed search data does not restore independence.',
         'No privacy-based query mechanism is implemented here; ordinary bootstrap intervals inherit no adaptive guarantee.')]
    for ident,title,authors,year,url,locator,mechanism,useful,limitations in definitions:
        card={'schema_version':1,'card_id':ident,'version':1,'card_type':'paper','family':'experimental_methodology',
            'tags':['selection','uncertainty','generalization','research'],'title':title,'authors':authors,'year':year,
            'source_url':url,'source_locator':locator,'curator':'Codex, primary-source review','verified_on':'2026-10-02',
            'claim_confidence':'moderate','mechanism':mechanism,'useful_when':useful,
            'diagnostic_signatures':'Curator inference: many near-tied candidates on a reused finite search set.',
            'assumptions':'Paper assumptions need separate verification for pooled clipped correlation and dependent rows.',
            'limitations':limitations,'causality':'Curator inference: fixes concern offline selection, not inference state.',
            'incremental_inference':'No inference mechanism.','training_compute':'Selection and resampling only.',
            'cpu_deployment':'No added inference cost.','dependencies':'No paper code imported.',
            'licensing':'Original curator paraphrase and bibliographic metadata.',
            'concise_relevance':'Diagnoses research reliability without changing the official score.'}
        write_json(destination/(ident+'.json'),validate_card(card))
    cards,manifest=load_library(destination)
    write_json(campaign.OUT/'library_v7_manifest.json',{'parent_manifest_sha256':load_library(parent)[1],
        'manifest_sha256':manifest,'count':len(cards),'added':[d[0] for d in definitions],
        'files':{p.name:campaign.sha(p) for p in sorted(destination.glob('*.json'))}})
    campaign.event('library_expanded',version='v7',count=len(cards),manifest_sha256=manifest)
    print(manifest)

if __name__=='__main__':
    main()
