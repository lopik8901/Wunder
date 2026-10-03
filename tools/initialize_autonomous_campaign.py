"""Create a new versioned card library without modifying existing evidence."""
import json
import shutil
from pathlib import Path

from competition_engineering.autonomous_campaign import CAMPAIGN, LIBRARY, ROOT, initialize, event, sha
from competition_engineering.pipeline import write_json
from connectome.research_foundation import validate_card
from connectome.research_retrieval import load_library


def main():
    initialize()
    if LIBRARY.exists():
        raise ValueError('Library version exists; never overwrite')
    source = ROOT / 'competition_engineering/research_cards/v1'
    shutil.copytree(source, LIBRARY)
    card = {
        'schema_version': 1, 'card_id': 'nonlinear_vector_autoregression', 'version': 1,
        'card_type': 'paper', 'family': 'polynomial_memory',
        'tags': ['temporal', 'nonlinear', 'polynomial', 'memory', 'causal'],
        'title': 'Next generation reservoir computing',
        'authors': ['Daniel J. Gauthier', 'Erik Bollt', 'Aaron Griffith', 'Wendson A. S. Barbosa'],
        'year': 2021, 'source_url': 'https://www.nature.com/articles/s41467-021-25801-2',
        'source_locator': 'Introduction, Equations 3-6; Results; Discussion',
        'curator': 'Codex, primary full-text mechanism review', 'verified_on': '2026-10-01',
        'claim_confidence': 'moderate',
        'mechanism': 'A regularized linear readout uses delayed observations and polynomial features of their joint history.',
        'useful_when': 'Curator inference: test interactions between current inputs and causal history after additive lag models fail.',
        'diagnostic_signatures': 'Curator inference: persistent residuals with weak linear feature correlations motivate a falsifiable nonlinear-memory comparison.',
        'assumptions': 'Paper benchmarks are low-dimensional dynamical systems; transfer to this task is unestablished.',
        'limitations': 'Full polynomial bases grow combinatorially. A featurewise EMA interaction is a restricted adaptation, not a reproduction or universality guarantee.',
        'causality': 'Curator adaptation: use present/past inputs only, zero state at each sequence, and never feed target residuals into online state.',
        'incremental_inference': 'Curator adaptation: bounded causal state and a polynomial readout permit incremental computation.',
        'training_compute': 'Regularized least squares fits the output weights.',
        'cpu_deployment': 'Curator inference: diagonal interactions have linear feature cost, but benchmark the actual callback.',
        'dependencies': 'Proposed adaptation uses NumPy and SciPy; the paper does not require these implementations.',
        'licensing': 'Metadata and original curator paraphrase; no copied implementation.',
        'concise_relevance': 'A structured alternative to generic random nonlinear features; task success requires measured search evidence.',
    }
    write_json(LIBRARY / (card['card_id'] + '.json'), validate_card(card))
    cards, manifest = load_library(LIBRARY)
    write_json(CAMPAIGN / 'library_manifest.json', {
        'version': 'v2', 'parent_version': 'v1', 'parent_manifest_sha256': load_library(source)[1],
        'manifest_sha256': manifest, 'count': len(cards),
        'files': {p.name: sha(p) for p in sorted(LIBRARY.glob('*.json'))},
        'added': [card['card_id']],
        'diagnosis_before_retrieval': 'Late t0 and persistent residuals remain after generic current-feature nonlinear expansion, additive lag models, and late linear expert tests.',
    })
    event('library_expanded', manifest_sha256=manifest, cards=len(cards), added=card['card_id'])
    print(json.dumps({'count': len(cards), 'manifest_sha256': manifest}))


if __name__ == '__main__':
    main()
