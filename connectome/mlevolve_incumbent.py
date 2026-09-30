"""Score MLEvolve's actual root with the frozen search-split incumbent."""

from utils.metric import MetricValue

from connectome.mlevolve_bounded import INCUMBENT_TUNE_WP


def seed_incumbent(agent, journal):
    code = '''EXPERIMENT = {
    "kind": "ridge",
    "hypothesis": "Reproduce the frozen target-specific 1024-sequence ridge incumbent.",
    "train_tier": 1024,
    "target_mode": "t0_raw_t1_clipped",
    "sample_stride": 10,
    "ridge_penalty": 0.001,
}'''
    node = agent.virtual_root
    node.code = code
    node.plan = "Frozen ridge1024_targetwise incumbent; scored on the search split."
    node.metric = MetricValue(INCUMBENT_TUNE_WP, maximize=True)
    node.is_buggy = False
    node.is_valid = True
    node.analysis = "Search WP 0.654865500. Submitted artifact and checkpoint are frozen."
    node.connectome_metrics = {"wp_t0": 0.6459635357038319,
                               "wp_t1": 0.6637674635202809,
                               "combined_wp": INCUMBENT_TUNE_WP}
    node.branch_id = 0
    agent.branch_all_nodes[0] = [node]
    agent.branch_successful_nodes[0] = [node]
    agent.best_node = node
    agent.best_metric = INCUMBENT_TUNE_WP
    agent.top_candidates.append(node)
    return node
