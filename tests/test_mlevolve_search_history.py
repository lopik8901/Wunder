from types import SimpleNamespace

from connectome.mlevolve_history import (
    format_search_history,
    next_parent_for_followup,
    planner_history_section,
    proposal_signature,
    load_prior_search_records,
)


def _record(*, score=0.66, result_extra=None):
    result = {
        "status": "success",
        "metrics": {"WP_t0": score - 0.01, "WP_t1": score + 0.01, "combined_WP": score},
        "metric_domain": "fixed development tuning sequences",
        "stage": "completed",
    }
    result.update(result_extra or {})
    return {
        "node_id": "node-1",
        "parent_id": "root",
        "status": "success",
        "runtime_seconds": 90,
        "delta_vs_incumbent_tune": score - 0.6548655,
        "delta_vs_parent_tune": score - 0.6548655,
        "spec": {"kind": "ridge", "hypothesis": "Test a concrete causal feature hypothesis."},
        "config": {"kind": "ridge", "target_mode": "raw", "sample_stride": 10, "ridge_penalty": 0.001},
        "result": result,
    }


def test_connectome_followup_reselects_from_tree_but_generic_run_keeps_chain():
    node = object()
    assert next_parent_for_followup(True, node) is None
    assert next_parent_for_followup(False, node) is node


def test_planner_history_exposes_search_metrics_and_failure_semantics_only():
    records = [_record(result_extra={"holdout": {"combined_WP": 0.99}, "leaderboard": 0.98})]
    rendered = format_search_history(records)
    assert "WP_t0=0.650000" in rendered
    assert "WP_t1=0.670000" in rendered
    assert "combined_WP=0.660000" in rendered
    assert "0.99" not in rendered
    assert "0.98" not in rendered
    assert "Implementation/validation failure is evidence about execution" in rendered
    assert "identical implementation/specification" in rendered
    assert "Matching ids mean the same bounded settings or generated train/callback code" in rendered


def test_search_history_never_serializes_unknown_diagnostic_or_promotion_fields():
    record = _record(result_extra={
        "search_error_diagnostics": {"evidence": "reused_search_evidence_not_independent_validation",
                                     "holdout_wp": "PROTECTED_SENTINEL"},
        "promotion": {"combined_WP": "PROTECTED_SENTINEL"},
    })
    rendered = format_search_history([record])
    assert "PROTECTED_SENTINEL" not in rendered
    assert "holdout_wp" not in rendered
    assert '"promotion":' not in rendered


def test_history_suppresses_metrics_from_unknown_or_promotion_domain():
    record = _record(result_extra={"metric_domain": "complete validation",
                                   "metrics": {"WP_t0": .98, "WP_t1": .97, "combined_WP": .975}})
    rendered = format_search_history([record])
    assert "0.980000" not in rendered and "0.975000" not in rendered
    assert "no score" in rendered


def test_history_prompt_uses_provider_only_in_connectome_mode():
    agent = SimpleNamespace(
        cfg=SimpleNamespace(connectome_mode=True),
        connectome_search_history_provider=lambda: "Attempt 1 search WP combined_WP=0.66",
    )
    assert "combined_WP=0.66" in planner_history_section(agent)
    agent.cfg.connectome_mode = False
    assert planner_history_section(agent) == ""


def test_bounded_signature_ignores_hypothesis_and_fills_semantic_defaults():
    first = "EXPERIMENT = {'kind': 'ridge', 'hypothesis': 'First testable idea'}"
    second = "EXPERIMENT = {'hypothesis': 'A differently worded idea', 'kind': 'ridge', 'sample_stride': 10, 'ridge_penalty': 1e-3}"
    assert proposal_signature(first) == proposal_signature(second)


def test_generated_signature_ignores_formatting_and_hypothesis():
    source_a = """CANDIDATE = {'hypothesis': 'Try a new causal learner', 'train_tier': 256, 'train_source': 'def train(train_files, output_dir):\\n    return {}', 'callback_source': 'class PredictionModel:\\n    pass'}"""
    source_b = """CANDIDATE = {'hypothesis': 'A new phrasing for the same implementation', 'train_tier': 256, 'train_source': 'def train(train_files, output_dir):\\n    return {}', 'callback_source': 'class PredictionModel:\\n    pass'}"""
    assert proposal_signature(source_a) == proposal_signature(source_b)


def test_history_loader_reads_only_mlevolve_search_history(tmp_path):
    run = tmp_path / "run-a" / "experiments"
    run.mkdir(parents=True)
    (run / "history.jsonl").write_text('{"node_id":"1","parent_id":"root","status":"success","spec":{},"result":{}}\n')
    protected = tmp_path / "run-a" / "promotion.json"
    protected.write_text('{"holdout_wp":0.99}')
    assert len(load_prior_search_records(tmp_path)) == 1
