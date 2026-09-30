import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))

from competition_engineering.search_error_diagnostics import (  # noqa: E402
    EVIDENCE_LABEL,
    FEATURE_GROUPS,
    FEATURE_LAGS,
    MAGNITUDE_BUCKETS,
    POSITION_BUCKETS,
    PERSISTENCE_LAGS,
    SearchErrorDiagnostics,
    format_search_error_diagnostics,
    sanitize_search_error_diagnostics,
)
from connectome.mlevolve_adapter import (  # noqa: E402
    RESULT_PREFIX,
    adapt_execution_result,
    parse_result_output,
)
from connectome.mlevolve_bounded import model_visible_report  # noqa: E402


def _synthetic_sequence(sequence_id, n=240):
    step = np.arange(n)
    x = np.zeros((n, 112), dtype=np.float32)
    x[:, 0] = np.sin(step / 9)
    x[:, 52] = np.cos(step / 13)
    y = np.column_stack((0.8 * x[:, 0] + 0.03 * np.sin(step),
                         0.7 * x[:, 52] + 0.04 * np.cos(step / 3))).astype(np.float32)
    root = y + np.column_stack((0.18 * np.sin(step / 4), 0.16 * np.cos(step / 5))).astype(np.float32)
    mask = np.ones(n, dtype=bool)
    mask[:5] = False
    return {"seq": sequence_id, "step": step, "x": x, "y": y, "p": np.zeros_like(y),
            "root": root, "need": np.ones(n, dtype=bool), "mask": mask}


def test_fixed_search_error_atlas_has_all_predetermined_summaries_and_counts():
    diagnostics = SearchErrorDiagnostics(root_predictor=lambda z: z["root"])
    for seq in range(3):
        z = _synthetic_sequence(seq)
        candidate = z["root"] + 0.5 * (z["y"] - z["root"])
        diagnostics.add(z, candidate)
    result = diagnostics.result()

    assert result["evidence"] == EVIDENCE_LABEL
    assert result["counts"] == {"sequences": 3, "scored_rows": 3 * 235}
    assert set(result["target_performance"]) == {"t0", "t1"}
    assert set(result["position_buckets"]) == {name for name, _, _ in POSITION_BUCKETS}
    assert len(result["target_magnitude_buckets"]["t0"]) == len(MAGNITUDE_BUCKETS)
    assert len(result["root_prediction_magnitude_buckets"]["t1"]) == len(MAGNITUDE_BUCKETS)
    assert result["per_sequence_wp"]["t0"]["n_sequences"] == 3
    assert result["residual_bias_scale"]["t0"]["candidate_weighted_scale"] < result["residual_bias_scale"]["t0"]["root_weighted_scale"]
    assert result["residual_persistence"]["t0"][0]["lag_rows"] == PERSISTENCE_LAGS[0]
    assert result["residual_persistence"]["t0"][-1]["n_pairs"] == 3 * (240 - 100 - 5)
    assert result["feature_residual_relationships"]["lags"] == list(FEATURE_LAGS)
    assert len(result["feature_residual_relationships"]["t0"]) == len(FEATURE_LAGS) * len(FEATURE_GROUPS)
    assert result["feature_residual_relationships"]["t0"][-1]["n_rows"] == 3 * (240 - 100)
    assert sanitize_search_error_diagnostics(result) == result
    rendered = format_search_error_diagnostics(result)
    assert "not independent validation" in rendered
    assert "what candidate fixed" in rendered.lower() or "root→candidate" in rendered


def test_allowlist_rejects_protected_or_unknown_fields_at_every_level():
    diagnostics = SearchErrorDiagnostics(root_predictor=lambda z: z["root"])
    z = _synthetic_sequence(0)
    diagnostics.add(z, z["root"])
    clean = diagnostics.result()
    for mutate in (
        lambda d: d.update(holdout_wp=0.99),
        lambda d: d["target_performance"]["t0"].update(leaderboard_score=0.99),
        lambda d: d["feature_residual_relationships"]["t1"][0].update(promotion_metric=0.99),
    ):
        poisoned = copy.deepcopy(clean)
        mutate(poisoned)
        assert sanitize_search_error_diagnostics(poisoned) is None


def test_model_visible_report_withholds_score_when_diagnostics_are_missing_or_poisoned():
    report = {"status": "success", "metric_name": "combined_WP", "maximize": True,
              "primary_metric": .66, "metrics": {"WP_t0": .65, "WP_t1": .67, "combined_WP": .66},
              "metric_domain": "fixed development tuning sequences",
              "search_error_diagnostics": {"evidence": "holdout", "leaderboard_score": 0.99}}
    visible = model_visible_report(report, {"runtime_seconds": 1,
        "delta_vs_incumbent_tune": .005, "delta_vs_parent_tune": .005})
    assert visible["status"] == "error"
    assert "0.99" not in str(visible)


def test_failure_feedback_uses_allowlisted_stage_and_generic_reason():
    visible = model_visible_report({"status": "error", "stage": "training",
        "error": "holdout_wp=0.99 leaderboard=0.98 secret/path"}, {"runtime_seconds": 1})
    assert visible["stage"] == "training or search evaluation"
    assert "holdout" not in str(visible) and "leaderboard" not in str(visible)
    assert "secret/path" not in str(visible)


def test_diagnostics_flow_once_through_allowlisted_mlevolve_feedback():
    diagnostics = SearchErrorDiagnostics(root_predictor=lambda z: z["root"])
    z = _synthetic_sequence(0)
    diagnostics.add(z, 0.5 * (z["root"] + z["y"]))
    result = {"status": "success", "metric_name": "combined_WP", "maximize": True,
              "primary_metric": .5, "metrics": {"WP_t0": .4, "WP_t1": .6, "combined_WP": .5},
              "metric_domain": "fixed development tuning sequences",
              "search_error_diagnostics": diagnostics.result(), "holdout_wp": 0.99}
    marker = RESULT_PREFIX + __import__("json").dumps(result)
    execution = type("Exec", (), {"term_out": [marker], "exc_type": None, "exc_info": None})()
    adapt_execution_result(execution)
    parsed_result = parse_result_output(execution.term_out)
    assert sanitize_search_error_diagnostics(parsed_result["search_error_diagnostics"]) == diagnostics.result()
    assert "holdout_wp" not in parsed_result
    feedback_line = next(line for line in execution.term_out[-1].splitlines()
                         if line.startswith("MLEVOLVE_FEEDBACK_JSON="))
    feedback = __import__("json").loads(feedback_line.split("=", 1)[1])
    assert sanitize_search_error_diagnostics(feedback["search_error_diagnostics"]) == diagnostics.result()
    assert "holdout_wp" not in feedback_line
    assert "reused search evidence, not independent validation" in feedback["summary"]
