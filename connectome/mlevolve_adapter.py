"""Thin bridge from Connectome evaluation output to vanilla MLEvolve execution."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Iterable
from typing import Any, TextIO


RESULT_PREFIX = "CONNECTOME_RESULT_JSON="
FEEDBACK_PREFIX = "MLEVOLVE_FEEDBACK_JSON="
_SAFE_METRIC_DOMAIN = "fixed development tuning sequences"
_TRAIN_DIAGNOSTIC_KEYS = {"training_rows", "train_seconds", "updates", "loss_first80", "loss_last80",
                          "loss_start", "loss_end", "rows_per_second", "sample_stride",
                          "ridge_penalty", "target_mode"}
_RESOURCE_KEYS = {"peak_gpu_allocated_bytes", "artifact_bytes", "cpu_microseconds_per_row",
                   "candidate_wall_seconds"}


def emit_result(result: dict[str, object], *, stream: TextIO = sys.stdout) -> None:
    """Print one machine-readable result line from candidate evaluation code."""
    _validate_result(result)
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":")), file=stream)


def parse_result_output(output: str | Iterable[str]) -> dict[str, Any]:
    """Read the last Connectome result marker from captured candidate output."""
    lines = output.splitlines() if isinstance(output, str) else (
        line for chunk in output for line in str(chunk).splitlines()
    )
    payload = None
    for line in lines:
        if line.startswith(RESULT_PREFIX):
            payload = line[len(RESULT_PREFIX):]
    if payload is None:
        raise ValueError(f"candidate output has no {RESULT_PREFIX} marker")
    result = json.loads(payload)
    _validate_result(result)
    return result


def to_mlevolve_feedback(result: dict[str, Any]) -> dict[str, object]:
    """Map a validated result to MLEvolve result_parse_agent's existing schema."""
    _validate_result(result)
    metrics = result["metrics"]
    domain = result.get("metric_domain")
    label = f"Connectome {domain}" if domain else "Official Connectome validation"
    delta_incumbent = result.get("delta_vs_incumbent_tune")
    delta_parent = result.get("delta_vs_parent_tune")
    deltas = ""
    if isinstance(delta_incumbent, (int, float)) and isinstance(delta_parent, (int, float)):
        deltas = f" Delta versus incumbent={delta_incumbent:+.9f}; versus parent={delta_parent:+.9f}."
    feedback = {
        "status": "success",
        "is_bug": False,
        "summary": (
            f"{label}: "
            f"WP_t0={metrics['WP_t0']}, WP_t1={metrics['WP_t1']}, "
            f"combined_WP={metrics['combined_WP']}.{deltas}"
        ),
        "metric": result["primary_metric"],
        "lower_is_better": False,
        "wp_t0": metrics["WP_t0"],
        "wp_t1": metrics["WP_t1"],
        "combined_wp": metrics["combined_WP"],
    }
    if "search_error_diagnostics" in result:
        from competition_engineering.search_error_diagnostics import sanitize_search_error_diagnostics
        diagnostics = sanitize_search_error_diagnostics(result["search_error_diagnostics"])
        if diagnostics is None:
            raise ValueError("search error diagnostics failed the fixed allowlist/schema")
        feedback["search_error_diagnostics"] = diagnostics
        feedback["summary"] += " Fixed-root error atlas attached; reused search evidence, not independent validation."
    return feedback


def _safe_success_marker(result: dict[str, Any]) -> dict[str, Any]:
    """Copy only fixed search-result fields into MLEvolve's visible journal."""
    _validate_result(result)
    if result.get("metric_domain", _SAFE_METRIC_DOMAIN) != _SAFE_METRIC_DOMAIN:
        raise ValueError("routine feedback must be from the fixed search split")
    safe = {
        "status": "success", "metric_name": "combined_WP", "maximize": True,
        "primary_metric": float(result["primary_metric"]),
        "metrics": {key: float(result["metrics"][key]) for key in ("WP_t0", "WP_t1", "combined_WP")},
        "metric_domain": _SAFE_METRIC_DOMAIN,
    }
    for key in ("runtime_seconds", "delta_vs_incumbent_tune", "delta_vs_parent_tune"):
        value = result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            safe[key] = float(value)
    for key in ("artifact_bytes", "training_sequences"):
        value = result.get(key)
        if type(value) is int and value >= 0:
            safe[key] = value
    kind = result.get("experiment_kind")
    if kind in {"ridge", "gpu_residual", "calibration", "generated"}:
        safe["experiment_kind"] = kind
    for container, allowed in (("training_diagnostics", _TRAIN_DIAGNOSTIC_KEYS),
                               ("resource_estimate", _RESOURCE_KEYS)):
        values = result.get(container)
        if isinstance(values, dict):
            clean = {}
            for key, value in values.items():
                if key not in allowed:
                    continue
                if key == "target_mode" and value in {"raw", "clipped", "t0_raw_t1_clipped", "t0_clipped_t1_raw"}:
                    clean[key] = value
                elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                    clean[key] = value
            if clean:
                safe[container] = clean
    if "search_error_diagnostics" in result:
        from competition_engineering.search_error_diagnostics import sanitize_search_error_diagnostics
        diagnostics = sanitize_search_error_diagnostics(result["search_error_diagnostics"])
        if diagnostics is None:
            raise ValueError("search error diagnostics failed the fixed allowlist/schema")
        safe["search_error_diagnostics"] = diagnostics
    return safe


def _safe_failure_marker(result: dict[str, Any]) -> dict[str, Any]:
    status = "timeout" if result.get("status") == "timeout" else "error"
    raw_stage = str(result.get("stage", "")).lower()
    if status == "timeout" or "timeout" in raw_stage:
        stage = "candidate timeout"
    elif "gpu" in raw_stage:
        stage = "GPU preflight"
    elif "valid" in raw_stage:
        stage = "candidate validation"
    elif "train" in raw_stage or "evaluation" in raw_stage:
        stage = "training or search evaluation"
    else:
        stage = "candidate specification or launcher"
    error = f"Candidate did not complete during {stage}."
    # This exact bounded-runner timeout wording contains no candidate-controlled
    # text beyond an integer duration, and is useful to MLEvolve when revising.
    raw_error = result.get("error")
    if status == "timeout" and isinstance(raw_error, str):
        import re
        if re.fullmatch(r"\d+-second training budget exhausted", raw_error):
            error = raw_error
    return {"status": status, "metric_name": "combined_WP", "maximize": True,
            "stage": stage, "error": error}


def _replace_result_markers(execution_result: Any, payload: dict[str, Any]) -> None:
    marker = RESULT_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    rewritten = []
    replaced = False
    for chunk in execution_result.term_out:
        for line in str(chunk).splitlines(keepends=True):
            if line.startswith(RESULT_PREFIX):
                ending = "\n" if line.endswith("\n") else ""
                rewritten.append(marker + ending)
                replaced = True
            else:
                rewritten.append(line)
    if replaced:
        execution_result.term_out = ["".join(rewritten)]


def adapt_execution_result(execution_result: Any) -> Any:
    """Append deterministic MLEvolve feedback without replacing ExecutionResult."""
    try:
        parsed = parse_result_output(execution_result.term_out)
        feedback = to_mlevolve_feedback(parsed)
        safe_marker = _safe_success_marker(parsed)
        _replace_result_markers(execution_result, safe_marker)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        failure = execution_result.exc_type or type(error).__name__
        detail = f"Candidate execution failed ({failure})."
        # Preserve bounded-runner failures while keeping them ineligible for a
        # score. The successful-result validation contract remains unchanged.
        lines = [line for chunk in execution_result.term_out for line in str(chunk).splitlines()]
        for line in reversed(lines):
            if line.startswith(RESULT_PREFIX):
                try:
                    failed = json.loads(line[len(RESULT_PREFIX):])
                    if failed.get("status") in {"error", "timeout"}:
                        safe_failure = _safe_failure_marker(failed)
                        detail = safe_failure["error"]
                        _replace_result_markers(execution_result, safe_failure)
                except (ValueError, TypeError, AttributeError):
                    pass
                break
        feedback = {
            "status": "failure",
            "is_bug": True,
            "summary": f"{failure}: {detail}",
            "metric": None,
            "lower_is_better": False,
            "wp_t0": None,
            "wp_t1": None,
            "combined_wp": None,
        }
    execution_result.term_out.append(
        FEEDBACK_PREFIX + json.dumps(feedback, sort_keys=True, separators=(",", ":"))
    )
    return execution_result


def make_exec_callback(interpreter: Any) -> Callable[..., Any]:
    """Wrap MLEvolve Interpreter.run while preserving its ExecutionResult object."""
    def exec_callback(*args: Any, **kwargs: Any) -> Any:
        return adapt_execution_result(interpreter.run(*args, **kwargs))

    return exec_callback


def _validate_result(result: dict[str, Any]) -> None:
    if result.get("status") != "success":
        raise ValueError("Connectome result status must be success")
    if result.get("metric_name") != "combined_WP" or result.get("maximize") is not True:
        raise ValueError("Connectome primary metric must be maximizing combined_WP")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"WP_t0", "WP_t1", "combined_WP"}:
        raise ValueError("Connectome result must contain exactly WP_t0, WP_t1, and combined_WP")
    values = [metrics["WP_t0"], metrics["WP_t1"], metrics["combined_WP"], result.get("primary_metric")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("Connectome metrics must be finite numbers")
    expected = (metrics["WP_t0"] + metrics["WP_t1"]) / 2.0
    if not math.isclose(metrics["combined_WP"], expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("combined_WP must be the mean of WP_t0 and WP_t1")
    if not math.isclose(result["primary_metric"], metrics["combined_WP"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("primary_metric must equal combined_WP")
