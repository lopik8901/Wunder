"""Thin bridge from Connectome evaluation output to vanilla MLEvolve execution."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Iterable
from typing import Any, TextIO


RESULT_PREFIX = "CONNECTOME_RESULT_JSON="
FEEDBACK_PREFIX = "MLEVOLVE_FEEDBACK_JSON="


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
    return {
        "status": "success",
        "is_bug": False,
        "summary": (
            "Official Connectome validation: "
            f"WP_t0={metrics['WP_t0']}, WP_t1={metrics['WP_t1']}, "
            f"combined_WP={metrics['combined_WP']}."
        ),
        "metric": result["primary_metric"],
        "lower_is_better": False,
        "wp_t0": metrics["WP_t0"],
        "wp_t1": metrics["WP_t1"],
        "combined_wp": metrics["combined_WP"],
    }


def adapt_execution_result(execution_result: Any) -> Any:
    """Append deterministic MLEvolve feedback without replacing ExecutionResult."""
    try:
        feedback = to_mlevolve_feedback(parse_result_output(execution_result.term_out))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        failure = execution_result.exc_type or type(error).__name__
        detail = execution_result.exc_info or str(error)
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
