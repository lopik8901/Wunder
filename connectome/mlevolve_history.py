"""Search-only history supplied to the Connectome MLEvolve planner."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = {"WP_t0", "WP_t1", "combined_WP"}
SEARCH_METRIC_DOMAIN = "fixed development tuning sequences"


def next_parent_for_followup(connectome_mode: bool, completed_node: Any) -> Any:
    """Connectome must let the search policy reselect from the full scored tree."""
    return None if connectome_mode else completed_node


def _canonical_source(source: str) -> str:
    """Remove formatting-only changes while retaining program structure."""
    return ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)


def _payload_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def proposal_signature(code: str) -> str | None:
    """Stable identity for an experiment, excluding its prose hypothesis."""
    try:
        from connectome.mlevolve_bounded import is_generated_candidate, parse_spec
        if is_generated_candidate(code):
            from connectome.mlevolve_generated import parse_candidate
            candidate = parse_candidate(code)
            payload = {
                "mechanism": "generated",
                "train_tier": candidate["train_tier"],
                "training_device": candidate.get("training_device", "cpu"),
                "train": _canonical_source(candidate["train_source"]),
                "callback": _canonical_source(candidate["callback_source"]),
            }
        else:
            spec = parse_spec(code)
            from connectome.mlevolve_bounded import validated_config
            config = validated_config(spec, "0" * 32, "signature")
            keys = ("kind", "target_mode", "base_scale", "base_bias", "sample_stride",
                    "ridge_penalty", "epochs", "amp", "base_model")
            payload = {"mechanism": "experiment", **{key: config[key] for key in keys if key in config}}
        return _payload_signature(payload)
    except Exception:
        return None


def _read_search_records(history_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = history_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return records
    for line in lines:
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            continue
        # Only consume bounded-runner search history with the expected schema.
        # Promotion reports live elsewhere and are never scanned here.
        if not isinstance(record, dict) or not {"node_id", "status", "spec", "result"}.issubset(record):
            continue
        records.append(record)
    return records


def load_prior_search_records(runs_dir: Path) -> list[dict[str, Any]]:
    """Read only experiment history.jsonl files from previous MLEvolve runs."""
    records = []
    try:
        files = sorted(runs_dir.glob("*/experiments/history.jsonl"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return records
    for history_file in files:
        records.extend(_read_search_records(history_file))
    return records


def _record_summary(record: dict[str, Any]) -> tuple[str, str, str | None]:
    spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    status = str(record.get("status", "unknown"))
    mechanism = str(spec.get("kind", "generated" if "train_source" in spec else "unknown"))
    hypothesis = " ".join(str(spec.get("hypothesis", "(no hypothesis recorded)")).split())[:220]
    if "train_source" in spec:
        settings = f"tier={spec.get('train_tier')}, device={spec.get('training_device', 'cpu')}"
    else:
        config = record.get("config") if isinstance(record.get("config"), dict) else {}
        settings_keys = ("target_mode", "sample_stride", "ridge_penalty", "epochs", "amp", "base_model", "base_scale", "base_bias")
        settings = ", ".join(f"{key}={config[key]}" for key in settings_keys if key in config) or ", ".join(
            f"{key}={spec[key]}" for key in settings_keys if key in spec
        )
    signature = None
    if "train_source" in spec and "callback_source" in spec:
        payload = {
            "mechanism": "generated",
            "train_tier": spec.get("train_tier"),
            "training_device": spec.get("training_device", "cpu"),
        }
        try:
            payload["train"] = _canonical_source(spec["train_source"])
            payload["callback"] = _canonical_source(spec["callback_source"])
            signature = _payload_signature(payload)
        except (SyntaxError, TypeError):
            pass
    else:
        # Existing bounded records retain the canonical runner config. Strip
        # paths and run-specific output fields so semantic configuration repeats
        # remain identifiable across runs.
        config = record.get("config") if isinstance(record.get("config"), dict) else {}
        keys = ("kind", "target_mode", "base_scale", "base_bias", "sample_stride",
                "ridge_penalty", "epochs", "amp", "base_model")
        payload = {"mechanism": "experiment", **{key: config[key] for key in keys if key in config}}
        if payload:
            signature = _payload_signature(payload)

    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    # Legacy records may contain auxiliary promotion diagnostics. Only show
    # scores with the exact routine-search domain tag; unknown domains are not
    # safe to infer from the file location alone.
    visible_metrics = ({key: metrics[key] for key in METRIC_KEYS
                       if isinstance(metrics.get(key), (int, float))}
                       if result.get("metric_domain") == SEARCH_METRIC_DOMAIN else {})
    runtime = record.get("runtime_seconds")
    details = []
    if visible_metrics:
        details.append("search WP " + ", ".join(f"{k}={visible_metrics[k]:.6f}" for k in ("WP_t0", "WP_t1", "combined_WP") if k in visible_metrics))
        incumbent_delta = record.get("delta_vs_incumbent_tune")
        parent_delta = record.get("delta_vs_parent_tune")
        if isinstance(incumbent_delta, (int, float)):
            details.append(f"vs search root={incumbent_delta:+.6f}")
        if isinstance(parent_delta, (int, float)):
            details.append(f"vs parent={parent_delta:+.6f}")
    else:
        # Do not surface raw exception text or paths to the planner.
        stage = result.get("stage")
        category = "timeout" if "timeout" in str(stage).lower() or status == "timeout" else "implementation/validation failure"
        details.append(f"no score ({category})")
    if isinstance(runtime, (int, float)):
        details.append(f"{runtime:.0f}s")
    training = record.get("training_diagnostics") if isinstance(record.get("training_diagnostics"), dict) else {}
    training_facts = []
    for key in ("training_rows", "updates", "train_seconds", "rows_per_second"):
        value = training.get(key)
        if isinstance(value, (int, float)):
            training_facts.append(f"{key}={value:.0f}" if isinstance(value, float) else f"{key}={value}")
    for key in ("loss_first80", "loss_last80"):
        value = training.get(key)
        if isinstance(value, (int, float)):
            training_facts.append(f"{key}={value:.4g}")
    resources = record.get("resource_estimate") if isinstance(record.get("resource_estimate"), dict) else {}
    cpu_us = resources.get("cpu_microseconds_per_row")
    if isinstance(cpu_us, (int, float)):
        training_facts.append(f"CPU inference={cpu_us:.2f}us/row")
    if training_facts:
        details.append("training/runtime diagnostics: " + ", ".join(training_facts))
    diagnostics = result.get("search_error_diagnostics")
    if diagnostics is not None:
        from competition_engineering.search_error_diagnostics import format_search_error_diagnostics
        details.append(format_search_error_diagnostics(diagnostics))
    setting_text = f" ({settings})" if settings else ""
    return f"{mechanism}{setting_text}: {hypothesis}", "; ".join(details), signature


def format_search_history(records: list[dict[str, Any]], *, limit: int = 18) -> str:
    """Compact planner context made only from search-split attempt records."""
    if not records:
        return "No prior scored search attempts are available in this run."
    selected = records[-limit:]
    lines = [
        "These are routine fixed-search-split outcomes only. Use them to compare hypotheses across branches; they contain no promotion or holdout metrics.",
        "The short id on each attempt is a canonical implementation/specification fingerprint. Matching ids mean the same bounded settings or generated train/callback code, even if the hypothesis wording differs.",
        "Implementation/validation failure is evidence about execution, not evidence that the underlying ML hypothesis is poor.",
        "Do not repeat a candidate with an identical implementation/specification on the same deterministic search data; change a scientifically meaningful assumption, or explain a specific reproducibility question.",
    ]
    skipped = len(records) - len(selected)
    if skipped:
        lines.append(f"({skipped} earlier attempts omitted here; historical search findings remain summarized in the task description.)")
    for index, record in enumerate(selected, start=max(1, len(records) - len(selected) + 1)):
        proposal, outcome, signature = _record_summary(record)
        parent = str(record.get("parent_id", "unknown"))[:8]
        lines.append(f"Attempt {index} [parent {parent}, id {signature[:10] if signature else 'unknown'}]: {proposal} — {outcome}.")
    return "\n".join(lines)


def planner_history_section(agent: Any) -> str:
    """Return planner-visible history only when running bounded Connectome mode."""
    if not getattr(getattr(agent, "cfg", None), "connectome_mode", False):
        return ""
    provider = getattr(agent, "connectome_search_history_provider", None)
    if not callable(provider):
        return "\n# Search-only experiment history\nNo previous attempt records are available.\n"
    return "\n# Search-only experiment history\n" + provider() + "\n"


def research_plan_guideline() -> dict[str, list[str]]:
    """Ask for a testable explanation without choosing the ML approach."""
    return {"Research plan": [
        "Before the code, state a concrete observation from the fixed search-split error atlas: name the target, position/magnitude bucket or lag, its sample count, and what the candidate fixed versus what remains unexplained. These are reused search observations, not independent validation.",
        "Explain the causal hypothesis suggested by that observation and what result of this experiment would distinguish the hypothesis from its alternative.",
        "If another search-only finding is more informative, name that finding and explain why it takes priority. If no candidate atlas is available yet, use the frozen-root search diagnostics and say so. Do not invent an atlas result or a sample count.",
        "Choose the model and execution mechanism that best test the hypothesis. Refining a known method and branching to a different method are equally valid.",
    ]}


def known_signatures(records: list[dict[str, Any]]) -> set[str]:
    signatures = set()
    for record in records:
        _, _, signature = _record_summary(record)
        if signature:
            signatures.add(signature)
    return signatures
