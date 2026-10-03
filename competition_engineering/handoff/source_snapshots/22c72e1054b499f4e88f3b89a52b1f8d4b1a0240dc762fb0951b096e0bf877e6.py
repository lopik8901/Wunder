"""Search-only history supplied to the Connectome MLEvolve planner."""

from __future__ import annotations

import ast
import hashlib
import json
import re
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
    from connectome.research_foundation import FORBIDDEN_TERMS
    if FORBIDDEN_TERMS.search(hypothesis) or re.search(r"https?://|@|\.\.[/\\]", hypothesis):
        hypothesis = "(unsafe hypothesis text omitted)"
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


def compact_search_history(records: list[dict[str, Any]], *, max_utf8_bytes: int = 8000) -> str:
    """Bounded search context: latest atlas plus short, deterministic attempt ledger."""
    if not records:
        return "No prior scored search attempts are available in this run."
    from competition_engineering.search_error_diagnostics import format_search_error_diagnostics
    header = ["Reused fixed-search evidence, not independent validation.",
              "Implementation failure does not disprove an ML hypothesis.",
              f"Total prior attempts: {len(records)}; latest 8 outcomes follow."]
    latest_atlas = next((r["result"]["search_error_diagnostics"] for r in reversed(records)
                         if isinstance(r.get("result"), dict) and
                         r["result"].get("metric_domain") == SEARCH_METRIC_DOMAIN and
                         r["result"].get("search_error_diagnostics") is not None), None)
    if latest_atlas is not None:
        header.append("Latest candidate error atlas: " + format_search_error_diagnostics(latest_atlas))
    briefs = []
    for index, record in enumerate(records[-8:], start=len(records) - min(8, len(records)) + 1):
        safe = dict(record)
        result = dict(record.get("result") or {})
        result.pop("search_error_diagnostics", None)
        safe["result"] = result
        proposal, outcome, signature = _record_summary(safe)
        briefs.append(f"Attempt {index} [parent {str(record.get('parent_id', 'unknown'))[:8]}, id {signature[:10] if signature else 'unknown'}]: {proposal} — {outcome}.")
    while briefs and len("\n".join(header + briefs).encode("utf-8")) > max_utf8_bytes:
        briefs.pop(0)
    rendered = "\n".join(header + briefs)
    if len(rendered.encode("utf-8")) > max_utf8_bytes:
        raise ValueError("search atlas exceeds the fixed history budget")
    return rendered


def planner_history_section(agent: Any) -> str:
    """Return planner-visible history only when running bounded Connectome mode."""
    if not getattr(getattr(agent, "cfg", None), "connectome_mode", False):
        return ""
    lessons = persistent_search_lessons()
    provider = getattr(agent, "connectome_search_history_provider", None)
    if not callable(provider):
        return lessons + "\n# Search-only experiment history\nNo previous attempt records are available.\n"
    return lessons + "\n# Search-only experiment history\n" + provider() + "\n"


def persistent_search_lessons(knowledge_path: Path | None = None) -> str:
    """Expose bounded manual-research lessons without reading mixed ledgers.

    This adds context, not a synthetic scored tree node. The historical runner
    root retains its actual model and metric identity.
    """
    import math
    from connectome.research_foundation import FORBIDDEN_TERMS
    path = knowledge_path or Path(__file__).resolve().parents[1] / "competition_engineering/search_knowledge.json"
    try:
        knowledge = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(knowledge, dict) or knowledge.get("schema_version") != 1:
        return ""
    incumbent = knowledge.get("incumbent")
    if not isinstance(incumbent, dict):
        return ""
    wp = incumbent.get("wp")
    sha = incumbent.get("sha256")
    if type(wp) not in (float, int) or not math.isfinite(wp) or not -1 <= wp <= 1 or not re.fullmatch(r"[a-f0-9]{64}", str(sha)):
        return ""
    lines = ["\n# Persistent search-only research lessons",
             f"Practical frozen search benchmark WP={wp:.9f}, artifact SHA256={sha}. This context does not rescore or replace the historical tree root.",
             "Historical adaptive point maxima are exploratory evidence. Use official WP for model/hyperparameter selection; lower residual MSE need not improve WP."]
    for key, field, prefix in (("established", "lesson", "Established"), ("weakened", "hypothesis", "Weakened")):
        items = knowledge.get(key)
        if not isinstance(items, list):
            continue
        for item in items[:6]:
            text = item.get(field) if isinstance(item, dict) else None
            if isinstance(text, str) and 0 < len(text) <= 600 and not FORBIDDEN_TERMS.search(text) and not re.search(r"https?://|@|\.\.[/\\]", text):
                lines.append(prefix + ": " + " ".join(text.split()))
    branches = knowledge.get("closed_branches")
    if isinstance(branches, list):
        safe = [" ".join(s.split()) for s in branches[:6] if isinstance(s, str) and len(s) <= 200 and not FORBIDDEN_TERMS.search(s)]
        if safe:
            lines.append("Branches requiring new measured evidence before reopening: " + "; ".join(safe))
    return "\n".join(lines) + "\n"


def research_plan_guideline() -> dict[str, list[str]]:
    """Ask for a testable explanation without choosing the ML approach."""
    return {"Research plan": [
        "Before the code, state a concrete observation from the fixed search-split error atlas: name the target, position/magnitude bucket or lag, its sample count, and what the candidate fixed versus what remains unexplained. These are reused search observations, not independent validation.",
        "Explain the causal hypothesis suggested by that observation and what result of this experiment would distinguish the hypothesis from its alternative.",
        "If another search-only finding is more informative, name that finding and explain why it takes priority. If no candidate atlas is available yet, use the frozen-root search diagnostics and say so. Do not invent an atlas result or a sample count.",
        "Choose the model and execution mechanism that best test the hypothesis. Refining a known method and branching to a different method are equally valid.",
        "Use official clipped weighted Pearson for model and hyperparameter selection whenever feasible. Residual MSE is a training/diagnostic tool, not a reliable ranking substitute. Diagnose uncertainty before interpreting point-gradient conflict.",
    ]}


def known_signatures(records: list[dict[str, Any]]) -> set[str]:
    signatures = set()
    for record in records:
        _, _, signature = _record_summary(record)
        if signature:
            signatures.add(signature)
    return signatures
