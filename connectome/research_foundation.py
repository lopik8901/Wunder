"""Search-only contracts for the supervisor's offline research retriever."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from competition_engineering.search_error_diagnostics import sanitize_search_error_diagnostics
from connectome.mlevolve_history import SEARCH_METRIC_DOMAIN


SCHEMA_VERSION = 1
EVIDENCE = "reused_search_evidence_not_independent_validation"
CARD_FIELDS = frozenset({
    "schema_version", "card_id", "version", "card_type", "family", "tags",
    "title", "authors", "year", "source_url", "source_locator", "curator",
    "verified_on", "claim_confidence", "mechanism", "useful_when",
    "diagnostic_signatures", "assumptions", "limitations", "causality",
    "incremental_inference", "training_compute", "cpu_deployment",
    "dependencies", "licensing", "concise_relevance",
})
QUERY_FIELDS = frozenset({"schema_version", "evidence", "atlas", "search_outcomes"})
OUTCOME_FIELDS = frozenset({"status", "mechanism", "WP_t0", "WP_t1", "combined_WP"})
FORBIDDEN_TERMS = re.compile(
    r"holdout|promotion|complete.validation|broad.gate|submission|leaderboard|"
    r"file://|(?:[a-z]:\\)|(?:/home/)|(?:/mnt/)|"
    r"ignore (?:all )?(?:previous|prior) instructions|system prompt|"
    r"(?:^|\n)\s*(?:assistant|developer|system)\s*:", re.IGNORECASE
)


def _nonempty(value: Any, *, max_chars: int = 600) -> bool:
    return isinstance(value, str) and 0 < len(value.strip()) <= max_chars and not FORBIDDEN_TERMS.search(value)


def validate_card(card: Any) -> dict[str, Any]:
    """Fail closed on provenance, schema drift, instructions and oversized text."""
    if not isinstance(card, dict) or set(card) != CARD_FIELDS or card.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("research card schema mismatch")
    if card["card_type"] not in {"paper", "concept"}:
        raise ValueError("invalid card type")
    if card["claim_confidence"] not in {"high", "moderate", "limited"}:
        raise ValueError("invalid claim confidence")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", str(card["card_id"])):
        raise ValueError("invalid card ID")
    if type(card["version"]) is not int or card["version"] < 1:
        raise ValueError("invalid card version")
    if type(card["year"]) is not int or not 1900 <= card["year"] <= 2100:
        raise ValueError("invalid publication year")
    for key in CARD_FIELDS - {"schema_version", "version", "year", "tags", "authors"}:
        if not _nonempty(card[key]):
            raise ValueError(f"invalid card field: {key}")
    for key in ("tags", "authors"):
        if not isinstance(card[key], list) or not card[key] or len(card[key]) > 12 or not all(
            _nonempty(item, max_chars=120) for item in card[key]
        ):
            raise ValueError(f"invalid card field: {key}")
    if not re.fullmatch(r"https://[^\s]+", card["source_url"]):
        raise ValueError("source must have an HTTPS provenance URL")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", card["verified_on"]):
        raise ValueError("verification date must be ISO formatted")
    if card["card_type"] == "paper" and not card["source_locator"].strip():
        raise ValueError("paper claim needs a precise source locator")
    encoded = json.dumps(card, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if len(encoded) > 5000:
        raise ValueError("card too large")
    return json.loads(encoded)


def card_sha256(card: dict[str, Any]) -> str:
    valid = validate_card(card)
    return hashlib.sha256(json.dumps(valid, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def build_search_query(atlas: Any, records: list[dict[str, Any]], *, last: int = 6) -> dict[str, Any]:
    """Copy only exact atlas fields and fixed search-domain score/status fields.

    Free-text hypotheses, code, paths, logs and arbitrary result dictionaries are
    deliberately absent. The planner may pass at most twelve validated,
    normalized hypothesis terms separately to the offline retriever.
    """
    if type(last) is not int or not 1 <= last <= 18:
        raise ValueError("search history window must be between 1 and 18")
    clean_atlas = sanitize_search_error_diagnostics(atlas)
    if clean_atlas is None:
        raise ValueError("query requires a valid fixed search-only atlas")
    outcomes = []
    for record in records[-last:]:
        result = record.get("result") if isinstance(record, dict) else None
        if not isinstance(result, dict) or result.get("metric_domain") != SEARCH_METRIC_DOMAIN:
            continue
        metrics = result.get("metrics")
        if not isinstance(metrics, dict) or any(
            not isinstance(metrics.get(k), (int, float)) or isinstance(metrics.get(k), bool) or
            not math.isfinite(metrics[k]) for k in ("WP_t0", "WP_t1", "combined_WP")
        ):
            continue
        spec = record.get("spec")
        mechanism = spec.get("kind", "generated") if isinstance(spec, dict) else "unknown"
        if mechanism not in {"ridge", "calibration", "gpu_residual", "generated"}:
            mechanism = "unknown"
        status = record.get("status")
        if status not in {"success", "error", "timeout"}:
            status = "unknown"
        outcomes.append({"status": status, "mechanism": mechanism,
                         **{key: float(metrics[key]) for key in ("WP_t0", "WP_t1", "combined_WP")}})
    query = {"schema_version": SCHEMA_VERSION, "evidence": EVIDENCE,
             "atlas": clean_atlas, "search_outcomes": outcomes}
    assert set(query) == QUERY_FIELDS and all(set(row) == OUTCOME_FIELDS for row in outcomes)
    return query
