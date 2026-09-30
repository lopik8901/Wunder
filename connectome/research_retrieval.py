"""Deterministic, offline research retrieval; deliberately not wired to MLEvolve.

The library is supervisor-owned. Candidate workspaces must never mount it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from competition_engineering.search_error_diagnostics import sanitize_search_error_diagnostics
from connectome.research_foundation import (
    EVIDENCE, OUTCOME_FIELDS, QUERY_FIELDS, card_sha256, validate_card,
)


OBSERVATION_FIELDS = frozenset({"target", "position", "signal", "lag_rows", "feature_group"})
SIGNALS = frozenset({"residual_persistence", "residual_bias", "residual_scale",
                     "position_error", "magnitude_error", "feature_residual_relation",
                     "sequence_variation", "cpu_inference_cost"})
TARGETS = frozenset({"t0", "t1", "both"})
POSITIONS = frozenset({"early", "middle", "late", "all"})
TOKEN_RE = re.compile(r"[a-z0-9]+")
SYNONYMS = {
    "lag": "temporal", "lagged": "temporal", "long": "long", "memory": "memory",
    "persistent": "persistence", "persistence": "persistence", "scale": "calibration",
    "bias": "calibration", "magnitude": "nonlinear", "sequence": "sequence",
    "causal": "causal", "online": "incremental", "runtime": "inference",
}
INDEX_FIELDS = ("mechanism", "useful_when", "diagnostic_signatures", "limitations",
                "causality", "incremental_inference", "cpu_deployment", "concise_relevance")


def validate_observation(value: Any) -> dict[str, Any]:
    """Constrained, pre-retrieval observation; no hypothesis or raw labels."""
    if not isinstance(value, dict) or set(value) != OBSERVATION_FIELDS:
        raise ValueError("observation schema mismatch")
    if value["target"] not in TARGETS or value["position"] not in POSITIONS or value["signal"] not in SIGNALS:
        raise ValueError("unsupported observation category")
    lag = value["lag_rows"]
    if lag is not None and (type(lag) is not int or lag not in {1, 10, 100}):
        raise ValueError("lag must be a fixed atlas lag")
    group = value["feature_group"]
    if group is not None:
        from competition_engineering.search_error_diagnostics import FEATURE_GROUPS
        if group not in FEATURE_GROUPS:
            raise ValueError("feature group must be a fixed atlas group")
    if value["signal"] == "residual_persistence" and lag is None:
        raise ValueError("persistence observation requires a lag")
    if value["signal"] != "feature_residual_relation" and group is not None:
        raise ValueError("feature group only applies to feature residual relations")
    return dict(value)


def _tokens(text: str) -> list[str]:
    return [SYNONYMS.get(t, t) for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]


def _observation_tokens(observation: dict[str, Any]) -> list[str]:
    item = validate_observation(observation)
    terms = [item["signal"].replace("_", " "), item["position"], item["target"]]
    if item["lag_rows"] is not None:
        terms += [f"lag {item['lag_rows']}", "temporal memory"]
    if item["feature_group"] is not None:
        terms.append(item["feature_group"].replace("_", " "))
    return _tokens(" ".join(terms))


def validate_retrieval_query(query: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Require an existing fixed-search atlas and a populated observed bucket."""
    item = validate_observation(observation)
    if not isinstance(query, dict) or set(query) != QUERY_FIELDS or query.get("schema_version") != 1 or query.get("evidence") != EVIDENCE:
        raise ValueError("retrieval requires an allowlisted search query")
    atlas = sanitize_search_error_diagnostics(query.get("atlas"))
    if atlas is None or not isinstance(query.get("search_outcomes"), list):
        raise ValueError("invalid search atlas or history")
    for row in query["search_outcomes"]:
        if not isinstance(row, dict) or set(row) != OUTCOME_FIELDS or row["status"] not in {"success", "error", "timeout", "unknown"} or row["mechanism"] not in {"ridge", "calibration", "gpu_residual", "generated", "unknown"}:
            raise ValueError("invalid search-only outcome")
        if any(type(row[key]) is not float or not math.isfinite(row[key]) for key in ("WP_t0", "WP_t1", "combined_WP")):
            raise ValueError("invalid search-only score")
    targets = ("t0", "t1") if item["target"] == "both" else (item["target"],)
    if item["position"] != "all" and not all(atlas["position_buckets"][item["position"]][t]["n"] > 0 for t in targets):
        raise ValueError("observation names an empty position bucket")
    if item["signal"] == "residual_persistence" and not all(any(
        row["lag_rows"] == item["lag_rows"] and row["n_pairs"] > 0
        for row in atlas["residual_persistence"][t]) for t in targets):
        raise ValueError("observation names an empty persistence lag")
    if item["signal"] == "feature_residual_relation" and not all(any(
        row["feature_group"] == item["feature_group"] and row["n_rows"] > 0
        for row in atlas["feature_residual_relationships"][t]) for t in targets):
        raise ValueError("observation names an empty feature group")
    return {"atlas": atlas, "search_outcomes": query["search_outcomes"]}


def manifest_sha256(cards: list[dict[str, Any]]) -> str:
    """Content hash independent of filesystem enumeration order."""
    identities = sorted((card["card_id"], card["version"], card_sha256(card)) for card in cards)
    return hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()


def load_library(directory: Path) -> tuple[list[dict[str, Any]], str]:
    """Read only a dedicated card directory; no recursive scan or network."""
    if not directory.is_dir():
        raise ValueError("research library directory missing")
    cards = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("research cards must be regular files")
        card = validate_card(json.loads(path.read_text(encoding="utf-8")))
        if path.stem != card["card_id"]:
            raise ValueError("card filename must match card ID")
        cards.append(card)
    if len({card["card_id"] for card in cards}) != len(cards):
        raise ValueError("duplicate card ID")
    if len({card["source_url"] for card in cards}) != len(cards):
        raise ValueError("duplicate source URL; combine related claims into one card")
    titles = [set(_tokens(card["title"])) for card in cards]
    for i, left in enumerate(titles):
        for right in titles[i + 1:]:
            if left and right and len(left & right) / len(left | right) >= .85:
                raise ValueError("near-duplicate research title; review source overlap")
    return cards, manifest_sha256(cards)


def retrieve(cards: list[dict[str, Any]], observation: dict[str, Any], search_query: dict[str, Any], *,
             max_cards: int = 5, max_utf8_bytes: int = 2800) -> dict[str, Any]:
    """BM25 plus deterministic family diversity; no family is privileged.

    UTF-8 is a strict byte cap, not an asserted model-token count. The caller
    must additionally enforce an actual tokenizer budget before prompt use.
    """
    if type(max_cards) is not int or not 1 <= max_cards <= 5 or type(max_utf8_bytes) is not int or max_utf8_bytes < 100:
        raise ValueError("invalid retrieval budget")
    validated_query = validate_retrieval_query(search_query, observation)
    query = _observation_tokens(observation)
    query_sha256 = hashlib.sha256(json.dumps(
        {"observation": observation, "atlas": validated_query["atlas"],
         "search_outcomes": validated_query["search_outcomes"]},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    validated = [validate_card(card) for card in cards]
    if len({c["card_id"] for c in validated}) != len(validated):
        raise ValueError("duplicate card ID")
    if not validated:
        return {"status": "empty_library", "evidence": EVIDENCE,
                "manifest_sha256": manifest_sha256([]), "query_sha256": query_sha256,
                "cards": [], "rendered": ""}
    documents = [_tokens(" ".join(str(c[key]) for key in INDEX_FIELDS) + " " + " ".join(c["tags"]))
                 for c in validated]
    mean_length = sum(len(doc) for doc in documents) / len(documents)
    df = Counter(token for doc in documents for token in set(doc))
    scores = []
    for card, doc in zip(validated, documents):
        tf = Counter(doc)
        score = 0.0
        for term in set(query):
            if not tf[term]:
                continue
            idf = math.log1p((len(documents) - df[term] + .5) / (df[term] + .5))
            score += idf * tf[term] * 2.2 / (tf[term] + 1.2 * (.25 + .75 * len(doc) / mean_length))
        if score > 0:
            scores.append((score, card))
    scores.sort(key=lambda pair: (-pair[0], pair[1]["card_id"]))
    selected: list[dict[str, Any]] = []
    used_families: Counter[str] = Counter()
    rendered_parts: list[str] = []
    for score, card in scores:
        if len(selected) >= max_cards:
            break
        # Two cards per family is a soft exposure limit, not a quota.
        if used_families[card["family"]] >= 2:
            continue
        excerpt = (f"[{card['card_id']} v{card['version']}] {card['title']} "
                   f"({card['source_url']}, {card['source_locator']}). "
                   f"Mechanism: {card['mechanism']} Useful when: {card['useful_when']} "
                   f"Limits: {card['limitations']} CPU/causality: {card['cpu_deployment']}; {card['causality']}.")
        if len(("\n".join([*rendered_parts, excerpt])).encode("utf-8")) > max_utf8_bytes:
            continue
        rendered_parts.append(excerpt)
        used_families[card["family"]] += 1
        selected.append({"card_id": card["card_id"], "version": card["version"],
                         "sha256": card_sha256(card), "score": round(score, 8),
                         "family": card["family"]})
    rendered = "\n".join(rendered_parts)
    return {"status": "ok" if selected else "no_result", "evidence": EVIDENCE,
            "manifest_sha256": manifest_sha256(validated), "query_sha256": query_sha256,
            "cards": selected, "rendered": rendered}
