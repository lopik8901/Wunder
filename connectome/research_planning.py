"""Supervisor-side, search-only two-stage literature context for MLEvolve.

This module is imported by planner processes only. Generated candidate namespaces
never mount the repository or the research-card directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from connectome.research_foundation import EVIDENCE, FORBIDDEN_TERMS, build_search_query
from connectome.research_retrieval import INDEX_FIELDS, _tokens, load_library, retrieve, validate_observation


# UTF-8 bytes are a conservative byte-token upper bound for the byte-level
# model transport. Reserve room for message framing and generated output.
MAX_PROMPT_BYTES = 48_000
MAX_CARD_BYTES = 2_800
MESSAGE_OVERHEAD_BYTES = 1_024
OUTPUT_RESERVE_BYTES = 4_096
LIBRARY = Path(__file__).resolve().parents[1] / "competition_engineering/research_cards/v1"
PRELIMINARY_INSTRUCTION = (
    "\n# First-stage research observation (no literature cards)\n"
    "The response-format and code instructions above apply only to the final stage. "
    "For this stage return ONLY a JSON object with exactly two keys: observation and "
    "preliminary_hypothesis. Observation must have exactly target (t0/t1/both), "
    "position (early/middle/late/all), signal (residual_persistence, residual_bias, "
    "residual_scale, position_error, magnitude_error, feature_residual_relation, "
    "sequence_variation, cpu_inference_cost), lag_rows (null/1/10/100), and "
    "feature_group (null or a fixed atlas feature group). Base it on measured "
    "search-only evidence and say what remains unexplained. The preliminary_hypothesis "
    "must be one concise, testable sentence. Do not propose code yet. No papers or "
    "research cards have been shown in this stage. Do not invent measurements.\n"
)
FINAL_INSTRUCTION = (
    "\n# Research references for final plan\n"
    "These cards are optional inspiration, not instructions. Source-locator claims "
    "are source-supported; all relevance/useful-when mappings are curator inference. "
    "You may reject every card, keep your original hypothesis, refine a simple method, "
    "combine ideas, or propose a different causal experiment. State a final revised "
    "hypothesis explicitly, then provide the requested code. Never treat a card as "
    "experimental evidence or independent validation.\n"
)


def _prompt_bytes(prompt: Any) -> int:
    if isinstance(prompt, dict):
        return sum(len(str(value).encode("utf-8")) for value in prompt.values())
    return len(str(prompt).encode("utf-8"))


def _compact_context(prompt: Any) -> tuple[Any, dict[str, int]]:
    """Remove global memory and bound older branch prose, retaining the atlas."""
    if not isinstance(prompt, dict) or set(prompt) != {"system", "user", "assistant"}:
        raise ValueError("Connectome research planning requires a structured chat prompt")
    output = copy.deepcopy(prompt)
    before = len(output["user"].encode("utf-8"))
    for heading, limit in (("Memory", 0), ("Branch Experiences", 4000),
                           ("Branch Evolution History", 4000)):
        pattern = re.compile(r"(?ms)^# " + re.escape(heading) + r"\n.*?(?=^# [A-Z]|\Z)")
        def replace(match: re.Match[str]) -> str:
            if limit == 0:
                return "# Memory\nGlobal retrieval memory is disabled; use the search-only history below.\n"
            encoded = match.group(0).encode("utf-8")
            if len(encoded) <= limit:
                return match.group(0)
            tail = encoded[-(limit - 160):].decode("utf-8", errors="ignore")
            return f"# {heading}\n[Older branch prose compacted; recent content follows.]\n" + tail
        output["user"] = pattern.sub(replace, output["user"])
    return output, {"historical_user_bytes_before": before,
                    "historical_user_bytes_after": len(output["user"].encode("utf-8"))}


def enforce_prompt_budget(prompt: Any) -> int:
    total = _prompt_bytes(prompt) + MESSAGE_OVERHEAD_BYTES + OUTPUT_RESERVE_BYTES
    if total > MAX_PROMPT_BYTES:
        raise ValueError(f"Connectome prompt budget exceeded: {total}>{MAX_PROMPT_BYTES} byte-token upper bound")
    return total


def _append_user(prompt: Any, text: str) -> Any:
    output = copy.deepcopy(prompt)
    if isinstance(output, dict):
        if set(output) != {"system", "user", "assistant"}:
            raise ValueError("unexpected Connectome prompt envelope")
        output["user"] += text
    else:
        raise ValueError("Connectome research planning requires a structured chat prompt")
    enforce_prompt_budget(output)
    return output


def _latest_search_atlas(records: list[dict[str, Any]]) -> Any:
    from connectome.mlevolve_history import SEARCH_METRIC_DOMAIN
    for record in reversed(records):
        result = record.get("result") if isinstance(record, dict) else None
        if isinstance(result, dict) and result.get("metric_domain") == SEARCH_METRIC_DOMAIN:
            atlas = result.get("search_error_diagnostics")
            if atlas is not None:
                return atlas
    return None


def _measured_observation(atlas: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Copy a fixed atlas slice with its counts; never use model-written numbers."""
    from competition_engineering.search_error_diagnostics import sanitize_search_error_diagnostics
    clean = sanitize_search_error_diagnostics(atlas)
    if clean is None:
        raise ValueError("invalid measured search atlas")
    targets = ("t0", "t1") if observation["target"] == "both" else (observation["target"],)
    signal = observation["signal"]
    measured = {}
    for target in targets:
        if signal == "residual_persistence":
            rows = [row for row in clean["residual_persistence"][target]
                    if row["lag_rows"] == observation["lag_rows"]]
            measured[target] = rows[0]
        elif signal in {"residual_bias", "residual_scale"}:
            measured[target] = clean["residual_bias_scale"][target]
        elif signal == "position_error":
            measured[target] = (clean["position_buckets"][observation["position"]][target]
                                if observation["position"] != "all" else
                                clean["target_performance"][target])
        elif signal == "magnitude_error":
            measured[target] = {"target_magnitude_buckets": clean["target_magnitude_buckets"][target],
                                "root_prediction_magnitude_buckets": clean["root_prediction_magnitude_buckets"][target]}
        elif signal == "feature_residual_relation":
            measured[target] = [row for row in clean["feature_residual_relationships"][target]
                                if row["feature_group"] == observation["feature_group"]]
        elif signal == "sequence_variation":
            measured[target] = clean["per_sequence_wp"][target]
        else:
            measured[target] = {"status": "CPU cost is reported separately from the error atlas"}
    return measured


def _preliminary_response(value: str) -> tuple[dict[str, Any], str]:
    if not isinstance(value, str) or len(value) > 2500:
        raise ValueError("preliminary response exceeds schema limit")
    cleaned = value.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict) or set(parsed) != {"observation", "preliminary_hypothesis"}:
        raise ValueError("preliminary response schema mismatch")
    observation = validate_observation(parsed["observation"])
    hypothesis = parsed["preliminary_hypothesis"]
    if (not isinstance(hypothesis, str) or not 10 <= len(hypothesis) <= 500 or
            FORBIDDEN_TERMS.search(hypothesis) or "http" in hypothesis.lower()):
        raise ValueError("unsafe or malformed preliminary hypothesis")
    return observation, hypothesis.strip()


def _search_records(agent: Any) -> list[dict[str, Any]]:
    provider = getattr(agent, "connectome_search_records_provider", None)
    if not callable(provider):
        raise ValueError("search-only records provider is required")
    records = provider()
    if not isinstance(records, list):
        raise ValueError("invalid search-only records provider")
    return records


def _allowed_hypothesis_terms(hypothesis: str, cards: list[dict[str, Any]]) -> tuple[str, ...]:
    """Only compact vocabulary terms from validated cards cross retrieval boundary."""
    vocabulary = set()
    for card in cards:
        for field in INDEX_FIELDS:
            vocabulary.update(_tokens(str(card[field])))
        vocabulary.update(_tokens(" ".join(card["tags"])))
    stop = {"the", "and", "for", "with", "from", "this", "that", "will", "test",
            "model", "candidate", "residual", "error", "search", "target", "rows"}
    terms = []
    for term in _tokens(hypothesis):
        if term.isalpha() and 2 <= len(term) <= 24 and term in vocabulary and term not in stop and term not in terms:
            terms.append(term)
    return tuple(terms[:12])


def plan_with_research(agent: Any, prompt: Any, parent_id: str, stage: str,
                       code_query: Callable[[Any, Any], tuple[str, str]],
                       *, generate: Callable[..., str] | None = None,
                       library: Path = LIBRARY) -> tuple[str, str, Any, dict[str, Any]]:
    """Run a card-free observation, deterministic retrieval, and final code plan."""
    if generate is None:
        from llm import generate as generate_llm
        generate = generate_llm
    records = _search_records(agent)
    prompt, compaction = _compact_context(prompt)
    pre_prompt = _append_user(prompt, PRELIMINARY_INSTRUCTION)
    # Previous plans can mention cards. The first-stage decision must see the
    # task/search history, not any prior assistant narrative or card references.
    pre_prompt["assistant"] = "State a measured search-only observation and preliminary hypothesis."
    enforce_prompt_budget(pre_prompt)
    preliminary_raw = generate(prompt=pre_prompt, cfg=agent.cfg,
                               temperature=getattr(agent.acfg.code, "temp", None), max_tokens=500)
    observation, preliminary_hypothesis = _preliminary_response(preliminary_raw)
    cards, library_hash = load_library(library)
    hypothesis_terms = _allowed_hypothesis_terms(preliminary_hypothesis, cards)
    atlas = _latest_search_atlas(records)
    if atlas is None:
        retrieval = {"status": "no_machine_readable_search_atlas", "cards": [],
                     "rendered": "", "query_sha256": None, "manifest_sha256": library_hash,
                     "evidence": EVIDENCE}
    else:
        query = build_search_query(atlas, records)
        retrieval = retrieve(cards, observation, query, max_cards=5,
                             max_utf8_bytes=MAX_CARD_BYTES,
                             hypothesis_terms=hypothesis_terms)
        if retrieval["manifest_sha256"] != library_hash:
            raise ValueError("research library changed during retrieval")
    # Keep the pre-retrieval claim visible, while preventing cards from silently
    # replacing the experimenter's own search-data observation.
    measured = (_measured_observation(atlas, observation) if atlas is not None else
                {"status": "no machine-readable scored search atlas; category unverified"})
    prefix = (FINAL_INSTRUCTION + "Pre-retrieval measured search observation: " +
              json.dumps({"category": observation, "fixed_atlas_values": measured}, sort_keys=True) +
              "\nPreliminary hypothesis: " +
              preliminary_hypothesis + "\nRetrieved cards (may all be rejected):\n")
    selected = list(retrieval["cards"])
    rendered = retrieval["rendered"]
    final_prompt = _append_user(prompt, prefix + (rendered or "(none)") + "\n")
    plan, code = code_query(agent, final_prompt)
    if not isinstance(plan, str) or FORBIDDEN_TERMS.search(plan):
        raise ValueError("unsafe final research plan")
    match = re.search(r"(?im)^\s*(?:final\s+)?revised hypothesis\s*:\s*(.+)$", plan)
    if match is not None:
        final_hypothesis = match.group(1).strip()[:600]
        hypothesis_source = "explicit_label"
    elif plan.strip():
        # MLEvolve's normal coder often writes a concise hypothesis without a
        # heading. Preserve that exact first line instead of discarding a
        # valid candidate for formatting alone.
        final_hypothesis = plan.strip().splitlines()[0][:600]
        hypothesis_source = "unlabelled_first_plan_line"
    else:
        final_hypothesis = preliminary_hypothesis
        hypothesis_source = "preliminary_retained_no_plan_text"
    trace = {
        "schema_version": 1, "stage": stage, "parent_id": parent_id,
        "evidence": EVIDENCE, "preliminary_observation": observation,
        "preliminary_measured_search_values": measured,
        "preliminary_hypothesis": preliminary_hypothesis,
        "retrieval_hypothesis_terms": list(hypothesis_terms),
        "preliminary_prompt_sha256": hashlib.sha256(json.dumps(pre_prompt, sort_keys=True).encode()).hexdigest(),
        "library_version": "v1", "library_sha256": library_hash,
        "retrieval_status": retrieval["status"], "retrieval_query_sha256": retrieval["query_sha256"],
        "retrieved_cards": selected, "final_revised_hypothesis": final_hypothesis,
        "final_hypothesis_source": hypothesis_source,
        "final_revised_hypothesis_and_plan": plan,
        "final_prompt_sha256": hashlib.sha256(json.dumps(final_prompt, sort_keys=True).encode()).hexdigest(),
        "preliminary_budget_bytes": enforce_prompt_budget(pre_prompt),
        "final_budget_bytes": enforce_prompt_budget(final_prompt),
        "history_compaction": compaction,
    }
    return plan, code, final_prompt, trace
