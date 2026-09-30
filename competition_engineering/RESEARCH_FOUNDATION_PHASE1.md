# Research retrieval foundation — Phase 1 only

This document is a design contract. Search prompts, search history, tree policy,
candidate execution, and global memory are unchanged. There is no research
library or live retriever.

## Prompt budget and history compaction

Use `tools/measure_mlevolve_prompt_budget.py` on a local journal. It reports
characters and UTF-8 bytes without printing prompt content. An exact content
token count requires an explicitly supplied *local* model tokenizer JSON; chat
framing and provider-specific overhead need separate calibration. Never call
character or byte counts exact model tokens. Measure every planning path and
the longest realistic parent-code/history combination before assigning a card
budget.

Measured on the saved `20260930_134047_connectome_gpu_search` journal: four
planning prompts were 46,874, 51,151, 59,361, and 62,531 characters
(47,872, 52,305, 60,663, and 63,985 UTF-8 bytes). Rendering the current
last-18-attempt history from saved search records produced 42,309 characters
and 43,865 UTF-8 bytes. No matching local model tokenizer was available, so
these are deliberately **not** reported as token counts. This is a measured
baseline for a later token-calibrated budget, not a safe final prompt ceiling.

Proposed future ceiling: reserve at most 800 content tokens for 3–5 cards,
with an additional fixed allowance for provenance. Before enabling retrieval,
set a model-specific total prompt ceiling, measured with its actual tokenizer.
If the base prompt already exceeds it, show zero cards and record why. No
silent truncation of the current observation, selected parent's code, or safety
instructions.

History compaction should be deterministic and search-only. Preserve the
selected parent's full immediate result and atlas; retain the recent few
attempts with their measured outcomes; reduce older attempts to bounded
fingerprint, parent, mechanism, outcome, target WP, runtime, and a fixed small
set of atlas deltas. Preserve failures as implementation evidence, not negative
ML evidence. Retain explicit counts of omitted attempts and duplicate
fingerprints. The compacted representation must be built from a validated
allowlist, never an LLM-generated summary, raw journal, or protected report.
Compaction should be separately versioned and tested before replacing the
current `format_search_history` output.

## Card and provenance rules

`connectome.research_foundation.validate_card` specifies the exact v1 fields.
Each card must have a stable ID/version and a source URL with an exact claim
locator, curator, verification date, confidence, limitations, causal and CPU
deployment constraints. Paper claims require a reviewer to check the cited
source location; concepts require a canonical public source and must be labelled
as concepts. Do not add unsupported claims, paper text dumps, model commands,
internal scores, or instructions to the planner. A card hash covers canonical
content. Freeze a manifest hash for an entire future run; preserve retrieved
card IDs/versions/hashes and prompt hash per decision.

No papers are curated or downloaded in Phase 1.

## Query allowlist and trust boundary

`build_search_query` currently accepts only the exact validated search atlas
and fixed-domain numeric search outcomes with enumerated mechanisms/statuses.
It discards free-text hypotheses, code, paths, logs, arbitrary result fields,
and all non-search metric domains. The query has no access to protected files.
This is an inactive contract, not an integration with MLEvolve. A future short
preliminary-hypothesis field requires its own reviewed schema and injection/
leakage tests before entering a query.

Retrieval must occur in the supervisor, after parent selection and a card-free
observation, before final plan-and-code. Candidate sandboxes must never mount
the research store. Empty retrieval falls back to the existing planner without
web access or global memory. `use_global_memory: false` remains unchanged.
