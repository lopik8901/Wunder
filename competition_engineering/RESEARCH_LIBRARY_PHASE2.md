# Offline research library and retriever — Phase 2

This phase does **not** connect retrieval to MLEvolve prompts, search history,
candidate execution, or global memory. It does not launch any experiment.
`use_global_memory: false` remains unchanged.

`research_cards/v1/` contains 23 paper cards spanning ten families: attention
(4), state space (4), recurrence (3), convolution (3), residual (2), simple
models (2), calibration (2), gating (1), reservoir (1), and nonlinear correction
(1). The corpus manifest SHA256 is
`2f079c894c3b6d33660ca21bba815110f78cc5875361394869e10227ef1196e7`.
Each JSON file has its own stable ID/version, canonical source URL, source
locator, provenance, causal and CPU caveats, and content hash computed by
`card_sha256`. `tools/build_research_cards.py` is the reproducible source for
the cards and refuses to overwrite an existing library.

Paper-mechanism claims were checked against primary publisher or author-posted
abstracts. `source_locator` is therefore `Abstract`, not a claim to have
reviewed the full paper. `useful_when` and diagnostic signatures are explicitly
marked curator inferences. No article text or paper PDF is stored. Published
results on other tasks are not claims about Connectome WP. Methods requiring
prior target labels at inference were excluded from this retrievable starter
set. Before live use, a second curator should review card wording, omissions,
licensing, and family balance.

`connectome.research_retrieval` accepts only validated search-atlas queries and
a constrained pre-retrieval observation. It uses deterministic BM25-style
lexical ranking with a small fixed synonym map, stable ID tie-breaking, at
most two cards per family, at most five cards total, and a 2,800-byte rendered
output ceiling. A byte limit is **not** an exact model-token limit. The result
records library and query hashes plus selected card IDs, versions, hashes,
families, and ranking scores. Empty libraries and no-result queries return
explicit statuses; malformed or protected context fails closed. The retriever
never uses the network, search data files, protected data, or candidate code.

The library remains supervisor-owned. The production bubblewrap sandbox mounts
one candidate workspace, runtime, and designated training data only; it must
never mount `research_cards`. Search outcomes are validated as search-only
context but do not boost previously successful architecture families in
ranking. This reduces feedback-driven literature dominance.

Before live integration: measure actual model tokens; implement deterministic
history compaction; obtain a card-free preliminary observation from MLEvolve and
verify it against the atlas; insert cards through one shared adapter across
draft/improve/aggregation/fusion/evolution; capture exact retrieval traces in
the journal; then run Windows/WSL boundary and isolation preflights. No card
should enter a prompt until those gates pass.
