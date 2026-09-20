# Phase 1 Implementation

Current status: the first vanilla MLEvolve autonomous loop completed through one evaluated candidate and one generated successor. ReasoningBank remains absent.

## Implemented

- `connectome/evaluator.py`: causal callback replay and exact authenticated Global Weighted Pearson scoring.
- `connectome/mlevolve_adapter.py`: converts evaluator output and execution failures into structured MLEvolve feedback without replacing `ExecutionResult`.
- `connectome/task.md`: complete competition contract and the MLEvolve candidate execution boundary.
- `upstream/MLEvolve` pinned at `9c5c8a3b23f0361708b59a401452dddc00f97189`.
- `connectome_mode`: narrowly skips Kaggle CSV existence/format-server checks, stores `wp_t0`, `wp_t1`, and `combined_wp` on the search node, and preserves ordinary MLEvolve validation when disabled.
- Explicit task files remain verbatim instead of passing through the lossy LLM task cleaner.
- The executor omits Linux CPU affinity calls on Windows while preserving Linux affinity behavior.

Search selection, Progressive MCGS, backpropagation, draft/evolution/fusion behavior, and MLEvolve memory code are unchanged.

## Environment

The experiment environment is `.venv` with Python 3.11.5. Dependencies came from the pinned checkout's `requirements_base.txt`; `antlr4-python3-runtime==4.9.3`, `black==24.3.0`, CPU-capable `torch==2.7.1`, `lightgbm==4.6.0`, and `scikit-learn==1.6.1` use exact pins from its other authoritative requirement files.

The upstream dependency set is Linux/CUDA-oriented. Installing all of `requirements_ml.txt` or `requirements_domain.txt` on Windows would incorrectly pull CUDA and unrelated domain stacks. `requirements_base.txt` also uses `--no-deps`, leaving optional Windows/transitive packages reported by `pip check`; none are on the verified MLEvolve import or Codex adapter path. The pinned `aiohttp==3.11.13` is yanked upstream but was retained rather than silently upgraded.

## Codex Backend

`llm_backend=codex` selects `upstream/MLEvolve/llm/codex.py`; `auto` preserves the original Gemini/OpenAI dispatch. Both MLEvolve interfaces are supported:

- `query(...)`: full compiled system and user prompts on stdin; `FunctionSpec.json_schema` is passed through `codex exec --output-schema`; returns a string or parsed dictionary exactly as before.
- `generate(...)`: full compiled prompt on stdin; returns the final text consumed by existing plan, code, JSON, and diff parsers.

Invocation uses `codex-cli 0.154.0`, ChatGPT login, model `gpt-5.6-luna`, medium reasoning, an ephemeral session, ignored project rules/user configuration, an empty read-only working directory, approval policy `never`, and `--output-last-message`. The adapter explicitly removes `OPENAI_API_KEY` from the child environment.

Unlike the SDK backends, Codex CLI does not expose temperature, output-token accounting, or a native stop-token option; stop tokens are applied to returned text. Each request is an isolated subprocess, so there is startup overhead and no conversational session reuse. Independent processes are concurrency-safe, though the Phase 1 smoke run is configured for one worker.

Real harmless checks returned:

```text
structured query: {'status': 'ready'}
normal plan/code parser: plan='Plan: Print the word ready.', code='print("ready")'
```

These calls used Codex's existing ChatGPT authentication. They made Codex service requests, but no direct OpenAI API-key request and no Connectome data was sent.

## Smoke Data

`smoke_data/connectome/` contains authenticated row groups 0 and 1 from both `train.parquet` and `valid.parquet`: two complete sequences and 40,000 rows per file, with every 20,000-row sequence boundary, 99-row warm-up, feature, target, and validation-mask value preserved. Scores on this engineering subset are not leaderboard evidence.

## Verification

Run:

```text
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv\Scripts\python.exe -m pytest -q
```

Current result: **21 passed, 0 failed**. Coverage includes official scorer parity, causality and resets, result propagation/failures, Connectome-only CSV bypass, unchanged non-Connectome rejection, task-context delivery, Codex opt-in dispatch, schema handling, stdin prompt preservation, API-key removal, debug-result recovery after backpropagation, and candidate subprocess access to the evaluator package.

## Autonomous Smoke Experiment #1

Run directory: `smoke_runs/20260920_111359_connectome-codex-phase1`.

Configuration: Codex CLI 0.154.0 through `llm_backend=codex`, model `gpt-5.6-luna`, medium reasoning, one worker, global memory disabled, existing ChatGPT authentication, and `OPENAI_API_KEY` removed. The command was:

```powershell
$env:OPENAI_API_KEY=$null
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe connectome\run_mlevolve_smoke.py data_dir=smoke_data/connectome desc_file=connectome/task.md log_dir=smoke_runs workspace_dir=smoke_runs exp_name=connectome-codex-phase1 connectome_mode=true llm_backend=codex codex_reasoning_effort=medium agent.code.model=gpt-5.6-luna agent.feedback.model=gpt-5.6-luna agent.search.parallel_search_num=1 agent.search.num_gpus=0 agent.use_global_memory=false agent.use_stepwise_generation=false preprocess_data=false copy_data=true exec.timeout=600 cpu_number=1
```

The deterministic subset used train row groups 0-1 (`seq_ix` 43601 and 50562) and validation row groups 0-1 (`seq_ix` 256678 and 751005). Every sequence retained 20,000 rows. Validation contained 40,000 rows, 39,802 prediction callbacks after the two 99-row warm-ups, and 3,210 rows selected by `need_prediction AND is_scored` for scoring.

### Candidate #1

MLEvolve drafted a deterministic echo-state reservoir with a ridge readout. Its causal features used all 112 i0/i1/additional inputs, training-only standardization, the current state, one EMA channel, first differences, documented-group summaries, and recurrent reservoir state. It reset feature and reservoir state on every sequence boundary and trained only from `train.parquet`.

The draft node `ed4fcd2dd23b4190b1aed3e17c93cc57` trained but passed a DataFrame to the evaluator. MLEvolve's native debug action consumed that traceback and produced node `d58a2c50c330424f8a45bcbc06263f07`, which constructed `SequenceData` and completed successfully. Executor runtime was 66.855 seconds; the evaluator reported 27.033 seconds for causal replay/scoring.

Smoke-test metrics, **not a leaderboard estimate**:

```text
WP_t0       = 0.4879655787025124
WP_t1       = 0.26740648246668985
combined_WP = 0.37768603058460115
```

The result path was candidate subprocess output `CONNECTOME_RESULT_JSON` -> `connectome/mlevolve_adapter.py` -> `agents/result_parse_agent.py` -> `SearchNode.metric` and `SearchNode.connectome_metrics` -> `engine/evaluation.py` -> best-node/search state. The stored node contains `metric.value=0.37768603058460115`, `maximize=true`, and all three component metrics. MLEvolve logged the parent draft as a debug success and promoted the repaired node to best.

### Candidate #2

Normal search selected an `improve` action with candidate #1 as parent and generated node `028fb69b755c4390b6f8e0b61a4328d1`. It proposed replacing the single EMA with fast (`0.80`) and slow (`0.99`) causal EMAs and adding acceleration features, aimed particularly at the weaker t1 score while retaining the reservoir, ridge readout, split, resets, and official evaluator. Its prompt included candidate #1's code, execution output, `WP_t0`, `WP_t1`, and combined score. Execution was deliberately deferred and the run stopped at the acceptance boundary.

### Integration Fixes and Residuals

- Codex schemas now copy upstream schemas into Codex-strict form, including nullable representations for upstream optional properties.
- The smoke runner follows MLEvolve's native debug chain and recovers the just-executed journal node when `AgentSearch.step()` returns the virtual root as a backpropagation control signal.
- Candidate subprocesses receive the repository root on `PYTHONPATH`, making the documented `connectome` evaluator import available from the isolated workspace.
- Candidate-required `lightgbm` and `scikit-learn` were installed at the versions already pinned by MLEvolve.
- MLEvolve's dynamic planner schema is not representable by Codex's strict output-schema subset. The planner exhausted its existing retries and used MLEvolve's existing full-rewrite fallback; candidate #2 was still generated successfully. This is noisy and slower but did not block the verified loop.

Progressive MCGS, node selection, backpropagation, draft/evolution/fusion logic, the competition metric, and MLEvolve memory code were not modified. Deployment feasibility checks remain a later requirement before packaging a competition submission.
