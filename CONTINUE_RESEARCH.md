# Cross-device Connectome bootstrap

Bootstrap is read-only verification and synthetic regression. Do not launch
experiments, old campaign phases, protected evaluation or submissions. Read
`competition_engineering/RESEARCH_HANDOFF.md` first; summarize and wait for
user approval before research resumes.

## 1. Clone / pull

```sh
git clone https://github.com/lopik8901/Wunder.git
cd Wunder
git status --short
git log -3 --oneline
```

If already cloned, save local changes and use `git pull --ff-only`. Do not discard
local work or force-push. The handoff commit/final push details are reported in
the ending session; verify it is present in the clone.

## 2. Install an environment

Windows (Python3.12 was tested; use Python3.11 for competition-like runtime):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-handoff.txt
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Native Linux or WSL Ubuntu (Python3.11; create the venv on the Linux filesystem):

```sh
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv bubblewrap
python3.11 -m venv "$HOME/.venvs/connectome"
"$HOME/.venvs/connectome/bin/python" -m pip install -r requirements-handoff.txt
"$HOME/.venvs/connectome/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
export WUNDER_WSL_PYTHON="$HOME/.venvs/connectome/bin/python"
```

If Ubuntu's configured repositories lack3.11, install3.11 through your approved
Python distribution first; do not substitute an unverified environment silently.
The requirements file specifies the minimal dependency set, not every optional
upstream ML backend. Tested version snapshots are in
`competition_engineering/handoff/environment_versions_*.json`. CPU PyTorch
is sufficient; AMD/ROCm installation is optional and device-specific. The old
large upstream requirement files can impose conflicting pins and are not the
minimal bootstrap. No cloud API keys are needed for verification.

For a Windows-launched MLEvolve bridge, explicitly set `WUNDER_WSL_PYTHON` to the
Linux absolute path of your new interpreter in the Windows environment too.
The historical fallback venv name is not portable setup. Invoke commands
directly inside WSL until that bridge is configured. `WUNDER_WSL_DISTRO` support,
if absent in the bridge, means its Ubuntu distribution assumption needs review
before launching MLEvolve; no search launch is part of bootstrap.

## 3. Restore external data and exact artifacts

Transfer `connectome_required_artifacts.zip` from the ending device; its SHA256
is in `competition_engineering/handoff/external_artifacts.json`. It is a small
private transfer bundle and is intentionally excluded from Git.

```sh
python -m tools.verify_research_handoff --metadata-only
python -m tools.verify_research_handoff --restore-bundle /path/to/connectome_required_artifacts.zip
python -m tools.verify_research_handoff --external
```

Windows: substitute `.\.venv\Scripts\python.exe` for `python` in these commands.
The restore command verifies the archive, exact member allowlist and individual
hashes, rejects traversal and refuses to overwrite a different existing artifact.
It does not restore datasets, caches, credentials or protected results.

Privately copy or obtain the official starter-pack dataset through your existing
authorized competition access. Place exact files at:

```text
competition_engineering/assets/wnn_connectome_starterpack/datasets/train.parquet
competition_engineering/assets/wnn_connectome_starterpack/datasets/valid.parquet
competition_engineering/assets/wnn_connectome_starterpack/datasets/valid_mask.parquet
```

For an external data disk, use an explicit directory junction/symlink at that
repository-relative location; verify its target before using it. No historical
personal path is authoritative. Private dataset credentials/download URLs are
not included. Opaque integrity hashing does not expose protected outcomes:

```sh
python -m tools.verify_research_handoff --datasets
```

This hashes bytes and checks train metadata/sequence-ID assignments only;
it performs no target analysis, evaluation or scoring. Optional historical
artifacts and original bulk caches are not needed to understand research state.
For exact old replay, transfer their manifest-listed identities privately as
well. Do not recreate an incumbent by retraining or claim absent caches verified.

## 4. Regression and production-isolation preflight

```sh
python -m tools.verify_research_handoff --tests
python -m tools.verify_research_handoff --isolation
```

Run tests on Windows and Linux/WSL. The Linux isolation check reaches the trusted
worker's intentional probe rejection with no candidate/data mounted. User
namespaces and bubblewrap must work; stop on failure. Windows synthetic test
success does not satisfy production isolation. Preserve network/mount/process
limits; never fall back to unisolated real-data fitting. Tests use synthetic
fixtures or immutable model parity; they run no ML competition experiments.

## 5. Load and confirm research/data state

Read, in order:

1. `competition_engineering/RESEARCH_HANDOFF.md`.
2. `competition_engineering/handoff/research_state.json` and `data_roles.json`.
3. `competition_engineering/search_knowledge.json` (v25).
4. `competition_engineering/handoff/experiment_ledger.jsonl`, compact82-attempt
   MLEvolve history, and relevant evidence reports/protocols.
5. `competition_engineering/research_cards/v16/` (39 cards; paper versus inference).
6. `competition_engineering/handoff/artifact_manifest.json` and external manifest.

The512 replication sequences in the frozen specification must remain untouched
until a candidate passes a justified development gate and is frozen. An older
additional256 block also remains reserved. The role registry lists1391
unassigned train groups. Reserve new roles before outcomes are inspected.
Never treat reused tune64, consumed development/replication or unknown GRU
pretraining exposure as independent validation.

The exported ledger has a new verifiable chain and records original event/source
hashes. Exported JSON normalizes personal paths and excludes explicit bulky/private
fields; frozen reservation/source snapshots retain exact original bytes.
`tools.verify_*checkpoint` scripts require the original local run trees/caches;
use the portable verifier for this clone. Do not run old `finalize`, test-recording,
`prepare`, `fit` or `train` phases inside frozen campaign directories.

## 6. Next-device prompt

The historical mask-protocol integration test additionally needs the old ridge
checkpoint and complete `cache/gru_tune` fixture. Those optional historical
artifacts are excluded from the required transfer bundle; the test explicitly
skips when absent. Synthetic mask/boundary regressions still run. Do not rebuild
the cache during bootstrap or interpret this skip as a scientific result.

### NEXT_DEVICE_PROMPT

```text
This is a read-only Connectome cross-device bootstrap, not authorization for experiments.
Read CONTINUE_RESEARCH.md, competition_engineering/RESEARCH_HANDOFF.md,
competition_engineering/handoff/research_state.json, its data_roles.json,
experiment_ledger.jsonl and mlevolve_attempts.json, competition_engineering/search_knowledge.json
and competition_engineering/research_cards/v16.
Verify Git state, the portable evidence manifest, exact incumbent/external artifact hashes,
data reservation identities, synthetic regression and Linux isolation where available.
Do not access protected feedback or run experiments, promotion, evaluation or submission.
Report missing artifacts honestly. Summarize the incumbent0.658835322, uncertainty and
negative/inconclusive findings, closed/open hypotheses, and512 untouched replication
sequences. Then wait for my approval before starting new research.
```
