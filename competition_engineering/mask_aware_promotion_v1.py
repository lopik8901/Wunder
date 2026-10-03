"""Supervisor-only v1 promotion gate; no candidate or protected data is read on import.

The caller must supply frozen, CPU-validated predictors. This module evaluates
them only when explicitly invoked by the supervisor. It is never a search kind.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from competition_engineering.mask_protocol import SPEC_SHA256, proxy_mask
from competition_engineering.residual import from_stats, sufficient

ROOT_SEARCH_WP = 0.6548654996120564
MIN_SEARCH_GAIN = 0.001
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_DRAWS = 1000
TEST_ROWS = 39_400_000
MAX_TEST_SECONDS = 3600
MAX_ZIP_BYTES = 20_000_000
ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_cpu_attestation(attestation: dict, candidate_sha256: str) -> None:
    if attestation.get("candidate_sha256") != candidate_sha256:
        raise ValueError("CPU attestation is not tied to the frozen candidate")
    required = ("python311_linux", "one_cpu", "offline", "causal", "reset",
                "deterministic", "finite_float32_output", "package_compatible")
    if any(attestation.get(key) is not True for key in required):
        raise ValueError("CPU callback/deployment attestation incomplete")
    if not (0 < attestation.get("zip_bytes", 0) <= MAX_ZIP_BYTES):
        raise ValueError("package size exceeds deployment bound")
    us = attestation.get("microseconds_per_row")
    if not isinstance(us, (float, int)) or not np.isfinite(us) or us <= 0:
        raise ValueError("missing standardized CPU callback timing")
    if us * TEST_ROWS / 1e6 >= MAX_TEST_SECONDS:
        raise ValueError("CPU callback does not plausibly meet the runtime limit")


def _predict_checked(predictor: Callable, z: dict) -> np.ndarray:
    # The predictor receives only inference-visible fields. Targets, scoring
    # mask, and cached supervisor metadata never cross this function boundary.
    visible = {}
    for key in ("x", "p", "need", "step", "seq"):
        if key in z:
            value = np.asarray(z[key]).copy()
            value.setflags(write=False)
            visible[key] = value
    prediction = np.asarray(predictor(visible), dtype=np.float32)
    if prediction.shape != z["y"].shape or not np.isfinite(prediction[z["need"]]).all():
        raise ValueError("invalid frozen prediction shape or finite required-row output")
    return prediction


def _stage(sequences: Iterable[tuple[int, dict]], reference: Callable,
           candidate: Callable, mask_provider: Callable, *, stage: str) -> dict:
    per_sequence = []
    mask_hasher = hashlib.sha256()
    total_rows = 0
    start = time.perf_counter()
    for group, z in sequences:
        mask = np.asarray(mask_provider(group, z), dtype=bool)
        if mask.shape != z["need"].shape or np.any(mask & ~z["need"]):
            raise ValueError("invalid scoring mask")
        if not mask.any():
            raise ValueError("empty scored sequence")
        # Bind both group identity and exact selected steps to the report.
        mask_hasher.update(int(group).to_bytes(8, "little", signed=True))
        mask_hasher.update(np.packbits(mask, bitorder="little").tobytes())
        root_p = _predict_checked(reference, z)
        candidate_p = _predict_checked(candidate, z)
        per_sequence.append([sufficient(z["y"][mask], root_p[mask]),
                             sufficient(z["y"][mask], candidate_p[mask])])
        total_rows += int(mask.sum())
    if not per_sequence:
        raise ValueError("empty supervisor evaluation set")
    moments = np.asarray(per_sequence)
    pooled = moments.sum(axis=0)
    scores = [from_stats(pooled[i]) for i in (0, 1)]
    delta = scores[1] - scores[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(len(moments), size=len(moments))
        pair = moments[indices].sum(axis=0)
        differences.append((from_stats(pair[1]) - from_stats(pair[0])).mean())
    interval = np.quantile(differences, [.025, .975])
    return {"stage": stage, "reference_wp": scores[0].tolist(),
            "candidate_wp": scores[1].tolist(), "delta_per_target": delta.tolist(),
            "delta_combined": float(delta.mean()), "paired_95ci_combined": interval.tolist(),
            "sequences": len(moments), "scored_rows": total_rows,
            "mask_sha256": mask_hasher.hexdigest(),
            "runtime_seconds": time.perf_counter() - start,
            "evidence": "reused_supervisor_only_not_search_feedback"}


def run_protocol(*, spec: dict, spec_bytes: bytes, search_wp: float,
                 candidate_sha256: str, cpu_attestation: dict,
                 reference: Callable, candidate: Callable,
                 confirm_sequences: Callable[[], Iterable[tuple[int, dict]]],
                 holdout_sequences: Callable[[], Iterable[tuple[int, dict]]],
                 proxy_reference_model: dict) -> dict:
    """Execute the locked gates in order, stopping before each ineligible read.

    Iterables are lazy factories: a failed search/CPU/confirm gate cannot even
    open the next protected cache. No protected result is returned to search.
    """
    if (_sha256(spec_bytes) != SPEC_SHA256 or
            json.loads(spec_bytes) != spec or spec.get("protocol_id") != "mask_aware_v1"):
        raise ValueError("frozen v1 specification mismatch")
    if not isinstance(search_wp, (float, int)) or not np.isfinite(search_wp):
        raise ValueError("invalid search result")
    report = {"protocol_id": "mask_aware_v1", "spec_sha256": SPEC_SHA256,
              "candidate_sha256": candidate_sha256,
              "search_wp": float(search_wp),
              "search_gain_vs_submitted_root": float(search_wp - ROOT_SEARCH_WP),
              "evidence": "supervisor_only_never_search_feedback"}
    if search_wp < ROOT_SEARCH_WP + MIN_SEARCH_GAIN:
        return {**report, "status": "failed_search_gate"}
    _validate_cpu_attestation(cpu_attestation, candidate_sha256)
    report["cpu_attestation"] = cpu_attestation.copy()
    confirm = _stage(confirm_sequences(), reference, candidate,
                     lambda group, z: z["need"] & z["mask"], stage="actual_official_mask_confirm")
    report["confirm"] = confirm
    if confirm["delta_combined"] <= 0:
        return {**report, "status": "failed_masked_confirm_gate"}
    proxy = _stage(holdout_sequences(), reference, candidate,
                   lambda group, z: proxy_mask(z, proxy_reference_model, spec, group),
                   stage="search_propensity_proxy_holdout")
    report["proxy"] = proxy
    if proxy["paired_95ci_combined"][0] <= 0:
        return {**report, "status": "failed_proxy_interval_gate"}
    return {**report, "status": "passed_v1_promotion_gates"}


def run_local_supervisor(*, search_wp: float, candidate_artifact: Path,
                         cpu_attestation: dict, candidate_predictor: Callable) -> dict:
    """Production entrypoint; invoke explicitly only for a frozen eligible candidate.

    Candidate execution must have been vetted separately. The supplied
    predictor must be a supervisor-owned adapter verified against this frozen
    artifact and its CPU callback; this function does not import candidate
    source, execute an archive, or publish feedback.
    """
    from competition_engineering.compare_ridges import predict as ridge_predict
    from competition_engineering.mask_protocol import SPEC, verify_frozen_spec
    from competition_engineering.pipeline import cached

    spec = verify_frozen_spec()
    spec_bytes = SPEC.read_bytes()
    candidate_artifact = Path(candidate_artifact)
    candidate_sha256 = _sha256(candidate_artifact.read_bytes())
    root_path = ROOT / spec["frozen_root_path"]
    with np.load(root_path) as archive:
        root = {key: archive[key].copy() for key in archive.files}

    def read_exact(cache_name: str, expected_split: str, expected_count: int):
        cache_path = ROOT / "competition_engineering/cache" / cache_name
        identity = json.loads((cache_path / "identity.json").read_text(encoding="utf-8"))
        if identity.get("split") != expected_split or len(identity.get("groups", [])) != expected_count:
            raise ValueError("supervisor cache does not match frozen split")
        yield from cached(cache_path)

    return run_protocol(
        spec=spec, spec_bytes=spec_bytes, search_wp=search_wp,
        candidate_sha256=candidate_sha256, cpu_attestation=cpu_attestation,
        reference=lambda z: ridge_predict(z, root), candidate=candidate_predictor,
        confirm_sequences=lambda: read_exact("gru_confirm", "confirm", 64),
        holdout_sequences=lambda: read_exact("gru_train_holdout_phase2", "train_holdout", 512),
        proxy_reference_model=root,
    )
