"""Search-derived scoring-mask proxy for supervisor-only future promotions.

This module never reads evaluation targets or candidate predictions. The proxy
is a robustness check, not the hidden test mask or an independent holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from competition_engineering.compare_ridges import predict
from competition_engineering.pipeline import cached

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "competition_engineering/protocols/mask_aware_v1.json"
SPEC_SHA256 = "7210810f5b4a43f47299d4d6883f932831d5eb9cae6975cd9e2d9e3cd10d3862"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bins(z: dict, root: dict, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    # The frozen root is causal; neither labels nor candidate outputs define a bin.
    magnitude = np.max(np.abs(predict(z, root)), axis=1)
    step = z["step"] if "step" in z else np.arange(len(magnitude))
    position = np.searchsorted(spec["step_edges"][1:-1], step, side="right")
    amplitude = np.searchsorted(spec["magnitude_edges"][1:], magnitude, side="right")
    return position, amplitude


def search_counts(search_cache: Path, root: dict, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(spec["step_edges"]) - 1, len(spec["magnitude_edges"]))
    required = np.zeros(shape, dtype=np.int64)
    selected = np.zeros(shape, dtype=np.int64)
    for _, z in cached(search_cache):
        position, amplitude = _bins(z, root, spec)
        need = z["need"]
        scored = need & z["mask"]
        np.add.at(required, (position[need], amplitude[need]), 1)
        np.add.at(selected, (position[scored], amplitude[scored]), 1)
    return required, selected


def proxy_mask(z: dict, root: dict, spec: dict, group: int) -> np.ndarray:
    """Stable candidate-blind Bernoulli mask for a training holdout sequence."""
    position, amplitude = _bins(z, root, spec)
    required = np.asarray(spec["search_required_counts"], dtype=np.float64)
    selected = np.asarray(spec["search_scored_counts"], dtype=np.float64)
    if required.shape != selected.shape or np.any(selected > required) or np.any(required <= 0):
        raise ValueError("invalid frozen search mask counts")
    probability = (selected + 0.5) / (required + 1.0)
    # A fixed cryptographic digest avoids platform-specific random streams and
    # lets any supervisor reproduce the same mask without storing row labels.
    key = f"{spec['proxy_seed']}:{group}:".encode("ascii")
    uniform = np.fromiter(
        (int.from_bytes(hashlib.blake2b(key + str(i).encode("ascii"), digest_size=8).digest(),
                        "little") / 2**64 for i in range(len(position))),
        dtype=np.float64, count=len(position),
    )
    return z["need"] & (uniform < probability[position, amplitude])


def verify_frozen_spec(spec_path: Path = SPEC) -> dict:
    if file_sha256(spec_path) != SPEC_SHA256:
        raise ValueError("frozen mask-aware protocol specification changed")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    root_path = ROOT / spec["frozen_root_path"]
    if file_sha256(root_path) != spec["frozen_root_sha256"]:
        raise ValueError("frozen root changed")
    search_cache = ROOT / spec["search_cache"]
    if file_sha256(search_cache / "identity.json") != spec["search_cache_identity_sha256"]:
        raise ValueError("search cache identity changed")
    with np.load(root_path) as archive:
        root = {key: archive[key] for key in archive.files}
    required, selected = search_counts(search_cache, root, spec)
    if required.tolist() != spec["search_required_counts"] or selected.tolist() != spec["search_scored_counts"]:
        raise ValueError("search-derived mask counts changed")
    return spec
