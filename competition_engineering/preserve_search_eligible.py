"""Copy search-qualified artifacts without opening any protected evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from competition_engineering.pipeline import write_json

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.001
INCUMBENT_SEARCH_WP = 0.6548654996120564
ARTIFACTS = {"ridge": "ridge.npz", "calibration": "calibration.npz",
             "gpu_residual": "checkpoint.pt"}


def preserve(run_dir: Path) -> list[dict]:
    run_dir = run_dir.resolve()
    history = run_dir / "experiments/history.jsonl"
    if not history.is_file():
        return []
    destination_root = ROOT / "competition_engineering/checkpoints" / ("promotion_eligible_" + run_dir.name)
    saved = []
    for line in history.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") != "success":
            continue
        score = record["result"]["primary_metric"]
        if score < INCUMBENT_SEARCH_WP + THRESHOLD:
            continue
        node_id = record["node_id"]
        kind = record["config"]["kind"]
        source_dir = ROOT / record["config"]["output"]
        source = source_dir / ARTIFACTS[kind]
        destination = destination_root / node_id
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != source_hash:
                raise RuntimeError(f"preserved artifact hash mismatch: {node_id}")
        else:
            shutil.copy2(source, target)
        details = {"run_id": run_dir.name, "node_id": node_id, "kind": kind,
                   "search_wp": score, "search_gain": score - INCUMBENT_SEARCH_WP,
                   "search_metrics": record["result"]["metrics"],
                   "spec": record["spec"], "artifact_sha256": source_hash,
                   "artifact": str(target), "protected_evaluation_performed": False}
        metadata = destination / "search_eligibility.json"
        if metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")) != details:
            raise RuntimeError(f"preserved metadata mismatch: {node_id}")
        if not metadata.exists():
            write_json(metadata, details)
        saved.append(details)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(preserve(args.run_dir), indent=2))
