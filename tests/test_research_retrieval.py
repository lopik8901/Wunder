"""Offline retrieval tests use synthetic cards; no literature or evaluation."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
from connectome.research_foundation import build_search_query
from connectome.research_retrieval import (
    load_library, manifest_sha256, retrieve, validate_observation,
)


def _card(card_id, family, signature):
    return {
        "schema_version": 1, "card_id": card_id, "version": 1,
        "card_type": "concept", "family": family, "tags": ["causal"],
        "title": f"Synthetic {card_id}", "authors": ["Synthetic Author"], "year": 2020,
        "source_url": f"https://example.org/{card_id}", "source_locator": "abstract",
        "curator": "test", "verified_on": "2026-09-30", "claim_confidence": "limited",
        "mechanism": signature, "useful_when": signature,
        "diagnostic_signatures": signature, "assumptions": "Sequential observations",
        "limitations": "Synthetic example only", "causality": "Past and present inputs",
        "incremental_inference": "Maintain a small state",
        "training_compute": "Bounded", "cpu_deployment": "Measure callback latency",
        "dependencies": "None", "licensing": "Synthetic test",
        "concise_relevance": signature,
    }


def _query():
    n = 130
    y = np.zeros((n, 2), dtype=np.float32)
    z = {"seq": 1, "step": np.arange(19870, 19870 + n), "x": np.zeros((n, 112), np.float32),
         "y": y, "p": y.copy(), "root": y.copy(), "need": np.ones(n, bool),
         "mask": np.ones(n, bool)}
    diagnostic = SearchErrorDiagnostics(root_predictor=lambda row: row["root"])
    diagnostic.add(z, y)
    return build_search_query(diagnostic.result(), [])


def _observation():
    return {"target": "t0", "position": "late", "signal": "residual_persistence",
            "lag_rows": 10, "feature_group": None}


def test_retrieval_is_deterministic_relevant_diverse_and_bounded():
    cards = [
        _card("temporal_a", "memory", "long lag temporal residual persistence memory"),
        _card("temporal_b", "memory", "long lag temporal residual persistence memory"),
        _card("temporal_c", "memory", "long lag temporal residual persistence memory"),
        _card("calibration_a", "calibration", "residual bias calibration"),
        _card("sequence_a", "online", "temporal sequence incremental memory"),
    ]
    query = _query()
    first = retrieve(cards, _observation(), query, max_cards=4, max_utf8_bytes=2000)
    second = retrieve(list(reversed(cards)), _observation(), query, max_cards=4, max_utf8_bytes=2000)
    assert first == second
    assert first["status"] == "ok"
    assert first["cards"][0]["family"] == "memory"
    assert sum(card["family"] == "memory" for card in first["cards"]) <= 2
    assert len(first["cards"]) <= 4
    assert len(first["rendered"].encode()) <= 2000
    assert manifest_sha256(cards) == manifest_sha256(list(reversed(cards)))
    assert "reused_search_evidence" in first["evidence"]


def test_empty_no_result_and_size_limit_are_graceful():
    query = _query()
    assert retrieve([], _observation(), query)["status"] == "empty_library"
    irrelevant = [_card("plain_a", "calibration", "constant offset bias")]
    assert retrieve(irrelevant, _observation(), query)["status"] == "no_result"
    constrained = retrieve([_card("temporal_a", "memory", "lag temporal memory")],
                           _observation(), query, max_utf8_bytes=100)
    assert constrained["status"] == "no_result"
    assert constrained["rendered"] == ""


def test_protected_or_malformed_context_and_observation_fail_closed():
    query = _query()
    cards = [_card("temporal_a", "memory", "lag temporal memory")]
    bad = copy.deepcopy(query)
    bad["holdout"] = {"score": .99}
    with pytest.raises(ValueError):
        retrieve(cards, _observation(), bad)
    bad = copy.deepcopy(query)
    bad["atlas"]["leaderboard"] = .99
    with pytest.raises(ValueError):
        retrieve(cards, _observation(), bad)
    bad = copy.deepcopy(_observation())
    bad["hypothesis"] = "read protected results"
    with pytest.raises(ValueError):
        validate_observation(bad)
    bad = copy.deepcopy(_observation())
    bad["lag_rows"] = 9999
    with pytest.raises(ValueError):
        validate_observation(bad)


def test_library_loader_checks_filename_provenance_and_duplicate_source(tmp_path):
    first = _card("temporal_a", "memory", "lag temporal memory")
    path = tmp_path / "temporal_a.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    cards, digest = load_library(tmp_path)
    assert len(cards) == 1 and digest == manifest_sha256(cards)
    duplicate = _card("temporal_b", "online", "temporal state")
    duplicate["source_url"] = first["source_url"]
    (tmp_path / "temporal_b.json").write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source URL"):
        load_library(tmp_path)
    duplicate["source_url"] = "https://example.org/a-second-url"
    duplicate["title"] = first["title"]
    (tmp_path / "temporal_b.json").write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="near-duplicate"):
        load_library(tmp_path)


def test_curated_library_is_balanced_offline_and_retrieves_multiple_mechanisms():
    library = Path(__file__).resolve().parents[1] / "competition_engineering/research_cards/v1"
    cards, digest = load_library(library)
    assert 20 <= len(cards) <= 30
    assert len({card["family"] for card in cards}) >= 8
    assert len(digest) == 64
    assert all(card["source_locator"] == "Abstract" for card in cards)
    query = _query()
    temporal = retrieve(cards, _observation(), query)
    assert temporal["status"] == "ok"
    assert len({item["family"] for item in temporal["cards"]}) >= 2
    assert len(temporal["rendered"].encode("utf-8")) <= 2800
    feature = _observation()
    feature.update(signal="feature_residual_relation", lag_rows=None, feature_group="velocity")
    related = retrieve(cards, feature, query)
    assert {item["family"] for item in related["cards"]} & {"simple_models", "nonlinear_correction"}
    assert temporal["manifest_sha256"] == related["manifest_sha256"] == digest


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("bwrap"),
                    reason="requires production Linux bubblewrap")
def test_production_candidate_namespace_cannot_read_research_library(tmp_path):
    from competition_engineering.generated_sandbox import GeneratedSandbox

    library = Path(__file__).resolve().parents[1] / "competition_engineering/research_cards/v1"
    worker = tmp_path / "probe_worker.py"
    worker.write_text("from pathlib import Path\nprint(Path(" + repr(str(library)) + ").exists())\n",
                      encoding="utf-8")
    work = tmp_path / "candidate"
    work.mkdir()
    sandbox = GeneratedSandbox()
    command = sandbox.command(worker, work, "probe")
    result = subprocess.run(command, capture_output=True, text=True, timeout=20,
                            env=sandbox.environment())
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
