"""Preflight the actual literal-spec and model-visible search boundary."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))

from connectome.mlevolve_bounded import BoundedInterpreter, model_visible_report, parse_spec, validated_config
from connectome.mlevolve_incumbent import seed_incumbent
from competition_engineering.run_experiment import validate_inputs
from competition_engineering import run_experiment
from competition_engineering.promote_candidate import promote
from engine.search_node import Journal, SearchNode


def test_planning_description_and_pilot_memory_have_only_search_numbers():
    description = (ROOT / "competition_engineering/mlevolve_task/description.md").read_text(encoding="utf-8")
    for line in description.splitlines():
        if any(domain in line.lower() for domain in ("holdout", "promotion", "complete validation", "leaderboard")):
            assert re.search(r"\b0\.\d{3,}\b", line) is None
    for search in ("0.6548654996120564", "0.654845419", "0.655221919", "0.656311883", "0.656343684"):
        assert search in description
    config = (ROOT / "competition_engineering/mlevolve_search_config.yaml").read_text(encoding="utf-8")
    assert "use_global_memory: false" in config
    assert "use_coldstart: false" in config
    assert sorted(path.name for path in (ROOT / "competition_engineering/mlevolve_task").iterdir()) == ["description.md"]


def test_synthetic_spec_cannot_request_protected_evaluation():
    node_id = "a" * 32
    base = {"kind": "ridge", "hypothesis": "Synthetic bounded search preflight."}
    for forbidden in ("confirm", "holdout", "promotion", "final_gate", "leaderboard"):
        with pytest.raises(ValueError, match="unsupported experiment fields"):
            validated_config({**base, forbidden: True}, node_id, "preflight")
    for kind in ("ridge", "calibration", "gpu_residual"):
        config = validated_config({**base, "kind": kind}, node_id, "preflight")
        assert "confirm" not in config
        assert config["tune"].endswith("gru_tune")
    with pytest.raises(ValueError, match="only EXPERIMENT"):
        parse_spec("EXPERIMENT = {}\nopen('holdout')")
    with pytest.raises(ValueError, match="routine search"):
        validate_inputs({"confirm": "protected"})


def test_result_object_excludes_protected_fields_even_if_runner_regresses():
    report = {"status": "success", "metric_name": "combined_WP", "maximize": True,
              "primary_metric": .66, "metrics": {"WP_t0": .65, "WP_t1": .67, "combined_WP": .66},
              "metric_domain": "fixed development tuning sequences",
              "heldout_diagnostics": {"sentinel": "PROTECTED_SENTINEL"},
              "leaderboard_score": "PROTECTED_SENTINEL", "complete_validation": "PROTECTED_SENTINEL"}
    record = {"runtime_seconds": 1, "delta_vs_incumbent_tune": .005,
              "delta_vs_parent_tune": .005}
    assert "PROTECTED_SENTINEL" not in json.dumps(model_visible_report(report, record))


def test_scored_root_is_actual_candidate_parent():
    root = SearchNode(code="", stage="root")
    journal = Journal()
    journal.append(root)
    agent = SimpleNamespace(virtual_root=root, branch_all_nodes={}, branch_successful_nodes={},
                            top_candidates=[], best_node=None, best_metric=None)
    seed_incumbent(agent, journal)
    child = SearchNode(code="EXPERIMENT = {}", stage="draft", parent=root)
    assert child.parent is root and root.metric.value == .6548654996120564
    assert len(journal) == 1 and agent.branch_successful_nodes[0] == [root]
    agent.branch_all_nodes[1] = [child]
    interpreter = object.__new__(BoundedInterpreter)
    interpreter.agent = agent
    assert interpreter._parent(child.id) == (root.id, root.metric.value)
    assert "holdout" not in root.analysis.lower()
    failed_child = SearchNode(code="EXPERIMENT = {}", stage="debug", parent=child)
    agent.branch_all_nodes[1].append(failed_child)
    assert interpreter._parent(failed_child.id) == (child.id, None)


def test_synthetic_routine_execution_never_opens_or_reports_holdout(tmp_path, monkeypatch, capsys):
    import numpy as np
    paths = {}
    for name, first, count in (("train", 0, 256), ("tune", 300, 2)):
        directory = tmp_path / name
        directory.mkdir()
        for group in range(count):
            np.savez(directory / f"{group:05d}.npz", seq=first + group)
        (directory / "identity.json").write_text(json.dumps({
            "split": "train" if name == "train" else "tune",
            "groups": list(range(count)),
        }), encoding="utf-8")
        paths[name] = str(directory)
    protected = tmp_path / "PROTECTED_SENTINEL_holdout"
    protected.mkdir()
    (protected / "identity.json").write_text("invalid protected cache", encoding="utf-8")
    (tmp_path / "competition_engineering").mkdir()
    (tmp_path / "competition_engineering/residual.py").write_text("# synthetic trainer", encoding="utf-8")
    monkeypatch.setattr(run_experiment, "ROOT", tmp_path)
    monkeypatch.setattr(run_experiment.shutil, "copy2", lambda *args: None)
    commands = []
    def fake_train(command, **kwargs):
        commands.append(command)
        assert "PROTECTED_SENTINEL" not in " ".join(command)
        output = tmp_path / "competition_engineering/checkpoints/synthetic"
        output.mkdir(parents=True)
        np.savez(output / "ridge.npz", coefficient=np.array([1.0]))
        (output / "report.json").write_text(json.dumps({
            "tune": {"grid": [{"t0": .65, "t1": .67}]}, "training_rows": 10,
        }), encoding="utf-8")
    monkeypatch.setattr(run_experiment.subprocess, "run", fake_train)
    config = {**paths, "kind": "ridge", "output": "competition_engineering/checkpoints/synthetic"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert run_experiment.run(config_path, 10)
    report_path = tmp_path / "competition_engineering/runs/synthetic/result.json"
    for visible in (capsys.readouterr().out, report_path.read_text(encoding="utf-8"),
                    (report_path.parent / "training.log").read_text(encoding="utf-8")):
        assert "PROTECTED_SENTINEL" not in visible
        assert "holdout" not in visible.lower()
    assert len(commands) == 1


def test_promotion_gate_rejects_ineligible_candidate_before_holdout(tmp_path):
    candidate = tmp_path / "result.json"
    candidate.write_text(json.dumps({"status": "success",
                                     "metric_domain": "fixed development tuning sequences",
                                     "primary_metric": .6549}), encoding="utf-8")
    with pytest.raises(ValueError, match="predetermined search gain"):
        promote(candidate, tmp_path / "promotion.json")
    assert not (tmp_path / "promotion.json").exists()
