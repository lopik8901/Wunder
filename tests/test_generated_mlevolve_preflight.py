"""Synthetic-only preflight for generated architectures and fail-closed policy."""
import json
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))

from connectome.mlevolve_generated import parse_candidate, validate_source
from connectome.mlevolve_bounded import BoundedInterpreter, model_visible_report
from connectome.mlevolve_adapter import parse_result_output, to_mlevolve_feedback
from competition_engineering.generated_runner import run_generated
from competition_engineering.generated_sandbox import GeneratedSandbox, IsolationUnavailable


def test_expanded_search_configuration_is_valid_and_bounded():
    settings = yaml.safe_load((ROOT/"competition_engineering/mlevolve_search_config.yaml").read_text())
    assert settings["generated_enabled"] is True
    assert 1 <= settings["agent"]["steps"] <= 10
    assert 1 <= settings["agent"]["time_limit"] <= 10800
    assert settings["exec"]["timeout"] == 1260
    assert settings["agent"]["search"]["parallel_search_num"] == 1
    assert settings["agent"]["use_global_memory"] is False

TRAIN = '''
import numpy as np
def train(train_files, output_dir):
    gram = np.eye(4, dtype=np.float64) * 0.1
    cross = np.zeros((4, 2), np.float64)
    rows = 0
    for path in train_files:
        with np.load(path) as z:
            x = np.tanh(z["x"][:, :4])[z["mask"]]
            y = z["y"][z["mask"]]
            gram += x.T @ x
            cross += x.T @ y
            rows += len(x)
    coef = np.linalg.solve(gram, cross).astype(np.float32)
    np.savez(output_dir + "/weights.npz", coef=coef)
    return {"training_rows": rows}
'''

CALLBACK = '''
import numpy as np
from pathlib import Path
class PredictionModel:
    def __init__(self):
        with np.load(Path(__file__).with_name("weights.npz")) as z:
            self.coef = z["coef"]
        self.history = np.zeros(4, np.float32)
        self.sequence = None
    def predict(self, point):
        if self.sequence != point.seq_ix:
            self.sequence = point.seq_ix
            self.history.fill(0)
        self.history = 0.8 * self.history + 0.2 * np.tanh(point.state[:4])
        if not point.need_prediction:
            return None
        return (self.history @ self.coef).astype(np.float32)
'''


def caches(tmp_path):
    paths = {}
    rng = np.random.default_rng(42)
    for split, group_count in (("train", 2), ("tune", 2)):
        directory = tmp_path/split
        directory.mkdir()
        for group in range(group_count):
            x = rng.standard_normal((20000, 112)).astype(np.float32)
            y = (np.tanh(x[:, :2]) + 0.05*rng.standard_normal((20000, 2))).astype(np.float32)
            need = np.arange(20000) >= 99
            np.savez(directory/f"{group:05d}.npz", x=x, y=y, need=need,
                     mask=need, seq=group+(0 if split == "train" else 1000))
        (directory/"identity.json").write_text(json.dumps({"split": split, "groups": list(range(group_count))}))
        paths[split] = directory
    return paths


def config(tmp_path, train_source=TRAIN, callback_source=CALLBACK, timeout=30):
    paths = caches(tmp_path)
    return {"hypothesis": "Causal learned exponential state filter with nonlinear features.",
            "train_tier": 256, "train_source": train_source,
            "callback_source": callback_source, "train_dir": str(paths["train"]),
            "search_dir": str(paths["tune"]), "attempt_dir": str(tmp_path/"attempt"),
            "timeout_seconds": timeout}


def test_novel_state_architecture_end_to_end(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    node = "a"*32
    cfg["attempt_dir"] = str(tmp_path/"experiments/generated"/node)
    proposal = "CANDIDATE = " + repr({k:cfg[k] for k in ("hypothesis", "train_tier", "train_source", "callback_source")})
    assert parse_candidate(proposal)["source_sha256"]
    result = run_generated(cfg, sandbox=GeneratedSandbox(synthetic_only=True), synthetic=True)
    assert result["status"] == "success", result.get("failure_reason")
    assert result["search_sequences"] == 2
    assert result["validation"]["causal_prefix"] and result["validation"]["sequence_reset"]
    assert result["validation"]["deterministic"] and result["validation"]["callback_us_per_row"] < 90
    assert set(result["metrics"]) == {"WP_t0", "WP_t1", "combined_WP"}
    assert result["source_sha256"] and result["artifact_sha256"] and result["deployment_sha256"]
    assert (Path(cfg["attempt_dir"])/"candidate.zip").is_file()
    assert (Path(cfg["attempt_dir"])/"identity.json").is_file()
    visible = model_visible_report(result, {"runtime_seconds": result["runtime_seconds"],
        "delta_vs_incumbent_tune": result["primary_metric"]-.6548654996120564,
        "delta_vs_parent_tune": result["primary_metric"]-.6548654996120564,
        "training_diagnostics": result["training_diagnostics"],
        "resource_estimate": result["validation"]})
    feedback = to_mlevolve_feedback(visible)
    assert feedback["status"] == "success" and feedback["metric"] == result["primary_metric"]
    assert "artifact_sha256" not in visible and "source_sha256" not in visible
    # Exercise the actual MLEvolve interpreter branch against the completed
    # synthetic attempt without launching a production untrusted process.
    import connectome.mlevolve_bounded as bounded
    class CompletedProcess:
        def __init__(self, *args, **kwargs):
            self.returncode = 0
        def communicate(self, timeout):
            return "", None
        def poll(self):
            return 0
    monkeypatch.setattr(bounded.subprocess, "Popen", CompletedProcess)
    interpreter = object.__new__(BoundedInterpreter)
    interpreter.root = tmp_path/"experiments"
    interpreter.run_id = "synthetic"
    interpreter.process = None
    interpreter._parent = lambda child: ("root", .6548654996120564)
    execution = interpreter.run(proposal, node)
    parsed = parse_result_output(execution.term_out)
    assert parsed["primary_metric"] == result["primary_metric"]
    assert "artifact_sha256" not in json.dumps(parsed)


@pytest.mark.parametrize("source,training,reason", [
    ('import numpy as np\ndef train(train_files, output_dir):\n np.load("holdout.npz")', True, "protected"),
    ('import socket\ndef train(train_files, output_dir):\n pass', True, "dependency"),
    ('import numpy as np\nclass PredictionModel:\n def predict(self, point):\n  return np.load("../../secret")', False, "escape"),
    ('import os\nclass PredictionModel:\n pass', False, "dependency"),
])
def test_static_gate_rejects_protected_escape_dependencies(source, training, reason):
    with pytest.raises(ValueError):
        validate_source(source, training=training)


def test_future_targets_missing_from_callback_contract(tmp_path):
    callback = CALLBACK.replace('return (self.history @ self.coef).astype(np.float32)',
                                'return point.targets[:2].astype(np.float32)')
    cfg = config(tmp_path, callback_source=callback)
    result = run_generated(cfg, sandbox=GeneratedSandbox(synthetic_only=True), synthetic=True)
    assert result["status"] == "error" and result["stage"] == "validation"
    assert "targets" in result["failure_reason"] or "callback" in result["failure_reason"]
    assert "primary_metric" not in result


def test_malformed_outputs_rejected_before_search(tmp_path):
    callback = CALLBACK.replace('return (self.history @ self.coef).astype(np.float32)',
                                'return np.zeros(3, np.float64)')
    cfg = config(tmp_path, callback_source=callback)
    result = run_generated(cfg, sandbox=GeneratedSandbox(synthetic_only=True), synthetic=True)
    assert result["status"] == "error" and result["stage"] == "validation"
    assert "primary_metric" not in result


@pytest.mark.parametrize("replacement", [
    'return np.array([np.nan, 0], np.float32)',
    'return np.zeros(2, np.float32)',
])
def test_nonfinite_and_warmup_violations(tmp_path, replacement):
    callback = CALLBACK.replace('return (self.history @ self.coef).astype(np.float32)', replacement)
    if "zeros" in replacement:
        callback = callback.replace('return None', 'return np.zeros(2, np.float32)')
    cfg = config(tmp_path, callback_source=callback)
    result = run_generated(cfg, sandbox=GeneratedSandbox(synthetic_only=True), synthetic=True)
    assert result["status"] == "error" and result["stage"] == "validation"
    assert "primary_metric" not in result


def test_timeout_rejected_before_search(tmp_path):
    source = TRAIN.replace('    gram = np.eye(4, dtype=np.float64) * 0.1',
                           '    while True:\n        pass\n    gram = np.eye(4, dtype=np.float64) * 0.1')
    cfg = config(tmp_path, train_source=source, timeout=2)
    result = run_generated(cfg, sandbox=GeneratedSandbox(synthetic_only=True), synthetic=True)
    assert result["status"] == "error" and result["stage"] == "training", result
    assert "primary_metric" not in result


def test_excessive_artifact_size_rejected_before_search(tmp_path, monkeypatch):
    import competition_engineering.generated_runner as runner
    source = TRAIN.replace('    np.savez(output_dir + "/weights.npz", coef=coef)',
                           '    np.savez(output_dir + "/weights.npz", coef=coef)\n    np.save(output_dir + "/oversized.npy", np.zeros(100000, np.float32))')
    cfg = config(tmp_path, train_source=source)
    monkeypatch.setattr(runner, "MAX_TOTAL_WORK_BYTES", 100_000)
    result = run_generated(cfg, sandbox=GeneratedSandbox(synthetic_only=True), synthetic=True)
    assert result["status"] == "error" and result["stage"] == "training"
    assert "primary_metric" not in result


def test_production_fails_closed_without_os_sandbox():
    if sys.platform != "linux":
        with pytest.raises(IsolationUnavailable):
            GeneratedSandbox()


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("bwrap"),
                    reason="requires an installed Linux bubblewrap sandbox")
def test_os_sandbox_cannot_read_unmounted_sentinel(tmp_path):
    sentinel = tmp_path/"outside_secret.npy"
    np.save(sentinel, np.array([314159], np.int64))
    source = TRAIN.replace('    gram = np.eye(4, dtype=np.float64) * 0.1',
                           f'    np.load({str(sentinel)!r})\n    gram = np.eye(4, dtype=np.float64) * 0.1')
    cfg = config(tmp_path, train_source=source)
    result = run_generated(cfg, sandbox=GeneratedSandbox(), synthetic=True)
    assert result["status"] == "error" and result["stage"] == "training", result.get("failure_reason")
    assert "primary_metric" not in result
    np.testing.assert_array_equal(np.load(sentinel), [314159])


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("bwrap"),
                    reason="requires production Linux bubblewrap")
@pytest.mark.parametrize("mode", ["train", "infer"])
def test_production_namespace_denies_host_protected_other_attempt_network_and_devices(tmp_path, mode):
    """Exercise the runner's actual mount command with a trusted adversarial probe."""
    work = tmp_path / "candidate"
    work.mkdir()
    train = tmp_path / "designated_train"
    train.mkdir()
    other = tmp_path / "other_candidate"
    other.mkdir()
    (other / "sentinel").write_text("other candidate")
    (tmp_path / "host_secret").write_text("host secret")
    sandbox = GeneratedSandbox()
    command = sandbox.command(ROOT / "competition_engineering/generated_worker.py", work,
                              mode, train if mode == "train" else None, gpu=False)
    marker = command.index(sys.executable)
    probe = '''
import json, pathlib, socket
paths = {
    "host": %r,
    "other_candidate": %r,
    "protected_holdout": %r,
    "complete_validation": %r,
    "promotion": %r,
    "submission": %r,
    "search_cache": %r,
    "windows_host": "/mnt/c/Windows/System32",
    "gpu_dxg": "/dev/dxg",
    "gpu_kfd": "/dev/kfd",
    "gpu_dri": "/dev/dri",
}
visible = {name: pathlib.Path(path).exists() for name, path in paths.items()}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=1)
    network = True
except OSError:
    network = False
print(json.dumps({"visible": visible, "network": network,
                  "work": pathlib.Path("/work").exists(),
                  "train": pathlib.Path("/data/train").exists()}))
''' % (str(tmp_path / "host_secret"), str(other / "sentinel"),
       str(ROOT / "competition_engineering/cache/gru_holdout"),
       str(ROOT / "competition_engineering/cache/gru_valid"),
       str(ROOT / "competition_engineering/promotions"),
       str(ROOT / "competition_engineering/submissions"),
       str(ROOT / "competition_engineering/cache/gru_tune"))
    command = command[:marker] + [sys.executable, "-I", "-c", probe]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20,
                            env=sandbox.environment())
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert not any(report["visible"].values()), report
    assert report["network"] is False
    assert report["work"] is True
    assert report["train"] is (mode == "train")


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("bwrap"),
                    reason="requires production Linux bubblewrap")
def test_novel_cpu_candidate_end_to_end_in_production_sandbox(tmp_path):
    cfg = config(tmp_path, timeout=120)
    result = run_generated(cfg, sandbox=GeneratedSandbox(), synthetic=True)
    assert result["status"] == "success", result.get("failure_reason")
    assert result["gpu_preflight"]["requested"] is False
    assert result["validation"]["causal_prefix"]
    assert result["validation"]["sequence_reset"]
    assert result["search_sequences"] == 2


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("bwrap"),
                    reason="requires production Linux bubblewrap")
def test_production_workspace_mount_on_wsl_project_drive():
    work = ROOT / "competition_engineering/mlevolve_runs"
    assert work.is_dir()
    GeneratedSandbox().preflight(ROOT / "competition_engineering/generated_worker.py", work)
