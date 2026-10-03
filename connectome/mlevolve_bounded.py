"""Execute MLEvolve's experiment choices through the fixed Connectome runner.

Generated Python is parsed as a literal experiment specification and is never
executed. This keeps the existing MLEvolve planner and search tree in charge of
choosing experiments while preserving the data and resource boundary.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from engine.executor import ExecutionResult
from connectome.mlevolve_generated import parse_candidate
from connectome.mlevolve_history import load_prior_search_records, format_search_history


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "competition_engineering"
INCUMBENT_TUNE_WP = 0.6548654996120564
RESULT_PREFIX = "CONNECTOME_RESULT_JSON="
_SEARCH_DOMAIN = "fixed development tuning sequences"
_TRAIN_KEYS = {"training_rows", "train_seconds", "updates", "loss_first80", "loss_last80",
               "sample_stride", "ridge_penalty", "target_mode", "rows_per_second"}
_RESOURCE_KEYS = {"peak_gpu_allocated_bytes", "artifact_bytes", "cpu_microseconds_per_row",
                  "candidate_wall_seconds"}


def wsl_path(path: Path) -> str:
    """Map a Windows workspace path into WSL without shell interpolation."""
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError("WSL bridge requires a local drive path")
    return f"/mnt/{drive}/" + resolved.as_posix().split(":/", 1)[1]


def wsl_python_command() -> list[str]:
    """Use an explicit interpreter, or the current Ubuntu user's venv."""
    configured = os.environ.get("WUNDER_WSL_PYTHON")
    if configured:
        return [configured]
    return ["sh", "-c", 'exec "$HOME/.venvs/wunder311/bin/python" "$@"', "wunder-python"]


def parse_spec(code: str) -> dict:
    if len(code) > 16000:
        raise ValueError("experiment specification is too long")
    tree = ast.parse(code)
    body = [node for node in tree.body if not isinstance(node, ast.Expr)
            or not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str)]
    if len(body) != 1 or not isinstance(body[0], ast.Assign):
        raise ValueError("provide only EXPERIMENT = {...}; generated code is not executed")
    assignment = body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "EXPERIMENT":
        raise ValueError("expected one literal EXPERIMENT assignment")
    spec = ast.literal_eval(assignment.value)
    if not isinstance(spec, dict):
        raise ValueError("EXPERIMENT must be a dictionary")
    if len(json.dumps(spec)) > 4096:
        raise ValueError("experiment dictionary is too large")
    return spec


def is_generated_candidate(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CANDIDATE" for t in node.targets)
               for node in tree.body)


def validated_config(spec: dict, node_id: str, run_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", node_id):
        raise ValueError("invalid MLEvolve node ID")
    allowed = {"kind", "hypothesis", "train_tier", "target_mode", "sample_stride",
               "ridge_penalty", "epochs", "calibration_scale", "calibration_bias",
               "base_model", "mixed_precision"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unsupported experiment fields: {sorted(unknown)}")
    kind = spec.get("kind")
    if kind not in {"ridge", "gpu_residual", "calibration"}:
        raise ValueError("kind must be ridge, gpu_residual, or calibration")
    hypothesis = spec.get("hypothesis")
    if not isinstance(hypothesis, str) or not 10 <= len(hypothesis) <= 400:
        raise ValueError("give a concrete hypothesis of 10-400 characters")
    tier = spec.get("train_tier", 1024)
    if type(tier) is not int or tier not in {256, 1024}:
        raise ValueError("train_tier must be 256 or 1024")
    mode = spec.get("target_mode", "raw")
    if mode not in {"raw", "clipped", "t0_raw_t1_clipped", "t0_clipped_t1_raw"}:
        raise ValueError("unsupported target_mode")
    if kind == "gpu_residual" and mode not in {"raw", "clipped"}:
        raise ValueError("GPU TCN currently supports raw or clipped target mode")
    config = {
        "kind": kind,
        "train": f"competition_engineering/cache/gru_{'train_pilot' if tier == 256 else 'train_scale_phase2'}",
        "tune": "competition_engineering/cache/gru_tune",
        "output": f"competition_engineering/checkpoints/mlevolve_{run_id}_{node_id}",
        "target_mode": mode,
        "base_scale": [0.75, 0.75],
        "base_bias": [-0.1, -0.1],
    }
    if kind in {"ridge", "calibration"}:
        if kind == "calibration" and any(k in spec for k in ("sample_stride", "ridge_penalty", "target_mode")):
            raise ValueError("calibration does not train a residual model")
        if kind == "ridge":
            stride = spec.get("sample_stride", 10)
            penalty = spec.get("ridge_penalty", 1e-3)
            if type(stride) is not int or stride not in {5, 10, 20, 40}:
                raise ValueError("sample_stride must be 5, 10, 20, or 40")
            if type(penalty) not in {int, float} or float(penalty) not in {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}:
                raise ValueError("ridge_penalty must be one of 1e-5, 1e-4, 1e-3, 1e-2, 1e-1")
            config.update(sample_stride=stride, ridge_penalty=float(penalty))
        for spec_key, config_key, low, high in [
            ("calibration_scale", "base_scale", .5, 1.25),
            ("calibration_bias", "base_bias", -.25, .25),
        ]:
            if spec_key in spec:
                values = spec[spec_key]
                if not isinstance(values, (list, tuple)) or len(values) != 2 or any(
                    type(v) not in {int, float} or not low <= float(v) <= high for v in values
                ):
                    raise ValueError(f"{spec_key} must contain two bounded numbers")
                config[config_key] = [float(v) for v in values]
        if "epochs" in spec:
            raise ValueError("epochs applies only to gpu_residual")
        if "base_model" in spec or "mixed_precision" in spec:
            raise ValueError("GPU base_model and mixed_precision apply only to gpu_residual")
    else:
        if any(k in spec for k in ("sample_stride", "ridge_penalty", "calibration_scale", "calibration_bias")):
            raise ValueError("ridge-only fields are unsupported for gpu_residual")
        epochs = spec.get("epochs", 1)
        if type(epochs) is not int or epochs not in {1, 2, 3}:
            raise ValueError("epochs must be 1, 2, or 3")
        base_model = spec.get("base_model", "incumbent_ridge")
        if base_model not in {"gru", "incumbent_ridge", "pilot_best_ridge"}:
            raise ValueError("unsupported GPU residual base_model")
        amp = spec.get("mixed_precision", False)
        if type(amp) is not bool:
            raise ValueError("mixed_precision must be boolean")
        config.update(epochs=epochs, amp=amp,
                      gru_solution="competition_engineering/assets/wnn_connectome_starterpack/baseline/solution.py")
        if base_model != "gru":
            config["base_ridge"] = ("competition_engineering/checkpoints/ridge1024_targetwise_v1/ridge.npz"
                                    if base_model == "incumbent_ridge" else
                                    "competition_engineering/checkpoints/mlevolve_best_20260929_005856/ridge.npz")
    return config


def model_visible_report(report: dict, record: dict) -> dict:
    """Only the search metric and training facts may reach MLEvolve prompts."""
    if report["status"] != "success":
        raw_stage = str(report.get("stage", "")).lower()
        if report["status"] == "timeout" or "timeout" in raw_stage:
            stage, error = "candidate timeout", "Candidate did not complete within its runtime budget."
        elif "valid" in raw_stage:
            stage, error = "candidate validation", "Candidate failed a pre-score validation gate."
        elif "train" in raw_stage or "evaluation" in raw_stage:
            stage, error = "training or search evaluation", "Candidate failed during training or search-only evaluation."
        else:
            stage, error = "candidate specification or launcher", "Candidate did not complete in the bounded runner."
        return {"status": "timeout" if report["status"] == "timeout" else "error",
                "metric_name": "combined_WP", "maximize": True,
                "stage": stage, "error": error}
    from competition_engineering.search_error_diagnostics import sanitize_search_error_diagnostics
    metrics = report.get("metrics")
    if report.get("metric_domain") != _SEARCH_DOMAIN or not isinstance(metrics, dict) or set(metrics) != {"WP_t0", "WP_t1", "combined_WP"}:
        return {"status": "error", "metric_name": "combined_WP", "maximize": True,
                "stage": "search result contract", "error": "candidate score withheld because the result was outside the fixed search-only metric contract"}
    metric_values = [metrics[key] for key in ("WP_t0", "WP_t1", "combined_WP")]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in metric_values):
        return {"status": "error", "metric_name": "combined_WP", "maximize": True,
                "stage": "search result contract", "error": "candidate score withheld because search metrics were invalid"}
    if (not math.isclose(metrics["combined_WP"], (metrics["WP_t0"] + metrics["WP_t1"]) / 2.0,
                         rel_tol=0.0, abs_tol=1e-12) or
            not math.isclose(report.get("primary_metric", math.nan), metrics["combined_WP"],
                             rel_tol=0.0, abs_tol=1e-12)):
        return {"status": "error", "metric_name": "combined_WP", "maximize": True,
                "stage": "search result contract", "error": "candidate score withheld because search metrics were inconsistent"}
    diagnostics = sanitize_search_error_diagnostics(report.get("search_error_diagnostics"))
    if diagnostics is None:
        return {"status": "error", "metric_name": "combined_WP", "maximize": True,
                "stage": "search error diagnostics contract",
                "error": "candidate score withheld because fixed-schema search error diagnostics are missing or invalid"}
    visible = {"status": "success", "metric_name": "combined_WP", "maximize": True,
               "primary_metric": float(metrics["combined_WP"]),
               "metrics": {key: float(metrics[key]) for key in ("WP_t0", "WP_t1", "combined_WP")},
               "metric_domain": _SEARCH_DOMAIN,
               "search_error_diagnostics": diagnostics}
    rows = report.get("training_sequences")
    if type(rows) is int and rows >= 0:
        visible["training_sequences"] = rows
    kind = report.get("experiment_kind")
    if kind in {"ridge", "gpu_residual", "calibration", "generated"}:
        visible["experiment_kind"] = kind
    for key, value in (("runtime_seconds", record.get("runtime_seconds")),
                       ("delta_vs_incumbent_tune", record.get("delta_vs_incumbent_tune")),
                       ("delta_vs_parent_tune", record.get("delta_vs_parent_tune"))):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            visible[key] = float(value)
    for field, allowed in (("training_diagnostics", _TRAIN_KEYS), ("resource_estimate", _RESOURCE_KEYS)):
        values = record.get(field)
        if isinstance(values, dict):
            clean = {}
            for key, value in values.items():
                if key not in allowed:
                    continue
                if key == "target_mode" and value in {"raw", "clipped", "t0_raw_t1_clipped", "t0_clipped_t1_raw"}:
                    clean[key] = value
                elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                    clean[key] = value
            if clean:
                visible[field] = clean
    return visible


class BoundedInterpreter:
    max_parallel_run = 1

    def __init__(self, cfg):
        self.cfg = cfg
        self.run_id = cfg.exp_name.replace("_", "")[:18]
        self.root = Path(cfg.log_dir).parent / "experiments"
        self.root.mkdir(parents=True, exist_ok=False)
        self.process = None
        self.agent = None
        runs_dir = Path(cfg.log_dir).parent.parent
        self.search_records = load_prior_search_records(runs_dir)

    def planning_history(self):
        """Return compact, search-only history for the next MLEvolve decision."""
        from connectome.mlevolve_history import compact_search_history
        return compact_search_history(self.search_records)

    def _parent(self, node_id):
        if self.agent is None:
            raise RuntimeError("search tree is unavailable")
        for branch in self.agent.branch_all_nodes.values():
            for node in branch:
                if node.id == node_id and node.parent is not None:
                    parent = node.parent
                    metric = parent.metric.value if parent.metric and parent.metric.value is not None else None
                    return parent.id, float(metric) if metric is not None else None
        raise RuntimeError("candidate not registered in scored search tree")

    def run(self, code: str, id, reset_session=True):
        del reset_session
        started = time.perf_counter()
        node_id = str(id)
        parent_id, parent_wp = self._parent(node_id)
        spec = None
        config = None
        report = None
        output = ""
        try:
            generated = is_generated_candidate(code)
            host_attempt_dir = self.root / "generated" / node_id if generated else None
            if generated:
                spec = parse_candidate(code)
                config = {"kind": "generated", "hypothesis": spec["hypothesis"],
                          "train_tier": spec["train_tier"],
                          "training_device": spec.get("training_device", "cpu"),
                          "train_source": spec["train_source"],
                          "callback_source": spec["callback_source"],
                          "attempt_dir": wsl_path(host_attempt_dir),
                          "timeout_seconds": 1200}
            else:
                spec = parse_spec(code)
                config = validated_config(spec, node_id, self.run_id)
            config_dir = self.root / "configs"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / f"{node_id}.json"
            config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            if generated and os.name == "nt":
                command = ["wsl", "-d", "Ubuntu", "--cd", wsl_path(ROOT), "--exec",
                           *wsl_python_command(), "-m", "competition_engineering.generated_runner",
                           wsl_path(config_file)]
            elif generated:
                command = [sys.executable, "-m", "competition_engineering.generated_runner", str(config_file)]
            else:
                command = [sys.executable, "-m", "competition_engineering.run_experiment",
                           str(config_file), "--timeout", "600"]
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                            errors="replace")
            try:
                output, _ = self.process.communicate(timeout=1260 if generated else 660)
            except subprocess.TimeoutExpired:
                self.process.kill()
                output, _ = self.process.communicate()
                raise TimeoutError("outer runner watchdog exceeded its candidate budget")
            result_path = (host_attempt_dir / "result.json" if generated else
                           ENGINE / "runs" / Path(config["output"]).name / "result.json")
            if result_path.exists():
                report = json.loads(result_path.read_text(encoding="utf-8"))
            if report is None:
                raise RuntimeError("candidate runner produced no result.json")
            if report.get("status") == "success":
                from competition_engineering.search_error_diagnostics import sanitize_search_error_diagnostics
                diagnostics = sanitize_search_error_diagnostics(report.get("search_error_diagnostics"))
                if diagnostics is None:
                    report = {"status": "error", "metric_name": "combined_WP", "maximize": True,
                              "stage": "search error diagnostics contract",
                              "error": "score withheld because fixed-schema candidate-vs-root diagnostics are missing or invalid"}
                else:
                    report["search_error_diagnostics"] = diagnostics
            if report.get("status") != "success":
                report["error"] = report.get("failure_reason", report.get("error", "candidate failed"))
            if report.get("status") != "success" and not any(
                line.startswith(RESULT_PREFIX) for line in output.splitlines()
            ):
                output += "\n" + RESULT_PREFIX + json.dumps(report) + "\n"
        except Exception as exc:
            report = {"status": "error", "metric_name": "combined_WP", "maximize": True,
                      "stage": "bounded specification or launcher", "error": repr(exc)}
            output += "\n" + RESULT_PREFIX + json.dumps(report) + "\n"
        finally:
            self.process = None
        elapsed = time.perf_counter() - started
        record = {"node_id": node_id, "parent_id": parent_id, "parent_tune_wp": parent_wp,
                  "incumbent_tune_wp": INCUMBENT_TUNE_WP, "spec": spec, "config": config,
                  "status": report["status"], "runtime_seconds": elapsed,
                  "failure_reason": None, "result": {},
                  "delta_vs_incumbent_tune": None, "delta_vs_parent_tune": None}
        # Decision provenance is search-only and remains in the supervisor
        # journal. The compact planner history intentionally does not replay
        # card text or this trace into subsequent planning prompts.
        if getattr(self, "agent", None) is not None:
            for branch in self.agent.branch_all_nodes.values():
                match = next((node for node in branch if node.id == node_id), None)
                if match is not None:
                    record["research_trace"] = getattr(match, "connectome_research_trace", None)
                    break
        if report["status"] == "success":
            metric = report["primary_metric"]
            record["delta_vs_incumbent_tune"] = metric - INCUMBENT_TUNE_WP
            record["delta_vs_parent_tune"] = metric - parent_wp if parent_wp is not None else None
            if config["kind"] == "generated":
                record["training_diagnostics"] = report.get("training_diagnostics")
            else:
                trained = ENGINE / "checkpoints" / Path(config["output"]).name / "report.json"
                if trained.exists():
                    training = json.loads(trained.read_text(encoding="utf-8"))
                    record["training_diagnostics"] = {key: training.get(key) for key in
                        ("training_rows", "train_seconds", "loss_first80", "loss_last80",
                         "sample_stride", "ridge_penalty", "target_mode", "updates", "rows_per_second")}
            record["resource_estimate"] = {
                "peak_gpu_allocated_bytes": report.get("peak_gpu_allocated_bytes"),
                "artifact_bytes": report.get("artifact_bytes", report.get("package_bytes")),
                "cpu_microseconds_per_row": report.get("cpu_microseconds_per_row", report.get("validation", {}).get("callback_us_per_row")),
                "candidate_wall_seconds": elapsed,
            }
        visible = model_visible_report(report, record)
        record["result"] = visible
        if report["status"] != "success":
            record["failure_reason"] = visible["stage"]
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, default=str) + "\n")
        if not hasattr(self, "search_records"):
            self.search_records = []
        self.search_records.append(record)
        # Persist only the same allowlisted record MLEvolve receives. Raw
        # candidate stdout can contain arbitrary candidate-controlled material.
        (self.root / f"{node_id}.stdout.log").write_text(
            RESULT_PREFIX + json.dumps(visible, sort_keys=True) + "\n", encoding="utf-8")
        # MLEvolve's improve/debug prompts inspect node.term_out as well as the
        # parsed feedback, so both channels use this search-only projection.
        model_output = RESULT_PREFIX + json.dumps(visible, sort_keys=True) + "\n"
        return ExecutionResult(term_out=[model_output], exec_time=elapsed, exc_type=None,
                               exc_info=None, exc_stack=None)

    def terminate_all_subprocesses(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()

    def cleanup_session(self, process_id=-1):
        del process_id
