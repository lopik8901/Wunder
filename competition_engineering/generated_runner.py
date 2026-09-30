"""Supervisor runner for MLEvolve-authored causal training/callback programs.

Only synthetic fixtures may use the explicit test sandbox. Production always
requires the OS-isolated sandbox and scores only the designated search cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from wnn_connectome_starterpack.utils import DataPoint, GlobalAccumulator

from competition_engineering.generated_sandbox import GeneratedSandbox, IsolationUnavailable
from competition_engineering.generated_worker import REQUEST, FEATURE_BYTES
from competition_engineering.pipeline import cached, write_json
from competition_engineering.run_experiment import validate_inputs
from connectome.mlevolve_generated import validate_source

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("generated_worker.py")
ALLOWED_ARTIFACTS = {".npz", ".npy", ".onnx", ".json", ".txt"}
MAX_PACKAGE_BYTES = 20_000_000
MAX_TOTAL_WORK_BYTES = 128_000_000
MAX_CALLBACK_US = 90.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_failure(exc: BaseException) -> str:
    # Candidate and subprocess exceptions are never allowed to return arbitrary
    # file contents or paths from the host to the MLEvolve planning context.
    message = str(exc).replace(str(ROOT), "<project>")
    return f"{type(exc).__name__}: {message[:300]}"


def _work_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file() and not p.is_symlink())


def _watch(process: subprocess.Popen, work: Path, deadline: float, limit_bytes: int, result: dict):
    while process.poll() is None:
        if time.monotonic() >= deadline:
            result["reason"] = "timeout"
            process.kill()
            break
        try:
            p = psutil.Process(process.pid)
            rss = sum(child.memory_info().rss for child in [p, *p.children(recursive=True)] if child.is_running())
            if rss > limit_bytes:
                result["reason"] = "memory limit"
                process.kill()
                break
            if _work_bytes(work) > MAX_TOTAL_WORK_BYTES:
                result["reason"] = "workspace size limit"
                process.kill()
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(.1)


def _launch(sandbox: GeneratedSandbox, work: Path, mode: str, deadline: float,
            train: Path | None = None, gpu: bool = False) -> tuple[subprocess.Popen, dict, threading.Thread]:
    command = sandbox.command(WORKER, work, mode, train, gpu=gpu)
    environment = sandbox.environment() if not sandbox.synthetic_only else {
        **os.environ, **sandbox.environment(), "HOME": str(work), "TMPDIR": str(work)}
    def limits():
        import resource
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(allowed)})
        resource.setrlimit(resource.RLIMIT_AS, (28_000_000_000 if mode == "train" else 14_000_000_000,)*2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (128_000_000,)*2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (128,)*2)
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (64,)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(deadline-time.monotonic())),)*2)
    if mode == "infer":
        environment.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    environment["CANDIDATE_GPU_REQUIRED"] = "1" if gpu else "0"
    process = subprocess.Popen(command, cwd=work,
                               stdin=subprocess.PIPE if mode == "infer" else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=environment,
                               preexec_fn=limits if sys.platform == "linux" and not sandbox.synthetic_only else None)
    status = {}
    watcher = threading.Thread(target=_watch, args=(process, work, deadline,
                            20_000_000_000 if mode == "train" else 14_000_000_000, status), daemon=True)
    watcher.start()
    return process, status, watcher


def _read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        part = stream.read(count-len(result))
        if not part:
            raise RuntimeError("callback exited or returned incomplete output")
        result.extend(part)
    return bytes(result)


def replay(sandbox: GeneratedSandbox, work: Path, sequences: list[dict], deadline: float):
    process, status, watcher = _launch(sandbox, work, "infer", deadline)
    outputs = []
    started = time.perf_counter()
    try:
        for z in sequences:
            result = np.full((len(z["x"]), 2), np.nan, np.float32)
            seq = int(z["seq"])
            for step, row in enumerate(z["x"]):
                need = bool(z["need"][step])
                process.stdin.write(REQUEST.pack(seq, step, need) + np.asarray(row, dtype="<f4").tobytes())
                process.stdin.flush()
                marker = _read_exact(process.stdout, 1)
                if need:
                    if marker != b"P":
                        raise ValueError("callback omitted required prediction")
                    result[step] = np.frombuffer(_read_exact(process.stdout, 8), dtype="<f4")
                    if not np.isfinite(result[step]).all():
                        raise ValueError("nonfinite callback prediction")
                elif marker != b"N":
                    raise ValueError("callback predicted during warm-up")
            outputs.append(result)
        process.stdin.close()
        process.wait(timeout=max(1, deadline-time.monotonic()))
        stderr = process.stderr.read().decode(errors="replace")[-2000:]
        if process.returncode != 0 or status:
            raise RuntimeError(status.get("reason") or stderr.splitlines()[-1:][0] if stderr else "inference failed")
        timing = None
        for line in stderr.splitlines():
            if line.startswith("CALLBACK_STATS="):
                timing = json.loads(line.removeprefix("CALLBACK_STATS="))
        if timing is None:
            raise RuntimeError("missing trusted callback timing")
        return outputs, {"wall_seconds": time.perf_counter()-started, **timing}
    finally:
        if process.poll() is None:
            process.kill()
        watcher.join(timeout=1)


def _check_artifacts(work: Path, original: dict[str, str]) -> dict[str, str]:
    if _work_bytes(work) > MAX_TOTAL_WORK_BYTES:
        raise ValueError("workspace size limit exceeded")
    for name, expected in original.items():
        if digest(work/name) != expected:
            raise ValueError(f"candidate changed immutable source {name}")
    artifact_dir = work / "artifacts"
    artifacts = sorted(artifact_dir.rglob("*"))
    if not artifacts or len(artifacts) > 32:
        raise ValueError("candidate must produce 1-32 artifact files")
    identities = {}
    for path in artifacts:
        if not path.is_file() or path.is_symlink() or path.suffix not in ALLOWED_ARTIFACTS:
            raise ValueError("unsupported artifact file or symlink")
        if path.parent != artifact_dir:
            raise ValueError("nested artifact paths are unsupported")
        identities[path.name] = digest(path)
    deploy = work / "deploy"
    deploy.mkdir(exist_ok=False)
    shutil.copyfile(work/"callback.py", deploy/"solution.py")
    for path in artifacts:
        shutil.copyfile(path, deploy/path.name)
    return identities


def _package(work: Path) -> tuple[Path, dict[str, str]]:
    files = sorted((work/"deploy").iterdir())
    identities = {p.name: digest(p) for p in files}
    archive = work/"candidate.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            z.write(path, path.name)
    if archive.stat().st_size > MAX_PACKAGE_BYTES:
        raise ValueError("deployment ZIP exceeds 20 MB")
    return archive, identities


def _probe_data(train: Path) -> list[dict]:
    _, z = next(cached(train))
    n = min(len(z["x"]), 512)
    original = {"seq": int(z["seq"]), "x": z["x"][:n], "need": z["need"][:n]}
    other = {"seq": int(z["seq"])+10_000_000, "x": z["x"][:n], "need": z["need"][:n]}
    complete = {"seq": int(z["seq"])+20_000_000, "x": z["x"], "need": z["need"]}
    return [original, other, complete]


def validate_callback(sandbox: GeneratedSandbox, work: Path, train: Path, deadline: float) -> dict:
    a, b, complete = _probe_data(train)
    first, timing = replay(sandbox, work, [a], deadline)
    repeated, _ = replay(sandbox, work, [a], deadline)
    if not np.array_equal(first[0][a["need"]], repeated[0][a["need"]]):
        raise ValueError("nondeterministic callback replay")
    reset, _ = replay(sandbox, work, [a, b, a], deadline)
    if not np.array_equal(first[0][a["need"]], reset[2][a["need"]]):
        raise ValueError("sequence state failed to reset; reset when point.seq_ix changes or point.step_in_seq is zero")
    # Future rows are not sent until the current prediction is received. Also
    # check a mutated suffix cannot affect its already-produced prefix.
    changed = {**a, "x": a["x"].copy()}
    changed["x"][len(a["x"])//2:] += 1
    modified, _ = replay(sandbox, work, [changed], deadline)
    mid = len(a["x"])//2
    if not np.array_equal(first[0][:mid], modified[0][:mid], equal_nan=True):
        raise ValueError("causal-prefix invariance failed")
    _, benchmark = replay(sandbox, work, [complete], deadline)
    callback_us = benchmark["callback_seconds"] / benchmark["rows"] * 1e6
    if callback_us > MAX_CALLBACK_US:
        raise ValueError(f"callback throughput {callback_us:.1f} us/row exceeds {MAX_CALLBACK_US:.0f} us/row gate")
    return {"causal_prefix": True, "sequence_reset": True, "deterministic": True,
            "warmup_and_output_contract": True, "benchmark_rows": benchmark["rows"],
            "callback_us_per_row": callback_us, "wall_us_per_row": benchmark["wall_seconds"]/benchmark["rows"]*1e6}


def run_generated(config: dict, *, sandbox: GeneratedSandbox | None = None, synthetic: bool = False) -> dict:
    started = time.perf_counter()
    timeout = int(config.get("timeout_seconds", 1200))
    deadline = time.monotonic()+timeout
    work = Path(config["attempt_dir"]).resolve()
    if work.exists():
        raise ValueError("generated attempt directory already exists")
    if not synthetic:
        expected_root = ROOT / "competition_engineering/mlevolve_runs"
        if not work.is_relative_to(expected_root):
            raise ValueError("generated attempt must live in the MLEvolve run tree")
        # Absolute caller-supplied paths cannot redirect a production run.
        train = ROOT / ("competition_engineering/cache/gru_train_pilot" if config["train_tier"] == 256 else
                        "competition_engineering/cache/gru_train_scale_phase2")
        search = ROOT / "competition_engineering/cache/gru_tune"
        validate_inputs({"train": str(train), "tune": str(search)})
    else:
        train, search = Path(config["train_dir"]).resolve(), Path(config["search_dir"]).resolve()
    sandbox = sandbox or GeneratedSandbox()
    work.mkdir(parents=True)
    source = {"train.py": config["train_source"], "callback.py": config["callback_source"]}
    for name, body in source.items():
        validate_source(body, training=name == "train.py")
        (work/name).write_text(body, encoding="utf-8")
    source_hashes = {name: digest(work/name) for name in source}
    report = {"status": "error", "stage": "preflight", "metric_name": "combined_WP", "maximize": True,
              "metric_domain": "fixed development tuning sequences", "hypothesis": config["hypothesis"],
              "train_tier": config["train_tier"], "source_sha256": source_hashes,
              "failure_reason": None}
    try:
        sandbox.preflight(WORKER, work)
        report["stage"] = "training"
        process, status, watcher = _launch(sandbox, work, "train", deadline, train,
                                          gpu=config.get("training_device", "cpu") == "gpu")
        try:
            out, err = process.communicate(timeout=max(1, deadline-time.monotonic()))
        finally:
            if process.poll() is None:
                process.kill()
            watcher.join(timeout=1)
        if process.returncode != 0 or status:
            raise RuntimeError(status.get("reason") or err.decode(errors="replace").splitlines()[-1:][0] if err else "training failed")
        diagnostics = {}
        gpu_probe = {}
        for line in out.decode(errors="replace").splitlines():
            if line.startswith("TRAIN_RESULT="):
                diagnostics = json.loads(line.removeprefix("TRAIN_RESULT="))
            elif line.startswith("TRAIN_ENV="):
                gpu_probe = json.loads(line.removeprefix("TRAIN_ENV="))
        artifacts = _check_artifacts(work, source_hashes)
        archive, deploy_hashes = _package(work)
        report.update(stage="validation", artifact_sha256=artifacts,
                      deployment_sha256=deploy_hashes, package_sha256=digest(archive),
                      package_bytes=archive.stat().st_size, gpu_preflight=gpu_probe)
        checks = validate_callback(sandbox, work, train, deadline)
        report["validation"] = checks
        report["stage"] = "search_evaluation"
        accumulator = GlobalAccumulator()
        from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
        if synthetic:
            error_diagnostics = SearchErrorDiagnostics(root_predictor=lambda z: np.zeros_like(z["y"], dtype=np.float32))
        else:
            error_diagnostics = SearchErrorDiagnostics()
        count = 0
        for _, z in cached(search):
            predictions, _ = replay(sandbox, work, [z], deadline)
            accumulator.add(z["y"], predictions[0], z["mask"])
            error_diagnostics.add(z, predictions[0])
            count += 1
        scores = accumulator.result()
        metric = (scores["t0"]+scores["t1"])/2
        for name, expected in deploy_hashes.items():
            if digest(work/"deploy"/name) != expected:
                raise ValueError("deployment file changed after validation")
        if digest(archive) != report["package_sha256"]:
            raise ValueError("candidate ZIP changed after validation")
        report.update(status="success", stage="complete", primary_metric=metric,
                      metrics={"WP_t0": scores["t0"], "WP_t1": scores["t1"], "combined_WP": metric},
                      training_diagnostics={k:v for k,v in diagnostics.items() if k in {"training_rows", "train_seconds", "updates", "loss_start", "loss_end"}},
                      search_error_diagnostics=error_diagnostics.result(),
                      search_sequences=count, artifact_bytes=sum((work/"artifacts"/name).stat().st_size for name in artifacts))
    except Exception as exc:
        report["failure_reason"] = safe_failure(exc)
    report["runtime_seconds"] = time.perf_counter()-started
    write_json(work/"result.json", report)
    write_json(work/"identity.json", {"source_sha256": source_hashes,
        "artifact_sha256": report.get("artifact_sha256"),
        "deployment_sha256": report.get("deployment_sha256"),
        "package_sha256": report.get("package_sha256"),
        "search_domain": "fixed development tuning sequences"})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = run_generated(json.loads(args.config.read_text(encoding="utf-8")))
    print("CONNECTOME_RESULT_JSON="+json.dumps(result, sort_keys=True))
    sys.exit(0 if result["status"] == "success" else 1)
