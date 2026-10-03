"""Fail-closed OS isolation for MLEvolve-generated programs.

Production requires Linux bubblewrap/user namespaces. Windows synthetic mode
is only for disposable preflight fixtures and is never callable by the search
adapter. The production sandbox mounts no project tree, protected data, search
targets, journals, other candidate workspaces, or network.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class IsolationUnavailable(RuntimeError):
    pass


class GeneratedSandbox:
    def __init__(self, *, synthetic_only: bool = False):
        self.synthetic_only = synthetic_only
        self.bwrap = shutil.which("bwrap") if sys.platform == "linux" else None
        if not synthetic_only and not self.bwrap:
            raise IsolationUnavailable("generated-code search requires a verified Linux bubblewrap sandbox; this host cannot isolate protected files")

    def command(self, worker: Path, work: Path, mode: str, train: Path | None = None, gpu: bool = False) -> list[str]:
        worker, work = worker.resolve(), work.resolve()
        if self.synthetic_only:
            return [sys.executable, "-I", str(worker), mode, str(work), *([str(train.resolve())] if train else [])]
        if sys.platform != "linux" or not self.bwrap:
            raise IsolationUnavailable("production sandbox unavailable")
        # Only interpreter/runtime libraries, one candidate workspace and the
        # designated training cache are visible. Search cache is never mounted.
        args = [self.bwrap, "--die-with-parent", "--new-session", "--unshare-user",
                "--unshare-pid", "--unshare-ipc", "--unshare-net", "--unshare-uts",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for path in ("/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache", "/etc/localtime"):
            if Path(path).exists():
                args += ["--ro-bind", path, path]
        runtime_paths = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
        if Path("/opt/rocm").exists():
            runtime_paths.add(Path("/opt/rocm"))
        runtime_paths = {p for p in runtime_paths if not p.is_relative_to(Path("/usr"))
                         and not p.is_relative_to(Path("/lib"))}
        parents = {parent for path in runtime_paths for parent in path.parents
                   if parent != Path("/") and not parent.is_relative_to(Path("/usr"))
                   and not parent.is_relative_to(Path("/lib"))}
        for parent in sorted(parents, key=lambda p: len(p.parts)):
            args += ["--dir", str(parent)]
        for path in sorted(runtime_paths, key=lambda p: len(p.parts)):
            args += ["--ro-bind", str(path), str(path)]
        # A managed Python venv may point at a version alias (for example uv's
        # cpython-3.11-linux-x86_64-gnu) while sys.base_prefix names the
        # resolved version directory. Recreate only that alias in the sandbox;
        # never bind its parent or any part of the host home directory.
        executable = Path(sys.executable)
        if executable.is_symlink():
            target = executable.readlink()
            for alias in (target, *target.parents):
                if alias.is_symlink() and alias.resolve().is_relative_to(Path(sys.base_prefix).resolve()):
                    args += ["--symlink", str(alias.resolve()), str(alias)]
        args += ["--dir", "/runtime", "--ro-bind", str(worker), "/runtime/worker.py",
                 "--dir", "/work", "--bind" if mode == "train" else "--ro-bind", str(work), "/work"]
        if train is not None:
            args += ["--dir", "/data", "--ro-bind", str(train.resolve()), "/data/train"]
        if mode == "train" and gpu and Path("/dev/kfd").exists():
            args += ["--dev-bind", "/dev/kfd", "/dev/kfd"]
            if Path("/dev/dri").exists():
                args += ["--dev-bind", "/dev/dri", "/dev/dri"]
        if mode == "train" and gpu and Path("/dev/dxg").exists():
            args += ["--dev-bind", "/dev/dxg", "/dev/dxg"]
        if mode == "train" and gpu:
            for device in sorted(Path("/dev").glob("nvidia*")):
                args += ["--dev-bind", str(device), str(device)]
        args += ["--chdir", "/tmp" if mode == "infer" else "/work",
                 sys.executable, "-I", "/runtime/worker.py", mode, "/work"]
        if train is not None:
            args.append("/data/train")
        return args

    def preflight(self, worker: Path, work: Path) -> None:
        if self.synthetic_only:
            return
        probe = self.command(worker, work, "probe")
        # The worker intentionally rejects probe; a running sandbox that reaches
        # the worker proves namespaces and mounts are usable on this host.
        result = subprocess.run(probe, capture_output=True, timeout=20, env=self.environment())
        if b"unknown worker mode" not in result.stderr:
            raise IsolationUnavailable("bubblewrap preflight failed: " + result.stderr.decode(errors="replace")[-400:])

    @staticmethod
    def environment() -> dict[str, str]:
        return {"PATH": os.environ.get("PATH", ""), "HOME": "/tmp", "TMPDIR": "/tmp",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
                "CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}
