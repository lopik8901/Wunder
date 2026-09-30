"""Launch the configured bounded MLEvolve Connectome search."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))
os.environ["MLEVOLVE_CONFIG_PATH"] = str(ROOT / "competition_engineering" / "mlevolve_search_config.yaml")
from competition_engineering.gpu_environment import configure_miopen_cache  # noqa: E402
configure_miopen_cache()

from competition_engineering.generated_sandbox import GeneratedSandbox  # noqa: E402


def preflight_generated_isolation():
    """Expanded searches fail before MLEvolve starts if isolation is absent."""
    config = (ROOT / "competition_engineering/mlevolve_search_config.yaml").read_text(encoding="utf-8")
    if "generated_enabled: true" in config:
        if os.name == "nt":
            from connectome.mlevolve_bounded import wsl_path, wsl_python_command
            command = ["wsl", "-d", "Ubuntu", "--cd", wsl_path(ROOT), "--exec",
                       *wsl_python_command(),
                       "-m", "pytest", "-p", "no:cacheprovider", "-q",
                       "tests/test_generated_mlevolve_preflight.py", "-k",
                       "os_sandbox_cannot_read_unmounted_sentinel or production_namespace_denies_host_protected_other_attempt_network_and_devices or novel_cpu_candidate_end_to_end_in_production_sandbox or production_workspace_mount_on_wsl_project_drive"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError("WSL production isolation preflight failed: " + (result.stdout + result.stderr)[-2000:])
            return
        with tempfile.TemporaryDirectory(prefix="mlevolve_sandbox_preflight_") as directory:
            sandbox = GeneratedSandbox()
            sandbox.preflight(ROOT / "competition_engineering/generated_worker.py", Path(directory))

from run import run  # noqa: E402


if __name__ == "__main__":
    preflight_generated_isolation()
    os.chdir(ROOT)
    run()
