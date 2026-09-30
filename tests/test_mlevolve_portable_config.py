"""Static configuration checks; these tests launch no search or evaluation."""

import json
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def test_connectome_search_config_uses_repository_relative_paths():
    config = OmegaConf.load(ROOT / "competition_engineering/mlevolve_search_config.yaml")
    for key in ("data_dir", "dataset_dir", "desc_file", "log_dir", "workspace_dir"):
        value = Path(config[key])
        assert not value.is_absolute()
        assert (ROOT / value).resolve().is_relative_to(ROOT)
    assert (ROOT / config.desc_file).is_file()
    assert config.log_dir == config.workspace_dir


def test_committed_experiment_configs_have_no_protected_cache_field():
    files = sorted((ROOT / "competition_engineering/configs").glob("*.json"))
    assert len(files) == 8
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "confirm" not in data
        assert "promotion" not in data
        assert "train" in data and "tune" in data


def test_wsl_interpreter_default_uses_current_linux_home_and_env_override(monkeypatch):
    # Importing the bounded runner requires MLEvolve's engine on the path.
    import sys
    monkeypatch.syspath_prepend(str(ROOT / "upstream/MLEvolve"))
    from connectome.mlevolve_bounded import wsl_python_command

    monkeypatch.delenv("WUNDER_WSL_PYTHON", raising=False)
    assert wsl_python_command() == [
        "sh", "-c", 'exec "$HOME/.venvs/wunder311/bin/python" "$@"', "wunder-python"
    ]
    monkeypatch.setenv("WUNDER_WSL_PYTHON", "/opt/venv/bin/python")
    assert wsl_python_command() == ["/opt/venv/bin/python"]
