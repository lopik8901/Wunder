import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))

from connectome.run_mlevolve_smoke import _executed_node, _expose_repository  # noqa: E402


def test_executed_node_uses_journal_result_after_backprop_returns_root():
    root = SimpleNamespace(stage="root")
    executed = SimpleNamespace(stage="debug")
    journal = SimpleNamespace(nodes=[root, executed])

    assert _executed_node(root, journal) is executed
    assert _executed_node(executed, journal) is executed


def test_expose_repository_prepends_root_to_pythonpath(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "existing")

    _expose_repository(tmp_path)

    assert os.environ["PYTHONPATH"] == f"{tmp_path}{os.pathsep}existing"
