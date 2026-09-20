import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))

from agents.result_parse_agent import _parse_connectome_feedback  # noqa: E402
from engine.execution import validate_executed_node  # noqa: E402
from engine.search_node import SearchNode  # noqa: E402
from utils.metric import MetricValue  # noqa: E402


def _agent(tmp_path, *, connectome_mode):
    return SimpleNamespace(
        cfg=SimpleNamespace(connectome_mode=connectome_mode, workspace_dir=tmp_path),
        branch_successful_nodes={},
    )


def _node():
    return SearchNode(code="pass", stage="draft", branch_id=1)


def test_connectome_feedback_populates_search_node_metrics_and_bypasses_csv(tmp_path):
    payload = {
        "status": "success", "is_bug": False, "summary": "real score",
        "metric": 0.2, "lower_is_better": False,
        "wp_t0": 0.1, "wp_t1": 0.3, "combined_wp": 0.2,
    }
    response = _parse_connectome_feedback(["MLEVOLVE_FEEDBACK_JSON=" + json.dumps(payload)])
    node = _node()
    node.is_buggy = response["is_bug"]
    node.metric = MetricValue(response["metric"], maximize=True)
    node.connectome_metrics = {
        "wp_t0": response["wp_t0"], "wp_t1": response["wp_t1"],
        "combined_wp": response["combined_wp"],
    }

    agent = _agent(tmp_path, connectome_mode=True)
    validate_executed_node(agent, node)

    assert node.is_buggy is False
    assert node.metric.value == 0.2
    assert node.connectome_metrics == {"wp_t0": 0.1, "wp_t1": 0.3, "combined_wp": 0.2}
    assert agent.branch_successful_nodes[1] == [node]


def test_non_connectome_node_still_requires_csv(tmp_path):
    node = _node()
    node.is_buggy = False
    node.metric = MetricValue(0.2, maximize=True)

    validate_executed_node(_agent(tmp_path, connectome_mode=False), node)

    assert node.is_buggy is True
    assert node.metric.value is None


def test_connectome_failure_cannot_become_valid_score():
    payload = {
        "status": "failure", "is_bug": True, "summary": "RuntimeError: boom",
        "metric": None, "lower_is_better": False,
        "wp_t0": None, "wp_t1": None, "combined_wp": None,
    }
    response = _parse_connectome_feedback(["MLEVOLVE_FEEDBACK_JSON=" + json.dumps(payload)])
    assert response["is_bug"] is True
    assert response["metric"] is None
