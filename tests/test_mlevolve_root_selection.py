"""Regression for completed root drafts becoming selectable search parents."""

from types import SimpleNamespace
import threading
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "upstream" / "MLEvolve"))
from engine import agent_search, node_selection


def test_completed_draft_is_unlocked_even_without_backpropagation(monkeypatch):
    root = SimpleNamespace(id="root", stage="root", is_terminal=False, is_buggy=False,
                           reached_child_limit=lambda scfg: False)
    child = SimpleNamespace(id="child", stage="draft", code="EXPERIMENT = {}",
                            lock=False, metric=SimpleNamespace(value=.65, maximize=True),
                            is_buggy=False)
    agent = object.__new__(agent_search.AgentSearch)
    agent.virtual_root = root
    agent.scfg = SimpleNamespace()
    agent.acfg = SimpleNamespace(use_aggregation=False)
    agent.best_node = None
    agent.journal_lock = threading.Lock()
    agent.journal = []
    monkeypatch.setattr(agent_search.draft_agent, "run", lambda *args, **kwargs: child)
    monkeypatch.setattr(agent_search.code_review_agent, "run", lambda *args, **kwargs: child.code)
    monkeypatch.setattr(agent_search.result_parse_agent, "run", lambda *args, **kwargs: child)
    monkeypatch.setattr(agent_search.execution, "validate_executed_node", lambda *args: None)
    monkeypatch.setattr(agent_search.evaluation, "check_improvement", lambda *args: False)
    _, result = agent._run_single_step(root, exec_callback=lambda *args: None)
    assert result is child and child.lock is False
    assert agent.journal == [child]


def test_root_with_full_draft_slots_selects_completed_child():
    root = SimpleNamespace(id="root", stage="root", is_terminal=False, children=[],
                           reached_child_limit=lambda scfg: True)
    child = SimpleNamespace(id="child", stage="draft", is_terminal=False, lock=False,
                            is_buggy=False, continue_improve=False, children=[],
                            reached_child_limit=lambda scfg: False)
    root.children.append(child)
    agent = SimpleNamespace(virtual_root=root,
        is_root=lambda node: node is root,
        cfg=SimpleNamespace(connectome_mode=True,
            agent=SimpleNamespace(decay=SimpleNamespace(phase_ratios=[.5, .8],
                exploration_constant=1.414, alpha=.01, lower_bound=.5))),
        acfg=SimpleNamespace(use_aggregation=False, steps=6),
        scfg=SimpleNamespace(num_drafts=2, num_improves=3), current_step=3)
    child.uct_value = lambda exploration_constant: 1
    assert node_selection.select(agent, root) is child
    child.lock = True
    with pytest.raises(RuntimeError, match="no selectable child"):
        node_selection.select(agent, root)
