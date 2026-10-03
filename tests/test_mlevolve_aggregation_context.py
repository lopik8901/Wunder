"""Aggregation must receive the same sanitized search evidence as draft planning."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "upstream" / "MLEvolve"))

from agents import aggregation_agent
from connectome.mlevolve_history import format_search_history
from competition_engineering.search_error_diagnostics import SearchErrorDiagnostics
from engine.search_node import SearchNode


def _search_history_with_atlas():
    rows = 120
    y = np.stack([np.linspace(-1, 1, rows), np.linspace(1, -1, rows)], axis=1).astype("float32")
    z = {"x": np.zeros((rows, 112), dtype="float32"), "y": y,
         "root": y * .8, "mask": np.ones(rows, bool), "need": np.ones(rows, bool),
         "step": np.arange(rows)}
    atlas = SearchErrorDiagnostics(root_predictor=lambda batch: batch["root"])
    atlas.add(z, y * .9)
    record = {"node_id": "safe-node", "parent_id": "root", "status": "success",
              "runtime_seconds": 1, "spec": {"kind": "ridge", "hypothesis": "Search-only test"},
              "config": {"kind": "ridge"},
              "result": {"metric_domain": "fixed development tuning sequences",
                         "metrics": {"WP_t0": .6, "WP_t1": .7, "combined_WP": .65},
                         "search_error_diagnostics": atlas.result(),
                         "holdout_wp": .999}}
    return format_search_history([record])


def test_aggregation_prompt_includes_search_atlas_and_diagnostic_plan(monkeypatch):
    root = SearchNode(stage="root", code="")
    representatives = [SimpleNamespace(branch_id=i, metric=SimpleNamespace(value=.65 + i * .001),
                       generate_node_trajectory=lambda need_code: "Search-only branch summary") for i in (1, 2)]
    captured = {}
    def fake_build(model, introduction, user_prompt, assistant_prefix):
        captured["assistant_prefix"] = assistant_prefix
        return user_prompt
    monkeypatch.setattr(aggregation_agent, "build_chat_prompt_for_model", fake_build)
    def fake_query(agent, prompt):
        captured["prompt"] = prompt
        return "Search observation; hypothesis; distinguishing test.", "EXPERIMENT = {'kind': 'ridge', 'hypothesis': 'test'}"
    monkeypatch.setattr(aggregation_agent, "plan_and_code_query", fake_query)
    from connectome import research_planning
    def fake_research_hook(agent, prompt, parent_id, stage, code_query):
        plan, code = code_query(agent, prompt)
        return plan, code, prompt, {"stage": stage, "parent_id": parent_id}
    monkeypatch.setattr(research_planning, "plan_with_research", fake_research_hook)
    monkeypatch.setattr(aggregation_agent, "register_node", lambda *args, **kwargs: None)
    agent = SimpleNamespace(
        virtual_root=root, is_root=lambda node: node is root,
        branch_successful_nodes={1: [representatives[0]], 2: [representatives[1]]},
        metric_maximize=True, fusion_draft_count=0, max_fusion_drafts=2,
        cfg=SimpleNamespace(connectome_mode=True, exec=SimpleNamespace(timeout=1260)),
        acfg=SimpleNamespace(code=SimpleNamespace(model="test"), steps=6, time_limit=7200),
        start_time=time.time(), current_step=3, task_desc="Search-only Connectome task",
        data_preview="", connectome_search_history_provider=_search_history_with_atlas,
    )
    node = aggregation_agent.run(agent, mode="node")
    prompt = captured["prompt"]
    assert node.stage == "fusion_draft"
    assert "REUSED SEARCH EVIDENCE" in prompt
    assert "Position early" in prompt and "Residual persistence" in prompt
    assert "concrete observation from the fixed search-split error atlas" in prompt
    assert "what the candidate fixed versus what remains unexplained" in prompt
    assert "distinguish the hypothesis" in prompt
    assert "refine a known approach or branch" in captured["assistant_prefix"]
    assert "0.999" not in prompt and "holdout_wp" not in prompt
