"""Regression tests for literal candidate review and its information boundary."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))

from agents import code_review_agent
from agents import draft_agent, debug_agent
from agents.prompts.impl_guideline import get_impl_guideline_from_agent
from connectome.mlevolve_generated import parse_candidate
from engine.search_node import SearchNode


def test_review_applies_patch_even_when_diff_generation_is_disabled():
    original = "EXPERIMENT = {'kind': 'ridge', 'hypothesis': 'A bounded test.', 'sample_stride': 10}"
    patch = "<<<<<<< SEARCH\n'sample_stride': 10\n=======\n'sample_stride': 20\n>>>>>>> REPLACE"
    revised = code_review_agent.apply_review_revision(
        original, patch, connectome_mode=True, use_diff_mode=False)
    assert "'sample_stride': 20" in revised
    assert "<<<<<<<" not in revised


def test_review_rejects_unapplied_or_malformed_patch():
    original = "EXPERIMENT = {'kind': 'ridge', 'hypothesis': 'A bounded test.'}"
    for revision in (
        "<<<<<<< SEARCH\nmissing text\n=======\nreplacement\n>>>>>>> REPLACE",
        "<<<<<<< SEARCH\n'kind': 'ridge'\n=======\n'kind': 'ridge'",
        "EXPERIMENT = {'kind': 'ridge'\n<<<<<<< SEARCH",
    ):
        assert code_review_agent.apply_review_revision(
            original, revision, connectome_mode=True, use_diff_mode=False) == original


def test_review_keeps_generated_interface_and_source_gate():
    source = "CANDIDATE = " + repr({
        "hypothesis": "A synthetic causal model for review.",
        "train_tier": 256,
        "train_source": "def train(train_files, output_dir):\n return {}",
        "callback_source": "class PredictionModel:\n pass",
    })
    assert parse_candidate(source)
    unsafe = source.replace("class PredictionModel:\\n pass", "class PredictionModel:\\n def predict(self, p): return getattr(p, 'state')")
    assert code_review_agent.apply_review_revision(
        source, unsafe, connectome_mode=True, use_diff_mode=False) == source


def test_connectome_review_prompt_omits_generic_submission_and_holdout(monkeypatch):
    captured = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return {"needs_revision": False, "reasoning": "No critical defect.", "revised_code": None}

    monkeypatch.setattr(code_review_agent, "query", fake_query)
    cfg = SimpleNamespace(connectome_mode=True, pretrain_model_dir="")
    acfg = SimpleNamespace(use_diff_mode=False, code=SimpleNamespace(model="test", temp=0))
    agent = SimpleNamespace(task_desc="Search-only Connectome task.", cfg=cfg, acfg=acfg)
    node = SimpleNamespace(id="synthetic", code="EXPERIMENT = {'kind': 'ridge'}")
    assert code_review_agent.run(agent, node) == node.code
    prompt = str(captured["system_message"]).lower()
    assert "submission.csv" not in prompt
    assert "hold-out validation" not in prompt
    assert "network resources" in prompt


def test_connectome_guideline_names_supervisor_and_literal_contract():
    agent = SimpleNamespace(
        cfg=SimpleNamespace(connectome_mode=True, exec=SimpleNamespace(timeout=1260)),
        acfg=SimpleNamespace(time_limit=7200, steps=4),
        start_time=0, current_step=1,
    )
    text = str(get_impl_guideline_from_agent(agent)).lower()
    assert "literal candidate" in text
    assert "supervisor alone" in text
    assert "submission.csv" not in text
    assert "hold-out validation" not in text


def test_candidate_contract_exposes_actual_fields_and_limits():
    description = (ROOT / "competition_engineering/mlevolve_task/description.md").read_text()
    for name in ("seq_ix", "step_in_seq", "need_prediction", "state", "getattr", "1,200-second"):
        assert name in description


def test_draft_and_debug_prompts_use_literal_contract_without_generic_mlebench_text(monkeypatch):
    captured = []

    def fake_plan_and_code(_agent, prompt):
        captured.append(str(prompt).lower())
        return "Repair the interface.", "EXPERIMENT = {'kind': 'ridge'}"

    monkeypatch.setattr(draft_agent, "plan_and_code_query", fake_plan_and_code)
    monkeypatch.setattr(debug_agent, "plan_and_code_query", fake_plan_and_code)
    monkeypatch.setattr(draft_agent, "register_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(debug_agent, "register_node", lambda *args, **kwargs: None)
    root = SearchNode(code="EXPERIMENT = {'kind': 'ridge'}", stage="root")
    acfg = SimpleNamespace(time_limit=7200, steps=4, use_diff_mode=False,
                           code=SimpleNamespace(model="test", temp=0))
    cfg = SimpleNamespace(connectome_mode=True, exec=SimpleNamespace(timeout=1260),
                          pretrain_model_dir="")
    agent = SimpleNamespace(
        cfg=cfg, acfg=acfg, start_time=0, current_step=0,
        task_desc="Search-only Connectome task.", data_preview="No data preview.",
        virtual_root=root, use_coldstart=False, coldstart_description="None model",
        use_stepwise_generation=False, global_memory=None,
    )
    draft_agent.run(agent)
    failed = SearchNode(code="EXPERIMENT = {'kind': 'ridge'}", stage="draft", parent=root,
                        analysis="literal parser failure", _term_out=["safe failure"])
    debug_agent.run(agent, failed)
    assert len(captured) == 2
    for prompt in captured:
        assert "literal candidate" in prompt or "literal `candidate`" in prompt or "literal candidate" in prompt.replace("`", "")
        assert "submission.csv" not in prompt
        assert "hold-out validation" not in prompt
        assert "real leaderboard" not in prompt
