from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_connectome_contract_reaches_all_mlevolve_candidate_prompts():
    task = (ROOT / "connectome" / "task.md").read_text(encoding="utf-8")
    required = (
        "use only the current observation and earlier observations",
        "Reset all recurrent, rolling, normalization, and cache state",
        "Global Weighted Pearson",
        "60-minute limit",
    )
    assert all(text in task for text in required)

    mlevolve = ROOT / "upstream" / "MLEvolve"
    config = (mlevolve / "config" / "__init__.py").read_text(encoding="utf-8")
    run = (mlevolve / "run.py").read_text(encoding="utf-8")
    search = (mlevolve / "engine" / "agent_search.py").read_text(encoding="utf-8")
    assert "with open(cfg.desc_file) as f:" in config and "return f.read()" in config
    assert "task_desc = load_task_desc(cfg)" in run and "task_desc=task_desc" in run
    assert "task_desc if cfg.desc_file is not None" in search

    for filename in ("draft_agent.py", "evolution_agent.py", "fusion_agent.py"):
        prompt = (mlevolve / "agents" / filename).read_text(encoding="utf-8")
        assert '\"Task description\": agent.task_desc' in prompt
        assert "{prompt['Task description']}" in prompt


def test_search_context_presents_equal_mechanisms_without_claiming_generated_gpu():
    description = (ROOT / "competition_engineering/mlevolve_task/description.md").read_text(encoding="utf-8")
    improve = (ROOT / "upstream/MLEvolve/agents/improve_agent.py").read_text(encoding="utf-8")
    draft = (ROOT / "upstream/MLEvolve/agents/draft_agent.py").read_text(encoding="utf-8")
    assert "two first-class execution mechanisms" in description
    assert "implementation failure" in description.lower()
    assert "Generated training\nis currently CPU-only" in description
    assert "If your method does not fit this fallback" not in description
    assert "refine the parent or branch" in improve
    assert "parent is a scored comparison" in improve
    assert "both first-class mechanisms" in draft


def test_scientific_diagnostics_read_only_search_cache():
    source = (ROOT / "tools/search_only_incumbent_diagnostics.py").read_text(encoding="utf-8")
    assert 'SEARCH = ROOT / "competition_engineering/cache/gru_tune"' in source
    assert 'identity.get("split") != "tune"' in source
    assert "valid.parquet" not in source
    assert "final_gate" not in source
    assert "holdout" not in source


def test_research_prompt_paths_share_search_history_and_diagnostic_plan():
    agents = ROOT / "upstream" / "MLEvolve" / "agents"
    for name in ("draft_agent.py", "improve_agent.py", "evolution_agent.py",
                 "aggregation_agent.py", "fusion_agent.py"):
        source = (agents / name).read_text(encoding="utf-8")
        assert "planner_history_section(agent)" in source, name
        assert "**research_plan_guideline()" in source, name
