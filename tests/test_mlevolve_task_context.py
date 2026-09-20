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
