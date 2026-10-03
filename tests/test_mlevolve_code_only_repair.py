"""Code-only repair replies must not be passed raw to the candidate parser."""

import sys
import ast
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "upstream" / "MLEvolve"))
from agents.coder import base_coder


def test_fenced_code_only_reply_is_extracted_without_retries(monkeypatch):
    calls = []
    def fake_generate(**kwargs):
        calls.append(kwargs)
        return "```python\nCANDIDATE = {'hypothesis': 'synthetic repair'}\n```"
    monkeypatch.setattr(base_coder, "generate", fake_generate)
    agent = SimpleNamespace(acfg=SimpleNamespace(code=SimpleNamespace(temp=0)), cfg=object())
    plan, code = base_coder.plan_and_code_query(agent, {"user": "synthetic"})
    assert plan == ""
    parsed = ast.parse(code)
    assert ast.literal_eval(parsed.body[0].value) == {"hypothesis": "synthetic repair"}
    assert len(calls) == 1
