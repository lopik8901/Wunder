import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))

from llm import _provider  # noqa: E402
from llm import codex  # noqa: E402
from llm.gemini import FunctionSpec  # noqa: E402


def _cfg(*, backend="codex", model="gpt-5.6-luna"):
    stage = SimpleNamespace(model=model)
    return SimpleNamespace(
        llm_backend=backend, codex_reasoning_effort="medium",
        agent=SimpleNamespace(code=stage, feedback=stage),
    )


def test_codex_backend_is_opt_in():
    assert _provider("gemini-3-pro-preview", _cfg(backend="auto")) == "gemini"
    assert _provider("gpt-4.1", _cfg(backend="auto")) == "openai"
    assert _provider("gpt-5.6-luna", _cfg()) == "codex"


def test_codex_query_uses_stdin_schema_and_removes_api_key(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        schema = Path(command[command.index("--output-schema") + 1])
        observed["schema"] = json.loads(schema.read_text(encoding="utf-8"))
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps({"answer": "ok"}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    spec = FunctionSpec(
        name="reply",
        description="Return a reply",
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )

    response, _, _, _, info = codex.query(
        system_message="system text", user_message="user text",
        func_spec=spec, cfg=_cfg(), model="gpt-5.6-luna",
    )

    assert response == {"answer": "ok"}
    assert info["backend"] == "codex-cli"
    assert observed["kwargs"]["input"].find("system text") >= 0
    assert observed["kwargs"]["input"].find("user text") >= 0
    assert "--output-schema" in observed["command"]
    assert observed["schema"]["additionalProperties"] is False
    assert observed["kwargs"]["env"].get("OPENAI_API_KEY") is None
    assert observed["kwargs"]["shell"] is False


def test_codex_schema_makes_optional_properties_required_and_nullable():
    original = {
        "type": "object",
        "properties": {
            "needs_revision": {"type": "boolean"},
            "revised_code": {"type": "string"},
        },
        "required": ["needs_revision"],
    }

    normalized = codex._strict_object_schema(original)

    assert normalized["required"] == ["needs_revision", "revised_code"]
    assert normalized["properties"]["revised_code"]["type"] == ["string", "null"]
    assert original["required"] == ["needs_revision"]
