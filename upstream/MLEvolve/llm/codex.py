"""Codex CLI backend for MLEvolve's existing query/generate contract."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .gemini import FunctionSpec, compile_prompt_to_md


def _strict_object_schema(schema: dict) -> dict:
    """Return the Codex-required strict form without mutating MLEvolve's schema."""
    normalized = json.loads(json.dumps(schema))

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                properties = value.get("properties", {})
                originally_required = set(value.get("required", []))
                for name, property_schema in properties.items():
                    if name not in originally_required:
                        property_type = property_schema.get("type")
                        if isinstance(property_type, str):
                            property_schema["type"] = [property_type, "null"]
                        elif isinstance(property_type, list) and "null" not in property_type:
                            property_schema["type"] = [*property_type, "null"]
                        else:
                            properties[name] = {
                                "anyOf": [property_schema, {"type": "null"}]
                            }
                value["required"] = list(properties)
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(normalized)
    return normalized


def _invoke(prompt: str, *, cfg, model: str, schema: dict | None = None) -> tuple[str, float]:
    with tempfile.TemporaryDirectory(prefix="mlevolve-codex-") as directory:
        root = Path(directory)
        output_path = root / "response.txt"
        command = [
            "codex", "exec", "-", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--sandbox", "read-only",
            "--model", model, "--config", f'model_reasoning_effort="{cfg.codex_reasoning_effort}"',
            "--config", 'approval_policy="never"',
            "--cd", str(root), "--output-last-message", str(output_path),
        ]
        if schema is not None:
            schema_path = root / "schema.json"
            schema_path.write_text(json.dumps(_strict_object_schema(schema)), encoding="utf-8")
            command.extend(["--output-schema", str(schema_path)])

        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        started = time.perf_counter()
        completed = subprocess.run(
            command, input=prompt, cwd=root, env=env, shell=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(f"Codex CLI failed ({completed.returncode}): {completed.stderr.strip()}")
        if not output_path.exists():
            raise RuntimeError("Codex CLI produced no final response file")
        return output_path.read_text(encoding="utf-8").strip(), elapsed


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg=None,
    **model_kwargs,
):
    model = model_kwargs.get("model") or cfg.agent.code.model
    parts = [
        "Act only as MLEvolve's language-model backend. Do not use tools. Return only the requested final response.",
    ]
    if system_message:
        parts.extend(["\n# System message\n", system_message])
    if user_message:
        parts.extend(["\n# User message\n", user_message])
    if func_spec is not None:
        parts.extend([
            "\n# Structured response\n",
            f"Return the arguments for `{func_spec.name}` as JSON matching the supplied schema. ",
            func_spec.description,
        ])
    text, elapsed = _invoke("".join(parts), cfg=cfg, model=model,
                            schema=func_spec.json_schema if func_spec else None)
    output = json.loads(text) if func_spec else text
    return output, elapsed, 0, 0, {"model": model, "backend": "codex-cli"}


def generate(
    prompt,
    cfg,
    temperature=None,
    max_tokens=None,
    stop_tokens=None,
    json_schema=None,
    max_retries=20,
    retry_delay=3,
):
    del temperature, max_tokens
    model = cfg.agent.code.model
    compiled = compile_prompt_to_md(prompt)
    request = (
        "Act only as MLEvolve's language-model backend. Do not use tools. Preserve every instruction "
        "in the request and return only the requested final response.\n\n" + compiled
    )
    last_error = None
    for attempt in range(max_retries):
        try:
            text, _ = _invoke(request, cfg=cfg, model=model, schema=json_schema)
            if stop_tokens:
                positions = [text.find(token) for token in stop_tokens if token in text]
                if positions:
                    text = text[:min(positions)]
            return text
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            last_error = error
            if attempt + 1 < max_retries:
                time.sleep(retry_delay)
    raise RuntimeError(f"Codex generation failed after {max_retries} attempts: {last_error}")
