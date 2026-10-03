"""Parse MLEvolve-authored programs without executing the proposal in the planner."""
from __future__ import annotations

import ast
import hashlib
import re

ALLOWED_TRAIN_IMPORTS = {"numpy", "torch", "math", "typing", "collections", "dataclasses", "itertools", "functools", "scipy", "pathlib", "json"}
ALLOWED_CALLBACK_IMPORTS = {"numpy", "math", "typing", "collections", "dataclasses", "itertools", "functools", "onnxruntime", "pathlib", "json", "struct"}
FORBIDDEN_NAMES = {"__import__", "eval", "exec", "compile", "open", "input", "breakpoint", "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "help"}
FORBIDDEN_TEXT = re.compile(r"holdout|final_gate|complete.validation|promotion|submission|leaderboard|valid\.parquet|\.\.[/\\]", re.I)


def parse_candidate(code: str) -> dict:
    if len(code.encode("utf-8")) > 100_000:
        raise ValueError("generated candidate exceeds 100 KB")
    tree = ast.parse(code)
    body = [node for node in tree.body if not isinstance(node, ast.Expr)
            or not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str)]
    if len(body) != 1 or not isinstance(body[0], ast.Assign):
        raise ValueError("provide one literal CANDIDATE assignment")
    assignment = body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "CANDIDATE":
        raise ValueError("provide one literal CANDIDATE assignment")
    candidate = ast.literal_eval(assignment.value)
    required = {"hypothesis", "train_tier", "train_source", "callback_source"}
    if not isinstance(candidate, dict) or not required.issubset(candidate) or set(candidate)-required-{"training_device"}:
        raise ValueError("CANDIDATE requires hypothesis, train_tier, train_source, callback_source; optional training_device")
    if not isinstance(candidate["hypothesis"], str) or not 10 <= len(candidate["hypothesis"]) <= 400:
        raise ValueError("hypothesis must be 10-400 characters")
    if type(candidate["train_tier"]) is not int or candidate["train_tier"] not in {256, 1024}:
        raise ValueError("train_tier must be 256 or 1024")
    if candidate.get("training_device", "cpu") not in {"cpu", "gpu"}:
        raise ValueError("training_device must be cpu or gpu")
    if "cuda" in candidate["train_source"].lower() and candidate.get("training_device", "cpu") != "gpu":
        raise ValueError("CUDA/HIP training requires training_device='gpu'")
    validate_source(candidate["train_source"], training=True)
    validate_source(candidate["callback_source"], training=False)
    candidate["source_sha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return candidate


def validate_source(source: str, *, training: bool) -> None:
    role = "training" if training else "callback"
    if not isinstance(source, str) or len(source.encode("utf-8")) > 60_000:
        raise ValueError(f"{role} source must be a string of at most 60 KB")
    if FORBIDDEN_TEXT.search(source):
        raise ValueError(f"{role} source names a protected data domain or filesystem escape")
    tree = ast.parse(source)
    imports = ALLOWED_TRAIN_IMPORTS if training else ALLOWED_CALLBACK_IMPORTS
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""])
            if any(name.split(".")[0] not in imports for name in modules):
                raise ValueError(f"unsupported {role} dependency: {modules}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"unsafe {role} primitive: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in {"system", "popen", "spawn", "fork", "environ", "getenv", "chdir", "remove", "unlink", "rmtree", "set_per_process_memory_fraction"}):
            raise ValueError(f"unsafe {role} attribute: {node.attr}")
    required = "train" if training else "PredictionModel"
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if (required not in functions) if training else (required not in classes):
        raise ValueError(f"{role} source must define {required}")
