"""Trusted entrypoint inside the generated-candidate sandbox.

Inference uses a one-row request/response protocol: the next feature vector is
never sent until the current prediction has returned. Search targets and masks
are never mounted inside this process.
"""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

REQUEST = struct.Struct("<qI?")
FEATURE_BYTES = 112 * 4


class DataPoint(NamedTuple):
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def load(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exactly(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError("incomplete inference request")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def main():
    mode = sys.argv[1]
    work = Path(sys.argv[2]).resolve()
    if mode == "train":
        train_dir = Path(sys.argv[3]).resolve()
        artifacts = work / "artifacts"
        artifacts.mkdir(exist_ok=False)
        gpu = {"requested": False, "available": False, "tensor_ok": False}
        if os.environ.get("CANDIDATE_GPU_REQUIRED") == "1":
            gpu["requested"] = True
            import torch
            gpu["available"] = bool(torch.cuda.is_available())
            if not gpu["available"]:
                raise RuntimeError("GPU requested but unavailable")
            torch.cuda.set_per_process_memory_fraction(.75)
            x = torch.arange(16, device="cuda", dtype=torch.float32)
            gpu["tensor_ok"] = bool((x*x).sum().item() == 1240.0)
            if not gpu["tensor_ok"]:
                raise RuntimeError("GPU tensor preflight failed")
        print("TRAIN_ENV=" + json.dumps(gpu), flush=True)
        identity = json.loads((train_dir / "identity.json").read_text())
        train_files = [str(train_dir / f"{group:05d}.npz") for group in identity["groups"]]
        result = load(work / "train.py").train(train_files, str(artifacts))
        print("TRAIN_RESULT=" + json.dumps(result if isinstance(result, dict) else {}, default=str), flush=True)
        return
    if mode != "infer":
        raise ValueError("unknown worker mode")
    model = load(work / "deploy" / "solution.py").PredictionModel()
    stream_in, stream_out = sys.stdin.buffer, sys.stdout.buffer
    count = 0
    callback_seconds = 0.0
    while True:
        first = stream_in.read(1)
        if not first:
            break
        header = first + exactly(stream_in, REQUEST.size-1)
        seq, step, need = REQUEST.unpack(header)
        state = np.frombuffer(exactly(stream_in, FEATURE_BYTES), dtype="<f4").copy()
        started = time.perf_counter()
        value = model.predict(DataPoint(seq, step, need, state))
        callback_seconds += time.perf_counter() - started
        if not need:
            if value is not None:
                raise ValueError("warm-up prediction must be None")
            stream_out.write(b"N")
        else:
            result = np.asarray(value)
            if result.shape != (2,) or result.dtype != np.float32 or not np.isfinite(result).all():
                raise ValueError("required prediction must be finite float32 shape (2,)")
            stream_out.write(b"P" + result.tobytes())
        stream_out.flush()
        count += 1
    print("CALLBACK_STATS=" + json.dumps({"rows": count, "callback_seconds": callback_seconds}), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
