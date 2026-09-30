"""Record a real GPU training smoke test; exit nonzero if unavailable."""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from competition_engineering.gpu_environment import configure_miopen_cache
configure_miopen_cache()

report = {"python": sys.version, "executable": sys.executable,
          "platform": platform.platform(), "passed": False}
try:
    import torch
    report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                  hip_build=torch.version.hip, available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("No usable GPU backend; CPU fallback prohibited")
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append({"index": i, "name": props.name, "vram_bytes": props.total_memory})
    report["devices"] = devices
    index = max(devices, key=lambda d: d["vram_bytes"])["index"]
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    torch.manual_seed(20260928)
    model = torch.nn.Sequential(torch.nn.Conv1d(112, 64, 3), torch.nn.ReLU(),
                                torch.nn.Conv1d(64, 2, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 112, 512, device=device)
    target = torch.randn(8, 2, 510, device=device)
    before = next(model.parameters()).detach().clone()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and not torch.equal(before, next(model.parameters()))
    report.update(passed=True, selected_device=str(device), loss=loss.item(),
                  five_steps_seconds=time.perf_counter()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as exc:
    report["error"] = repr(exc)
output = Path(__file__).resolve().parents[1] / "competition_engineering" / "environment"
output.mkdir(parents=True, exist_ok=True)
destination = output / f"gpu_check_{time.time_ns()}.json"
destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"Report: {destination}")
sys.exit(0 if report["passed"] else 1)
