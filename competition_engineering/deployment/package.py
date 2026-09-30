"""Package a versioned CPU callback derived from the immutable checkpoint."""
import hashlib
import json
import zipfile
from pathlib import Path
from competition_engineering.deployment.build_and_test import CHECKPOINT, DEPLOY, export, sha


def main():
    export()
    expected = "afead41dbbe93cd888c59da19b24a5d416a2f007e4fbcbba386de4e97c3992e1"
    assert sha(CHECKPOINT) == expected
    files = ["solution.py", "gru.py", "baseline.onnx", "model.npz"]
    manifest = {"source_checkpoint_sha256": expected,
                "relationship": "model.npz is a lossless export of the frozen checkpoint arrays; solution.py is a streaming CPU implementation",
                "files": {name: sha(DEPLOY / name) for name in files}}
    (DEPLOY / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2))
    files.append("deployment_manifest.json")
    path = DEPLOY.parent / "submissions/promoted_20260929_014231_cpu_v1.zip"
    assert not path.exists(), "versioned ZIP already exists"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in files:
            z.write(DEPLOY / name, name)
    report = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path),
              "files": manifest["files"], "source_checkpoint_sha256": expected}
    (DEPLOY.parent / "reports/deployment_package.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
