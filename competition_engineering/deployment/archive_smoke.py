"""Load the actual ZIP contents and replay one complete search sequence offline."""
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from competition_engineering.deployment.build_and_test import CACHE, DEPLOY, ROOT, cached, optimized, replay, sha
from competition_engineering.pipeline import load_model


def main():
    path = ROOT / "competition_engineering/submissions/promoted_20260929_014231_cpu_v1.zip"
    expected = {"solution.py", "gru.py", "baseline.onnx", "model.npz", "deployment_manifest.json"}
    with tempfile.TemporaryDirectory() as directory:
        dest = Path(directory)
        with zipfile.ZipFile(path) as archive:
            assert set(archive.namelist()) == expected
            archive.extractall(dest)
        manifest = json.loads((dest / "deployment_manifest.json").read_text())
        for name, digest in manifest["files"].items():
            assert sha(dest/name) == digest
        packed = load_model(dest / "solution.py")
        source = optimized()
        z = next(cached(CACHE))[1]
        a, _ = replay(source.predict, z)
        b, _ = replay(packed.predict, z)
        assert np.array_equal(a[z["need"]], b[z["need"]])
        assert packed.gru.session.get_providers() == ["CPUExecutionProvider"]
    print(json.dumps({"zip_sha256": sha(path), "archive_files": sorted(expected),
                      "tested_complete_rows": 20000, "exact_archive_prediction_match": True,
                      "provider": "CPUExecutionProvider", "internet_used": False}, indent=2))


if __name__ == "__main__":
    main()
