"""Search-sequence-only parity, state, and timing checks for CPU deployment prototype."""
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits
from wnn_connectome_starterpack.utils import DataPoint

from competition_engineering.deployment.build_and_test import CACHE, CHECKPOINT, cached, export, frozen, optimized, replay, sha


def main():
    state, network = export()
    blocks = [z for i, (_, z) in enumerate(cached(CACHE)) if i < 4]
    reports = []
    times = {"frozen": [], "prototype": []}
    with threadpool_limits(limits=1):
        for run in range(3):
            reference = frozen(state, network)
            candidate = optimized()
            for z in blocks:
                a, ta = replay(reference, z)
                b, tb = replay(candidate.predict, z)
                times["frozen"].append(ta)
                times["prototype"].append(tb)
                np.testing.assert_allclose(a[z["need"]], b[z["need"]], atol=3e-5, rtol=3e-4)
                reports.append({"run": run, "sequence": int(z["seq"]), "rows": len(a),
                                "max_abs": float(np.max(np.abs(a[z["need"]]-b[z["need"]]))),
                                "frozen_seconds": ta, "prototype_seconds": tb})
        first = blocks[0]
        a, _ = replay(optimized().predict, first)
        b, _ = replay(optimized().predict, first)
        assert np.array_equal(a[first["need"]], b[first["need"]])
        reset = optimized()
        replay(reset.predict, blocks[1])
        c, _ = replay(reset.predict, first)
        assert np.array_equal(a[first["need"]], c[first["need"]])
        boundary = 10000
        probe = optimized()
        prefix = []
        for step in range(boundary):
            p = probe.predict(DataPoint(int(first["seq"]), step, bool(first["need"][step]), first["x"][step]))
            prefix.append(None if p is None else p.copy())
        altered = first["x"].copy()
        altered[boundary:] += 1.0
        probe2 = optimized()
        for step in range(boundary):
            p = probe2.predict(DataPoint(int(first["seq"]), step, bool(first["need"][step]), altered[step]))
            assert (p is None and prefix[step] is None) or np.array_equal(p, prefix[step])
    rows = sum(r["rows"] for r in reports)
    summary = {"source": "fixed search-tune sequences only", "checkpoint_sha256": sha(CHECKPOINT),
               "rows_each": rows, "sequences_each": 4, "repetitions": 3,
               "max_abs": max(r["max_abs"] for r in reports),
               "frozen_us_per_row": sum(times["frozen"])/rows*1e6,
               "prototype_us_per_row": sum(times["prototype"])/rows*1e6,
               "warmup_shape_finite": True, "deterministic": True,
               "sequence_reset": True, "causal_prefix": True, "details": reports}
    path = CHECKPOINT.parents[2] / "reports/deployment_prototype_verification.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
