"""Synthetic-only supervisor checks. No real candidate or protected cache is read."""
import json

import numpy as np
import pytest

from competition_engineering.mask_protocol import SPEC
from competition_engineering.mask_aware_promotion_v1 import (
    ROOT_SEARCH_WP, run_protocol,
)


def fixtures():
    spec_bytes = SPEC.read_bytes()
    spec = json.loads(spec_bytes)
    model = {"base_scale": np.ones(2), "base_bias": np.zeros(2),
             "strengths": np.zeros(2), "mean": np.zeros(112),
             "scale": np.ones(112), "coef": np.zeros((115, 2))}
    def sequences(count):
        for group in range(count):
            rng = np.random.default_rng(group + 1000)
            x = np.zeros((2000, 112), np.float32)
            x[:, :2] = rng.normal(size=(2000, 2))
            y = x[:, :2] + rng.normal(scale=.1, size=(2000, 2)).astype(np.float32)
            p = x[:, :2] + rng.normal(scale=.6, size=(2000, 2)).astype(np.float32)
            need = np.arange(2000) >= 99
            mask = need & (np.arange(2000) % 4 == 0)
            yield group, {"x": x, "y": y, "p": p, "need": need,
                          "mask": mask, "step": np.arange(2000)}
    attestation = {key: True for key in
                   ("python311_linux", "one_cpu", "offline", "causal", "reset",
                    "deterministic", "finite_float32_output", "package_compatible")}
    attestation.update(candidate_sha256="a" * 64, zip_bytes=123456,
                       microseconds_per_row=50.0)
    return dict(spec=spec, spec_bytes=spec_bytes,
                search_wp=ROOT_SEARCH_WP + .002,
                candidate_sha256="a" * 64, cpu_attestation=attestation,
                reference=lambda z: z["p"], candidate=lambda z: z["x"][:, :2],
                confirm_sequences=lambda: sequences(16),
                holdout_sequences=lambda: sequences(32),
                proxy_reference_model=model)


def test_synthetic_pass_is_paired_and_reproducible():
    args = fixtures()
    first = run_protocol(**args)
    second = run_protocol(**args)
    assert first["status"] == "passed_v1_promotion_gates"
    for stage in ("confirm", "proxy"):
        assert first[stage]["delta_combined"] > 0
        assert first[stage]["mask_sha256"] == second[stage]["mask_sha256"]
        assert first[stage]["reference_wp"] == second[stage]["reference_wp"]
    assert first["proxy"]["paired_95ci_combined"][0] > 0
    assert first["confirm"]["scored_rows"] != first["proxy"]["scored_rows"]


def test_failed_search_never_opens_protected_factories():
    args = fixtures()
    args["search_wp"] = ROOT_SEARCH_WP
    args["confirm_sequences"] = lambda: pytest.fail("confirmation opened")
    args["holdout_sequences"] = lambda: pytest.fail("holdout opened")
    assert run_protocol(**args)["status"] == "failed_search_gate"


def test_failed_cpu_check_never_opens_protected_factories():
    args = fixtures()
    args["cpu_attestation"]["microseconds_per_row"] = 200.0
    args["confirm_sequences"] = lambda: pytest.fail("confirmation opened")
    with pytest.raises(ValueError, match="runtime limit"):
        run_protocol(**args)


def test_failed_confirm_never_opens_proxy_holdout():
    args = fixtures()
    args["candidate"] = args["reference"]
    args["holdout_sequences"] = lambda: pytest.fail("holdout opened")
    assert run_protocol(**args)["status"] == "failed_masked_confirm_gate"


def test_tampered_spec_rejected_before_any_read():
    args = fixtures()
    args["spec_bytes"] += b" "
    args["confirm_sequences"] = lambda: pytest.fail("confirmation opened")
    with pytest.raises(ValueError, match="specification mismatch"):
        run_protocol(**args)


def test_malformed_output_rejected_before_proxy_holdout():
    args = fixtures()
    args["candidate"] = lambda z: np.zeros(len(z["need"]), np.float32)
    args["holdout_sequences"] = lambda: pytest.fail("holdout opened")
    with pytest.raises(ValueError, match="prediction shape"):
        run_protocol(**args)


def test_predictor_receives_no_target_or_mask_fields():
    args = fixtures()
    def inspect(z):
        assert set(z) == {"x", "p", "need", "step"}
        assert all(not value.flags.writeable for value in z.values())
        return z["x"][:, :2]
    args["candidate"] = inspect
    assert run_protocol(**args)["status"] == "passed_v1_promotion_gates"


def test_tampered_parsed_counts_rejected():
    args = fixtures()
    args["spec"]["search_scored_counts"][0][0] += 1
    args["confirm_sequences"] = lambda: pytest.fail("confirmation opened")
    with pytest.raises(ValueError, match="specification mismatch"):
        run_protocol(**args)
