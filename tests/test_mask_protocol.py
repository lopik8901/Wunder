import copy
import hashlib
import json

import numpy as np
import pytest

from competition_engineering.mask_protocol import ROOT, SPEC, proxy_mask, verify_frozen_spec


@pytest.mark.skipif(
    not (ROOT/'competition_engineering/checkpoints/ridge1024_targetwise_v1/ridge.npz').is_file()
    or not (ROOT/'competition_engineering/cache/gru_tune/identity.json').is_file(),
    reason='Optional historical root/search-cache integration fixture is not transferred; synthetic boundary tests remain required')
def test_frozen_search_only_spec_replays():
    spec = verify_frozen_spec()
    assert spec["status"] == "prospectively_frozen_not_applied_to_any_candidate"
    assert spec["search_required_total"] == 1_273_664
    assert spec["search_scored_total"] == 108_139
    assert min(map(min, spec["search_scored_counts"])) >= 15
    assert not any(word in json.dumps(spec).lower() for word in
                   ("holdout", "promotion_result", "leaderboard", "submission_score"))


def test_proxy_is_repeatable_candidate_and_label_blind():
    spec = {"step_edges": [99, 500, 2000, 5000, 10000, 20000],
            "magnitude_edges": [0, .25, .5, .75, 1, 1.5, 2],
            "search_required_counts": [[100] * 7] * 5,
            "search_scored_counts": [[40] * 7] * 5,
            "proxy_seed": "synthetic"}
    z = {"need": np.r_[np.zeros(99, bool), np.ones(901, bool)],
         "step": np.arange(1000), "y": np.random.default_rng(1).normal(size=(1000, 2)),
         "candidate_prediction": np.zeros((1000, 2))}
    root = {"base_scale": np.ones(2), "base_bias": np.zeros(2),
            "strengths": np.zeros(2), "mean": np.zeros(112), "scale": np.ones(112),
            "coef": np.zeros((115, 2))}
    z["x"] = np.zeros((1000, 112), np.float32)
    z["p"] = np.full((1000, 2), .5, np.float32)
    first = proxy_mask(z, root, spec, 7)
    changed = copy.deepcopy(z)
    changed["y"][:] = 999
    changed["candidate_prediction"][:] = -999
    np.testing.assert_array_equal(first, proxy_mask(changed, root, spec, 7))
    assert not first[:99].any()
    assert first.any()
    assert hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(
        proxy_mask(z, root, spec, 7).tobytes()).hexdigest()
