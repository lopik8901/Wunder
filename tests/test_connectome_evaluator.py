import math
import unittest

import numpy as np

from connectome.evaluator import (
    EvaluationError,
    SequenceData,
    evaluate_model,
    replay_model,
    weighted_pearson,
)
from wnn_connectome_starterpack.utils import weighted_pearson as official_weighted_pearson


class WeightedPearsonTests(unittest.TestCase):
    def test_uses_clipped_targets_as_weights_and_pools_globally(self):
        targets = np.array([[-3.0, 0.5], [-0.5, 1.0], [1.0, -2.5], [3.0, 2.0]])
        predictions = np.array([[-1.0, 0.25], [0.5, 0.5], [2.0, -1.5], [1.0, 3.0]])
        mask = np.array([True, True, False, True])

        score = weighted_pearson(targets[:, 0], predictions[:, 0], mask)

        y = np.clip(targets[mask, 0], -2.0, 2.0)
        p = np.clip(predictions[mask, 0], -2.0, 2.0)
        w = np.abs(y)
        y_mean = np.sum(w * y) / np.sum(w)
        p_mean = np.sum(w * p) / np.sum(w)
        expected = np.sum(w * (y - y_mean) * (p - p_mean)) / math.sqrt(
            np.sum(w * (y - y_mean) ** 2) * np.sum(w * (p - p_mean) ** 2)
        )
        self.assertAlmostEqual(score, expected)
        self.assertNotAlmostEqual(score, float(np.corrcoef(y, p)[0, 1]))

    def test_matches_authenticated_official_metric_and_degenerate_conventions(self):
        target = np.array([-3.0, -0.25, 0.0, 0.75, 4.0], dtype=np.float64)
        prediction = np.array([2.5, 0.5, -1.0, 0.25, -3.0], dtype=np.float64)
        mask = np.array([True, False, True, True, True])
        self.assertEqual(
            weighted_pearson(target, prediction, mask),
            official_weighted_pearson(target[mask], prediction[mask]),
        )
        self.assertEqual(weighted_pearson([1.0, 1.0], [0.0, 1.0], [True, True]), 0.0)
        self.assertEqual(weighted_pearson([0.0, 0.0], [0.0, 1.0], [True, True]), 0.0)
        with self.assertRaises(EvaluationError):
            weighted_pearson([1.0, np.nan], [0.0, 1.0], [True, True])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.data = SequenceData(
            seq_ix=np.array([4, 4, 4, 9, 9, 9]),
            step_in_seq=np.array([0, 1, 2, 0, 1, 2]),
            need_prediction=np.array([False, True, True, False, True, True]),
            states=np.array([[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]], dtype=np.float32),
            targets=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0], [0.0, 0.0], [-1.0, 2.0], [-2.0, -2.0]]),
            is_scored=np.array([False, True, True, False, True, True]),
        )

    def test_warmup_updates_state_and_model_resets_between_sequences(self):
        predictions = replay_model(ResettingRunningSumModel, self.data)
        np.testing.assert_allclose(
            predictions[self.data.need_prediction],
            [[3.0, -3.0], [6.0, -6.0], [30.0, -30.0], [60.0, -60.0]],
        )
        self.assertTrue(np.isnan(predictions[~self.data.need_prediction]).all())

    def test_only_current_row_is_exposed_so_future_changes_do_not_change_prefix(self):
        original = replay_model(ResettingRunningSumModel, self.data)
        changed_states = self.data.states.copy()
        changed_states[2:] += 10_000
        changed = replay_model(ResettingRunningSumModel, self.data.with_states(changed_states))
        np.testing.assert_array_equal(original[:2], changed[:2])

    def test_callback_cannot_access_hidden_labels_or_mutate_evaluator_state(self):
        replay_model(PrivacyCheckingModel, self.data)
        np.testing.assert_array_equal(
            self.data.states,
            [[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]],
        )

    def test_rejects_bad_predictions_and_non_contiguous_sequences(self):
        with self.assertRaises(EvaluationError):
            replay_model(BadShapeModel, self.data)
        with self.assertRaises(EvaluationError):
            replay_model(NonFiniteModel, self.data)
        with self.assertRaises(EvaluationError):
            replay_model(
                ResettingRunningSumModel,
                SequenceData(
                    seq_ix=np.array([1, 2, 1]),
                    step_in_seq=np.array([0, 0, 1]),
                    need_prediction=np.array([False, False, True]),
                    states=np.ones((3, 1), dtype=np.float32),
                    targets=np.ones((3, 2)),
                    is_scored=np.array([False, False, True]),
                ),
            )

    def test_evaluator_emits_mlevolve_compatible_result(self):
        result = evaluate_model(ScoringModel, self.data, artifact_bytes=123)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metric_name"], "combined_WP")
        self.assertTrue(result["maximize"])
        self.assertEqual(result["primary_metric"], result["metrics"]["combined_WP"])
        self.assertEqual(set(result["metrics"]), {"WP_t0", "WP_t1", "combined_WP"})
        self.assertEqual(result["artifact_bytes"], 123)
        self.assertGreaterEqual(result["runtime_seconds"], 0.0)
        self.assertEqual(result["rows_processed"], 6)
        self.assertEqual(result["predictions_made"], 4)


class ResettingRunningSumModel:
    def __init__(self):
        self.seq_ix = None
        self.total = 0.0

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.total = 0.0
        self.total += float(data_point.state[0])
        if not data_point.need_prediction:
            return None
        return np.array([self.total, -self.total], dtype=np.float32)


class ScoringModel:
    def predict(self, data_point):
        if not data_point.need_prediction:
            return None
        value = float(data_point.state[0]) / 20.0
        return np.array([value, -0.5 * value], dtype=np.float32)


class PrivacyCheckingModel:
    def predict(self, data_point):
        assert not hasattr(data_point, "targets")
        assert not hasattr(data_point, "is_scored")
        with np.testing.assert_raises(ValueError):
            data_point.state[0] = 999
        if not data_point.need_prediction:
            return None
        return np.array([0.0, 0.0], dtype=np.float32)


class BadShapeModel:
    def predict(self, data_point):
        return None if not data_point.need_prediction else np.zeros(3)


class NonFiniteModel:
    def predict(self, data_point):
        return None if not data_point.need_prediction else np.array([np.nan, 0.0])


if __name__ == "__main__":
    unittest.main()
