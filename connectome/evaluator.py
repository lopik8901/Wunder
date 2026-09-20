"""Official-metric scoring and causal row-wise replay for Connectome models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class EvaluationError(ValueError):
    """Raised when data or model output violates the competition contract."""


@dataclass(frozen=True)
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: NDArray[np.float32]


@dataclass(frozen=True)
class SequenceData:
    seq_ix: ArrayLike
    step_in_seq: ArrayLike
    need_prediction: ArrayLike
    states: ArrayLike
    targets: ArrayLike
    is_scored: ArrayLike

    def __post_init__(self) -> None:
        arrays = {
            "seq_ix": np.asarray(self.seq_ix),
            "step_in_seq": np.asarray(self.step_in_seq),
            "need_prediction": np.asarray(self.need_prediction, dtype=bool),
            "states": np.asarray(self.states, dtype=np.float32),
            "targets": np.asarray(self.targets, dtype=np.float64),
            "is_scored": np.asarray(self.is_scored, dtype=bool),
        }
        row_count = len(arrays["seq_ix"])
        if arrays["states"].ndim != 2:
            raise EvaluationError("states must have shape (rows, features)")
        if arrays["targets"].shape != (row_count, 2):
            raise EvaluationError("targets must have shape (rows, 2)")
        if any(len(value) != row_count for value in arrays.values()):
            raise EvaluationError("all inputs must contain the same number of rows")
        if row_count == 0:
            raise EvaluationError("at least one row is required")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def with_states(self, states: ArrayLike) -> "SequenceData":
        return replace(self, states=states)


class PredictionModel(Protocol):
    def predict(self, data_point: DataPoint) -> ArrayLike | None: ...


def weighted_pearson(target: ArrayLike, prediction: ArrayLike, mask: ArrayLike) -> float:
    """Compute one target's official clipped Global Weighted Pearson score."""
    target_array = np.asarray(target, dtype=np.float32)
    prediction_array = np.asarray(prediction, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=bool)
    if target_array.ndim != 1 or prediction_array.shape != target_array.shape or mask_array.shape != target_array.shape:
        raise EvaluationError("target, prediction, and mask must be equal-length vectors")
    if not np.any(mask_array):
        raise EvaluationError("scoring mask selects no rows")

    y = np.clip(target_array[mask_array], -2.0, 2.0).astype(np.float64)
    p = np.clip(prediction_array[mask_array], -2.0, 2.0).astype(np.float64)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise EvaluationError("scored targets and predictions must be finite")

    weights = np.abs(y)
    weight_sum = float(weights.sum())
    if weight_sum < 1e-8:
        return 0.0
    y_centered = y - np.sum(weights * y) / weight_sum
    p_centered = p - np.sum(weights * p) / weight_sum
    target_std = float(np.sqrt(np.sum(weights * y_centered**2) / weight_sum))
    prediction_std = float(np.sqrt(np.sum(weights * p_centered**2) / weight_sum))
    if target_std <= 1e-8 or prediction_std <= 1e-8:
        return 0.0
    covariance = float(np.sum(weights * y_centered * p_centered) / weight_sum)
    return float(np.clip(covariance / (target_std * prediction_std), -1.0, 1.0))


def replay_model(model_factory: Callable[[], PredictionModel], data: SequenceData) -> NDArray[np.float64]:
    """Replay rows through the official callback surface without exposing targets or masks."""
    _validate_sequence_order(data)
    model = model_factory()
    predictions = np.full((len(data.seq_ix), 2), np.nan, dtype=np.float64)

    for row_index in range(len(data.seq_ix)):
        state = data.states[row_index]
        point = DataPoint(
            seq_ix=int(data.seq_ix[row_index]),
            step_in_seq=int(data.step_in_seq[row_index]),
            need_prediction=bool(data.need_prediction[row_index]),
            state=state,
        )
        output = model.predict(point)
        if not point.need_prediction:
            if output is not None:
                raise EvaluationError(f"model returned a prediction during warm-up at row {row_index}")
            continue
        if output is None:
            raise EvaluationError(f"model returned no prediction at required row {row_index}")
        row_prediction = np.asarray(output, dtype=np.float64)
        if row_prediction.shape != (2,):
            raise EvaluationError(f"prediction at row {row_index} must have shape (2,)")
        if not np.isfinite(row_prediction).all():
            raise EvaluationError(f"prediction at row {row_index} must contain finite values")
        predictions[row_index] = row_prediction

    return predictions


def evaluate_model(
    model_factory: Callable[[], PredictionModel],
    data: SequenceData,
    *,
    artifact_bytes: int = 0,
) -> dict[str, object]:
    """Replay and score a model, returning MLEvolve's scalar metric plus diagnostics."""
    started = perf_counter()
    predictions = replay_model(model_factory, data)
    score_mask = data.need_prediction & data.is_scored
    wp_t0 = weighted_pearson(data.targets[:, 0], predictions[:, 0], score_mask)
    wp_t1 = weighted_pearson(data.targets[:, 1], predictions[:, 1], score_mask)
    combined = (wp_t0 + wp_t1) / 2.0
    return {
        "status": "success",
        "primary_metric": combined,
        "metric_name": "combined_WP",
        "maximize": True,
        "metrics": {"WP_t0": wp_t0, "WP_t1": wp_t1, "combined_WP": combined},
        "runtime_seconds": perf_counter() - started,
        "artifact_bytes": int(artifact_bytes),
        "rows_processed": len(data.seq_ix),
        "predictions_made": int(data.need_prediction.sum()),
    }


def _validate_sequence_order(data: SequenceData) -> None:
    completed: set[int] = set()
    current_seq: int | None = None
    previous_step = -1
    for row_index, (seq_value, step_value) in enumerate(zip(data.seq_ix, data.step_in_seq)):
        seq_ix = int(seq_value)
        step = int(step_value)
        if seq_ix != current_seq:
            if seq_ix in completed:
                raise EvaluationError(f"sequence {seq_ix} is not contiguous")
            if current_seq is not None:
                completed.add(current_seq)
            current_seq = seq_ix
            previous_step = -1
        if step != previous_step + 1:
            raise EvaluationError(f"row {row_index} has non-consecutive step_in_seq")
        previous_step = step
