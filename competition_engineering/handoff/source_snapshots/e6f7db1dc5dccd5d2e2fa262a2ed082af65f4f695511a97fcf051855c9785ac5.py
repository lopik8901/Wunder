"""Fixed-schema, aggregate-only comparison against the frozen search root."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np


EVIDENCE_LABEL = "reused_search_evidence_not_independent_validation"
POSITION_BUCKETS = (("early", 0, 6667), ("middle", 6667, 13333), ("late", 13333, 20000))
MAGNITUDE_BUCKETS = (("0_to_0.5", 0.0, 0.5), ("0.5_to_1", 0.5, 1.0),
                     ("1_to_1.5", 1.0, 1.5), ("1.5_to_2", 1.5, 2.000001))
PERSISTENCE_LAGS = (1, 10, 100)
FEATURE_LAGS = (0, 1, 10, 100)
FEATURE_GROUPS = {
    "position": tuple(list(range(0, 22)) + list(range(52, 74))),
    "velocity": tuple(list(range(22, 44)) + list(range(74, 96))),
    "delta_position": tuple(list(range(44, 48)) + list(range(96, 100))),
    "delta_velocity": tuple(list(range(48, 52)) + list(range(100, 104))),
    "auxiliary": tuple(range(104, 112)),
}
ROOT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/ridge1024_targetwise_v1/ridge.npz"


class _WeightedPearson:
    def __init__(self):
        self.s = np.zeros(6, dtype=np.float64)
        self.rows = 0

    def add(self, y, p, mask):
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return
        self.rows += int(m.sum())
        yy = np.clip(np.asarray(y, dtype=np.float32)[m], -2.0, 2.0).astype(np.float64)
        pp = np.clip(np.asarray(p, dtype=np.float32)[m], -2.0, 2.0).astype(np.float64)
        w = np.abs(yy)
        self.s += (w.sum(), (w * yy).sum(), (w * pp).sum(),
                   (w * yy * yy).sum(), (w * pp * pp).sum(), (w * yy * pp).sum())

    def value(self):
        w, sy, sp, syy, spp, syp = self.s
        if w < 1e-8:
            return 0.0
        vy = max(syy - sy * sy / w, 0.0)
        vp = max(spp - sp * sp / w, 0.0)
        if vy / w <= 1e-16 or vp / w <= 1e-16:
            return 0.0
        return float(np.clip((syp - sy * sp / w) / np.sqrt(vy * vp), -1.0, 1.0))


class _ResidualMoments:
    def __init__(self):
        self.weight = 0.0
        self.sum = 0.0
        self.square = 0.0
        self.rows = 0

    def add(self, residual, weight):
        self.weight += float(weight.sum())
        self.sum += float(np.dot(weight, residual))
        self.square += float(np.dot(weight, residual * residual))
        self.rows += len(residual)

    def value(self):
        if self.weight <= 1e-12:
            return 0.0, 0.0
        bias = self.sum / self.weight
        scale = np.sqrt(max(self.square / self.weight - bias * bias, 0.0))
        return float(bias), float(scale)


class _CorrelationMoments:
    def __init__(self, dimensions):
        self.w = np.zeros(dimensions, dtype=np.float64)
        self.sx = np.zeros(dimensions, dtype=np.float64)
        self.sxx = np.zeros(dimensions, dtype=np.float64)
        self.sxr = np.zeros(dimensions, dtype=np.float64)
        self.sr = 0.0
        self.srr = 0.0
        self.weight = 0.0
        self.rows = 0

    def add(self, x, residual, weight):
        if not len(residual):
            return
        x = np.asarray(x, dtype=np.float64)
        r = np.asarray(residual, dtype=np.float64)
        w = np.asarray(weight, dtype=np.float64)
        self.w += w.sum()
        self.sx += (x * w[:, None]).sum(axis=0)
        self.sxx += (x * x * w[:, None]).sum(axis=0)
        self.sxr += (x * (w * r)[:, None]).sum(axis=0)
        self.sr += float(np.dot(w, r))
        self.srr += float(np.dot(w, r * r))
        self.weight += float(w.sum())
        self.rows += len(r)

    def correlations(self):
        if self.weight <= 1e-12:
            return np.zeros_like(self.sx)
        vx = np.maximum(self.sxx - self.sx * self.sx / self.w.clip(1e-12), 0.0)
        vr = max(self.srr - self.sr * self.sr / self.weight, 0.0)
        cov = self.sxr - self.sx * self.sr / self.weight
        den = np.sqrt(vx * vr)
        out = np.zeros_like(cov)
        valid = (self.w > 1e-12) & (vx > 1e-16) & (vr > 1e-16)
        out[valid] = cov[valid] / den[valid]
        return np.clip(out, -1.0, 1.0)


def _pair_corr(pairs):
    n, sx, sy, sxx, syy, sxy = pairs
    vx = sxx - sx * sx / n if n else 0.0
    vy = syy - sy * sy / n if n else 0.0
    if n < 2 or vx <= 1e-16 or vy <= 1e-16:
        return 0.0
    return float(np.clip((sxy - sx * sy / n) / np.sqrt(vx * vy), -1, 1))


@lru_cache(maxsize=4)
def _load_root_model(checkpoint: str):
    with np.load(checkpoint) as model:
        return {key: model[key].copy() for key in model.files}


def frozen_root_predictions(z: dict[str, Any], checkpoint: Path = ROOT_CHECKPOINT):
    """Reconstruct the immutable root from the cache's raw official-GRU outputs."""
    from competition_engineering.residual import features

    model = _load_root_model(str(checkpoint))
    mean, scale, coef = model["mean"], model["scale"], model["coef"]
    strengths = model["strengths"]
    base_scale, base_bias = model["base_scale"], model["base_bias"]
    raw = np.asarray(z.get("gru_p", z["p"]), dtype=np.float32)
    base = (raw * base_scale.astype(np.float32) + base_bias.astype(np.float32)).astype(np.float32)
    local = dict(z)
    local["p"] = base
    correction = features(local, mean, scale) @ coef
    return (base + strengths * correction).astype(np.float32)


class SearchErrorDiagnostics:
    """Stream fixed aggregate comparisons; never retains row-level outcomes."""

    def __init__(self, root_predictor=frozen_root_predictions):
        self.root_predictor = root_predictor
        self.sequence_count = 0
        self.scored_rows = 0
        self.groups: dict[tuple[str, str, int, str], _WeightedPearson] = {}
        self.residuals: dict[tuple[int, str], _ResidualMoments] = {}
        self.sequence_scores = {k: {"root": [], "candidate": [], "better": 0} for k in range(2)}
        self.persistence: dict[tuple[int, int, str], list[float]] = {}
        self.feature_moments: dict[tuple[int, int, str], _CorrelationMoments] = {}

    def _score_add(self, key, y, root, candidate, mask):
        for model_name, prediction in (("root", root), ("candidate", candidate)):
            acc = self.groups.setdefault((*key, model_name), _WeightedPearson())
            acc.add(y, prediction, mask)

    def add(self, z: dict[str, Any], candidate_prediction):
        root = np.asarray(self.root_predictor(z), dtype=np.float32)
        candidate = np.asarray(candidate_prediction, dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.float32)
        x = np.asarray(z["x"], dtype=np.float32)
        mask = np.asarray(z["mask"], dtype=bool) & np.asarray(z.get("need", np.ones(len(y))), dtype=bool)
        steps = np.asarray(z.get("step", np.arange(len(y))), dtype=np.int64)
        if root.shape != y.shape or candidate.shape != y.shape or y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("diagnostic predictions must match (rows, 2) targets")
        if x.ndim != 2 or x.shape[1] != 112 or len(x) != len(y) or len(steps) != len(y):
            raise ValueError("diagnostic search rows have invalid feature/step shape")
        if not mask.any():
            return
        self.sequence_count += 1
        self.scored_rows += int(mask.sum())
        clipped_y = np.clip(y, -2, 2)
        clipped_root = np.clip(root, -2, 2)
        clipped_candidate = np.clip(candidate, -2, 2)
        root_residual = clipped_y - clipped_root
        candidate_residual = clipped_y - clipped_candidate

        for target in range(2):
            self._score_add(("all", "all", target), clipped_y[:, target], clipped_root[:, target],
                            clipped_candidate[:, target], mask)
            for name, lo, hi in POSITION_BUCKETS:
                selected = mask & (steps >= lo) & (steps < hi)
                self._score_add(("position", name, target), clipped_y[:, target], clipped_root[:, target],
                                clipped_candidate[:, target], selected)
            for name, lo, hi in MAGNITUDE_BUCKETS:
                target_bin = mask & (np.abs(clipped_y[:, target]) >= lo) & (np.abs(clipped_y[:, target]) < hi)
                prediction_bin = mask & (np.abs(clipped_root[:, target]) >= lo) & (np.abs(clipped_root[:, target]) < hi)
                self._score_add(("target_magnitude", name, target), clipped_y[:, target], clipped_root[:, target],
                                clipped_candidate[:, target], target_bin)
                self._score_add(("root_prediction_magnitude", name, target), clipped_y[:, target], clipped_root[:, target],
                                clipped_candidate[:, target], prediction_bin)

            for model_name, residual in (("root", root_residual[:, target]), ("candidate", candidate_residual[:, target])):
                stats = self.residuals.setdefault((target, model_name), _ResidualMoments())
                stats.add(residual[mask].astype(np.float64), np.abs(clipped_y[mask, target]).astype(np.float64))
            local_mask = mask
            local_root = _WeightedPearson(); local_root.add(y[:, target], root[:, target], local_mask)
            local_candidate = _WeightedPearson(); local_candidate.add(y[:, target], candidate[:, target], local_mask)
            root_wp, candidate_wp = local_root.value(), local_candidate.value()
            self.sequence_scores[target]["root"].append(root_wp)
            self.sequence_scores[target]["candidate"].append(candidate_wp)
            self.sequence_scores[target]["better"] += int(candidate_wp > root_wp)

            for lag in PERSISTENCE_LAGS:
                if len(y) <= lag:
                    continue
                pair_mask = mask[lag:] & mask[:-lag]
                r0, r1 = root_residual[lag:, target][pair_mask], root_residual[:-lag, target][pair_mask]
                c0, c1 = candidate_residual[lag:, target][pair_mask], candidate_residual[:-lag, target][pair_mask]
                for model_name, a, b in (("root", r0, r1), ("candidate", c0, c1)):
                    values = self.persistence.setdefault((target, lag, model_name), [0.0] * 6)
                    values[0] += len(a)
                    values[1:] = [u + v for u, v in zip(values[1:],
                        (a.sum(), b.sum(), np.dot(a, a), np.dot(b, b), np.dot(a, b)))]

            for lag in FEATURE_LAGS:
                if len(y) <= lag:
                    continue
                current = slice(lag, None) if lag else slice(None)
                past = slice(None, -lag) if lag else slice(None)
                valid = mask[current]
                weights = np.abs(clipped_y[current, target][valid]).astype(np.float64)
                features = x[past][valid]
                for model_name, residual in (("root", root_residual[:, target]), ("candidate", candidate_residual[:, target])):
                    values = self.feature_moments.setdefault((target, lag, model_name), _CorrelationMoments(112))
                    values.add(features, residual[current][valid], weights)

    def result(self):
        def paired(bucket, name, target):
            root_acc = self.groups.get((bucket, name, target, "root"), _WeightedPearson())
            candidate_acc = self.groups.get((bucket, name, target, "candidate"), _WeightedPearson())
            root = root_acc.value()
            candidate = candidate_acc.value()
            return {"n": root_acc.rows, "root_wp": root,
                    "candidate_wp": candidate, "delta_wp": candidate - root}

        # Row counts are tracked separately from weighted sums because zero-valued
        # targets are valid scored rows but carry zero official metric weight.
        output = {
            "schema_version": 1,
            "evidence": EVIDENCE_LABEL,
            "counts": {"sequences": self.sequence_count, "scored_rows": self.scored_rows},
            "target_performance": {},
            "position_buckets": {},
            "target_magnitude_buckets": {},
            "root_prediction_magnitude_buckets": {},
            "per_sequence_wp": {},
            "residual_bias_scale": {},
            "residual_persistence": {"lags": list(PERSISTENCE_LAGS), "t0": [], "t1": []},
            "feature_residual_relationships": {"lags": list(FEATURE_LAGS), "groups": list(FEATURE_GROUPS), "t0": [], "t1": []},
        }
        for target, label in enumerate(("t0", "t1")):
            output["target_performance"][label] = paired("all", "all", target)
            for bucket, _, _ in POSITION_BUCKETS:
                output["position_buckets"].setdefault(bucket, {})[label] = paired("position", bucket, target)
            for bucket, _, _ in MAGNITUDE_BUCKETS:
                output["target_magnitude_buckets"].setdefault(label, []).append(paired("target_magnitude", bucket, target))
                output["root_prediction_magnitude_buckets"].setdefault(label, []).append(paired("root_prediction_magnitude", bucket, target))
            root_seq = np.asarray(self.sequence_scores[target]["root"])
            candidate_seq = np.asarray(self.sequence_scores[target]["candidate"])
            output["per_sequence_wp"][label] = {
                "n_sequences": len(root_seq),
                "candidate_better_fraction": float(np.mean(candidate_seq > root_seq)) if len(root_seq) else 0.0,
                "root_q10_q50_q90": np.quantile(root_seq, [.1, .5, .9]).round(6).tolist() if len(root_seq) else [0.0] * 3,
                "candidate_q10_q50_q90": np.quantile(candidate_seq, [.1, .5, .9]).round(6).tolist() if len(candidate_seq) else [0.0] * 3,
            }
            rb, rs = self.residuals.get((target, "root"), _ResidualMoments()).value()
            cb, cs = self.residuals.get((target, "candidate"), _ResidualMoments()).value()
            output["residual_bias_scale"][label] = {
                "n": self.scored_rows,
                "root_weighted_bias": rb, "candidate_weighted_bias": cb, "delta_weighted_bias": cb - rb,
                "root_weighted_scale": rs, "candidate_weighted_scale": cs, "delta_weighted_scale": cs - rs,
            }
            for lag in PERSISTENCE_LAGS:
                root_corr = _pair_corr(self.persistence.get((target, lag, "root"), [0.0] * 6))
                cand_corr = _pair_corr(self.persistence.get((target, lag, "candidate"), [0.0] * 6))
                n = int(self.persistence.get((target, lag, "root"), [0])[0])
                output["residual_persistence"][label].append({"lag_rows": lag, "n_pairs": n,
                    "root_corr": root_corr, "candidate_corr": cand_corr, "delta_corr": cand_corr - root_corr})
            for lag in FEATURE_LAGS:
                root_corr = self.feature_moments.get((target, lag, "root"), _CorrelationMoments(112))
                cand_corr = self.feature_moments.get((target, lag, "candidate"), _CorrelationMoments(112))
                r_values, c_values = root_corr.correlations(), cand_corr.correlations()
                for group, indices in FEATURE_GROUPS.items():
                    r = float(np.sqrt(np.mean(r_values[list(indices)] ** 2)))
                    c = float(np.sqrt(np.mean(c_values[list(indices)] ** 2)))
                    output["feature_residual_relationships"][label].append({
                        "feature_group": group, "feature_lag_rows": lag,
                        "n_rows": root_corr.rows,
                        "root_rms_correlation": r, "candidate_rms_correlation": c,
                        "delta_rms_correlation": c - r,
                    })
        output["root_prediction_magnitude_reference"] = "frozen_root_abs_prediction"
        output["residual_definition"] = "clip(y,-2,2)-clip(prediction,-2,2); bias/scale weighted by abs(clipped target)"
        output["feature_relationship_definition"] = "RMS of all fixed-feature weighted correlations with residual using x[t-lag_rows]"
        return output


_TOP_KEYS = {
    "schema_version", "evidence", "counts", "target_performance", "position_buckets",
    "target_magnitude_buckets", "root_prediction_magnitude_buckets", "per_sequence_wp",
    "residual_bias_scale", "residual_persistence", "feature_residual_relationships",
    "root_prediction_magnitude_reference", "residual_definition", "feature_relationship_definition",
}
_SCORE_KEYS = {"n", "root_wp", "candidate_wp", "delta_wp"}
_TARGETS = {"t0", "t1"}


def sanitize_search_error_diagnostics(value):
    """Validate and copy only the exact fixed diagnostics schema to MLEvolve."""
    import math

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def score_row(row):
        return isinstance(row, dict) and set(row) == _SCORE_KEYS and type(row["n"]) is int and all(
            finite(row[k]) for k in ("root_wp", "candidate_wp", "delta_wp")
        )

    if not isinstance(value, dict) or set(value) != _TOP_KEYS or value.get("schema_version") != 1:
        return None
    if value.get("evidence") != EVIDENCE_LABEL:
        return None
    counts = value.get("counts")
    if not isinstance(counts, dict) or set(counts) != {"sequences", "scored_rows"} or any(type(v) is not int for v in counts.values()):
        return None
    target_perf = value.get("target_performance")
    if not isinstance(target_perf, dict) or set(target_perf) != _TARGETS or not all(score_row(v) for v in target_perf.values()):
        return None
    positions = value.get("position_buckets")
    if not isinstance(positions, dict) or set(positions) != {name for name, _, _ in POSITION_BUCKETS}:
        return None
    if any(not isinstance(v, dict) or set(v) != _TARGETS or not all(score_row(row) for row in v.values()) for v in positions.values()):
        return None
    for name in ("target_magnitude_buckets", "root_prediction_magnitude_buckets"):
        buckets = value.get(name)
        if not isinstance(buckets, dict) or set(buckets) != _TARGETS:
            return None
        if any(not isinstance(rows, list) or len(rows) != len(MAGNITUDE_BUCKETS) or not all(score_row(row) for row in rows) for rows in buckets.values()):
            return None
    per_seq = value.get("per_sequence_wp")
    if not isinstance(per_seq, dict) or set(per_seq) != _TARGETS:
        return None
    for row in per_seq.values():
        if not isinstance(row, dict) or set(row) != {"n_sequences", "candidate_better_fraction", "root_q10_q50_q90", "candidate_q10_q50_q90"}:
            return None
        if type(row["n_sequences"]) is not int or not finite(row["candidate_better_fraction"]):
            return None
        if any(not isinstance(row[key], list) or len(row[key]) != 3 or not all(finite(x) for x in row[key]) for key in ("root_q10_q50_q90", "candidate_q10_q50_q90")):
            return None
    residual = value.get("residual_bias_scale")
    residual_keys = {"n", "root_weighted_bias", "candidate_weighted_bias", "delta_weighted_bias", "root_weighted_scale", "candidate_weighted_scale", "delta_weighted_scale"}
    if not isinstance(residual, dict) or set(residual) != _TARGETS:
        return None
    if any(not isinstance(row, dict) or set(row) != residual_keys or type(row["n"]) is not int or not all(finite(row[k]) for k in residual_keys - {"n"}) for row in residual.values()):
        return None
    persistence = value.get("residual_persistence")
    if not isinstance(persistence, dict) or set(persistence) != {"lags", "t0", "t1"} or persistence["lags"] != list(PERSISTENCE_LAGS):
        return None
    persist_keys = {"lag_rows", "n_pairs", "root_corr", "candidate_corr", "delta_corr"}
    for target in _TARGETS:
        rows = persistence[target]
        if not isinstance(rows, list) or len(rows) != len(PERSISTENCE_LAGS) or any(
            not isinstance(row, dict) or set(row) != persist_keys or type(row["lag_rows"]) is not int or
            type(row["n_pairs"]) is not int or not all(finite(row[k]) for k in persist_keys - {"lag_rows", "n_pairs"})
            for row in rows
        ):
            return None
    features = value.get("feature_residual_relationships")
    if not isinstance(features, dict) or set(features) != {"lags", "groups", "t0", "t1"} or features["lags"] != list(FEATURE_LAGS) or features["groups"] != list(FEATURE_GROUPS):
        return None
    feature_keys = {"feature_group", "feature_lag_rows", "n_rows", "root_rms_correlation", "candidate_rms_correlation", "delta_rms_correlation"}
    for target in _TARGETS:
        rows = features[target]
        if not isinstance(rows, list) or len(rows) != len(FEATURE_LAGS) * len(FEATURE_GROUPS):
            return None
        if any(not isinstance(row, dict) or set(row) != feature_keys or row["feature_group"] not in FEATURE_GROUPS or
               row["feature_lag_rows"] not in FEATURE_LAGS or type(row["n_rows"]) is not int or
               not all(finite(row[k]) for k in feature_keys - {"feature_group", "feature_lag_rows", "n_rows"})
               for row in rows):
            return None
    if value.get("root_prediction_magnitude_reference") != "frozen_root_abs_prediction":
        return None
    if value.get("residual_definition") != "clip(y,-2,2)-clip(prediction,-2,2); bias/scale weighted by abs(clipped target)":
        return None
    if value.get("feature_relationship_definition") != "RMS of all fixed-feature weighted correlations with residual using x[t-lag_rows]":
        return None
    # Round-trip creates a detached primitive-only copy; no paths or caller-owned
    # objects can survive into prompts, journals, or memory.
    import json
    return json.loads(json.dumps(value, allow_nan=False))


def format_search_error_diagnostics(value) -> str:
    """Render the fixed-schema atlas compactly for MLEvolve's history prompt."""
    diag = sanitize_search_error_diagnostics(value)
    if diag is None:
        return "No valid fixed-schema search error atlas was recorded."
    parts = [f"REUSED SEARCH EVIDENCE (not independent validation); n={diag['counts']['scored_rows']} rows/{diag['counts']['sequences']} sequences."]
    for target in ("t0", "t1"):
        row = diag["target_performance"][target]
        parts.append(f"{target} WP root→candidate={row['root_wp']:.4f}→{row['candidate_wp']:.4f} (Δ{row['delta_wp']:+.4f}, n={row['n']}).")
    for name in ("early", "middle", "late"):
        pairs = []
        for target in ("t0", "t1"):
            row = diag["position_buckets"][name][target]
            pairs.append(f"{target} {row['root_wp']:.3f}→{row['candidate_wp']:.3f} Δ{row['delta_wp']:+.3f}/n{row['n']}")
        parts.append(f"Position {name}: " + "; ".join(pairs) + ".")
    for kind, label in (("target_magnitude_buckets", "|target|"), ("root_prediction_magnitude_buckets", "|frozen-root prediction|")):
        for target in ("t0", "t1"):
            rows = diag[kind][target]
            values = ", ".join(f"{name} Δ{row['delta_wp']:+.3f}/n{row['n']}" for (name, _, _), row in zip(MAGNITUDE_BUCKETS, rows))
            parts.append(f"{label} bins {target}: {values}.")
    for target in ("t0", "t1"):
        row = diag["per_sequence_wp"][target]
        parts.append(f"Per-sequence {target} q10/q50/q90 root={row['root_q10_q50_q90']} candidate={row['candidate_q10_q50_q90']}; candidate better on {row['candidate_better_fraction']:.1%} of n={row['n_sequences']}.")
        row = diag["residual_bias_scale"][target]
        parts.append(f"Weighted clipped residual {target} bias root→candidate={row['root_weighted_bias']:+.3f}→{row['candidate_weighted_bias']:+.3f}; scale={row['root_weighted_scale']:.3f}→{row['candidate_weighted_scale']:.3f} (n={row['n']}).")
        lag_rows = diag["residual_persistence"][target]
        parts.append(f"Residual persistence {target}: " + ", ".join(f"lag{r['lag_rows']} {r['root_corr']:+.3f}→{r['candidate_corr']:+.3f}/n{r['n_pairs']}" for r in lag_rows) + ".")
        features = diag["feature_residual_relationships"][target]
        parts.append(f"Fixed-feature RMS residual correlations {target} (root→candidate): " + "; ".join(
            f"{r['feature_group']}@lag{r['feature_lag_rows']} {r['root_rms_correlation']:.3f}→{r['candidate_rms_correlation']:.3f}/n{r['n_rows']}"
            for r in features) + ".")
    return " ".join(parts)
