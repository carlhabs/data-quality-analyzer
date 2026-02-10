from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreWeights:
    completeness: int = 25
    validity: int = 25
    uniqueness: int = 20
    consistency: int = 20
    outliers: int = 10


@dataclass(frozen=True)
class Scores:
    completeness: float
    validity: float
    uniqueness: float
    consistency: float
    outliers: float
    total: float


def _bounded(score: float) -> float:
    return max(0.0, min(100.0, score))


def compute_scores(
    *,
    missing_pct: float,
    invalid_pct: float,
    duplicate_rows_pct: float,
    duplicate_key_pct: float | None,
    consistency_invalid_pct: float,
    outlier_pct: float,
    weights: ScoreWeights | None = None,
) -> Scores:
    weights = weights or ScoreWeights()

    completeness_score = _bounded(100.0 * (1.0 - missing_pct))
    validity_score = _bounded(100.0 * (1.0 - invalid_pct))
    uniqueness_rate = max(duplicate_rows_pct, duplicate_key_pct or 0.0)
    uniqueness_score = _bounded(100.0 * (1.0 - uniqueness_rate))
    consistency_score = _bounded(100.0 * (1.0 - consistency_invalid_pct))
    outliers_score = _bounded(100.0 * (1.0 - outlier_pct))

    total_weight = (
        weights.completeness
        + weights.validity
        + weights.uniqueness
        + weights.consistency
        + weights.outliers
    )
    weighted_total = (
        completeness_score * weights.completeness
        + validity_score * weights.validity
        + uniqueness_score * weights.uniqueness
        + consistency_score * weights.consistency
        + outliers_score * weights.outliers
    ) / total_weight

    return Scores(
        completeness=completeness_score,
        validity=validity_score,
        uniqueness=uniqueness_score,
        consistency=consistency_score,
        outliers=outliers_score,
        total=_bounded(weighted_total),
    )
