from __future__ import annotations

from typing import Any

import pandas as pd

from dqa.io import rows_to_examples


def compute_outliers(df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    columns_metrics: dict[str, dict[str, Any]] = {}

    total_numeric = 0
    total_outliers = 0

    for col in df.columns:
        if df[col].dtype == 'bool' or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        total_numeric += int(series.notna().sum())
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (series < lower) | (series > upper)
        outlier_count = int(outliers.sum())
        outlier_pct = outlier_count / int(series.notna().sum()) if series.notna().sum() else 0.0
        total_outliers += outlier_count
        columns_metrics[col] = {
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
        }
        if outlier_count:
            issues.append(
                {
                    "type": "outliers",
                    "columns": [col],
                    "count": outlier_count,
                    "examples": rows_to_examples(series[outliers].dropna()),
                }
            )

    outlier_pct_total = total_outliers / total_numeric if total_numeric else 0.0
    metrics = {
        "global": {
            "outlier_count": total_outliers,
            "outlier_pct": outlier_pct_total,
        },
        "columns": columns_metrics,
    }
    return metrics, issues
