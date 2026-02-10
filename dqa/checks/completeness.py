from __future__ import annotations

from typing import Any

import pandas as pd

from dqa.io import rows_to_examples


def compute_completeness(df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    columns_metrics: dict[str, dict[str, Any]] = {}

    total_cells = len(df) * len(df.columns)
    missing_total = int(df.isna().sum().sum())
    missing_pct = missing_total / total_cells if total_cells else 0.0

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_col_pct = missing_count / len(df) if len(df) else 0.0
        columns_metrics[col] = {
            "missing_count": missing_count,
            "missing_pct": missing_col_pct,
        }
        if missing_count:
            examples = rows_to_examples(df.loc[df[col].isna(), col])
            issues.append(
                {
                    "type": "missing_values",
                    "columns": [col],
                    "count": missing_count,
                    "examples": examples,
                }
            )

    metrics = {
        "global": {
            "missing_count": missing_total,
            "missing_pct": missing_pct,
        },
        "columns": columns_metrics,
    }
    return metrics, issues
