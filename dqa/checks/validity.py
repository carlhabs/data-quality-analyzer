from __future__ import annotations

from typing import Any

import pandas as pd

from dqa.io import rows_to_examples
from dqa.rules import Rules


_BOOL_STRINGS = {"true", "false", "yes", "no", "0", "1"}


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    inferred: dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            inferred[col] = "string"
            continue
        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = numeric.notna().mean()
        if numeric_ratio >= 0.9:
            if (numeric.dropna() % 1 == 0).all():
                inferred[col] = "int"
            else:
                inferred[col] = "float"
            continue
        lower = non_null.astype(str).str.lower().str.strip()
        if lower.isin(_BOOL_STRINGS).mean() >= 0.9:
            inferred[col] = "bool"
            continue
        dates = pd.to_datetime(non_null, errors="coerce", format="mixed", utc=False)
        if dates.notna().mean() >= 0.9:
            inferred[col] = "date"
            continue
        inferred[col] = "string"
    return inferred


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _invalid_by_type(series: pd.Series, rule_type: str) -> pd.Series:
    if rule_type == "int":
        numeric = _coerce_numeric(series)
        return numeric.isna() | (numeric % 1 != 0)
    if rule_type == "float":
        numeric = _coerce_numeric(series)
        return numeric.isna()
    if rule_type == "bool":
        lower = series.astype(str).str.lower().str.strip()
        return ~lower.isin(_BOOL_STRINGS)
    if rule_type == "date":
        return _coerce_date(series).isna()
    if rule_type == "string":
        return pd.Series([False] * len(series), index=series.index)
    return pd.Series([False] * len(series), index=series.index)


def compute_validity(
    df: pd.DataFrame,
    rules: Rules | None,
    inferred_types: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    columns_metrics: dict[str, dict[str, Any]] = {}

    inferred_types = inferred_types or infer_column_types(df)
    total_cells = len(df) * len(df.columns)
    invalid_total = 0

    for col in df.columns:
        series = df[col]
        invalid_mask = pd.Series([False] * len(df), index=df.index)
        rules_for_col = rules.columns.get(col) if rules else None

        if rules_for_col and rules_for_col.required:
            missing_mask = series.isna()
            if missing_mask.any():
                count = int(missing_mask.sum())
                invalid_mask |= missing_mask
                issues.append(
                    {
                        "type": "required_missing",
                        "columns": [col],
                        "count": count,
                        "examples": rows_to_examples(series[missing_mask]),
                    }
                )

        rule_type = rules_for_col.type if rules_for_col else None
        if rule_type:
            mismatch = _invalid_by_type(series, rule_type)
            if mismatch.any():
                count = int(mismatch.sum())
                invalid_mask |= mismatch
                issues.append(
                    {
                        "type": "type_mismatch",
                        "columns": [col],
                        "count": count,
                        "examples": rows_to_examples(series[mismatch]),
                    }
                )

        if rules_for_col:
            if rules_for_col.min is not None:
                numeric = _coerce_numeric(series)
                too_low = numeric < rules_for_col.min
                if too_low.any():
                    count = int(too_low.sum())
                    invalid_mask |= too_low
                    issues.append(
                        {
                            "type": "below_min",
                            "columns": [col],
                            "count": count,
                            "examples": rows_to_examples(series[too_low]),
                        }
                    )
            if rules_for_col.max is not None:
                numeric = _coerce_numeric(series)
                too_high = numeric > rules_for_col.max
                if too_high.any():
                    count = int(too_high.sum())
                    invalid_mask |= too_high
                    issues.append(
                        {
                            "type": "above_max",
                            "columns": [col],
                            "count": count,
                            "examples": rows_to_examples(series[too_high]),
                        }
                    )
            if rules_for_col.regex:
                regex = rules_for_col.regex
                non_null = series.dropna().astype(str)
                mismatch = ~non_null.str.match(regex)
                if mismatch.any():
                    bad_values = non_null[mismatch]
                    count = int(mismatch.sum())
                    invalid_mask |= series.index.isin(bad_values.index)
                    issues.append(
                        {
                            "type": "regex_mismatch",
                            "columns": [col],
                            "count": count,
                            "examples": rows_to_examples(bad_values),
                        }
                    )
            if rules_for_col.allowed:
                allowed = set(rules_for_col.allowed)
                non_null = series.dropna()
                mismatch = ~non_null.isin(allowed)
                if mismatch.any():
                    bad_values = non_null[mismatch]
                    count = int(mismatch.sum())
                    invalid_mask |= series.index.isin(bad_values.index)
                    issues.append(
                        {
                            "type": "not_allowed",
                            "columns": [col],
                            "count": count,
                            "examples": rows_to_examples(bad_values),
                        }
                    )
            if rules_for_col.not_future:
                dates = _coerce_date(series)
                future = dates > pd.Timestamp.now()
                if future.any():
                    count = int(future.sum())
                    invalid_mask |= future
                    issues.append(
                        {
                            "type": "date_in_future",
                            "columns": [col],
                            "count": count,
                            "examples": rows_to_examples(series[future]),
                        }
                    )

        invalid_count = int(invalid_mask.sum())
        invalid_pct = invalid_count / len(df) if len(df) else 0.0
        invalid_total += invalid_count
        columns_metrics[col] = {
            "invalid_count": invalid_count,
            "invalid_pct": invalid_pct,
            "inferred_type": inferred_types.get(col, "string"),
        }

    invalid_pct_total = invalid_total / total_cells if total_cells else 0.0
    metrics = {
        "global": {
            "invalid_count": invalid_total,
            "invalid_pct": invalid_pct_total,
        },
        "columns": columns_metrics,
    }
    return metrics, issues
