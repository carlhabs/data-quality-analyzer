from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from dqa.io import rows_to_examples


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col not in df.columns]


def compute_uniqueness(
    df: pd.DataFrame,
    id_cols: list[str] | None = None,
    unique_keys: list[list[str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rows_pct = duplicate_rows / len(df) if len(df) else 0.0

    metrics: dict[str, Any] = {
        "duplicate_rows_count": duplicate_rows,
        "duplicate_rows_pct": duplicate_rows_pct,
        "duplicates_on_key": {},
    }

    if duplicate_rows:
        examples = rows_to_examples(df.loc[df.duplicated()].index)
        issues.append(
            {
                "type": "duplicate_rows",
                "columns": [],
                "count": duplicate_rows,
                "examples": examples,
            }
        )

    def handle_key(key_cols: list[str], label: str) -> None:
        missing = _missing_columns(df, key_cols)
        if missing:
            issues.append(
                {
                    "type": "missing_columns",
                    "columns": missing,
                    "count": len(missing),
                    "examples": [],
                }
            )
            return
        duplicates = int(df.duplicated(subset=key_cols).sum())
        pct = duplicates / len(df) if len(df) else 0.0
        metrics["duplicates_on_key"][label] = {
            "count": duplicates,
            "pct": pct,
            "columns": key_cols,
        }
        if duplicates:
            examples = rows_to_examples(df.loc[df.duplicated(subset=key_cols)].index)
            issues.append(
                {
                    "type": "duplicate_key",
                    "columns": key_cols,
                    "count": duplicates,
                    "examples": examples,
                }
            )

    if id_cols:
        handle_key(id_cols, "id_cols")

    if unique_keys:
        for idx, cols in enumerate(unique_keys, start=1):
            handle_key(cols, f"unique_key_{idx}")

    return metrics, issues
