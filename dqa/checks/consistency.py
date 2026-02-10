from __future__ import annotations

from typing import Any

import pandas as pd

from dqa.io import rows_to_examples
from dqa.rules import Rules, RulesError, evaluate_row_rule


def compute_consistency(df: pd.DataFrame, rules: Rules | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    total_invalid = 0

    if not rules or not rules.row_rules:
        return {
            "global": {"invalid_count": 0, "invalid_pct": 0.0},
            "rules": {},
        }, issues

    rule_metrics: dict[str, dict[str, Any]] = {}
    for rule in rules.row_rules:
        try:
            result = evaluate_row_rule(rule.expr, df)
        except RulesError as exc:
            issues.append(
                {
                    "type": "row_rule_error",
                    "columns": [],
                    "count": 0,
                    "examples": [str(exc)],
                }
            )
            rule_metrics[rule.name] = {
                "invalid_count": 0,
                "invalid_pct": 0.0,
                "expr": rule.expr,
            }
            continue

        invalid_mask = ~result.fillna(False)
        invalid_count = int(invalid_mask.sum())
        invalid_pct = invalid_count / len(df) if len(df) else 0.0
        total_invalid += invalid_count
        rule_metrics[rule.name] = {
            "invalid_count": invalid_count,
            "invalid_pct": invalid_pct,
            "expr": rule.expr,
        }
        if invalid_count:
            issues.append(
                {
                    "type": "row_rule_violation",
                    "columns": [],
                    "count": invalid_count,
                    "examples": rows_to_examples(df.index[invalid_mask]),
                }
            )

    total_invalid_pct = total_invalid / len(df) if len(df) else 0.0
    metrics = {
        "global": {"invalid_count": total_invalid, "invalid_pct": total_invalid_pct},
        "rules": rule_metrics,
    }
    return metrics, issues
