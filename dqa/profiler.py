from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dqa.checks import (
	compute_completeness,
	compute_consistency,
	compute_outliers,
	compute_uniqueness,
	compute_validity,
	infer_column_types,
)
from dqa.scoring import compute_scores
from dqa.rules import validate_column_rules


def run_profile(
	df: pd.DataFrame,
	*,
	rules: Any,
	id_cols: list[str] | None,
	target_col: str | None,
	logger: logging.Logger,
	rules_path: Path | None = None,
) -> dict[str, Any]:
	logger.info("Profiling dataset with %s rows", len(df))
	inferred_types = infer_column_types(df)

	missing_rule_cols = validate_column_rules(df.columns, rules) if rules else []
	if missing_rule_cols:
		logger.warning("Rules reference missing columns: %s", ", ".join(missing_rule_cols))

	completeness_metrics, completeness_issues = compute_completeness(df)
	uniqueness_metrics, uniqueness_issues = compute_uniqueness(
		df, id_cols=id_cols, unique_keys=rules.unique_keys if rules else None
	)
	validity_metrics, validity_issues = compute_validity(df, rules, inferred_types)
	outlier_metrics, outlier_issues = compute_outliers(df)
	consistency_metrics, consistency_issues = compute_consistency(df, rules)

	issues = (
		completeness_issues
		+ uniqueness_issues
		+ validity_issues
		+ outlier_issues
		+ consistency_issues
	)
	for col in missing_rule_cols:
		issues.append(
			{
				"type": "missing_rule_column",
				"columns": [col],
				"count": 1,
				"examples": [],
			}
		)
	issues = sorted(issues, key=lambda item: item.get("count", 0), reverse=True)

	duplicate_key_pct = None
	if uniqueness_metrics.get("duplicates_on_key"):
		duplicate_key_pct = max(
			(entry["pct"] for entry in uniqueness_metrics["duplicates_on_key"].values()),
			default=0.0,
		)

	scores = compute_scores(
		missing_pct=completeness_metrics["global"]["missing_pct"],
		invalid_pct=validity_metrics["global"]["invalid_pct"],
		duplicate_rows_pct=uniqueness_metrics["duplicate_rows_pct"],
		duplicate_key_pct=duplicate_key_pct,
		consistency_invalid_pct=consistency_metrics["global"]["invalid_pct"],
		outlier_pct=outlier_metrics["global"]["outlier_pct"],
	)

	summary_rows: list[dict[str, Any]] = []
	for col in df.columns:
		row = {
			"column": col,
			"inferred_type": inferred_types.get(col, "string"),
		}
		row.update(completeness_metrics["columns"].get(col, {}))
		row.update(validity_metrics["columns"].get(col, {}))
		row.update(outlier_metrics["columns"].get(col, {}))
		summary_rows.append(row)

	summary_rows.append(
		{
			"column": "__global__",
			"missing_count": completeness_metrics["global"]["missing_count"],
			"missing_pct": completeness_metrics["global"]["missing_pct"],
			"invalid_count": validity_metrics["global"]["invalid_count"],
			"invalid_pct": validity_metrics["global"]["invalid_pct"],
			"outlier_count": outlier_metrics["global"]["outlier_count"],
			"outlier_pct": outlier_metrics["global"]["outlier_pct"],
			"duplicate_rows_count": uniqueness_metrics["duplicate_rows_count"],
			"duplicate_rows_pct": uniqueness_metrics["duplicate_rows_pct"],
			"consistency_invalid_count": consistency_metrics["global"]["invalid_count"],
			"consistency_invalid_pct": consistency_metrics["global"]["invalid_pct"],
			"score_total": scores.total,
		}
	)

	summary_df = pd.DataFrame(summary_rows)
	issues_df = pd.DataFrame(issues)

	metadata = {
		"row_count": len(df),
		"column_count": len(df.columns),
		"memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
		"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
		"rules_path": str(rules_path) if rules_path else None,
		"id_cols": id_cols,
		"target_col": target_col,
	}

	return {
		"summary": summary_df,
		"issues": issues_df,
		"scores": scores,
		"metrics": {
			"completeness": completeness_metrics,
			"uniqueness": uniqueness_metrics,
			"validity": validity_metrics,
			"outliers": outlier_metrics,
			"consistency": consistency_metrics,
		},
		"metadata": metadata,
		"inferred_types": inferred_types,
	}


def build_report_context(results: dict[str, Any], plot_paths: list[str]) -> dict[str, Any]:
	completeness_rows = []
	for col, metrics in results["metrics"]["completeness"]["columns"].items():
		completeness_rows.append(
			{
				"column": col,
				"missing_count": metrics.get("missing_count", 0),
				"missing_pct": metrics.get("missing_pct", 0.0),
			}
		)

	issues = results["issues"].head(20).to_dict(orient="records")

	return {
		"scores": results["scores"],
		"metadata": results["metadata"],
		"inferred_types": results["inferred_types"],
		"completeness_table": completeness_rows,
		"issues": issues,
		"plots": plot_paths,
	}
