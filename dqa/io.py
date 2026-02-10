from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PlotPaths:
    missing_by_column: Path
    numeric_distributions: Path | None
    outliers_by_column: Path


def ensure_output_dirs(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def read_csv(input_path: Path, delimiter: str, encoding: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Reading CSV: %s", input_path)
    try:
        df = pd.read_csv(input_path, delimiter=delimiter, encoding=encoding)
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_path)
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to parse CSV: %s", exc)
        raise
    return df


def write_summary_csv(summary: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "summary.csv"
    summary.to_csv(path, index=False)
    return path


def write_issues_csv(issues: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "issues.csv"
    issues.to_csv(path, index=False)
    return path


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def generate_plots(df: pd.DataFrame, out_dir: Path, logger: logging.Logger) -> PlotPaths:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = ensure_output_dirs(out_dir)

    missing = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    missing.plot(kind="bar")
    plt.title("Missing values by column")
    plt.ylabel("Count")
    plt.tight_layout()
    missing_path = plots_dir / "missing_by_column.png"
    plt.savefig(missing_path)
    plt.close()

    numeric_cols = [c for c in df.columns if _safe_numeric_series(df[c]).notna().any() and df[c].dtype not in ['bool', 'object']]
    numeric_path: Path | None = None
    if numeric_cols:
        plt.figure(figsize=(10, 5))
        df[numeric_cols].apply(_safe_numeric_series).plot(kind="hist", bins=20, alpha=0.7)
        plt.title("Numeric distributions")
        plt.xlabel("Value")
        plt.tight_layout()
        numeric_path = plots_dir / "numeric_distributions.png"
        plt.savefig(numeric_path)
        plt.close()

    if numeric_cols:
        outlier_counts = {}
        for col in numeric_cols:
            series = _safe_numeric_series(df[col]).dropna()
            if series.empty:
                outlier_counts[col] = 0
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_counts[col] = int(((series < lower) | (series > upper)).sum())
        plt.figure(figsize=(10, 5))
        pd.Series(outlier_counts).sort_values(ascending=False).plot(kind="bar")
        plt.title("Outliers by column (IQR)")
        plt.ylabel("Count")
        plt.tight_layout()
        outliers_path = plots_dir / "outliers.png"
        plt.savefig(outliers_path)
        plt.close()
    else:
        outliers_path = plots_dir / "outliers.png"
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outliers_path)
        plt.close()

    logger.info("Plots generated under %s", plots_dir)
    return PlotPaths(
        missing_by_column=missing_path,
        numeric_distributions=numeric_path,
        outliers_by_column=outliers_path,
    )


def rows_to_examples(values: Iterable) -> list[str]:
    examples: list[str] = []
    for value in values:
        if len(examples) >= 3:
            break
        examples.append(str(value))
    return examples
