import pandas as pd

from dqa.checks.completeness import compute_completeness
from dqa.checks.outliers import compute_outliers
from dqa.checks.uniqueness import compute_uniqueness


def test_completeness_missing_counts():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
    metrics, issues = compute_completeness(df)
    assert metrics["global"]["missing_count"] == 3
    assert metrics["columns"]["a"]["missing_count"] == 1
    assert metrics["columns"]["b"]["missing_count"] == 2
    assert any(issue["type"] == "missing_values" for issue in issues)


def test_uniqueness_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2], "value": [10, 10, 20]})
    metrics, issues = compute_uniqueness(df, id_cols=["id"])
    assert metrics["duplicate_rows_count"] == 1
    assert metrics["duplicates_on_key"]["id_cols"]["count"] == 1
    assert any(issue["type"] == "duplicate_rows" for issue in issues)


def test_outliers_iqr():
    df = pd.DataFrame({"amount": [10, 12, 11, 13, 500]})
    metrics, issues = compute_outliers(df)
    assert metrics["columns"]["amount"]["outlier_count"] == 1
    assert any(issue["type"] == "outliers" for issue in issues)
