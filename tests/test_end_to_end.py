from pathlib import Path

from dqa.cli import run_cli


def test_end_to_end(tmp_path: Path):
    out_dir = tmp_path / "out"
    exit_code = run_cli(
        [
            "run",
            "--input",
            "examples/demo.csv",
            "--rules",
            "examples/rules.yml",
            "--out",
            str(out_dir),
            "--format",
            "html,csv",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "issues.csv").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "plots" / "missing_by_column.png").exists()
    assert (out_dir / "plots" / "outliers.png").exists()

    summary_content = (out_dir / "summary.csv").read_text(encoding="utf-8")
    assert "__global__" in summary_content
    assert "missing_count" in summary_content
