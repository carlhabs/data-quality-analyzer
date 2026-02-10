from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dqa import __version__
from dqa.io import generate_plots, read_csv, write_issues_csv, write_summary_csv
from dqa.profiler import build_report_context, run_profile
from dqa.reporting import render_html_report
from dqa.rules import RulesError, load_rules


def _configure_logging(verbose: bool) -> logging.Logger:
	logger = logging.getLogger("dqa")
	logger.setLevel(logging.DEBUG if verbose else logging.INFO)
	handler = logging.StreamHandler()
	handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
	logger.handlers = [handler]
	return logger


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="dqa", description="Data quality analyzer")
	parser.add_argument("--version", action="version", version=__version__)
	sub = parser.add_subparsers(dest="command", required=True)

	run = sub.add_parser("run", help="Run data quality analysis")
	run.add_argument("--input", required=True, help="Path to input CSV")
	run.add_argument("--out", required=True, help="Output directory")
	run.add_argument("--rules", help="Path to rules.yml")
	run.add_argument(
		"--format",
		default="html,csv",
		help="Output formats: html,csv",
	)
	run.add_argument("--delimiter", default=",", help="CSV delimiter")
	run.add_argument("--encoding", default="utf-8", help="CSV encoding")
	run.add_argument("--id-cols", default="", help="Comma-separated id columns")
	run.add_argument("--target-col", default=None, help="Target column")
	run.add_argument("--verbose", action="store_true", help="Verbose logging")
	return parser


def run_cli(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	logger = _configure_logging(args.verbose)
	input_path = Path(args.input)
	out_dir = Path(args.out)

	try:
		df = read_csv(input_path, args.delimiter, args.encoding, logger)
	except FileNotFoundError:
		return 2
	except Exception:
		return 2

	rules = None
	rules_path = None
	if args.rules:
		rules_path = Path(args.rules)
		try:
			rules = load_rules(rules_path)
		except (FileNotFoundError, RulesError) as exc:
			logger.error("Rules error: %s", exc)
			return 2

	id_cols = [col.strip() for col in args.id_cols.split(",") if col.strip()]
	formats = {fmt.strip().lower() for fmt in args.format.split(",") if fmt.strip()}

	results = run_profile(
		df,
		rules=rules,
		id_cols=id_cols or None,
		target_col=args.target_col,
		logger=logger,
		rules_path=rules_path,
	)

	out_dir.mkdir(parents=True, exist_ok=True)
	write_summary_csv(results["summary"], out_dir)
	write_issues_csv(results["issues"], out_dir)
	plots = generate_plots(df, out_dir, logger)

	if "html" in formats:
		plot_paths = [
			str(Path("plots") / plots.missing_by_column.name),
			str(Path("plots") / plots.outliers_by_column.name),
		]
		if plots.numeric_distributions:
			plot_paths.insert(1, str(Path("plots") / plots.numeric_distributions.name))
		context = build_report_context(results, plot_paths)
		render_html_report(out_dir / "report.html", context=context)

	return 0


def main() -> None:
	raise SystemExit(run_cli())
