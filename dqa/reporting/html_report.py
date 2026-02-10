from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from dqa.scoring import Scores


def render_html_report(
    output_path: Path,
    *,
    context: dict[str, Any],
) -> None:
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    safe_context = dict(context)
    scores = safe_context.get("scores")
    if isinstance(scores, Scores):
        safe_context["scores"] = asdict(scores)

    output_path.write_text(template.render(**safe_context), encoding="utf-8")
