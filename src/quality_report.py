"""Strict JSON and standalone HTML output for data-quality reports."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_quality import DataQualityConfig, DataQualityReport


STATUS_ORDER = {"error": 0, "warning": 1, "not_evaluated": 2, "pass": 3}
STATUS_LABELS = {
    "error": "Error",
    "warning": "Warning",
    "not_evaluated": "Not evaluated",
    "pass": "Pass",
}


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, Mapping):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, bool)):
        return value
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def config_to_dict(config: DataQualityConfig) -> dict[str, Any]:
    """Return a strict-JSON-safe snapshot of the applied configuration."""

    return _json_safe(asdict(config))


def build_json_payload(
    report: DataQualityReport,
    config: DataQualityConfig,
) -> dict[str, Any]:
    """Build a serializable report envelope without mutating its inputs."""

    return {
        "schema_version": "1.0",
        "config": config_to_dict(config),
        "report": report.to_dict(),
    }


def write_json_report(
    report: DataQualityReport,
    config: DataQualityConfig,
    output_path: str | Path,
) -> Path:
    """Write a deterministic, strict JSON report as UTF-8."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        build_json_payload(report, config),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{content}\n", encoding="utf-8")
    return path


def _pretty_json(value: Any) -> str:
    return escape(
        json.dumps(_json_safe(value), allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
    )


def _render_columns(columns: list[str], preview_limit: int = 8) -> str:
    escaped_columns = [escape(str(column)) for column in columns]
    if not escaped_columns:
        return '<span class="muted">None</span>'
    if len(escaped_columns) <= preview_limit:
        return ", ".join(escaped_columns)
    preview = ", ".join(escaped_columns[:preview_limit])
    full_list = "".join(f"<li>{column}</li>" for column in escaped_columns)
    return (
        f"<details><summary>{preview}, … ({len(columns)} columns total)</summary>"
        f"<ul>{full_list}</ul></details>"
    )


def _overall_status(summary: Mapping[str, int]) -> tuple[str, str]:
    if summary["error"]:
        return "Error", "error"
    if summary["warning"]:
        return "Passed with warnings", "warning"
    if summary["not_evaluated"]:
        return "Passed — checks not evaluated", "not_evaluated"
    return "Passed", "pass"


def render_html_report(
    report: DataQualityReport,
    config: DataQualityConfig,
    *,
    title: str = "Manufacturing Data Quality Report",
) -> str:
    """Render a self-contained HTML report with no external dependencies."""

    report_payload = report.to_dict()
    summary = report_payload["summary"]
    overall_text, overall_class = _overall_status(summary)
    sorted_checks = sorted(
        report_payload["checks"],
        key=lambda check: STATUS_ORDER[check["status"]],
    )
    cards = [
        ("Overall status", overall_text, overall_class),
        ("Error", str(summary["error"]), "error"),
        ("Warning", str(summary["warning"]), "warning"),
        ("Pass", str(summary["pass"]), "pass"),
        ("Not evaluated", str(summary["not_evaluated"]), "not_evaluated"),
    ]
    cards_html = "".join(
        f'<section class="summary-card {css_class}"><span>{escape(label)}</span>'
        f"<strong>{escape(value)}</strong></section>"
        for label, value, css_class in cards
    )
    checks_html = "".join(
        f'''<article class="check {escape(check["status"])}">
          <header><h2>{escape(check["name"])}</h2>
            <span class="badge {escape(check["status"])}">{STATUS_LABELS[check["status"]]}</span>
          </header>
          <p class="message">{escape(str(check["message"]))}</p>
          <dl>
            <dt>Criterion</dt><dd><pre>{_pretty_json(check["criterion"])}</pre></dd>
            <dt>Observed</dt><dd><pre>{_pretty_json(check["observed"])}</pre></dd>
            <dt>Related columns</dt><dd>{_render_columns(check["columns"])}</dd>
          </dl>
        </article>'''
        for check in sorted_checks
    )
    safe_title = escape(title)
    config_json = _pretty_json(config_to_dict(config))
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fa; --ink:#17202a; --muted:#667085;
      --error:#b42318; --error-bg:#fef3f2; --warning:#b54708; --warning-bg:#fffaeb;
      --pass:#067647; --pass-bg:#ecfdf3; --na:#475467; --na-bg:#f2f4f7; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
    main {{ width:min(1120px,calc(100% - 32px)); margin:32px auto 64px; }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    .subtitle,.muted {{ color:var(--muted); }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:24px 0; }}
    .summary-card,.check,.config {{ background:white; border:1px solid #d0d5dd; border-radius:10px; }}
    .summary-card {{ padding:16px; border-top:5px solid; }}
    .summary-card span {{ display:block; color:var(--muted); font-size:13px; }}
    .summary-card strong {{ display:block; margin-top:5px; font-size:21px; }}
    .error {{ border-color:var(--error); }} .warning {{ border-color:var(--warning); }}
    .pass {{ border-color:var(--pass); }} .not_evaluated {{ border-color:var(--na); }}
    .checks {{ display:grid; gap:14px; }} .check {{ padding:18px; border-left-width:6px; }}
    .check header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; }}
    h2 {{ margin:0; font-size:18px; }} .message {{ margin:8px 0 14px; }}
    .badge {{ border-radius:999px; padding:4px 10px; color:white; font-weight:700; white-space:nowrap; }}
    .badge.error {{ background:var(--error); }} .badge.warning {{ background:var(--warning); }}
    .badge.pass {{ background:var(--pass); }} .badge.not_evaluated {{ background:var(--na); }}
    dl {{ display:grid; grid-template-columns:140px minmax(0,1fr); gap:8px 14px; margin:0; }}
    dt {{ font-weight:700; }} dd {{ margin:0; min-width:0; }}
    pre {{ margin:0; padding:10px; overflow:auto; background:#f8fafc; border-radius:6px; white-space:pre-wrap; overflow-wrap:anywhere; }}
    details summary {{ cursor:pointer; color:#344054; }} ul {{ columns:2; padding-left:22px; }}
    .config {{ margin-top:24px; padding:18px; }}
    @media (max-width:640px) {{ dl {{ grid-template-columns:1fr; }} ul {{ columns:1; }} }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <p class="subtitle">Standalone report. Status is communicated with text and color.</p>
    <section class="summary" aria-label="Quality summary">{cards_html}</section>
    <section class="checks" aria-label="Quality checks">{checks_html}</section>
    <section class="config"><h2>Applied configuration</h2><pre>{config_json}</pre></section>
  </main>
</body>
</html>
'''


def write_html_report(
    report: DataQualityReport,
    config: DataQualityConfig,
    output_path: str | Path,
    *,
    title: str = "Manufacturing Data Quality Report",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(report, config, title=title), encoding="utf-8")
    return path


def write_quality_reports(
    report: DataQualityReport,
    config: DataQualityConfig,
    output_dir: str | Path,
    *,
    json_name: str = "data_quality_report.json",
    html_name: str = "data_quality_report.html",
    title: str = "Manufacturing Data Quality Report",
) -> tuple[Path, Path]:
    """Write both formats and return ``(json_path, html_path)``."""

    for name, suffix in ((json_name, ".json"), (html_name, ".html")):
        if Path(name).name != name or not name.lower().endswith(suffix):
            raise ValueError(f"{name!r} must be a plain {suffix} filename.")
    directory = Path(output_dir)
    json_path = write_json_report(report, config, directory / json_name)
    html_path = write_html_report(report, config, directory / html_name, title=title)
    return json_path, html_path
