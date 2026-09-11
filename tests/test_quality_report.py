import copy
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from src.data_quality import DataQualityConfig, DataQualityReport, QualityCheck, validate_data_quality
from src.quality_report import render_html_report, write_quality_reports
from scripts import generate_quality_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "scripts" / "generate_quality_report.py"


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["01/01/2024", "02/01/2024", "03/01/2024", "04/01/2024"],
            "sensor_a": [1.0, 2.0, 3.0, 4.0],
            "sensor_b": [4.0, 3.0, 2.0, 1.0],
            "class": [-1, -1, 1, 1],
        }
    )


def make_config(**overrides) -> DataQualityConfig:
    defaults = {"minimum_minority_samples": 1, "minimum_minority_ratio": 0.10}
    defaults.update(overrides)
    return DataQualityConfig.secom(**defaults)


def test_strict_json_contains_summary_and_applied_config(tmp_path):
    config = make_config(feature_missing_error_rate=0.25)
    report = validate_data_quality(make_frame(), config)

    json_path, _ = write_quality_reports(report, config, tmp_path)
    text = json_path.read_text(encoding="utf-8")
    payload = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    assert payload["schema_version"] == "1.0"
    assert payload["report"]["summary"] == report.to_dict()["summary"]
    assert set(payload["report"]["summary"]) == {"error", "warning", "pass", "not_evaluated"}
    assert payload["config"]["feature_missing_error_rate"] == 0.25
    assert text == json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def test_html_is_standalone_and_displays_all_checks_and_summary(tmp_path):
    config = make_config()
    report = validate_data_quality(make_frame(), config)

    _, html_path = write_quality_reports(report, config, tmp_path)
    html = html_path.read_text(encoding="utf-8")

    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()
    assert "http://" not in html.lower() and "https://" not in html.lower()
    assert html.count('<article class="check ') == 16
    for check in report.checks:
        assert f"<h2>{check.name}</h2>" in html
    for label in ("Overall status", "Error", "Warning", "Pass", "Not evaluated"):
        assert label in html
    for status, count in report.to_dict()["summary"].items():
        assert f'class="summary-card {status}"' in html
        assert f"<strong>{count}</strong>" in html

    actual_counts = {
        status: sum(check.status == status for check in report.checks)
        for status in ("error", "warning", "pass", "not_evaluated")
    }
    assert sum(actual_counts.values()) == 16
    assert report.to_dict()["summary"] == actual_counts


def test_html_orders_status_groups_while_preserving_schema_order():
    config = make_config()
    original = validate_data_quality(make_frame(), config)
    statuses = ("pass", "error", "warning", "not_evaluated")
    report = DataQualityReport(
        checks=[
            replace(check, status=statuses[index % len(statuses)])
            for index, check in enumerate(original.checks)
        ]
    )

    html = render_html_report(report, config)
    rendered_names = re.findall(
        r'<article class="check [^"]+">\s*<header><h2>(.*?)</h2>', html
    )
    expected_names = [
        check.name
        for status in ("error", "warning", "not_evaluated", "pass")
        for check in report.checks
        if check.status == status
    ]

    assert rendered_names == expected_names


def test_html_escapes_all_report_derived_fields():
    unsafe = "<tag attr=\"value\">Tom & Jerry's</tag>"
    config = DataQualityConfig(required_columns=(unsafe,))
    report = DataQualityReport(
        checks=[
            QualityCheck(
                name="escaping_check",
                status="error",
                criterion={"criterion": unsafe},
                observed={"observed": unsafe},
                message=unsafe,
                columns=(unsafe,),
            )
        ]
    )

    html = render_html_report(report, config, title=unsafe)

    escaped = escape(unsafe)
    criterion_html = escape(
        json.dumps({"criterion": unsafe}, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
    )
    observed_html = escape(
        json.dumps({"observed": unsafe}, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
    )
    assert html.count(escaped) >= 4
    assert criterion_html in html
    assert observed_html in html
    assert unsafe not in html
    assert '<tag attr="value">' not in html


def test_long_column_lists_use_collapsible_preview():
    frame = pd.DataFrame(
        {f"sensor_{index}": [f"value-{index}-{row}" for row in range(4)] for index in range(12)}
    )
    config = DataQualityConfig(feature_columns=tuple(frame.columns))
    report = validate_data_quality(frame, config)

    html = render_html_report(report, config)

    assert "<details>" in html
    assert "(12 columns total)" in html
    assert "sensor_11" in html


def test_error_report_has_error_overall_status():
    frame = make_frame()
    frame.loc[0, "sensor_a"] = np.inf
    config = make_config()
    report = validate_data_quality(frame, config)

    html = render_html_report(report, config)

    assert not report.passed
    assert '<section class="summary-card error"><span>Overall status</span><strong>Error</strong>' in html


def test_warning_and_not_evaluated_do_not_fail_report():
    frame = pd.DataFrame({"constant_sensor": [1.0, 1.0, 1.0]})
    config = DataQualityConfig()
    report = validate_data_quality(frame, config)

    html = render_html_report(report, config)

    assert report.passed
    assert report.warnings
    assert report.not_evaluated
    assert "Passed with warnings" in html


def test_report_generation_does_not_mutate_inputs(tmp_path):
    config = make_config()
    report = validate_data_quality(make_frame(), config)
    report_before = copy.deepcopy(report)
    config_before = copy.deepcopy(config)

    write_quality_reports(report, config, tmp_path)

    assert report == report_before
    assert config == config_before


def test_csv_cli_smoke_test(tmp_path):
    csv_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    pd.DataFrame({"sensor_a": [1.0, 2.0], "sensor_b": [3.0, 4.0]}).to_csv(
        csv_path, index=False
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--input-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--json-name",
            "smoke.json",
            "--html-name",
            "smoke.html",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "smoke.json").is_file()
    assert (output_dir / "smoke.html").is_file()
    assert "Passed: True" in result.stdout


def test_cli_rejects_mutually_exclusive_inputs(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("sensor\n1\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--input-csv", str(csv_path), "--uci-secom"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_reports_missing_csv_without_traceback(tmp_path):
    missing_path = tmp_path / "does-not-exist.csv"

    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--input-csv", str(missing_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "error:" in result.stderr
    assert missing_path.name in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_rejects_unsafe_output_names_without_writing_outside_output_dir(tmp_path):
    csv_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    csv_path.write_text("sensor\n1\n2\n", encoding="utf-8")
    cases = [
        ("--json-name", str(tmp_path / "absolute.json")),
        ("--html-name", str(tmp_path / "absolute.html")),
        ("--json-name", "../escaped.json"),
        ("--html-name", "../escaped.html"),
        ("--json-name", "/tmp/escaped.json"),
        ("--html-name", "C:\\temp\\escaped.html"),
    ]

    for option, unsafe_name in cases:
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--input-csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                option,
                unsafe_name,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "must be a plain" in result.stderr
        assert "Traceback" not in result.stderr

    assert not (tmp_path / "absolute.json").exists()
    assert not (tmp_path / "absolute.html").exists()
    assert not (tmp_path / "escaped.json").exists()
    assert not (tmp_path / "escaped.html").exists()


def test_uci_loader_combines_original_columns_without_download(monkeypatch):
    features = pd.DataFrame(
        {"timestamp": ["01/01/2024", "02/01/2024"], "Attribute 1": [1.0, 2.0]},
        index=[10, 20],
    )
    target = pd.DataFrame({"class": [-1, 1]}, index=[30, 40])
    monkeypatch.setattr(
        generate_quality_report,
        "load_secom_data",
        lambda: (features, target),
    )

    combined = generate_quality_report._load_uci_secom()

    assert combined.columns.tolist() == ["timestamp", "Attribute 1", "class"]
    assert combined.to_dict(orient="list") == {
        "timestamp": ["01/01/2024", "02/01/2024"],
        "Attribute 1": [1.0, 2.0],
        "class": [-1, 1],
    }
    assert combined.index.tolist() == [0, 1]


def test_cli_fail_on_error_returns_nonzero_after_writing_reports(tmp_path):
    csv_path = tmp_path / "bad.csv"
    output_dir = tmp_path / "output"
    pd.DataFrame({"text_sensor": ["a", "b"]}).to_csv(csv_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--input-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--fail-on-error",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert (output_dir / "data_quality_report.json").is_file()
    assert (output_dir / "data_quality_report.html").is_file()
    assert "Passed: False" in result.stdout


def test_warning_and_not_evaluated_cli_does_not_fail(tmp_path):
    csv_path = tmp_path / "warning.csv"
    pd.DataFrame({"constant_sensor": [1.0, 1.0]}).to_csv(csv_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--input-csv",
            str(csv_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--fail-on-error",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Passed: True" in result.stdout
