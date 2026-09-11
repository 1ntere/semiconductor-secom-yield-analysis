"""Generate strict JSON and standalone HTML data-quality reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from requests.exceptions import RequestException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_quality import DataQualityConfig, validate_data_quality
from src.load_data import load_secom_data
from src.quality_report import write_quality_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-csv", type=Path, help="CSV file to validate")
    inputs.add_argument("--uci-secom", action="store_true", help="Load UCI SECOM dataset 179")
    parser.add_argument(
        "--preset", choices=("general", "secom"),
        help="Validation preset; CSV defaults to general and UCI input defaults to secom",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "data_quality")
    parser.add_argument("--json-name", default="data_quality_report.json")
    parser.add_argument("--html-name", default="data_quality_report.html")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def _load_uci_secom() -> pd.DataFrame:
    features, target = load_secom_data()
    target_frame = target.to_frame() if isinstance(target, pd.Series) else target.copy()
    if len(features) != len(target_frame):
        raise ValueError("UCI SECOM features and target have different row counts.")
    overlap = set(features.columns) & set(target_frame.columns)
    if overlap:
        raise ValueError(f"UCI SECOM features and target overlap: {sorted(overlap)}")
    return pd.concat(
        [features.reset_index(drop=True), target_frame.reset_index(drop=True)], axis=1
    )


def _select_config(args: argparse.Namespace) -> DataQualityConfig:
    preset = args.preset or ("secom" if args.uci_secom else "general")
    if args.uci_secom and preset != "secom":
        raise ValueError("--uci-secom requires the secom preset.")
    return DataQualityConfig.secom() if preset == "secom" else DataQualityConfig()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _select_config(args)
        if args.uci_secom:
            frame = _load_uci_secom()
            title = "UCI SECOM Data Quality Report"
        else:
            frame = pd.read_csv(args.input_csv)
            title = "CSV Data Quality Report"
        report = validate_data_quality(frame, config)
        json_path, html_path = write_quality_reports(
            report,
            config,
            args.output_dir,
            json_name=args.json_name,
            html_name=args.html_name,
            title=title,
        )
    except (OSError, ValueError, RequestException, pd.errors.ParserError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    print(f"Passed: {report.passed}")
    if args.fail_on_error and report.errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
