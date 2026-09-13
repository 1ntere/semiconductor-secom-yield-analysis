"""Run nested-CV cost-based threshold sensitivity analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys

import pandas as pd
from requests.exceptions import RequestException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cost_threshold import (
    CostScenario, CostThresholdConfig, evaluate_cost_thresholds, write_cost_threshold_results,
)
from src.fold_safe_modeling import FoldSafeConfig
from src.load_data import load_secom_data


def _cost(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("cost must be a positive finite number")
    return parsed


def _scenario(value: str) -> CostScenario:
    try:
        cost_fn, cost_fp = value.split(":", 1)
        return CostScenario(_cost(cost_fp), _cost(cost_fn), f"fn{float(cost_fn):g}_fp{float(cost_fp):g}")
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("scenario must be positive finite FN:FP, for example 5:1") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--uci-secom", action="store_true", help="Load UCI SECOM dataset 179")
    inputs.add_argument("--input-csv", type=Path, help="CSV containing numeric features and target")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "cost_threshold")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--exclude-column", action="append", default=[])
    parser.add_argument("--cost-scenario", action="append", type=_scenario, metavar="FN:FP",
                        help="Positive finite cost ratio; repeat as needed (defaults: 1:1, 5:1, 10:1)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Use smaller estimator counts for tests only")
    return parser


def _uci_frame() -> pd.DataFrame:
    features, target = load_secom_data()
    target_frame = target.to_frame() if isinstance(target, pd.Series) else target.copy()
    if len(features) != len(target_frame):
        raise ValueError("UCI SECOM features and target have different row counts.")
    return pd.concat([features.reset_index(drop=True), target_frame.reset_index(drop=True)], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = args.output_dir.resolve()
        if output == PROJECT_ROOT or PROJECT_ROOT.is_relative_to(output):
            raise ValueError("--output-dir must not be the project root or one of its parents.")
        if shutil.disk_usage(PROJECT_ROOT).free < 100 * 1024 * 1024:
            raise OSError("Less than 100 MiB free; cost-threshold analysis was not started.")
        frame = _uci_frame() if args.uci_secom else pd.read_csv(args.input_csv)
        evaluation = CostThresholdConfig(
            random_state=args.random_state, outer_splits=args.outer_splits, inner_splits=args.inner_splits,
        )
        factory = FoldSafeConfig.quick if args.quick else FoldSafeConfig
        modeling = factory(random_state=args.random_state, n_splits=args.outer_splits)
        scenarios = tuple(args.cost_scenario) if args.cost_scenario else None
        payload, rows = evaluate_cost_thresholds(
            frame, target_column=args.target_column, timestamp_column=args.timestamp_column,
            excluded_columns=tuple(args.exclude_column), scenarios=scenarios,
            config=evaluation, modeling_config=modeling,
        )
        json_path, csv_path = write_cost_threshold_results(payload, rows, output)
    except (OSError, ValueError, TypeError, RequestException, pd.errors.ParserError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"JSON results: {json_path}")
    print(f"CSV results: {csv_path}")
    print("Outer hold-out evaluated: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
