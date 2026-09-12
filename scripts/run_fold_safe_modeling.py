"""Run fold-safe SECOM modeling and write JSON/CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from requests.exceptions import RequestException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fold_safe_modeling import (
    MODELS,
    STRATEGIES,
    FoldSafeConfig,
    evaluate_fold_safe,
    write_fold_safe_results,
)
from src.load_data import load_secom_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--uci-secom", action="store_true", help="Load UCI SECOM dataset 179")
    inputs.add_argument("--input-csv", type=Path, help="CSV containing features and target")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "fold_safe_modeling")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument(
        "--exclude-column",
        action="append",
        default=[],
        help="Identifier or derived-result column to exclude; repeat as needed",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use smaller iteration counts for smoke tests")
    parser.add_argument("--strategy", action="append", choices=STRATEGIES, help="Strategy to run; repeat as needed")
    parser.add_argument("--model", action="append", choices=MODELS, help="Model to run; repeat as needed")
    return parser


def _uci_frame() -> pd.DataFrame:
    features, target = load_secom_data()
    target_frame = target.to_frame() if isinstance(target, pd.Series) else target.copy()
    if len(features) != len(target_frame):
        raise ValueError("UCI SECOM features and target have different row counts.")
    overlap = set(features.columns) & set(target_frame.columns)
    if overlap:
        raise ValueError(f"UCI SECOM features and target overlap: {sorted(overlap)}")
    return pd.concat([features.reset_index(drop=True), target_frame.reset_index(drop=True)], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        frame = _uci_frame() if args.uci_secom else pd.read_csv(args.input_csv)
        config_factory = FoldSafeConfig.quick if args.quick else FoldSafeConfig
        config = config_factory(random_state=args.random_state)
        payload, fold_metrics = evaluate_fold_safe(
            frame,
            target_column=args.target_column,
            timestamp_column=args.timestamp_column,
            excluded_columns=tuple(args.exclude_column),
            config=config,
            strategies=tuple(args.strategy) if args.strategy else STRATEGIES,
            model_names=tuple(args.model) if args.model else MODELS,
        )
        json_path, csv_path = write_fold_safe_results(payload, fold_metrics, args.output_dir)
    except (OSError, ValueError, TypeError, RequestException, pd.errors.ParserError) as exc:
        parser.exit(1, f"error: {exc}\n")
    best = payload["best_mean_pr_auc"]
    print(f"JSON results: {json_path}")
    print(f"CSV results: {csv_path}")
    print(
        "Best mean PR-AUC: "
        f"{best['strategy']} / {best['model']} / {best['value']:.6f} "
        "(ties: strategy order, then model order)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
