"""Run exploratory SECOM temporal validation and feature drift analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd
from requests.exceptions import RequestException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fold_safe_modeling import FoldSafeConfig
from src.load_data import load_secom_data
from src.temporal_validation import TemporalConfig, evaluate_temporal_validation, write_temporal_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--uci-secom", action="store_true", help="Load UCI SECOM dataset 179")
    inputs.add_argument("--input-csv", type=Path, help="CSV containing features, timestamp, and target")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "temporal_validation")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--exclude-column", action="append", default=[])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--drift-bins", type=int, default=10)
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
            raise OSError("Less than 100 MiB free; temporal validation was not started.")
        frame = _uci_frame() if args.uci_secom else pd.read_csv(args.input_csv)
        config = TemporalConfig(
            random_state=args.random_state, n_splits=args.n_splits, drift_bins=args.drift_bins,
        )
        factory = FoldSafeConfig.quick if args.quick else FoldSafeConfig
        modeling = factory(random_state=args.random_state)
        payload, performance, drift = evaluate_temporal_validation(
            frame, target_column=args.target_column, timestamp_column=args.timestamp_column,
            excluded_columns=tuple(args.exclude_column), config=config, modeling_config=modeling,
        )
        paths = write_temporal_results(payload, performance, drift, output)
    except (OSError, ValueError, TypeError, RequestException, pd.errors.ParserError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"JSON results: {paths[0]}")
    print(f"Performance CSV: {paths[1]}")
    print(f"Drift CSV: {paths[2]}")
    print(f"Evaluated splits: {(performance.status == 'evaluated').sum()}/{len(performance)}")
    print("Random outer hold-out evaluated: no")
    print(performance.to_csv(index=False).strip())
    ranked_drift = drift.dropna(subset=["psi"]).sort_values(
        ["split", "psi", "feature"], ascending=[True, False, True]
    ).groupby("split", sort=True).head(5)
    print("Top PSI features by split:")
    print(ranked_drift.loc[:, ["split", "feature", "psi"]].to_csv(index=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
