"""Run resumable repeated fold-safe CV candidates and finalize stability results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from requests.exceptions import RequestException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fold_safe_modeling import FoldSafeConfig
from src.load_data import load_secom_data
from src.repeated_cv import (
    RepeatedCVConfig,
    evaluate_repeated_candidate,
    finalize_parts,
    load_validation_results,
    make_repeated_splits,
    prepare_repeated_data,
    select_candidates,
    split_fingerprint,
    validate_part,
    write_part,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-validation-results", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--uci-secom", action="store_true", help="Load UCI SECOM; default when no CSV is supplied")
    inputs.add_argument("--input-csv", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--candidate", help="Run one shortlist candidate as strategy:model")
    actions.add_argument("--all-candidates", action="store_true", help="Run all shortlist candidates sequentially")
    actions.add_argument("--finalize", action="store_true", help="Merge complete compatible parts")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "repeated_cv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--exclude-column", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
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


def _parse_candidate(value: str, shortlist: list[dict]) -> tuple[str, str]:
    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError("--candidate must use strategy:model format.")
    candidate = (pieces[0], pieces[1])
    allowed = {(item["strategy"], item["model"]) for item in shortlist}
    if candidate not in allowed:
        raise ValueError(f"Candidate is not in the deterministic shortlist: {value}")
    return candidate


def _safe_output_dir(value: Path) -> Path:
    output = value.resolve()
    if output == PROJECT_ROOT or PROJECT_ROOT.is_relative_to(output):
        raise ValueError("--output-dir must not be the project root or one of its parents.")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validation = load_validation_results(args.from_validation_results)
        shortlist = select_candidates(validation)
        output_dir = _safe_output_dir(args.output_dir)
        parts_dir = output_dir / "parts"
        if args.finalize:
            result, folds, repeats, pairs = finalize_parts(parts_dir, shortlist, output_dir)
            print(f"Candidates: {len(shortlist)}; completed splits: {len(folds)}")
            print(f"JSON results: {output_dir / 'repeated_cv_results.json'}")
            print(f"Best mean PR-AUC: {result['best_mean_pr_auc']}")
            return 0

        frame = pd.read_csv(args.input_csv) if args.input_csv else _uci_frame()
        repeated = RepeatedCVConfig(args.n_splits, args.n_repeats, args.random_state)
        modeling = FoldSafeConfig(random_state=args.random_state, n_jobs=1)
        candidates = (
            [_parse_candidate(args.candidate, shortlist)]
            if args.candidate
            else [(item["strategy"], item["model"]) for item in shortlist]
        )
        for candidate in candidates:
            part_dir = parts_dir / f"{candidate[0]}__{candidate[1]}"
            if part_dir.exists():
                if not args.resume:
                    raise ValueError(f"Completed or partial part already exists; use --resume after validation: {part_dir}")
                X, y, dataset = prepare_repeated_data(
                    frame,
                    repeated,
                    target_column=args.target_column,
                    timestamp_column=args.timestamp_column,
                    excluded_columns=tuple(args.exclude_column),
                )
                expected_splits = make_repeated_splits(X, y, repeated)
                payload, rows = validate_part(
                    part_dir,
                    expected_candidate=candidate,
                    expected_dataset_fingerprint=dataset["dataset_fingerprint"],
                    expected_split_fingerprint=split_fingerprint(expected_splits),
                )
                if payload["repeated_cv_config"] != vars(repeated) or payload["modeling_config"] != vars(modeling):
                    raise ValueError("Existing part configuration does not match the requested run.")
                print(f"Reused complete part: {candidate[0]}:{candidate[1]} ({len(rows)} splits)")
                continue
            payload, rows = evaluate_repeated_candidate(
                frame,
                candidate,
                repeated_config=repeated,
                modeling_config=modeling,
                target_column=args.target_column,
                timestamp_column=args.timestamp_column,
                excluded_columns=tuple(args.exclude_column),
                shortlist=shortlist,
            )
            json_path, csv_path = write_part(payload, rows, part_dir)
            validate_part(part_dir, expected_candidate=candidate)
            print(f"Completed {candidate[0]}:{candidate[1]} ({len(rows)} splits)")
            print(f"Part JSON: {json_path}")
            print(f"Part CSV: {csv_path}")
        if len(candidates) != len(shortlist):
            print("Partial execution only; run remaining candidates before --finalize.")
        return 0
    except (OSError, ValueError, TypeError, KeyError, RequestException, pd.errors.ParserError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
