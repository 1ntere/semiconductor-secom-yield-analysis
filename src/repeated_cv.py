"""Repeated fold-safe CV, resumable parts, and paired stability summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.fold_safe_modeling import (
    METRICS,
    MODELS,
    STRATEGIES,
    FoldSafeConfig,
    _feature_audit,
    _json_safe,
    _metric_values,
    _scores,
    _validate_numeric_features,
    build_feature_pipeline,
    build_models,
)


Candidate = tuple[str, str]
FOLD_COLUMNS = (
    "strategy", "model", "repeat_id", "fold_id", "split_id",
    "validation_fingerprint", "train_size", "validation_size",
    "feature_count_before", "feature_count_after_missing",
    "feature_count_after_constant", "feature_count_after_correlation",
    "feature_count_selected", "elapsed_seconds", *METRICS,
)
REPEAT_COLUMNS = ("strategy", "model", "repeat_id", *METRICS)
PAIRWISE_COLUMNS = (
    "reference_strategy", "reference_model", "comparison_strategy", "comparison_model",
    "split_id", "repeat_id", "fold_id", "reference_pr_auc", "comparison_pr_auc", "delta",
)


@dataclass(frozen=True)
class RepeatedCVConfig:
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42
    outer_holdout_size: float = 0.20
    std_ddof: int = 1

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")
        if not 0 < self.outer_holdout_size < 1:
            raise ValueError("outer_holdout_size must be between 0 and 1.")


def load_validation_results(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(
        source.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    results = payload.get("results")
    expected = {(strategy, model) for strategy in STRATEGIES for model in MODELS}
    actual = {
        (item.get("strategy"), item.get("model"))
        for item in results or []
    }
    if payload.get("schema_version") != "1.0" or len(results or []) != 16 or actual != expected:
        raise ValueError("Validation results must contain exactly the canonical 16 combinations.")
    return payload


def select_candidates(validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Apply the frozen deterministic shortlist rule to Validation 3.1 results."""

    results = list(validation["results"])
    order = {(strategy, model): (s, m) for s, strategy in enumerate(STRATEGIES) for m, model in enumerate(MODELS)}

    def key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        canonical = order[(item["strategy"], item["model"])]
        metric = item["metrics"]["pr_auc"]
        return (-metric["mean"], metric["std"], *canonical)

    dummy = next(item for item in results if item["strategy"] == "all_features" and item["model"] == "dummy_prior")
    non_dummy = sorted((item for item in results if item["model"] != "dummy_prior"), key=key)
    best = non_dummy[0]
    diverse = next(item for item in non_dummy if item["model"] != best["model"])
    selected = [dummy, best, diverse]
    feature_candidates = sorted(
        (
            item for item in non_dummy
            if item["strategy"] in {"correlation_pruned", "mi_selected", "model_selected"}
        ),
        key=key,
    )
    selected.append(next(item for item in feature_candidates if item not in selected))
    reasons = (
        "fixed all-features dummy baseline",
        "highest Validation 3.1 mean PR-AUC among non-dummy combinations",
        "highest mean PR-AUC using a classifier different from the overall best",
        "highest remaining feature-selection non-dummy combination",
    )
    return [
        {
            "strategy": item["strategy"],
            "model": item["model"],
            "validation_mean_pr_auc": item["metrics"]["pr_auc"]["mean"],
            "validation_std_pr_auc": item["metrics"]["pr_auc"]["std"],
            "reason": reason,
        }
        for item, reason in zip(selected, reasons)
    ]


def _fingerprint_values(values: Sequence[int]) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def prepare_repeated_data(
    frame: pd.DataFrame,
    config: RepeatedCVConfig,
    *,
    target_column: str = "class",
    timestamp_column: str | None = "timestamp",
    excluded_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    unknown = [column for column in excluded_columns if column not in frame.columns]
    if unknown:
        raise ValueError(f"Excluded columns were not found: {unknown}")
    excluded = {target_column, *excluded_columns}
    if timestamp_column is not None:
        excluded.add(timestamp_column)
    feature_columns = [column for column in frame.columns if column not in excluded]
    X = frame.loc[:, feature_columns].copy()
    y = frame[target_column].copy()
    _validate_numeric_features(X)
    if y.isna().any() or set(y.unique()) != {-1, 1}:
        raise ValueError("Target must contain exactly {-1, 1} with no missing values.")
    positions = np.arange(len(frame))
    train_positions, holdout_positions = train_test_split(
        positions,
        test_size=config.outer_holdout_size,
        random_state=config.random_state,
        stratify=y,
    )
    X_train = X.iloc[train_positions].reset_index(drop=True)
    y_train = y.iloc[train_positions].reset_index(drop=True)
    if int(y_train.value_counts().min()) < config.n_splits:
        raise ValueError("Each class must have at least n_splits outer-train samples.")
    metadata = {
        "dataset_shape": list(frame.shape),
        "dataset_fingerprint": _dataset_fingerprint(frame),
        "target_distribution": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "modeling_feature_count": len(feature_columns),
        "outer_train_size": len(train_positions),
        "outer_holdout_size": len(holdout_positions),
        "outer_holdout_evaluated": False,
    }
    return X_train, y_train, metadata


def make_repeated_splits(X: pd.DataFrame, y: pd.Series, config: RepeatedCVConfig) -> list[dict[str, Any]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )
    splits = []
    for split_id, (train_indices, validation_indices) in enumerate(splitter.split(X, y)):
        splits.append({
            "split_id": split_id,
            "repeat_id": split_id // config.n_splits,
            "fold_id": split_id % config.n_splits,
            "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
            "validation_fingerprint": _fingerprint_values(validation_indices),
        })
    return splits


def split_fingerprint(splits: Sequence[Mapping[str, Any]]) -> str:
    joined = "|".join(item["validation_fingerprint"] for item in splits)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def evaluate_repeated_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    repeated_config: RepeatedCVConfig | None = None,
    modeling_config: FoldSafeConfig | None = None,
    target_column: str = "class",
    timestamp_column: str | None = "timestamp",
    excluded_columns: Sequence[str] = (),
    shortlist: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    repeated_config = repeated_config or RepeatedCVConfig()
    modeling_config = modeling_config or FoldSafeConfig(random_state=repeated_config.random_state)
    strategy, model_name = candidate
    if strategy not in STRATEGIES or model_name not in MODELS:
        raise ValueError(f"Invalid repeated-CV candidate: {candidate}")
    original = frame.copy(deep=True)
    X, y, dataset = prepare_repeated_data(
        frame,
        repeated_config,
        target_column=target_column,
        timestamp_column=timestamp_column,
        excluded_columns=excluded_columns,
    )
    splits = make_repeated_splits(X, y, repeated_config)
    model_template = build_models(modeling_config)[model_name]
    rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []
    for split in splits:
        started = time.perf_counter()
        train_indices = np.asarray(split["train_indices"])
        validation_indices = np.asarray(split["validation_indices"])
        transformer = clone(build_feature_pipeline(strategy, modeling_config))
        train_data = transformer.fit_transform(X.iloc[train_indices], y.iloc[train_indices])
        validation_data = transformer.transform(X.iloc[validation_indices])
        estimator = clone(model_template)
        estimator.fit(train_data, y.iloc[train_indices])
        predictions = estimator.predict(validation_data)
        scores = _scores(estimator, validation_data)
        metrics = _metric_values(y.iloc[validation_indices], predictions, scores)
        audit = _feature_audit(transformer)
        elapsed = time.perf_counter() - started
        row = {
            "strategy": strategy,
            "model": model_name,
            "repeat_id": split["repeat_id"],
            "fold_id": split["fold_id"],
            "split_id": split["split_id"],
            "validation_fingerprint": split["validation_fingerprint"],
            "train_size": len(train_indices),
            "validation_size": len(validation_indices),
            **{key: audit[key] for key in FOLD_COLUMNS if key in audit},
            "elapsed_seconds": float(elapsed),
            **metrics,
        }
        rows.append(row)
        fold_details.append({**row, "selected_features": audit["selected_features"]})
        del estimator, transformer, train_data, validation_data, scores, predictions
    pd.testing.assert_frame_equal(frame, original)
    table = pd.DataFrame(rows).loc[:, FOLD_COLUMNS]
    payload = {
        "schema_version": "1.0",
        "status": "complete",
        "candidate": {"strategy": strategy, "model": model_name},
        "shortlist": list(shortlist or []),
        "candidate_selection_source": "Validation 3.1 exploratory shortlist; not external validation",
        "dataset": dataset,
        "repeated_cv_config": asdict(repeated_config),
        "modeling_config": asdict(modeling_config),
        "cv": {
            "type": "RepeatedStratifiedKFold",
            "split_count": len(splits),
            "split_fingerprint": split_fingerprint(splits),
            "splits": splits,
        },
        "positive_class": 1,
        "random_state_policy": "splitter, selectors, and stochastic models use the fixed configured random state",
        "execution": {
            "completed_splits": len(rows),
            "split_elapsed_seconds_total": float(sum(row["elapsed_seconds"] for row in rows)),
        },
        "folds": fold_details,
        "limitations": [
            "Overlapping training folds mean the 25 split results are not independent observations.",
            "No threshold tuning, temporal validation, or final hold-out evaluation is performed.",
        ],
    }
    return payload, table


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_part(payload: Mapping[str, Any], rows: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    json_path = directory / "part_results.json"
    csv_path = directory / "part_fold_metrics.csv"
    _atomic_text(json_path, json.dumps(_json_safe(payload), allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(csv_path, rows.loc[:, FOLD_COLUMNS].to_csv(index=False))
    return json_path, csv_path


def validate_part(
    directory: str | Path,
    *,
    expected_candidate: Candidate | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_split_fingerprint: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    directory = Path(directory)
    json_path = directory / "part_results.json"
    csv_path = directory / "part_fold_metrics.csv"
    if not json_path.is_file() or not csv_path.is_file():
        raise ValueError(f"Incomplete repeated-CV part: {directory}")
    payload = json.loads(
        json_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    rows = pd.read_csv(csv_path)
    candidate = (payload.get("candidate", {}).get("strategy"), payload.get("candidate", {}).get("model"))
    expected_rows = payload.get("repeated_cv_config", {}).get("n_splits", 0) * payload.get("repeated_cv_config", {}).get("n_repeats", 0)
    if payload.get("status") != "complete" or len(rows) != expected_rows or tuple(rows.columns) != FOLD_COLUMNS:
        raise ValueError(f"Invalid or incomplete repeated-CV part: {directory}")
    expected_splits = payload.get("cv", {}).get("splits", [])
    if len(expected_splits) != expected_rows or len(payload.get("folds", [])) != expected_rows:
        raise ValueError(f"Invalid or incomplete repeated-CV part: {directory}")
    expected_split_ids = list(range(expected_rows))
    if rows["split_id"].tolist() != expected_split_ids:
        raise ValueError("Repeated-CV part split IDs are incomplete or out of order.")
    expected_identity = [
        (
            split["split_id"],
            split["repeat_id"],
            split["fold_id"],
            split["validation_fingerprint"],
        )
        for split in expected_splits
    ]
    actual_identity = list(rows[
        ["split_id", "repeat_id", "fold_id", "validation_fingerprint"]
    ].itertuples(index=False, name=None))
    if actual_identity != expected_identity:
        raise ValueError("Repeated-CV part split metadata does not match its JSON schema.")
    if not (
        rows["strategy"].eq(candidate[0]).all()
        and rows["model"].eq(candidate[1]).all()
    ):
        raise ValueError("Repeated-CV part rows do not match its candidate metadata.")
    if expected_candidate is not None and candidate != expected_candidate:
        raise ValueError("Repeated-CV part candidate mismatch.")
    if expected_dataset_fingerprint is not None and payload["dataset"]["dataset_fingerprint"] != expected_dataset_fingerprint:
        raise ValueError("Repeated-CV part dataset fingerprint mismatch.")
    if expected_split_fingerprint is not None and payload["cv"]["split_fingerprint"] != expected_split_fingerprint:
        raise ValueError("Repeated-CV part split fingerprint mismatch.")
    return payload, rows


def _describe(values: pd.Series, *, quartiles: bool) -> dict[str, float]:
    result = {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }
    if quartiles:
        result.update({
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        })
    return result


def finalize_parts(
    part_root: str | Path,
    candidates: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_keys = [(item["strategy"], item["model"]) for item in candidates]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("Shortlist contains duplicate candidates.")
    payloads = []
    tables = []
    dataset_fingerprint = split_id = None
    repeated_config = modeling_config = None
    for candidate in candidate_keys:
        directory = Path(part_root) / f"{candidate[0]}__{candidate[1]}"
        payload, table = validate_part(
            directory,
            expected_candidate=candidate,
            expected_dataset_fingerprint=dataset_fingerprint,
            expected_split_fingerprint=split_id,
        )
        dataset_fingerprint = payload["dataset"]["dataset_fingerprint"]
        split_id = payload["cv"]["split_fingerprint"]
        if repeated_config is not None and payload["repeated_cv_config"] != repeated_config:
            raise ValueError("Repeated-CV parts use different repeated-CV configurations.")
        if modeling_config is not None and payload["modeling_config"] != modeling_config:
            raise ValueError("Repeated-CV parts use different modeling configurations.")
        repeated_config = payload["repeated_cv_config"]
        modeling_config = payload["modeling_config"]
        payloads.append(payload)
        tables.append(table)
    if len(payloads) != len(candidate_keys):
        raise ValueError("Repeated-CV parts are incomplete.")
    folds = pd.concat(tables, ignore_index=True)
    expected_count = len(candidate_keys) * payloads[0]["cv"]["split_count"]
    if len(folds) != expected_count or folds[["strategy", "model", "split_id"]].duplicated().any():
        raise ValueError("Merged repeated-CV parts have missing or duplicate rows.")

    repeat_summary = folds.groupby(["strategy", "model", "repeat_id"], sort=False)[list(METRICS)].mean().reset_index()
    repeat_summary = repeat_summary.loc[:, REPEAT_COLUMNS]
    split_summaries = {}
    repeat_summaries = {}
    for candidate in candidate_keys:
        key = f"{candidate[0]}__{candidate[1]}"
        subset = folds[(folds.strategy == candidate[0]) & (folds.model == candidate[1])]
        repeats = repeat_summary[(repeat_summary.strategy == candidate[0]) & (repeat_summary.model == candidate[1])]
        split_summaries[key] = {metric: _describe(subset[metric], quartiles=True) for metric in METRICS}
        repeat_summaries[key] = {metric: _describe(repeats[metric], quartiles=False) for metric in METRICS}

    reference = candidate_keys[1] if len(candidate_keys) > 1 else candidate_keys[0]
    reference_rows = folds[(folds.strategy == reference[0]) & (folds.model == reference[1])].set_index("split_id")
    pairwise_frames = []
    paired_summary = []
    for comparison in candidate_keys:
        if comparison == reference:
            continue
        compared = folds[(folds.strategy == comparison[0]) & (folds.model == comparison[1])].set_index("split_id")
        if not reference_rows["validation_fingerprint"].equals(compared["validation_fingerprint"]):
            raise ValueError("Paired comparison requires identical split fingerprints.")
        pairs = pd.DataFrame({
            "reference_strategy": reference[0], "reference_model": reference[1],
            "comparison_strategy": comparison[0], "comparison_model": comparison[1],
            "split_id": reference_rows.index,
            "repeat_id": reference_rows["repeat_id"].to_numpy(),
            "fold_id": reference_rows["fold_id"].to_numpy(),
            "reference_pr_auc": reference_rows["pr_auc"].to_numpy(),
            "comparison_pr_auc": compared["pr_auc"].to_numpy(),
        })
        pairs["delta"] = pairs["reference_pr_auc"] - pairs["comparison_pr_auc"]
        pairwise_frames.append(pairs)
        tolerance = 1e-12
        paired_summary.append({
            "reference": f"{reference[0]}__{reference[1]}",
            "comparison": f"{comparison[0]}__{comparison[1]}",
            "mean_delta": float(pairs.delta.mean()),
            "median_delta": float(pairs.delta.median()),
            "std_delta": float(pairs.delta.std(ddof=1)),
            "win_count": int((pairs.delta > tolerance).sum()),
            "tie_count": int((pairs.delta.abs() <= tolerance).sum()),
            "loss_count": int((pairs.delta < -tolerance).sum()),
            "win_rate": float((pairs.delta > tolerance).mean()),
            "repeat_mean_delta": {str(key): float(value) for key, value in pairs.groupby("repeat_id").delta.mean().items()},
        })
    pairwise = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame(columns=PAIRWISE_COLUMNS)
    pairwise = pairwise.loc[:, PAIRWISE_COLUMNS]

    rank_source = folds.pivot(index="split_id", columns=["strategy", "model"], values="pr_auc")
    split_ranks = rank_source.rank(axis=1, ascending=False, method="average")
    repeat_rank_source = repeat_summary.pivot(index="repeat_id", columns=["strategy", "model"], values="pr_auc")
    repeat_ranks = repeat_rank_source.rank(axis=1, ascending=False, method="average")
    rank_stability = {}
    for candidate in candidate_keys:
        ranks = split_ranks[candidate]
        repeat_values = repeat_ranks[candidate]
        rank_stability[f"{candidate[0]}__{candidate[1]}"] = {
            "split_first_count": int((ranks == 1).sum()),
            "repeat_first_count": int((repeat_values == 1).sum()),
            "mean_rank": float(ranks.mean()),
            "rank_std": float(ranks.std(ddof=1)),
            "split_ranks": {str(key): float(value) for key, value in ranks.items()},
            "repeat_ranks": {str(key): float(value) for key, value in repeat_values.items()},
        }
    best = min(candidate_keys, key=lambda candidate: (-split_summaries[f"{candidate[0]}__{candidate[1]}"]["pr_auc"]["mean"], candidate_keys.index(candidate)))
    result = {
        "schema_version": "1.0",
        "status": "complete",
        "dataset": payloads[0]["dataset"],
        "candidate_selection": {"source": payloads[0]["candidate_selection_source"], "candidates": list(candidates)},
        "repeated_cv_config": payloads[0]["repeated_cv_config"],
        "modeling_config": payloads[0]["modeling_config"],
        "cv": {key: payloads[0]["cv"][key] for key in ("type", "split_count", "split_fingerprint", "splits")},
        "positive_class": 1,
        "random_state_policy": payloads[0]["random_state_policy"],
        "execution": {
            "candidate_count": len(candidate_keys),
            "completed_splits": len(folds),
            "split_elapsed_seconds_total": float(folds["elapsed_seconds"].sum()),
        },
        "split_summary": split_summaries,
        "repeat_summary": repeat_summaries,
        "paired_reference": f"{reference[0]}__{reference[1]}",
        "paired_comparison": paired_summary,
        "rank_ties": "average rank",
        "rank_stability": rank_stability,
        "best_mean_pr_auc": {"strategy": best[0], "model": best[1], "value": split_summaries[f"{best[0]}__{best[1]}"]["pr_auc"]["mean"]},
        "parts": [payload["candidate"] for payload in payloads],
        "limitations": payloads[0]["limitations"],
    }
    directory = Path(output_dir)
    _atomic_text(directory / "repeated_cv_results.json", json.dumps(_json_safe(result), allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(directory / "repeated_cv_fold_metrics.csv", folds.loc[:, FOLD_COLUMNS].to_csv(index=False))
    _atomic_text(directory / "repeated_cv_repeat_summary.csv", repeat_summary.to_csv(index=False))
    _atomic_text(directory / "repeated_cv_pairwise.csv", pairwise.to_csv(index=False))
    return result, folds, repeat_summary, pairwise
