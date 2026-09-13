"""Exploratory forward validation and training-referenced feature drift for SECOM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.fold_safe_modeling import (
    FoldSafeConfig, _feature_audit, _json_safe, _validate_numeric_features,
    build_feature_pipeline, build_models,
)


PERFORMANCE_COLUMNS = (
    "split", "status", "reason", "train_size", "validation_size",
    "train_start", "train_end", "validation_start", "validation_end",
    "train_fail_count", "train_fail_rate", "validation_fail_count",
    "validation_fail_rate", "selected_feature_count", "pr_auc_defined",
    "pr_auc", "roc_auc_defined", "roc_auc", "tp", "fp", "tn", "fn",
    "recall", "precision", "f1", "balanced_accuracy",
    "predicted_positive_rate", "threshold",
)
DRIFT_COLUMNS = (
    "split", "feature", "train_size", "validation_size", "train_missing_rate",
    "validation_missing_rate", "missing_rate_change", "train_non_missing_count",
    "validation_non_missing_count", "train_reference_status", "bin_count",
    "psi", "validation_below_train_range_count", "validation_above_train_range_count",
)


@dataclass(frozen=True)
class TemporalConfig:
    random_state: int = 42
    n_splits: int = 3
    outer_holdout_size: float = 0.20
    drift_bins: int = 10
    psi_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits must be positive.")
        if not 0 < self.outer_holdout_size < 1:
            raise ValueError("outer_holdout_size must be between 0 and 1.")
        if self.drift_bins < 2:
            raise ValueError("drift_bins must be at least 2.")
        if not np.isfinite(self.psi_epsilon) or self.psi_epsilon <= 0:
            raise ValueError("psi_epsilon must be positive and finite.")


def parse_timestamps(values: pd.Series, *, dayfirst: bool = True) -> pd.Series:
    if values.isna().any():
        raise ValueError("Timestamp column contains missing values.")
    text = values.astype("string")
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]|$)", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(
        text.loc[iso_mask], format="mixed", dayfirst=False, errors="coerce"
    )
    parsed.loc[~iso_mask] = pd.to_datetime(
        text.loc[~iso_mask], format="mixed", dayfirst=dayfirst, errors="coerce"
    )
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"Timestamp parsing failed for {bad} row(s); values are not silently repaired.")
    return parsed


def assess_timestamps(values: pd.Series, *, dayfirst: bool = True) -> dict[str, Any]:
    total = len(values)
    try:
        parsed = parse_timestamps(values, dayfirst=dayfirst)
    except ValueError as exc:
        return {
            "verdict": "unsuitable", "reason": str(exc), "row_count": total,
            "parsed_count": 0, "parse_success_rate": 0.0,
            "missing_count": int(values.isna().sum()),
        }
    counts = parsed.value_counts()
    differences = parsed.diff()
    positive_gaps = differences[differences > pd.Timedelta(0)]
    return {
        "verdict": "limited_suitability",
        "reason": "UCI documents a timestamp at the specific test point but does not document timezone, production-order guarantees, or lot/wafer identifiers.",
        "row_count": total, "parsed_count": int(parsed.notna().sum()),
        "parse_success_rate": float(parsed.notna().mean()), "missing_count": 0,
        "minimum": parsed.min().isoformat(), "maximum": parsed.max().isoformat(),
        "unique_count": int(parsed.nunique()),
        "duplicate_extra_rows": int(total - parsed.nunique()),
        "duplicate_group_count": int((counts > 1).sum()),
        "maximum_timestamp_group_size": int(counts.max()),
        "original_order_time_reversals": int((differences < pd.Timedelta(0)).sum()),
        "maximum_positive_gap_seconds": float(positive_gaps.max().total_seconds()) if len(positive_gaps) else 0.0,
        "gaps_over_24_hours": int((positive_gaps > pd.Timedelta(hours=24)).sum()),
        "dayfirst": dayfirst, "timezone_documented": False,
        "lot_wafer_identifiers_available": False,
    }


def make_forward_splits(timestamps: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding splits over indivisible timestamp groups."""
    parsed = parse_timestamps(timestamps)
    unique = np.asarray(sorted(parsed.unique()))
    if len(unique) < n_splits + 1:
        raise ValueError("Not enough unique timestamp groups for requested forward splits.")
    blocks = np.array_split(unique, n_splits + 1)
    if any(len(block) == 0 for block in blocks):
        raise ValueError("Timestamp grouping produced an empty temporal block.")
    splits = []
    for index in range(1, n_splits + 1):
        train_times = np.concatenate(blocks[:index])
        validation_times = blocks[index]
        train = np.flatnonzero(parsed.isin(train_times).to_numpy())
        validation = np.flatnonzero(parsed.isin(validation_times).to_numpy())
        if parsed.iloc[train].max() >= parsed.iloc[validation].min():
            raise RuntimeError("Temporal split invariant failed: training must be strictly earlier.")
        splits.append((train, validation))
    return splits


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model(modeling: FoldSafeConfig) -> Pipeline:
    return Pipeline([
        ("features", build_feature_pipeline("all_features", modeling)),
        ("classifier", build_models(modeling)["random_forest_balanced"]),
    ])


def _positive_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("Fitted estimator does not expose positive class 1.")
    scores = np.asarray(estimator.predict_proba(X)[:, classes.index(1)], dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Estimator returned invalid positive-class probabilities.")
    return scores


def _distribution(values: pd.Series) -> dict[str, Any]:
    counts = values.value_counts().sort_index()
    return {
        "count": len(values), "class_counts": {str(key): int(value) for key, value in counts.items()},
        "fail_count": int((values == 1).sum()), "fail_rate": float((values == 1).mean()),
    }


def feature_drift(
    train: pd.DataFrame, validation: pd.DataFrame, *, split: int,
    bins: int = 10, epsilon: float = 1e-6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute PSI using only train-defined quantile edges and a missing bin."""
    _validate_numeric_features(train)
    _validate_numeric_features(validation)
    if np.isinf(train.to_numpy(dtype=float)).any() or np.isinf(validation.to_numpy(dtype=float)).any():
        raise ValueError("Drift features must not contain infinity.")
    rows, references = [], {}
    for feature in train.columns:
        left, right = train[feature], validation[feature]
        finite_left = left.dropna().to_numpy(dtype=float)
        finite_right = right.dropna().to_numpy(dtype=float)
        status = "variable"
        psi: float | None = None
        bin_count = 0
        below = above = 0
        edges: list[float] = []
        if finite_left.size == 0:
            status = "all_missing_training"
        elif np.unique(finite_left).size == 1:
            status = "constant_training"
            below = int((finite_right < finite_left[0]).sum())
            above = int((finite_right > finite_left[0]).sum())
        else:
            quantiles = np.linspace(0, 1, bins + 1)
            raw_edges = np.unique(np.quantile(finite_left, quantiles))
            internal = raw_edges[1:-1]
            edges_array = np.r_[-np.inf, internal, np.inf]
            edges = [float(value) for value in edges_array]
            bin_count = len(edges_array) - 1
            train_bins = pd.cut(left, edges_array, include_lowest=True).value_counts(sort=False).to_numpy(float)
            validation_bins = pd.cut(right, edges_array, include_lowest=True).value_counts(sort=False).to_numpy(float)
            train_counts = np.r_[train_bins, left.isna().sum()]
            validation_counts = np.r_[validation_bins, right.isna().sum()]
            train_share = np.maximum(train_counts / len(left), epsilon)
            validation_share = np.maximum(validation_counts / len(right), epsilon)
            psi = float(np.sum((validation_share - train_share) * np.log(validation_share / train_share)))
            below = int((finite_right < np.min(finite_left)).sum())
            above = int((finite_right > np.max(finite_left)).sum())
        rows.append({
            "split": split, "feature": str(feature), "train_size": len(train),
            "validation_size": len(validation), "train_missing_rate": float(left.isna().mean()),
            "validation_missing_rate": float(right.isna().mean()),
            "missing_rate_change": float(right.isna().mean() - left.isna().mean()),
            "train_non_missing_count": int(left.notna().sum()),
            "validation_non_missing_count": int(right.notna().sum()),
            "train_reference_status": status, "bin_count": bin_count, "psi": psi,
            "validation_below_train_range_count": below,
            "validation_above_train_range_count": above,
        })
        references[str(feature)] = {"status": status, "edges": edges}
    return rows, references


def evaluate_temporal_validation(
    frame: pd.DataFrame, *, target_column: str = "class", timestamp_column: str = "timestamp",
    excluded_columns: tuple[str, ...] = (), config: TemporalConfig | None = None,
    modeling_config: FoldSafeConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = config or TemporalConfig()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    required = [column for column in (target_column, timestamp_column) if column not in frame.columns]
    if required:
        raise ValueError(f"Required columns were not found: {required}")
    unknown = [column for column in excluded_columns if column not in frame.columns]
    if unknown:
        raise ValueError(f"Excluded columns were not found: {unknown}")
    y = frame[target_column].copy()
    if y.isna().any() or set(y.unique()) != {-1, 1}:
        raise ValueError("Target must contain exactly the SECOM classes {-1, 1}.")
    parsed = parse_timestamps(frame[timestamp_column])
    excluded = {target_column, timestamp_column, *excluded_columns}
    feature_columns = [column for column in frame.columns if column not in excluded]
    X = frame.loc[:, feature_columns].copy()
    _validate_numeric_features(X)
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Modeling and drift features must not contain infinity.")

    positions = np.arange(len(frame))
    outer_train_pos, holdout_pos = train_test_split(
        positions, test_size=config.outer_holdout_size, stratify=y,
        random_state=config.random_state,
    )
    working = pd.DataFrame({"original_row_id": outer_train_pos, "timestamp": parsed.iloc[outer_train_pos].to_numpy()})
    working["target"] = y.iloc[outer_train_pos].to_numpy()
    working = working.sort_values(["timestamp", "original_row_id"], kind="stable").reset_index(drop=True)
    X_work = X.iloc[working.original_row_id].reset_index(drop=True)
    y_work = working.target.reset_index(drop=True)
    time_work = working.timestamp.reset_index(drop=True)
    splits = make_forward_splits(time_work, config.n_splits)
    modeling = modeling_config or FoldSafeConfig(random_state=config.random_state)
    template = _model(modeling)
    performance_rows, drift_rows, split_details = [], [], []

    for split_number, (train_idx, validation_idx) in enumerate(splits, start=1):
        train_X, validation_X = X_work.iloc[train_idx], X_work.iloc[validation_idx]
        train_y, validation_y = y_work.iloc[train_idx], y_work.iloc[validation_idx]
        drift, references = feature_drift(
            train_X, validation_X, split=split_number,
            bins=config.drift_bins, epsilon=config.psi_epsilon,
        )
        drift_rows.extend(drift)
        base = {
            "split": split_number, "train_size": len(train_idx), "validation_size": len(validation_idx),
            "train_start": time_work.iloc[train_idx].min().isoformat(),
            "train_end": time_work.iloc[train_idx].max().isoformat(),
            "validation_start": time_work.iloc[validation_idx].min().isoformat(),
            "validation_end": time_work.iloc[validation_idx].max().isoformat(),
            "train_fail_count": int((train_y == 1).sum()), "train_fail_rate": float((train_y == 1).mean()),
            "validation_fail_count": int((validation_y == 1).sum()),
            "validation_fail_rate": float((validation_y == 1).mean()),
        }
        reason = None
        if set(train_y.unique()) != {-1, 1}:
            reason = "training interval does not contain both classes"
        elif int((validation_y == 1).sum()) == 0:
            reason = "validation interval contains no positive class 1; PR-AUC is undefined"
        elif int((validation_y == -1).sum()) == 0:
            reason = "validation interval contains no negative class -1; ROC-AUC is undefined"
        detail = {
            **base, "train_original_row_ids": working.original_row_id.iloc[train_idx].tolist(),
            "validation_original_row_ids": working.original_row_id.iloc[validation_idx].tolist(),
            "train_distribution": _distribution(train_y), "validation_distribution": _distribution(validation_y),
            "train_validation_timestamp_shared": False,
            "train_reference_fingerprint": _fingerprint(references),
        }
        if reason:
            row = {
                **base, "status": "not_evaluated", "reason": reason,
                "selected_feature_count": None, "pr_auc_defined": False, "pr_auc": None,
                "roc_auc_defined": False, "roc_auc": None, "tp": None, "fp": None,
                "tn": None, "fn": None, "recall": None, "precision": None, "f1": None,
                "balanced_accuracy": None, "predicted_positive_rate": None, "threshold": 0.5,
            }
            detail.update({"status": "not_evaluated", "reason": reason, "selected_features": []})
        else:
            fitted = clone(template).fit(train_X, train_y)
            scores = _positive_scores(fitted, validation_X)
            predictions = np.where(scores >= 0.5, 1, -1)
            tn, fp, fn, tp = confusion_matrix(validation_y, predictions, labels=[-1, 1]).ravel()
            audit = _feature_audit(fitted.named_steps["features"])
            row = {
                **base, "status": "evaluated", "reason": None,
                "selected_feature_count": audit["feature_count_selected"],
                "pr_auc_defined": True,
                "pr_auc": float(average_precision_score(validation_y, scores, pos_label=1)),
                "roc_auc_defined": True, "roc_auc": float(roc_auc_score(validation_y, scores)),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "recall": float(recall_score(validation_y, predictions, pos_label=1, zero_division=0)),
                "precision": float(precision_score(validation_y, predictions, pos_label=1, zero_division=0)),
                "f1": float(f1_score(validation_y, predictions, pos_label=1, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(validation_y, predictions)),
                "predicted_positive_rate": float((predictions == 1).mean()), "threshold": 0.5,
            }
            detail.update({"status": "evaluated", "reason": None, **audit})
        performance_rows.append(row)
        split_details.append(detail)

    performance = pd.DataFrame(performance_rows).loc[:, PERFORMANCE_COLUMNS]
    drift = pd.DataFrame(drift_rows).loc[:, DRIFT_COLUMNS]
    assessment = assess_timestamps(frame[timestamp_column])
    payload = {
        "schema_version": "1.0", "status": "complete",
        "timestamp_assessment": {
            **assessment,
            "source_meaning": "UCI: associated date time stamp for the specific pass/fail in-house line test point",
            "format": "DD/MM/YYYY HH:MM:SS", "timezone": None,
            "overall_period_counts": [
                {"period": str(period), "sample_count": int(len(group)),
                 "fail_count": int((group == 1).sum()), "fail_rate": float((group == 1).mean())}
                for period, group in y.groupby(parsed.dt.to_period("M"))
            ],
        },
        "metadata": {
            "scope": "Exploratory forward validation inside the randomly selected outer-training 80%; not independent future-production validation",
            "random_outer_holdout_evaluated": False,
            "outer_training_was_randomly_sampled_across_the_full_period": True,
            "model": "all_features × random_forest_balanced exploratory candidate from Improvement 4",
            "positive_class": 1, "classification_threshold": 0.5,
            "improvement_5_candidate_threshold_used": False,
            "hyperparameter_tuning_or_calibration_added": False,
            "gap_embargo": None,
            "limitations": [
                "Timestamp is documented only as the specific test-point timestamp; production ordering semantics and timezone are not documented.",
                "No lot, wafer, batch, equipment, or process-stage identifiers are available, so group leakage cannot be assessed.",
                "PSI is an exploratory distribution heuristic and does not identify causes, equipment faults, or causal effects.",
                "Random repeated-CV and temporal results use different sample partitions and must not be treated as a like-for-like degradation estimate.",
            ],
        },
        "config": asdict(config), "modeling_config": asdict(modeling),
        "dataset": {
            "shape": list(frame.shape), "modeling_feature_count": len(feature_columns),
            "excluded_columns": [timestamp_column, target_column, *excluded_columns],
            "outer_train_size": len(outer_train_pos), "outer_holdout_size": len(holdout_pos),
            "outer_train_timestamp_assessment": assess_timestamps(frame[timestamp_column].iloc[outer_train_pos].reset_index(drop=True)),
        },
        "split": {
            "type": "expanding_window_by_indivisible_timestamp_group",
            "fingerprint": _fingerprint([(a.tolist(), b.tolist()) for a, b in splits]),
            "outer_train_fingerprint": _fingerprint(outer_train_pos.tolist()),
            "outer_holdout_fingerprint": _fingerprint(holdout_pos.tolist()),
            "requested_splits": config.n_splits, "produced_splits": len(splits),
        },
        "drift_method": {
            "name": "population_stability_index",
            "reference": "training interval only",
            "bins": "training quantiles with duplicate edges removed; -inf/+inf endpoints; missing values are a separate bin",
            "formula": "sum((validation_share - train_share) * ln(validation_share / train_share))",
            "zero_count_epsilon": config.psi_epsilon,
            "thresholds": None,
            "interpretation": "No universal threshold asserted; rank changes for exploratory inspection only.",
        },
        "splits": split_details,
    }
    return payload, performance, drift


def write_temporal_results(
    payload: Mapping[str, Any], performance: pd.DataFrame, drift: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "temporal_validation_results.json"
    performance_path = directory / "temporal_performance.csv"
    drift_path = directory / "temporal_feature_drift.csv"
    json_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    performance.loc[:, PERFORMANCE_COLUMNS].to_csv(performance_path, index=False)
    drift.loc[:, DRIFT_COLUMNS].to_csv(drift_path, index=False)
    return json_path, performance_path, drift_path
