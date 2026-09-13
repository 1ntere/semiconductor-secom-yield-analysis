"""Leakage-safe nested-CV evaluation of cost-based probability thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.fold_safe_modeling import (
    FoldSafeConfig,
    _feature_audit,
    _json_safe,
    _validate_numeric_features,
    build_feature_pipeline,
    build_models,
)


POLICIES = ("cost_selected", "threshold_0_5", "all_normal", "all_fail")
CSV_COLUMNS = (
    "outer_fold", "scenario", "cost_fp", "cost_fn", "policy", "threshold",
    "threshold_source", "train_size", "validation_size", "inner_oof_size",
    "tp", "fp", "tn", "fn", "total_cost", "cost_units_per_1000_samples",
    "recall", "precision", "specificity", "f1", "balanced_accuracy",
    "predicted_positive_rate", "actual_fail_rate_among_predicted_positive",
    "pr_auc", "roc_auc", "outer_validation_fingerprint",
)


@dataclass(frozen=True)
class CostScenario:
    cost_fp: float
    cost_fn: float
    name: str | None = None

    def __post_init__(self) -> None:
        for field in ("cost_fp", "cost_fn"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{field} must be a positive finite number.")
        object.__setattr__(self, "cost_fp", float(self.cost_fp))
        object.__setattr__(self, "cost_fn", float(self.cost_fn))
        if self.name is not None and not self.name.strip():
            raise ValueError("Scenario name must not be empty.")

    @property
    def label(self) -> str:
        return self.name or f"fn{self.cost_fn:g}_fp{self.cost_fp:g}"


@dataclass(frozen=True)
class CostThresholdConfig:
    random_state: int = 42
    outer_splits: int = 5
    inner_splits: int = 3
    outer_holdout_size: float = 0.20

    def __post_init__(self) -> None:
        if self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("outer_splits and inner_splits must be at least 2.")
        if not 0 < self.outer_holdout_size < 1:
            raise ValueError("outer_holdout_size must be between 0 and 1.")


def validate_probabilities(probabilities: Any) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Probabilities must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return values


def _binary_target(target: Any) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim != 1 or values.size == 0 or pd.isna(values).any():
        raise ValueError("Target must be a non-empty one-dimensional array without missing values.")
    if not set(np.unique(values)).issubset({-1, 1}):
        raise ValueError("Target values must be SECOM classes {-1, 1}.")
    return values


def threshold_candidates(probabilities: Any) -> np.ndarray:
    """Return every distinct >= policy, including all-positive and all-negative."""
    values = validate_probabilities(probabilities)
    return np.unique(np.concatenate(([0.0, 1.0], values, [np.nextafter(1.0, 2.0)])))


def confusion_and_cost(y_true: Any, predictions: Any, cost_fp: float, cost_fn: float) -> dict[str, Any]:
    scenario = CostScenario(cost_fp, cost_fn)
    truth = _binary_target(y_true)
    predicted = np.asarray(predictions)
    if predicted.shape != truth.shape or not set(np.unique(predicted)).issubset({-1, 1}):
        raise ValueError("Predictions must match target shape and contain only {-1, 1}.")
    tp = int(((truth == 1) & (predicted == 1)).sum())
    fp = int(((truth == -1) & (predicted == 1)).sum())
    tn = int(((truth == -1) & (predicted == -1)).sum())
    fn = int(((truth == 1) & (predicted == -1)).sum())
    total = fp * scenario.cost_fp + fn * scenario.cost_fn
    positives = tp + fp
    actual_positive = tp + fn
    actual_negative = tn + fp
    precision = tp / positives if positives else 0.0
    recall = tp / actual_positive if actual_positive else 0.0
    specificity = tn / actual_negative if actual_negative else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_cost": float(total),
        "cost_units_per_1000_samples": float(total * 1000 / len(truth)),
        "recall": float(recall), "precision": float(precision),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "balanced_accuracy": float((recall + specificity) / 2),
        "predicted_positive_rate": float(positives / len(truth)),
        "actual_fail_rate_among_predicted_positive": float(precision),
    }


def select_cost_threshold(y_true: Any, probabilities: Any, cost_fp: float, cost_fn: float) -> dict[str, Any]:
    """Minimize cost; ties choose fewer FP, then fewer FN, then highest threshold."""
    truth = _binary_target(y_true)
    scores = validate_probabilities(probabilities)
    if truth.shape != scores.shape:
        raise ValueError("Target and probabilities must have the same shape.")
    candidates = threshold_candidates(scores)
    evaluated = []
    for threshold in candidates:
        metrics = confusion_and_cost(truth, np.where(scores >= threshold, 1, -1), cost_fp, cost_fn)
        evaluated.append({"threshold": float(threshold), **metrics})
    chosen = min(evaluated, key=lambda row: (row["total_cost"], row["fp"], row["fn"], -row["threshold"]))
    return {**chosen, "candidate_count": len(evaluated)}


def _fingerprint(parts: Iterable[Any]) -> str:
    encoded = json.dumps(list(parts), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_pipeline(modeling: FoldSafeConfig) -> Pipeline:
    return Pipeline([
        ("features", build_feature_pipeline("all_features", modeling)),
        ("classifier", build_models(modeling)["random_forest_balanced"]),
    ])


def _probability(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("Fitted estimator does not expose positive class 1.")
    return validate_probabilities(probabilities[:, classes.index(1)])


def _inner_oof(
    X: pd.DataFrame, y: pd.Series, *, splits: int, random_state: int,
    model_template: Pipeline,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if int(y.value_counts().min()) < splits:
        raise ValueError("Each class must have at least inner_splits samples in every outer training fold.")
    oof = np.full(len(y), np.nan)
    schema = []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, validation_idx) in enumerate(cv.split(X, y), start=1):
        fitted = clone(model_template).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[validation_idx] = _probability(fitted, X.iloc[validation_idx])
        schema.append({
            "inner_fold": fold,
            "train_indices": train_idx.tolist(),
            "validation_indices": validation_idx.tolist(),
            "validation_fingerprint": _fingerprint(validation_idx.tolist()),
        })
    return validate_probabilities(oof), schema


def _policy_row(
    *, fold: int, scenario: CostScenario, policy: str, threshold: float | None,
    threshold_source: str, y_true: pd.Series, scores: np.ndarray, train_size: int,
    inner_oof_size: int, validation_fingerprint: str,
) -> dict[str, Any]:
    if policy == "all_normal":
        predictions = np.full(len(y_true), -1)
    elif policy == "all_fail":
        predictions = np.full(len(y_true), 1)
    else:
        predictions = np.where(scores >= float(threshold), 1, -1)
    metrics = confusion_and_cost(y_true, predictions, scenario.cost_fp, scenario.cost_fn)
    return {
        "outer_fold": fold, "scenario": scenario.label,
        "cost_fp": scenario.cost_fp, "cost_fn": scenario.cost_fn,
        "policy": policy, "threshold": threshold, "threshold_source": threshold_source,
        "train_size": train_size, "validation_size": len(y_true),
        "inner_oof_size": inner_oof_size, **metrics,
        "pr_auc": float(average_precision_score(y_true, scores, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "outer_validation_fingerprint": validation_fingerprint,
    }


def evaluate_cost_thresholds(
    frame: pd.DataFrame, *, target_column: str = "class", timestamp_column: str | None = "timestamp",
    excluded_columns: tuple[str, ...] = (), scenarios: tuple[CostScenario, ...] | None = None,
    config: CostThresholdConfig | None = None, modeling_config: FoldSafeConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    config = config or CostThresholdConfig()
    scenarios = scenarios or tuple(CostScenario(1, value, f"fn{value:g}_fp1") for value in (1, 5, 10))
    if not scenarios or len({item.label for item in scenarios}) != len(scenarios):
        raise ValueError("At least one uniquely named cost scenario is required.")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
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
    _validate_numeric_features(X)
    y = pd.Series(_binary_target(frame[target_column]), index=frame.index)
    if set(y.unique()) != {-1, 1}:
        raise ValueError("Target must contain exactly the SECOM classes {-1, 1}.")

    positions = np.arange(len(frame))
    outer_train_pos, holdout_pos = train_test_split(
        positions, test_size=config.outer_holdout_size, stratify=y, random_state=config.random_state,
    )
    X_train = X.iloc[outer_train_pos].reset_index(drop=True)
    y_train = y.iloc[outer_train_pos].reset_index(drop=True)
    if int(y_train.value_counts().min()) < config.outer_splits:
        raise ValueError("Each class must have at least outer_splits samples in outer-training data.")
    modeling = modeling_config or FoldSafeConfig(
        random_state=config.random_state, n_splits=config.outer_splits,
        outer_holdout_size=config.outer_holdout_size,
    )
    template = _model_pipeline(modeling)
    outer_cv = StratifiedKFold(
        n_splits=config.outer_splits, shuffle=True, random_state=config.random_state,
    )
    rows: list[dict[str, Any]] = []
    fold_metadata = []
    for fold, (train_idx, validation_idx) in enumerate(outer_cv.split(X_train, y_train), start=1):
        fold_X = X_train.iloc[train_idx].reset_index(drop=True)
        fold_y = y_train.iloc[train_idx].reset_index(drop=True)
        inner_scores, inner_schema = _inner_oof(
            fold_X, fold_y, splits=config.inner_splits,
            random_state=config.random_state + fold, model_template=template,
        )
        selections = {
            scenario.label: select_cost_threshold(fold_y, inner_scores, scenario.cost_fp, scenario.cost_fn)
            for scenario in scenarios
        }
        fitted = clone(template).fit(fold_X, fold_y)
        validation_scores = _probability(fitted, X_train.iloc[validation_idx])
        validation_y = y_train.iloc[validation_idx]
        fingerprint = _fingerprint(validation_idx.tolist())
        for scenario in scenarios:
            selected = selections[scenario.label]["threshold"]
            specs = (
                ("cost_selected", selected, "inner_oof_cost_minimization"),
                ("threshold_0_5", 0.5, "fixed_not_optimized"),
                ("all_normal", None, "fixed_not_optimized"),
                ("all_fail", None, "fixed_not_optimized"),
            )
            for policy, threshold, source in specs:
                rows.append(_policy_row(
                    fold=fold, scenario=scenario, policy=policy, threshold=threshold,
                    threshold_source=source, y_true=validation_y, scores=validation_scores,
                    train_size=len(train_idx), inner_oof_size=len(inner_scores),
                    validation_fingerprint=fingerprint,
                ))
        fold_metadata.append({
            "outer_fold": fold, "train_indices": train_idx.tolist(),
            "validation_indices": validation_idx.tolist(),
            "validation_fingerprint": fingerprint, "inner_splits": inner_schema,
            "threshold_selections": selections,
            "feature_audit": _feature_audit(fitted.named_steps["features"]),
        })

    results = pd.DataFrame(rows).loc[:, CSV_COLUMNS]
    aggregate = []
    for (scenario, policy), group in results.groupby(["scenario", "policy"], sort=False):
        item = {"scenario": scenario, "policy": policy}
        for column in ("tp", "fp", "tn", "fn"):
            item[column] = int(group[column].sum())
        costs = group["total_cost"]
        item["pooled_total_cost"] = float(costs.sum())
        item["mean_fold_cost"] = float(costs.mean())
        item["pooled_cost_units_per_1000_samples"] = float(costs.sum() * 1000 / group.validation_size.sum())
        pooled = confusion_and_cost(
            np.r_[np.ones(item["tp"] + item["fn"]), -np.ones(item["tn"] + item["fp"])],
            np.r_[np.ones(item["tp"]), -np.ones(item["fn"]), -np.ones(item["tn"]), np.ones(item["fp"])],
            float(group.cost_fp.iloc[0]), float(group.cost_fn.iloc[0]),
        )
        item.update({key: pooled[key] for key in (
            "recall", "precision", "specificity", "f1", "balanced_accuracy",
            "predicted_positive_rate", "actual_fail_rate_among_predicted_positive",
        )})
        aggregate.append(item)

    final_oof, final_inner_schema = _inner_oof(
        X_train, y_train, splits=config.inner_splits, random_state=config.random_state,
        model_template=template,
    )
    candidate_thresholds = {
        scenario.label: {
            "status": "candidate_threshold_not_holdout_evaluated",
            **select_cost_threshold(y_train, final_oof, scenario.cost_fp, scenario.cost_fn),
        }
        for scenario in scenarios
    }
    split_definition = {
        "outer_train_positions": outer_train_pos.tolist(),
        "outer_holdout_positions": holdout_pos.tolist(),
        "outer_folds": [
            {"outer_fold": item["outer_fold"], "validation": item["validation_indices"],
             "inner": [part["validation_indices"] for part in item["inner_splits"]]}
            for item in fold_metadata
        ],
    }
    split_fingerprint = _fingerprint([split_definition])
    payload = {
        "schema_version": "1.0", "status": "complete",
        "metadata": {
            "scope": "Exploratory all_features × random_forest_balanced candidate selected by Improvement 4 mean PR-AUC; not proof of superiority over all models",
            "cost_interpretation": "Example cost units for sensitivity analysis; not currency or measured manufacturing loss",
            "positive_rule": "P(class=1) >= threshold",
            "threshold_candidates": "0, 1, every distinct inner-OOF probability, and nextafter(1,+inf) for all-negative",
            "tie_break": "minimum total cost, then fewer FP, then fewer FN, then highest threshold",
            "outer_holdout_used_for_split": True, "outer_holdout_evaluated": False,
            "probability_calibration_added": False,
            "limitations": [
                "Inner-OOF models and the model refit on the full outer-fold training portion can have different probability scales.",
                "No probability calibration or external hold-out evaluation is performed.",
                "A high PR-AUC does not guarantee minimum cost for a particular cost scenario.",
            ],
        },
        "dataset": {
            "shape": list(frame.shape), "modeling_feature_count": len(feature_columns),
            "target_distribution": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
            "outer_train_size": len(outer_train_pos), "outer_holdout_size": len(holdout_pos),
        },
        "config": asdict(config), "modeling_config": asdict(modeling),
        "cost_scenarios": [{**asdict(item), "name": item.label} for item in scenarios],
        "split": {
            "fingerprint": split_fingerprint,
            "outer_train_positions": outer_train_pos.tolist(),
            "outer_holdout_positions": holdout_pos.tolist(),
            "outer_train_fingerprint": _fingerprint(outer_train_pos.tolist()),
            "outer_holdout_fingerprint": _fingerprint(holdout_pos.tolist()),
        },
        "split_fingerprint": split_fingerprint, "outer_folds": fold_metadata,
        "aggregate": aggregate,
        "candidate_thresholds_for_future_external_validation": candidate_thresholds,
        "candidate_threshold_inner_splits": final_inner_schema,
    }
    return payload, results


def write_cost_threshold_results(
    payload: Mapping[str, Any], rows: pd.DataFrame, output_dir: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "cost_threshold_results.json"
    csv_path = directory / "cost_threshold_fold_policies.csv"
    json_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows.loc[:, CSV_COLUMNS].to_csv(csv_path, index=False)
    return json_path, csv_path
