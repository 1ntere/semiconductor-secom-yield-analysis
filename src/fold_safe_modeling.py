"""Fold-safe preprocessing, feature selection, and SECOM model evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


STRATEGIES = ("all_features", "correlation_pruned", "mi_selected", "model_selected")
MODELS = ("dummy_prior", "logistic_balanced", "random_forest_balanced", "histgb_balanced")
METRICS = ("pr_auc", "roc_auc", "recall", "precision", "f1", "balanced_accuracy")
CSV_COLUMNS = (
    "strategy", "model", "fold", "train_size", "validation_size",
    "feature_count_before", "feature_count_after_missing",
    "feature_count_after_constant", "feature_count_after_correlation",
    "feature_count_selected", *METRICS,
)


@dataclass(frozen=True)
class FoldSafeConfig:
    random_state: int = 42
    n_splits: int = 5
    outer_holdout_size: float = 0.20
    missing_threshold: float = 0.40
    correlation_threshold: float = 0.95
    mi_null_permutations: int = 20
    mi_null_quantile: float = 0.95
    selector_rf_estimators: int = 500
    model_rf_estimators: int = 300
    hist_max_iter: int = 150
    std_ddof: int = 1
    n_jobs: int = 1

    @classmethod
    def quick(cls, **overrides: Any) -> "FoldSafeConfig":
        """Return the same algorithms with smaller iteration counts for smoke tests."""

        values = {
            "mi_null_permutations": 2,
            "selector_rf_estimators": 5,
            "model_rf_estimators": 5,
            "hist_max_iter": 10,
        }
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not 0 < self.outer_holdout_size < 1:
            raise ValueError("outer_holdout_size must be between 0 and 1.")
        for name in ("missing_threshold", "correlation_threshold", "mi_null_quantile"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.mi_null_permutations < 1:
            raise ValueError("mi_null_permutations must be positive.")


def _require_dataframe(X: Any) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame so feature names can be tracked.")
    return X


def _validate_numeric_features(X: pd.DataFrame) -> None:
    invalid = [
        str(column) for column in X.columns
        if pd.api.types.is_bool_dtype(X[column].dtype)
        or not pd.api.types.is_numeric_dtype(X[column].dtype)
    ]
    if invalid:
        raise TypeError(f"All modeling features must be numeric and non-boolean: {invalid}")


class FoldPreprocessor(BaseEstimator, TransformerMixin):
    """Learn feature schema, missingness, constants, and medians from training data."""

    def __init__(self, missing_threshold: float = 0.40):
        self.missing_threshold = missing_threshold

    def fit(self, X: pd.DataFrame, y: Any = None):
        frame = _require_dataframe(X)
        if frame.empty or frame.shape[1] == 0:
            raise ValueError("Training data must contain rows and features.")
        if not frame.columns.is_unique:
            raise ValueError("Training feature names must be unique.")
        _validate_numeric_features(frame)
        if np.isinf(frame.to_numpy(dtype=float)).any():
            raise ValueError("Training features must not contain infinity.")

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.feature_count_before_ = frame.shape[1]
        missing_rates = frame.isna().mean()
        self.high_missing_features_ = tuple(
            column for column in frame.columns if missing_rates[column] > self.missing_threshold
        )
        after_missing = [column for column in frame.columns if column not in self.high_missing_features_]
        self.feature_count_after_missing_ = len(after_missing)
        self.constant_features_ = tuple(
            column for column in after_missing if frame[column].nunique(dropna=True) <= 1
        )
        self.feature_names_out_ = tuple(
            column for column in after_missing if column not in self.constant_features_
        )
        self.feature_count_after_constant_ = len(self.feature_names_out_)
        if not self.feature_names_out_:
            raise ValueError("All features were removed by missingness and constant checks.")
        self.medians_ = frame.loc[:, self.feature_names_out_].median(axis=0)
        if self.medians_.isna().any() or not np.isfinite(self.medians_.to_numpy()).all():
            raise ValueError("At least one retained training feature has no finite median.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        frame = _require_dataframe(X)
        missing = [column for column in self.feature_names_in_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Validation data is missing fitted feature columns: {missing}")
        selected = frame.loc[:, self.feature_names_out_]
        _validate_numeric_features(selected)
        if np.isinf(selected.to_numpy(dtype=float)).any():
            raise ValueError("Validation features must not contain infinity.")
        transformed = selected.fillna(self.medians_)
        if transformed.isna().any().any():
            raise ValueError("Median imputation left missing validation values.")
        return transformed.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Prune correlated features using training-only MI priority and original-order ties."""

    def __init__(self, threshold: float = 0.95, random_state: int = 42):
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any):
        frame = _require_dataframe(X)
        _validate_numeric_features(frame)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        scores = mutual_info_classif(frame, y, random_state=self.random_state)
        original_order = {column: index for index, column in enumerate(frame.columns)}
        priority = sorted(frame.columns, key=lambda column: (-scores[original_order[column]], original_order[column]))
        correlation = frame.corr().abs()
        kept: list[Any] = []
        for column in priority:
            if not any(correlation.loc[column, other] >= self.threshold for other in kept):
                kept.append(column)
        kept_set = set(kept)
        self.feature_names_out_ = tuple(column for column in frame.columns if column in kept_set)
        self.removed_features_ = tuple(column for column in frame.columns if column not in kept_set)
        if not self.feature_names_out_:
            raise ValueError("Correlation pruning removed all features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        frame = _require_dataframe(X)
        missing = [column for column in self.feature_names_in_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Validation data is missing fitted feature columns: {missing}")
        return frame.loc[:, self.feature_names_out_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


class MutualInformationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, permutations: int = 20, quantile: float = 0.95, random_state: int = 42):
        self.permutations = permutations
        self.quantile = quantile
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any):
        frame = _require_dataframe(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        scores = mutual_info_classif(frame, y, random_state=self.random_state)
        rng = np.random.default_rng(self.random_state)
        null_scores: list[float] = []
        target = np.asarray(y)
        for iteration in range(self.permutations):
            null_scores.extend(mutual_info_classif(
                frame, rng.permutation(target), random_state=self.random_state + iteration + 1
            ))
        self.threshold_ = float(np.quantile(null_scores, self.quantile))
        self.scores_ = pd.Series(scores, index=frame.columns)
        self.feature_names_out_ = tuple(column for column in frame.columns if self.scores_[column] > self.threshold_)
        if not self.feature_names_out_:
            raise ValueError("Mutual-information selection removed all features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        frame = _require_dataframe(X)
        missing = [column for column in self.feature_names_in_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Validation data is missing fitted feature columns: {missing}")
        return frame.loc[:, self.feature_names_out_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


class ModelBasedSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators: int = 500, random_state: int = 42, n_jobs: int = 1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y: Any):
        frame = _require_dataframe(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            max_features="sqrt",
        ).fit(frame, y)
        self.threshold_ = float(np.mean(self.estimator_.feature_importances_))
        self.feature_names_out_ = tuple(
            column for column, importance in zip(frame.columns, self.estimator_.feature_importances_)
            if importance > self.threshold_
        )
        if not self.feature_names_out_:
            raise ValueError("Model-based selection removed all features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        frame = _require_dataframe(X)
        missing = [column for column in self.feature_names_in_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Validation data is missing fitted feature columns: {missing}")
        return frame.loc[:, self.feature_names_out_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


def build_feature_pipeline(strategy: str, config: FoldSafeConfig) -> Pipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown feature strategy: {strategy}")
    steps: list[tuple[str, Any]] = [("preprocess", FoldPreprocessor(config.missing_threshold))]
    if strategy != "all_features":
        steps.append(("correlation", CorrelationPruner(config.correlation_threshold, config.random_state)))
    if strategy == "mi_selected":
        steps.append(("select", MutualInformationSelector(
            config.mi_null_permutations, config.mi_null_quantile, config.random_state
        )))
    elif strategy == "model_selected":
        steps.append(("select", ModelBasedSelector(
            config.selector_rf_estimators, config.random_state, config.n_jobs
        )))
    return Pipeline(steps)


def build_models(config: FoldSafeConfig) -> dict[str, BaseEstimator]:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_balanced": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=config.random_state)),
        ]),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=config.model_rf_estimators, class_weight="balanced_subsample",
            random_state=config.random_state, n_jobs=config.n_jobs, max_features="sqrt",
        ),
        "histgb_balanced": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=config.hist_max_iter, learning_rate=0.08,
            l2_regularization=1.0, random_state=config.random_state,
        ),
    }


def _feature_audit(pipeline: Pipeline) -> dict[str, Any]:
    preprocess = pipeline.named_steps["preprocess"]
    after_constant = preprocess.feature_count_after_constant_
    after_correlation = (
        len(pipeline.named_steps["correlation"].feature_names_out_)
        if "correlation" in pipeline.named_steps else after_constant
    )
    names = list(pipeline.steps[-1][1].get_feature_names_out())
    return {
        "feature_count_before": preprocess.feature_count_before_,
        "feature_count_after_missing": preprocess.feature_count_after_missing_,
        "feature_count_after_constant": after_constant,
        "feature_count_after_correlation": after_correlation,
        "feature_count_selected": len(names),
        "selected_features": [str(name) for name in names],
    }


def _scores(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        classes = list(estimator.classes_)
        return probabilities[:, classes.index(1)]
    scores = estimator.decision_function(X)
    classes = list(estimator.classes_)
    return np.asarray(scores) if classes[-1] == 1 else -np.asarray(scores)


def _metric_values(y_true: pd.Series, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, scores, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def evaluate_fold_safe(
    frame: pd.DataFrame,
    *,
    target_column: str = "class",
    timestamp_column: str | None = "timestamp",
    excluded_columns: tuple[str, ...] = (),
    config: FoldSafeConfig | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
    model_names: tuple[str, ...] = MODELS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate every strategy/model on shared folds inside an untouched outer hold-out."""

    config = config or FoldSafeConfig()
    unknown_strategies = [name for name in strategies if name not in STRATEGIES]
    unknown_models = [name for name in model_names if name not in MODELS]
    if not strategies or unknown_strategies:
        raise ValueError(f"Invalid feature strategies: {unknown_strategies or list(strategies)}")
    if not model_names or unknown_models:
        raise ValueError(f"Invalid models: {unknown_models or list(model_names)}")
    original = _require_dataframe(frame)
    if target_column not in original.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    excluded = {target_column}
    if timestamp_column is not None:
        excluded.add(timestamp_column)
    unknown_exclusions = [column for column in excluded_columns if column not in original.columns]
    if unknown_exclusions:
        raise ValueError(f"Excluded columns were not found: {unknown_exclusions}")
    excluded.update(excluded_columns)
    feature_columns = [column for column in original.columns if column not in excluded]
    X = original.loc[:, feature_columns].copy()
    _validate_numeric_features(X)
    y = original[target_column].copy()
    if y.isna().any() or set(y.unique()) != {-1, 1}:
        raise ValueError("Target must contain exactly the SECOM classes {-1, 1} with no missing values.")

    positions = np.arange(len(original))
    train_positions, holdout_positions = train_test_split(
        positions, test_size=config.outer_holdout_size, random_state=config.random_state, stratify=y
    )
    X_train = X.iloc[train_positions].reset_index(drop=True)
    y_train = y.iloc[train_positions].reset_index(drop=True)
    if int(y_train.value_counts().min()) < config.n_splits:
        raise ValueError("Each target class must have at least n_splits samples in outer-train.")
    cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    folds = [(train_idx, validation_idx) for train_idx, validation_idx in cv.split(X_train, y_train)]
    fold_schema = [
        {
            "fold": index,
            "train_indices": train_idx.tolist(),
            "validation_indices": validation_idx.tolist(),
        }
        for index, (train_idx, validation_idx) in enumerate(folds, start=1)
    ]

    rows: list[dict[str, Any]] = []
    result_groups: list[dict[str, Any]] = []
    models = build_models(config)
    for strategy in strategies:
        group_folds: dict[int, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
        for fold_number, (train_idx, validation_idx) in enumerate(folds, start=1):
            transformer = clone(build_feature_pipeline(strategy, config))
            train_transformed = transformer.fit_transform(X_train.iloc[train_idx], y_train.iloc[train_idx])
            validation_transformed = transformer.transform(X_train.iloc[validation_idx])
            group_folds[fold_number] = (train_transformed, validation_transformed, _feature_audit(transformer))
        for model_name in model_names:
            model_template = models[model_name]
            fold_results: list[dict[str, Any]] = []
            for fold_number, (train_idx, validation_idx) in enumerate(folds, start=1):
                train_transformed, validation_transformed, audit = group_folds[fold_number]
                estimator = clone(model_template)
                estimator.fit(train_transformed, y_train.iloc[train_idx])
                predictions = estimator.predict(validation_transformed)
                scores = _scores(estimator, validation_transformed)
                metrics = _metric_values(y_train.iloc[validation_idx], predictions, scores)
                fold_result = {
                    "fold": fold_number,
                    "train_size": len(train_idx),
                    "validation_size": len(validation_idx),
                    **audit,
                    **metrics,
                }
                fold_results.append(fold_result)
                rows.append({"strategy": strategy, "model": model_name, **fold_result})
            aggregate = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in fold_results])),
                    "std": float(np.std([row[metric] for row in fold_results], ddof=config.std_ddof)),
                }
                for metric in METRICS
            }
            result_groups.append({
                "strategy": strategy,
                "model": model_name,
                "model_parameters": _json_safe(models[model_name].get_params(deep=True)),
                "folds": fold_results,
                "metrics": aggregate,
            })

    csv_frame = pd.DataFrame(rows).loc[:, CSV_COLUMNS]
    ranking = sorted(
        result_groups,
        key=lambda item: (
            -item["metrics"]["pr_auc"]["mean"],
            STRATEGIES.index(item["strategy"]),
            MODELS.index(item["model"]),
        ),
    )
    payload = {
        "schema_version": "1.0",
        "metadata": {
            "scope": "Single fold-safe 5-fold CV on outer-train; hold-out is not evaluated",
            "positive_class": 1,
            "score_metrics": ["pr_auc", "roc_auc"],
            "classification_threshold": "estimator default (0.5 for probability classifiers)",
            "std_ddof": config.std_ddof,
            "outer_holdout_used": True,
            "outer_holdout_evaluated": False,
            "strategies": list(strategies),
            "models": list(model_names),
            "preprocessing_and_selection": {
                "missing": "remove when training-fold missing rate is greater than threshold; original column order",
                "constant": "remove when training-fold non-missing unique count is at most one",
                "imputation": "training-fold median",
                "correlation": "remove at absolute correlation greater than or equal to threshold; prefer higher training-fold MI, then original order",
                "mi": "training-fold MI greater than permutation-null quantile",
                "model": "training-fold balanced-subsample RF importance greater than mean importance",
                "validation": "transform only; extra columns ignored and fitted schema columns required",
            },
            "warnings": [
                "Single 5-fold estimate only; repeated CV, threshold tuning, and temporal validation are deferred."
            ],
        },
        "dataset": {
            "shape": list(original.shape),
            "modeling_feature_count": len(feature_columns),
            "excluded_columns": [str(column) for column in excluded_columns],
            "target_distribution": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
            "outer_train_size": len(train_positions),
            "outer_holdout_size": len(holdout_positions),
        },
        "config": asdict(config),
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": config.n_splits,
            "shuffle": True,
            "random_state": config.random_state,
            "folds": fold_schema,
        },
        "results": result_groups,
        "best_mean_pr_auc": {
            "strategy": ranking[0]["strategy"],
            "model": ranking[0]["model"],
            "value": ranking[0]["metrics"]["pr_auc"]["mean"],
            "tie_break": "strategy order, then model order",
        },
    }
    return payload, csv_frame


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
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def write_fold_safe_results(
    payload: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic strict JSON and fold-level CSV files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "fold_safe_results.json"
    csv_path = directory / "fold_safe_fold_metrics.csv"
    content = json.dumps(_json_safe(payload), allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
    json_path.write_text(f"{content}\n", encoding="utf-8")
    fold_metrics.loc[:, CSV_COLUMNS].to_csv(csv_path, index=False)
    return json_path, csv_path
