"""Reusable, dataframe-based data quality checks for manufacturing data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

CheckStatus = Literal["pass", "warning", "error"]


@dataclass(frozen=True)
class DataQualityConfig:
    """Schema expectations and thresholds for data-quality validation."""

    required_columns: tuple[str, ...] = ()
    expected_shape: tuple[int | None, int | None] | None = None
    target_column: str = "class"
    expected_target_values: frozenset[Any] = frozenset({-1, 1})
    timestamp_column: str | None = "timestamp"
    feature_columns: tuple[str, ...] | None = None
    timestamp_min_success_rate: float = 0.95
    feature_missing_warning_rate: float = 0.10
    feature_missing_error_rate: float = 0.40
    near_constant_rate: float = 0.99
    minimum_minority_samples: int = 30
    minimum_minority_ratio: float = 0.05
    require_chronological_order: bool = False
    dayfirst: bool = True

    def __post_init__(self) -> None:
        rates = {
            "timestamp_min_success_rate": self.timestamp_min_success_rate,
            "feature_missing_warning_rate": self.feature_missing_warning_rate,
            "feature_missing_error_rate": self.feature_missing_error_rate,
            "near_constant_rate": self.near_constant_rate,
            "minimum_minority_ratio": self.minimum_minority_ratio,
        }
        for name, value in rates.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.feature_missing_warning_rate > self.feature_missing_error_rate:
            raise ValueError("The missing-value warning rate cannot exceed the error rate.")
        if self.minimum_minority_samples < 0:
            raise ValueError("minimum_minority_samples cannot be negative.")


@dataclass(frozen=True)
class QualityCheck:
    """One structured data-quality decision."""

    name: str
    status: CheckStatus
    criterion: Any
    observed: Any
    message: str
    columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["columns"] = list(self.columns)
        return result


@dataclass
class DataQualityReport:
    """A collection of checks suitable for Python use or serialization."""

    checks: list[QualityCheck] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(check.status == "error" for check in self.checks)

    @property
    def warnings(self) -> list[QualityCheck]:
        return [check for check in self.checks if check.status == "warning"]

    @property
    def errors(self) -> list[QualityCheck]:
        return [check for check in self.checks if check.status == "error"]

    def get(self, name: str) -> QualityCheck:
        matches = [check for check in self.checks if check.name == name]
        if len(matches) != 1:
            raise KeyError(f"Expected one check named {name!r}, found {len(matches)}.")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": {
                "pass": sum(check.status == "pass" for check in self.checks),
                "warning": len(self.warnings),
                "error": len(self.errors),
            },
            "metadata": self.metadata,
            "checks": [check.to_dict() for check in self.checks],
        }


def _check(
    name: str,
    status: CheckStatus,
    criterion: Any,
    observed: Any,
    message: str,
    columns: Sequence[str] = (),
) -> QualityCheck:
    return QualityCheck(name, status, criterion, observed, message, tuple(columns))


def _duplicate_feature_groups(frame: pd.DataFrame) -> list[list[str]]:
    groups: list[list[str]] = []
    remaining = list(frame.columns)
    while remaining:
        base = remaining.pop(0)
        group = [base]
        for candidate in remaining.copy():
            if frame[base].equals(frame[candidate]):
                group.append(candidate)
                remaining.remove(candidate)
        if len(group) > 1:
            groups.append(group)
    return groups


def validate_data_quality(
    data: pd.DataFrame,
    config: DataQualityConfig | None = None,
) -> DataQualityReport:
    """Validate a manufacturing dataframe without mutating it."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    config = config or DataQualityConfig()
    report = DataQualityReport(
        metadata={"rows": int(data.shape[0]), "columns": int(data.shape[1])}
    )

    required = set(config.required_columns) | {config.target_column}
    if config.timestamp_column is not None:
        required.add(config.timestamp_column)
    missing_required = sorted(required - set(data.columns))
    report.checks.append(_check(
        "required_columns", "error" if missing_required else "pass",
        {"required": sorted(required)}, {"missing": missing_required},
        "Required columns are missing." if missing_required else "All required columns are present.",
        missing_required,
    ))

    if config.expected_shape is not None:
        expected_rows, expected_columns = config.expected_shape
        shape_ok = (
            (expected_rows is None or data.shape[0] == expected_rows)
            and (expected_columns is None or data.shape[1] == expected_columns)
        )
        report.checks.append(_check(
            "expected_shape", "pass" if shape_ok else "error",
            {"rows": expected_rows, "columns": expected_columns},
            {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
            "Dataframe shape matches the expectation." if shape_ok else "Unexpected dataframe shape.",
        ))

    excluded = {config.target_column}
    if config.timestamp_column is not None:
        excluded.add(config.timestamp_column)
    if config.feature_columns is None:
        features = [column for column in data.columns if column not in excluded]
    else:
        features = [column for column in config.feature_columns if column in data.columns]
        absent_features = sorted(set(config.feature_columns) - set(data.columns))
        if absent_features:
            report.checks.append(_check(
                "configured_feature_columns", "error",
                {"configured": list(config.feature_columns)}, {"missing": absent_features},
                "Configured feature columns are missing.", absent_features,
            ))
    feature_frame = data[features]

    non_numeric = [column for column in features if not is_numeric_dtype(data[column])]
    report.checks.append(_check(
        "numeric_feature_dtypes", "error" if non_numeric else "pass",
        "All configured feature columns must have a numeric dtype.",
        {column: str(data[column].dtype) for column in non_numeric},
        "Non-numeric feature columns found." if non_numeric else "All feature columns are numeric.",
        non_numeric,
    ))

    if config.target_column in data:
        target = data[config.target_column]
        target_missing = int(target.isna().sum())
        report.checks.append(_check(
            "target_missing", "error" if target_missing else "pass", 0, target_missing,
            "Target contains missing values." if target_missing else "Target has no missing values.",
            [config.target_column] if target_missing else (),
        ))
        observed_values = set(target.dropna().unique().tolist())
        unexpected = sorted(observed_values - set(config.expected_target_values), key=str)
        missing_classes = sorted(set(config.expected_target_values) - observed_values, key=str)
        target_ok = not unexpected and not missing_classes
        report.checks.append(_check(
            "target_values", "pass" if target_ok else "error",
            {"expected": sorted(config.expected_target_values, key=str)},
            {"observed": sorted(observed_values, key=str), "unexpected": unexpected,
             "missing_expected_classes": missing_classes},
            "Target values match the expectation." if target_ok else "Target classes do not match the expectation.",
            [config.target_column],
        ))
        counts = target.dropna().value_counts()
        ratios = target.dropna().value_counts(normalize=True)
        minority_count = int(counts.min()) if len(counts) else 0
        minority_ratio = float(ratios.min()) if len(ratios) else 0.0
        minority_warning = (
            minority_count < config.minimum_minority_samples
            or minority_ratio < config.minimum_minority_ratio
        )
        report.checks.append(_check(
            "target_class_distribution", "warning" if minority_warning else "pass",
            {"minimum_minority_samples": config.minimum_minority_samples,
             "minimum_minority_ratio": config.minimum_minority_ratio},
            {"counts": {str(key): int(value) for key, value in counts.items()},
             "ratios": {str(key): float(value) for key, value in ratios.items()},
             "minority_samples": minority_count, "minority_ratio": minority_ratio},
            "Minority class is below a configured minimum." if minority_warning
            else "Target class distribution meets the configured minimums.",
            [config.target_column],
        ))

    if config.timestamp_column is not None and config.timestamp_column in data:
        raw_timestamp = data[config.timestamp_column]
        parsed = pd.to_datetime(
            raw_timestamp, format="mixed", dayfirst=config.dayfirst, errors="coerce"
        )
        non_missing_input = int(raw_timestamp.notna().sum())
        parsed_count = int(parsed.notna().sum())
        success_rate = parsed_count / non_missing_input if non_missing_input else 0.0
        parse_ok = success_rate >= config.timestamp_min_success_rate
        report.checks.append(_check(
            "timestamp_parsing", "pass" if parse_ok else "error",
            {"minimum_success_rate": config.timestamp_min_success_rate},
            {"non_missing_input": non_missing_input, "parsed": parsed_count,
             "success_rate": float(success_rate)},
            "Timestamp parsing meets the configured success rate." if parse_ok
            else "Timestamp parsing success rate is too low.",
            [config.timestamp_column],
        ))
        valid = parsed.dropna()
        chronological = bool(valid.is_monotonic_increasing)
        order_status: CheckStatus = "pass" if chronological else (
            "error" if config.require_chronological_order else "warning"
        )
        report.checks.append(_check(
            "timestamp_range_and_order", order_status,
            {"chronological_order_required": config.require_chronological_order},
            {"minimum": valid.min().isoformat() if len(valid) else None,
             "maximum": valid.max().isoformat() if len(valid) else None,
             "chronological": chronological},
            "Timestamp values are chronological." if chronological
            else "Timestamp values are not chronological.",
            [config.timestamp_column],
        ))

    missing_rates = feature_frame.isna().mean()
    error_missing = missing_rates[missing_rates > config.feature_missing_error_rate]
    warning_missing = missing_rates[
        (missing_rates > config.feature_missing_warning_rate)
        & (missing_rates <= config.feature_missing_error_rate)
    ]
    missing_status: CheckStatus = "error" if len(error_missing) else (
        "warning" if len(warning_missing) else "pass"
    )
    affected_missing = list(error_missing.index) + list(warning_missing.index)
    report.checks.append(_check(
        "feature_missing_rates", missing_status,
        {"warning_above": config.feature_missing_warning_rate,
         "error_above": config.feature_missing_error_rate},
        {column: float(rate) for column, rate in missing_rates.items()},
        "Feature missing rates exceed configured thresholds." if affected_missing
        else "Feature missing rates meet the configured thresholds.",
        affected_missing,
    ))

    constant_columns: list[str] = []
    near_constant: dict[str, float] = {}
    for column in features:
        non_missing = data[column].dropna()
        unique_count = int(non_missing.nunique())
        if unique_count <= 1:
            constant_columns.append(column)
        elif len(non_missing):
            dominant_rate = float(non_missing.value_counts(normalize=True).iloc[0])
            if dominant_rate >= config.near_constant_rate:
                near_constant[column] = dominant_rate
    report.checks.append(_check(
        "constant_features", "warning" if constant_columns else "pass",
        "More than one observed value", constant_columns,
        "Constant features found." if constant_columns else "No constant features found.",
        constant_columns,
    ))
    report.checks.append(_check(
        "near_constant_features", "warning" if near_constant else "pass",
        {"dominant_value_rate_below": config.near_constant_rate}, near_constant,
        "Near-constant features found." if near_constant else "No near-constant features found.",
        list(near_constant),
    ))

    duplicate_groups = _duplicate_feature_groups(feature_frame)
    duplicate_columns = [column for group in duplicate_groups for column in group]
    report.checks.append(_check(
        "exact_duplicate_features", "warning" if duplicate_groups else "pass",
        "No feature columns with identical values and missing-value positions", duplicate_groups,
        "Exact duplicate feature groups found." if duplicate_groups
        else "No exact duplicate features found.", duplicate_columns,
    ))
    duplicate_rows = int(data.duplicated(keep="first").sum())
    report.checks.append(_check(
        "exact_duplicate_rows", "warning" if duplicate_rows else "pass", 0, duplicate_rows,
        "Exact duplicate rows found." if duplicate_rows else "No exact duplicate rows found.",
    ))

    numeric_frame = feature_frame.select_dtypes(include=[np.number])
    infinity_counts = {
        column: int(np.isinf(numeric_frame[column].to_numpy(dtype=float, na_value=np.nan)).sum())
        for column in numeric_frame.columns
    }
    infinity_counts = {column: count for column, count in infinity_counts.items() if count}
    report.checks.append(_check(
        "infinite_values", "error" if infinity_counts else "pass", 0, infinity_counts,
        "Positive or negative infinity found." if infinity_counts else "No infinite values found.",
        list(infinity_counts),
    ))
    return report
