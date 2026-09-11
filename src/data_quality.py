"""Reusable, dataframe-based data quality checks for manufacturing data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

CheckStatus = Literal["pass", "warning", "error", "not_evaluated"]


def _json_safe(value: Any) -> Any:
    """Recursively convert report values to strict-JSON-compatible values."""

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
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, bool)):
        return value
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


@dataclass(frozen=True)
class DataQualityConfig:
    """Schema expectations and thresholds for generic dataframe validation."""

    required_columns: tuple[str, ...] = ()
    expected_shape: tuple[int | None, int | None] | None = None
    target_column: str | None = None
    expected_target_values: frozenset[Any] | None = None
    timestamp_column: str | None = None
    feature_columns: tuple[str, ...] | None = None
    timestamp_min_success_rate: float = 0.95
    feature_missing_warning_rate: float = 0.10
    feature_missing_error_rate: float = 0.40
    near_constant_rate: float = 0.99
    minimum_minority_samples: int = 30
    minimum_minority_ratio: float = 0.05
    require_chronological_order: bool = False
    dayfirst: bool = False

    @classmethod
    def secom(cls, **overrides: Any) -> "DataQualityConfig":
        """Return a SECOM-oriented preset while allowing threshold overrides."""

        values: dict[str, Any] = {
            "target_column": "class",
            "expected_target_values": frozenset({-1, 1}),
            "timestamp_column": "timestamp",
            "dayfirst": True,
        }
        values.update(overrides)
        return cls(**values)

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
        return _json_safe(asdict(self))


@dataclass
class DataQualityReport:
    """A fixed-schema collection of checks suitable for serialization."""

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

    @property
    def not_evaluated(self) -> list[QualityCheck]:
        return [check for check in self.checks if check.status == "not_evaluated"]

    def get(self, name: str) -> QualityCheck:
        matches = [check for check in self.checks if check.name == name]
        if len(matches) != 1:
            raise KeyError(f"Expected one check named {name!r}, found {len(matches)}.")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "passed": self.passed,
            "summary": {
                status: sum(check.status == status for check in self.checks)
                for status in ("pass", "warning", "error", "not_evaluated")
            },
            "metadata": self.metadata,
            "checks": [check.to_dict() for check in self.checks],
        }
        return _json_safe(payload)


def _check(
    name: str,
    status: CheckStatus,
    criterion: Any,
    observed: Any,
    message: str,
    columns: Sequence[str] = (),
) -> QualityCheck:
    return QualityCheck(name, status, criterion, observed, message, tuple(columns))


def _not_evaluated(name: str, reason: str, columns: Sequence[str] = ()) -> QualityCheck:
    return _check(name, "not_evaluated", None, None, reason, columns)


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
    is_empty = data.empty
    report.checks.append(_check(
        "non_empty", "error" if is_empty else "pass", "At least one row", int(len(data)),
        "Dataframe is empty." if is_empty else "Dataframe contains rows.",
    ))

    required = set(config.required_columns)
    if config.target_column is not None:
        required.add(config.target_column)
    if config.timestamp_column is not None:
        required.add(config.timestamp_column)
    missing_required = sorted(required - set(data.columns))
    report.checks.append(_check(
        "required_columns", "error" if missing_required else "pass",
        {"required": sorted(required)}, {"missing": missing_required},
        "Required columns are missing." if missing_required else "All required columns are present.",
        missing_required,
    ))

    if config.expected_shape is None:
        report.checks.append(_not_evaluated("expected_shape", "No expected shape was configured."))
    else:
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

    excluded = {column for column in (config.target_column, config.timestamp_column) if column}
    if config.feature_columns is None:
        features = [column for column in data.columns if column not in excluded]
        report.checks.append(_check(
            "configured_feature_columns", "pass", "Infer features from non-target columns",
            {"inferred": features}, "Feature columns were inferred.", features,
        ))
    else:
        absent_features = sorted(set(config.feature_columns) - set(data.columns))
        features = [column for column in config.feature_columns if column in data.columns]
        report.checks.append(_check(
            "configured_feature_columns", "error" if absent_features else "pass",
            {"configured": list(config.feature_columns)}, {"missing": absent_features},
            "Configured feature columns are missing." if absent_features
            else "All configured feature columns are present.", absent_features,
        ))
    feature_frame = data[features]

    if is_empty:
        reason = "Dataframe is empty; this check cannot be evaluated."
        for name in (
            "numeric_feature_dtypes", "target_missing", "target_values",
            "target_class_distribution", "timestamp_parsing", "timestamp_range_and_order",
            "feature_missing_rates", "constant_features", "near_constant_features",
            "exact_duplicate_features", "exact_duplicate_rows", "infinite_values",
        ):
            report.checks.append(_not_evaluated(name, reason))
        return report

    non_numeric = [
        column for column in features
        if is_bool_dtype(data[column].dtype) or not is_numeric_dtype(data[column].dtype)
    ]
    report.checks.append(_check(
        "numeric_feature_dtypes", "error" if non_numeric else "pass",
        "Feature dtypes must be numeric and non-boolean.",
        {column: str(data[column].dtype) for column in non_numeric},
        "Non-numeric or boolean feature columns found." if non_numeric
        else "All feature columns are numeric and non-boolean.", non_numeric,
    ))

    if config.target_column is None:
        reason = "No target column was configured."
        report.checks.extend(_not_evaluated(name, reason) for name in (
            "target_missing", "target_values", "target_class_distribution"
        ))
    elif config.target_column not in data.columns:
        reason = "The configured target column is missing."
        report.checks.extend(_not_evaluated(name, reason, [config.target_column]) for name in (
            "target_missing", "target_values", "target_class_distribution"
        ))
    else:
        target = data[config.target_column]
        target_missing = int(target.isna().sum())
        report.checks.append(_check(
            "target_missing", "error" if target_missing else "pass", 0, target_missing,
            "Target contains missing values." if target_missing else "Target has no missing values.",
            [config.target_column],
        ))
        observed_values = set(target.dropna().unique().tolist())
        if config.expected_target_values is None:
            report.checks.append(_not_evaluated(
                "target_values", "No expected target values were configured.", [config.target_column]
            ))
            class_values = sorted(observed_values, key=str)
        else:
            expected_values = set(config.expected_target_values)
            unexpected = sorted(observed_values - expected_values, key=str)
            missing_classes = sorted(expected_values - observed_values, key=str)
            target_ok = not unexpected and not missing_classes
            report.checks.append(_check(
                "target_values", "pass" if target_ok else "error",
                {"expected": sorted(expected_values, key=str)},
                {"observed": sorted(observed_values, key=str), "unexpected": unexpected,
                 "missing_expected_classes": missing_classes},
                "Target values match the expectation." if target_ok
                else "Target classes do not match the expectation.", [config.target_column],
            ))
            class_values = sorted(expected_values | observed_values, key=str)
        non_missing_target = target.dropna()
        raw_counts = non_missing_target.value_counts()
        total = int(len(non_missing_target))
        counts = {value: int(raw_counts.get(value, 0)) for value in class_values}
        ratios = {value: (count / total if total else 0.0) for value, count in counts.items()}
        positive_counts = [count for count in counts.values() if count > 0]
        observed_class_count = len(positive_counts)
        minority_count = min(positive_counts) if observed_class_count >= 2 else 0
        minority_ratio = min(
            (ratio for value, ratio in ratios.items() if counts[value] > 0), default=0.0
        ) if observed_class_count >= 2 else 0.0
        if observed_class_count < 2:
            distribution_status: CheckStatus = "error"
            distribution_message = "Target must contain at least two observed classes."
        elif (minority_count < config.minimum_minority_samples
              or minority_ratio < config.minimum_minority_ratio):
            distribution_status = "warning"
            distribution_message = "Minority class is below a configured minimum."
        else:
            distribution_status = "pass"
            distribution_message = "Target class distribution meets the configured minimums."
        report.checks.append(_check(
            "target_class_distribution", distribution_status,
            {"minimum_observed_classes": 2,
             "minimum_minority_samples": config.minimum_minority_samples,
             "minimum_minority_ratio": config.minimum_minority_ratio},
            {"counts": {str(key): value for key, value in counts.items()},
             "ratios": {str(key): float(value) for key, value in ratios.items()},
             "observed_class_count": observed_class_count,
             "minority_samples": minority_count, "minority_ratio": float(minority_ratio)},
            distribution_message, [config.target_column],
        ))

    if config.timestamp_column is None:
        reason = "No timestamp column was configured."
        report.checks.extend(_not_evaluated(name, reason) for name in (
            "timestamp_parsing", "timestamp_range_and_order"
        ))
    elif config.timestamp_column not in data.columns:
        reason = "The configured timestamp column is missing."
        report.checks.extend(_not_evaluated(name, reason, [config.timestamp_column]) for name in (
            "timestamp_parsing", "timestamp_range_and_order"
        ))
    else:
        raw_timestamp = data[config.timestamp_column]
        parsed = pd.to_datetime(
            raw_timestamp, format="mixed", dayfirst=config.dayfirst, errors="coerce"
        )
        non_missing_input = int(raw_timestamp.notna().sum())
        parsed_count = int(parsed.notna().sum())
        success_rate = parsed_count / non_missing_input if non_missing_input else 0.0
        parse_ok = success_rate >= config.timestamp_min_success_rate and parsed_count > 0
        report.checks.append(_check(
            "timestamp_parsing", "pass" if parse_ok else "error",
            {"minimum_success_rate": config.timestamp_min_success_rate,
             "at_least_one_valid_timestamp": True},
            {"rows": int(len(raw_timestamp)), "non_missing_input": non_missing_input,
             "parsed": parsed_count, "success_rate": float(success_rate)},
            "Timestamp parsing meets the configured success rate." if parse_ok
            else "Timestamp parsing success rate is too low or no timestamp parsed.",
            [config.timestamp_column],
        ))
        valid = parsed.dropna()
        if valid.empty:
            report.checks.append(_check(
                "timestamp_range_and_order", "not_evaluated",
                {"chronological_order_required": config.require_chronological_order},
                {"minimum": None, "maximum": None, "chronological": None},
                "No valid timestamps are available for range or order evaluation.",
                [config.timestamp_column],
            ))
        else:
            chronological = bool(valid.is_monotonic_increasing)
            order_status: CheckStatus = "pass" if chronological else (
                "error" if config.require_chronological_order else "warning"
            )
            report.checks.append(_check(
                "timestamp_range_and_order", order_status,
                {"chronological_order_required": config.require_chronological_order,
                 "invalid_timestamps_excluded": int(len(raw_timestamp) - len(valid))},
                {"minimum": valid.min(), "maximum": valid.max(),
                 "chronological": chronological},
                "Valid timestamp values are chronological." if chronological
                else "Valid timestamp values are not chronological.", [config.timestamp_column],
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
        else "Feature missing rates meet the configured thresholds.", affected_missing,
    ))

    constant_columns: list[str] = []
    near_constant: dict[str, float] = {}
    not_observed: list[str] = []
    for column in features:
        non_missing = data[column].dropna()
        if non_missing.empty:
            not_observed.append(column)
            continue
        unique_count = int(non_missing.nunique())
        if unique_count == 1:
            constant_columns.append(column)
        else:
            dominant_rate = float(non_missing.value_counts(normalize=True).iloc[0])
            if dominant_rate >= config.near_constant_rate:
                near_constant[column] = dominant_rate
    report.checks.append(_check(
        "constant_features", "warning" if constant_columns else "pass",
        {"minimum_distinct_non_missing_values": 2,
         "all_missing_features_not_evaluated": not_observed}, constant_columns,
        "Constant features found." if constant_columns else "No constant features found.",
        constant_columns,
    ))
    report.checks.append(_check(
        "near_constant_features", "warning" if near_constant else "pass",
        {"dominant_non_missing_value_rate_at_least": config.near_constant_rate,
         "missing_values_excluded_from_denominator": True,
         "all_missing_features_not_evaluated": not_observed}, near_constant,
        "Near-constant features found." if near_constant
        else "No near-constant features found among evaluable features.", list(near_constant),
    ))

    duplicate_groups = _duplicate_feature_groups(feature_frame)
    duplicate_columns = [column for group in duplicate_groups for column in group]
    report.checks.append(_check(
        "exact_duplicate_features", "warning" if duplicate_groups else "pass",
        "Identical values, dtypes, indexes, and missing-value positions", duplicate_groups,
        "Exact duplicate feature groups found." if duplicate_groups
        else "No exact duplicate features found.", duplicate_columns,
    ))
    duplicate_rows = int(data.duplicated(keep="first").sum())
    report.checks.append(_check(
        "exact_duplicate_rows", "warning" if duplicate_rows else "pass", 0, duplicate_rows,
        "Exact duplicate rows found." if duplicate_rows else "No exact duplicate rows found.",
    ))

    numeric_frame = feature_frame.select_dtypes(include=[np.number]).select_dtypes(exclude=["bool"])
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
