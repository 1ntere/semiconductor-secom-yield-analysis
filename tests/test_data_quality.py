import json

import numpy as np
import pandas as pd
import pytest

from src.data_quality import DataQualityConfig, validate_data_quality


CHECK_NAMES = {
    "non_empty",
    "required_columns",
    "expected_shape",
    "configured_feature_columns",
    "numeric_feature_dtypes",
    "target_missing",
    "target_values",
    "target_class_distribution",
    "timestamp_parsing",
    "timestamp_range_and_order",
    "feature_missing_rates",
    "constant_features",
    "near_constant_features",
    "exact_duplicate_features",
    "exact_duplicate_rows",
    "infinite_values",
}


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h").astype(str),
            "sensor_a": np.arange(10, dtype=float),
            "sensor_b": np.linspace(1, 2, 10),
            "class": [-1] * 5 + [1] * 5,
        }
    )


def secom_config(**overrides) -> DataQualityConfig:
    defaults = {
        "required_columns": ("sensor_a", "sensor_b"),
        "expected_shape": (10, 4),
        "minimum_minority_samples": 2,
        "minimum_minority_ratio": 0.10,
        "near_constant_rate": 0.90,
    }
    defaults.update(overrides)
    return DataQualityConfig.secom(**defaults)


def assert_fixed_schema(report) -> None:
    assert {check.name for check in report.checks} == CHECK_NAMES
    assert len(report.checks) == len(CHECK_NAMES)


def test_clean_secom_frame_returns_strict_json_serializable_report():
    report = validate_data_quality(make_frame(), secom_config())

    assert report.passed
    assert not report.errors
    assert_fixed_schema(report)
    payload = report.to_dict()
    assert payload["metadata"] == {"rows": 10, "columns": 4}
    assert payload["summary"]["error"] == 0
    assert payload["summary"]["not_evaluated"] == 0
    json.dumps(payload, allow_nan=False)


def test_empty_dataframe_is_error_and_data_checks_are_not_evaluated():
    frame = pd.DataFrame(columns=["timestamp", "sensor_a", "sensor_b", "class"])

    report = validate_data_quality(frame, secom_config(expected_shape=(0, 4)))

    assert report.get("non_empty").status == "error"
    for name in CHECK_NAMES - {
        "non_empty", "required_columns", "expected_shape", "configured_feature_columns"
    }:
        assert report.get(name).status == "not_evaluated"
    assert report.get("exact_duplicate_features").observed is None
    assert report.get("constant_features").observed is None
    assert_fixed_schema(report)
    json.dumps(report.to_dict(), allow_nan=False)


def test_missing_target_returns_fixed_schema_without_key_error():
    frame = make_frame().drop(columns="class")

    report = validate_data_quality(frame, secom_config(expected_shape=None))

    assert report.get("required_columns").status == "error"
    for name in ("target_missing", "target_values", "target_class_distribution"):
        assert report.get(name).status == "not_evaluated"
        assert report.get(name).columns == ("class",)
    assert_fixed_schema(report)


def test_missing_timestamp_returns_fixed_schema_without_key_error():
    frame = make_frame().drop(columns="timestamp")

    report = validate_data_quality(frame, secom_config(expected_shape=None))

    assert report.get("required_columns").status == "error"
    assert report.get("timestamp_parsing").status == "not_evaluated"
    assert report.get("timestamp_range_and_order").status == "not_evaluated"
    assert_fixed_schema(report)


def test_single_target_class_includes_zero_count_expected_class():
    frame = make_frame()
    frame["class"] = -1

    report = validate_data_quality(frame, secom_config())
    distribution = report.get("target_class_distribution")

    assert report.get("target_values").status == "error"
    assert distribution.status == "error"
    assert distribution.observed["counts"] == {"-1": 10, "1": 0}
    assert distribution.observed["ratios"] == {"-1": 1.0, "1": 0.0}
    assert distribution.observed["minority_samples"] == 0
    assert distribution.observed["minority_ratio"] == 0.0


def test_target_missing_and_low_minority_are_reported_separately():
    frame = make_frame()
    frame["class"] = [-1] * 9 + [1]
    frame.loc[0, "class"] = np.nan

    report = validate_data_quality(
        frame,
        secom_config(minimum_minority_samples=3, minimum_minority_ratio=0.20),
    )

    assert report.get("target_missing").status == "error"
    assert report.get("target_class_distribution").status == "warning"
    assert report.get("target_class_distribution").observed["minority_samples"] == 1


def test_all_timestamp_parsing_failure_skips_range_and_order():
    frame = make_frame()
    frame["timestamp"] = "not-a-date"

    report = validate_data_quality(frame, secom_config())
    range_check = report.get("timestamp_range_and_order")

    assert report.get("timestamp_parsing").status == "error"
    assert range_check.status == "not_evaluated"
    assert range_check.observed == {
        "minimum": None,
        "maximum": None,
        "chronological": None,
    }


def test_partial_timestamp_parsing_preserves_parse_and_order_decisions():
    frame = make_frame()
    frame.loc[2, "timestamp"] = "not-a-date"
    frame.loc[7, "timestamp"], frame.loc[8, "timestamp"] = (
        frame.loc[8, "timestamp"],
        frame.loc[7, "timestamp"],
    )

    permissive = validate_data_quality(
        frame,
        secom_config(timestamp_min_success_rate=0.80, require_chronological_order=False),
    )
    strict = validate_data_quality(
        frame,
        secom_config(timestamp_min_success_rate=0.95, require_chronological_order=True),
    )

    assert permissive.get("timestamp_parsing").status == "pass"
    assert permissive.get("timestamp_range_and_order").status == "warning"
    assert strict.get("timestamp_parsing").status == "error"
    assert strict.get("timestamp_range_and_order").status == "error"
    serialized_range = strict.get("timestamp_range_and_order").to_dict()["observed"]
    assert serialized_range["minimum"].startswith("2024-01-01")


@pytest.mark.parametrize(
    ("values", "expected_dtype"),
    [
        ([True, False] * 5, "bool"),
        (pd.array([True, False] * 5, dtype="boolean"), "boolean"),
    ],
)
def test_boolean_features_are_not_numeric(values, expected_dtype):
    frame = make_frame()
    frame["flag"] = values

    report = validate_data_quality(
        frame,
        secom_config(
            expected_shape=(10, 5),
            feature_columns=("sensor_a", "sensor_b", "flag"),
        ),
    )

    dtype_check = report.get("numeric_feature_dtypes")
    assert dtype_check.status == "error"
    assert dtype_check.columns == ("flag",)
    assert dtype_check.observed == {"flag": expected_dtype}


def test_string_feature_is_not_numeric():
    frame = make_frame()
    frame["sensor_a"] = frame["sensor_a"].astype(str)

    report = validate_data_quality(frame, secom_config())

    assert report.get("numeric_feature_dtypes").status == "error"
    assert report.get("numeric_feature_dtypes").columns == ("sensor_a",)


def test_missing_rates_distinguish_warning_error_and_remain_json_safe():
    frame = make_frame()
    frame.loc[:5, "sensor_a"] = np.nan

    report = validate_data_quality(
        frame,
        secom_config(feature_missing_warning_rate=0.10, feature_missing_error_rate=0.50),
    )

    assert report.get("feature_missing_rates").status == "error"
    assert report.get("feature_missing_rates").observed["sensor_a"] == pytest.approx(0.6)
    json.dumps(report.to_dict(), allow_nan=False)


def test_near_constant_excludes_missing_values_from_denominator():
    frame = make_frame()
    frame["mostly_same"] = [0.0] * 8 + [1.0, np.nan]
    frame["all_missing"] = np.nan
    report = validate_data_quality(
        frame,
        secom_config(
            expected_shape=(10, 6),
            feature_columns=("sensor_a", "sensor_b", "mostly_same", "all_missing"),
            near_constant_rate=0.85,
        ),
    )

    check = report.get("near_constant_features")
    assert check.observed["mostly_same"] == pytest.approx(8 / 9)
    assert "all_missing" not in check.columns
    assert check.criterion["missing_values_excluded_from_denominator"] is True
    assert check.criterion["all_missing_features_not_evaluated"] == ["all_missing"]


def test_duplicate_features_require_matching_nan_positions():
    frame = make_frame()
    frame["same_a"] = frame["sensor_a"]
    frame.loc[1, "sensor_a"] = np.nan
    frame.loc[2, "same_a"] = np.nan

    report = validate_data_quality(
        frame,
        secom_config(
            expected_shape=(10, 5),
            feature_columns=("sensor_a", "sensor_b", "same_a"),
        ),
    )

    assert report.get("exact_duplicate_features").status == "pass"
    assert report.get("exact_duplicate_features").observed == []


def test_duplicate_features_match_when_nan_positions_are_equal():
    frame = make_frame()
    frame.loc[2, "sensor_a"] = np.nan
    frame["same_a"] = frame["sensor_a"]

    report = validate_data_quality(
        frame,
        secom_config(
            expected_shape=(10, 5),
            feature_columns=("sensor_a", "sensor_b", "same_a"),
        ),
    )

    assert ["sensor_a", "same_a"] in report.get("exact_duplicate_features").observed


def test_nan_and_both_infinity_signs_are_separate_checks():
    frame = make_frame()
    frame.loc[2, "sensor_a"] = np.nan
    frame.loc[3, "sensor_a"] = np.inf
    frame.loc[4, "sensor_b"] = -np.inf

    report = validate_data_quality(
        frame,
        secom_config(feature_missing_warning_rate=0.0),
    )

    assert report.get("feature_missing_rates").observed["sensor_a"] == pytest.approx(0.1)
    assert report.get("infinite_values").observed == {"sensor_a": 1, "sensor_b": 1}
    json.dumps(report.to_dict(), allow_nan=False)


def test_generic_config_skips_target_and_timestamp_checks():
    frame = pd.DataFrame({"sensor_a": [1.0, 2.0], "sensor_b": [3.0, 4.0]})

    report = validate_data_quality(frame)

    assert report.passed
    for name in ("target_missing", "target_values", "target_class_distribution"):
        assert report.get(name).status == "not_evaluated"
    for name in ("timestamp_parsing", "timestamp_range_and_order"):
        assert report.get(name).status == "not_evaluated"
    assert_fixed_schema(report)


def test_secom_preset_and_overrides():
    preset = DataQualityConfig.secom(minimum_minority_samples=10)

    assert preset.target_column == "class"
    assert preset.expected_target_values == frozenset({-1, 1})
    assert preset.timestamp_column == "timestamp"
    assert preset.dayfirst is True
    assert preset.minimum_minority_samples == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"near_constant_rate": 1.1},
        {"minimum_minority_ratio": -0.1},
        {"feature_missing_warning_rate": 0.6, "feature_missing_error_rate": 0.5},
        {"minimum_minority_samples": -1},
    ],
)
def test_invalid_custom_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DataQualityConfig(**kwargs)
