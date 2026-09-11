import json

import numpy as np
import pandas as pd
import pytest

from src.data_quality import DataQualityConfig, validate_data_quality


def make_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="h").astype(str),
        "sensor_a": np.arange(10, dtype=float),
        "sensor_b": np.linspace(1, 2, 10),
        "class": [-1] * 5 + [1] * 5,
    })


def config(**overrides) -> DataQualityConfig:
    defaults = {
        "required_columns": ("sensor_a", "sensor_b"),
        "expected_shape": (10, 4),
        "minimum_minority_samples": 2,
        "minimum_minority_ratio": 0.10,
        "near_constant_rate": 0.90,
    }
    defaults.update(overrides)
    return DataQualityConfig(**defaults)


def test_clean_frame_returns_structured_serializable_report():
    report = validate_data_quality(make_frame(), config())
    assert report.passed
    assert not report.errors
    assert report.get("expected_shape").status == "pass"
    payload = report.to_dict()
    assert payload["metadata"] == {"rows": 10, "columns": 4}
    assert payload["summary"]["error"] == 0
    json.dumps(payload)


def test_schema_dtype_and_target_value_failures_are_errors():
    frame = make_frame().drop(columns="sensor_b")
    frame["sensor_a"] = frame["sensor_a"].astype(str)
    frame.loc[0, "class"] = 0
    report = validate_data_quality(frame, config())
    assert report.get("required_columns").status == "error"
    assert report.get("expected_shape").status == "error"
    assert report.get("numeric_feature_dtypes").status == "error"
    assert report.get("target_values").status == "error"
    assert not report.passed


def test_target_missing_and_minority_thresholds_are_separate():
    frame = make_frame()
    frame["class"] = [-1] * 9 + [1]
    frame.loc[0, "class"] = np.nan
    report = validate_data_quality(
        frame, config(minimum_minority_samples=3, minimum_minority_ratio=0.20)
    )
    assert report.get("target_missing").status == "error"
    distribution = report.get("target_class_distribution")
    assert distribution.status == "warning"
    assert distribution.observed["minority_samples"] == 1


def test_timestamp_parse_rate_range_order_and_custom_thresholds():
    frame = make_frame()
    frame.loc[2, "timestamp"] = "not-a-date"
    frame.loc[7, "timestamp"], frame.loc[8, "timestamp"] = (
        frame.loc[8, "timestamp"], frame.loc[7, "timestamp"]
    )
    permissive = validate_data_quality(
        frame, config(timestamp_min_success_rate=0.80, require_chronological_order=False)
    )
    strict = validate_data_quality(
        frame, config(timestamp_min_success_rate=0.95, require_chronological_order=True)
    )
    assert permissive.get("timestamp_parsing").status == "pass"
    assert permissive.get("timestamp_range_and_order").status == "warning"
    assert permissive.get("timestamp_range_and_order").observed["minimum"].startswith("2024-01-01")
    assert strict.get("timestamp_parsing").status == "error"
    assert strict.get("timestamp_range_and_order").status == "error"


def test_feature_missing_rates_distinguish_warning_and_error():
    warning_frame = make_frame()
    warning_frame.loc[:1, "sensor_a"] = np.nan
    error_frame = make_frame()
    error_frame.loc[:5, "sensor_a"] = np.nan
    thresholds = config(feature_missing_warning_rate=0.10, feature_missing_error_rate=0.50)
    warning_report = validate_data_quality(warning_frame, thresholds)
    error_report = validate_data_quality(error_frame, thresholds)
    assert warning_report.get("feature_missing_rates").status == "warning"
    assert error_report.get("feature_missing_rates").status == "error"
    assert error_report.get("feature_missing_rates").observed["sensor_a"] == pytest.approx(0.6)


def test_constant_near_constant_and_duplicate_features_are_detected():
    frame = make_frame()
    frame["constant"] = 3.0
    frame["near_constant"] = [0.0] * 9 + [1.0]
    frame["sensor_a_copy"] = frame["sensor_a"]
    quality_config = config(
        expected_shape=(10, 7),
        feature_columns=("sensor_a", "sensor_b", "constant", "near_constant", "sensor_a_copy"),
    )
    report = validate_data_quality(frame, quality_config)
    assert report.get("constant_features").columns == ("constant",)
    assert report.get("near_constant_features").columns == ("near_constant",)
    assert ["sensor_a", "sensor_a_copy"] in report.get("exact_duplicate_features").observed


def test_duplicate_rows_nan_and_both_infinity_signs_are_detected():
    frame = make_frame()
    frame.loc[1] = frame.loc[0]
    frame.loc[2, "sensor_a"] = np.nan
    frame.loc[3, "sensor_a"] = np.inf
    frame.loc[4, "sensor_b"] = -np.inf
    report = validate_data_quality(frame, config(feature_missing_warning_rate=0.0))
    assert report.get("exact_duplicate_rows").observed == 1
    assert report.get("feature_missing_rates").status == "warning"
    assert report.get("infinite_values").status == "error"
    assert report.get("infinite_values").observed == {"sensor_a": 1, "sensor_b": 1}


@pytest.mark.parametrize("kwargs", [
    {"near_constant_rate": 1.1},
    {"minimum_minority_ratio": -0.1},
    {"feature_missing_warning_rate": 0.6, "feature_missing_error_rate": 0.5},
    {"minimum_minority_samples": -1},
])
def test_invalid_custom_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DataQualityConfig(**kwargs)
