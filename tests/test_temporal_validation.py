from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.fold_safe_modeling import FoldSafeConfig
from src.temporal_validation import (
    DRIFT_COLUMNS, PERFORMANCE_COLUMNS, TemporalConfig, assess_timestamps,
    evaluate_temporal_validation, feature_drift, make_forward_splits,
    parse_timestamps, write_temporal_results,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_temporal_validation.py"


def frame(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    y = np.where(np.arange(rows) % 5 == 0, 1, -1)
    times = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(rows) // 2, unit="h")
    return pd.DataFrame({
        "signal": y + rng.normal(size=rows), "noise": rng.normal(size=rows),
        "constant": 3.0, "all_missing": np.nan, "identifier": np.arange(rows),
        "timestamp": times.strftime("%d/%m/%Y %H:%M:%S"), "class": y,
    })


def evaluated(source=None):
    return evaluate_temporal_validation(
        frame() if source is None else source, excluded_columns=("identifier",),
        config=TemporalConfig(random_state=4, n_splits=3, drift_bins=4),
        modeling_config=FoldSafeConfig.quick(random_state=4),
    )


def test_timestamp_parse_failure_and_unsuitable_assessment():
    with pytest.raises(ValueError, match="failed"):
        parse_timestamps(pd.Series(["31/01/2025", "not-a-time"]))
    result = assess_timestamps(pd.Series(["not-a-time"]))
    assert result["verdict"] == "unsuitable" and result["parsed_count"] == 0


def test_timestamp_parser_accepts_uci_dayfirst_and_loader_iso_forms():
    parsed = parse_timestamps(pd.Series(["19/07/2008 11:55:00", "2008-08-01 02:02:00"]))
    assert parsed.dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2008-07-19 11:55:00", "2008-08-01 02:02:00",
    ]


def test_forward_splits_keep_timestamp_groups_and_strict_order():
    timestamps = parse_timestamps(frame(40).timestamp)
    for train, validation in make_forward_splits(timestamps, 3):
        assert timestamps.iloc[train].max() < timestamps.iloc[validation].min()
        assert set(timestamps.iloc[train]).isdisjoint(timestamps.iloc[validation])


def test_excluded_timestamp_target_identifier_and_fresh_split_models(monkeypatch):
    import src.temporal_validation as module
    original, fitted_ids = module._model, []
    template = original(FoldSafeConfig.quick(random_state=4))
    original_fit = type(template).fit
    def fit_spy(self, X, y=None, **params):
        fitted_ids.append(id(self))
        assert "timestamp" not in X and "class" not in X and "identifier" not in X
        return original_fit(self, X, y, **params)
    monkeypatch.setattr(type(template), "fit", fit_spy)
    payload, performance, _ = evaluated()
    assert len(fitted_ids) == int((performance.status == "evaluated").sum())
    assert len(set(fitted_ids)) == len(fitted_ids)
    assert payload["dataset"]["excluded_columns"] == ["timestamp", "class", "identifier"]


def test_future_changes_do_not_change_past_split_fit(monkeypatch):
    import src.temporal_validation as module
    captured = []
    original = module._feature_audit
    def spy(pipeline):
        result = original(pipeline)
        captured.append(copy.deepcopy(result))
        return result
    monkeypatch.setattr(module, "_feature_audit", spy)
    source = frame()
    evaluated(source)
    first = captured[0]
    captured.clear()
    changed = source.copy()
    changed.loc[changed.index[-15:], "noise"] = 1e9
    changed.loc[changed.index[-15:], "class"] *= -1
    evaluated(changed)
    assert captured[0] == first


def test_positive_score_metrics_are_score_based():
    _, performance, _ = evaluated()
    assert (performance[performance.status == "evaluated"].pr_auc >= performance[performance.status == "evaluated"].validation_fail_rate).any()
    assert (performance.threshold == 0.5).all()


def test_single_class_temporal_interval_is_not_evaluated():
    source = frame()
    source.loc[source.index >= 75, "class"] = -1
    _, performance, _ = evaluated(source)
    assert "not_evaluated" in set(performance.status)
    row = performance[performance.status == "not_evaluated"].iloc[0]
    assert row.pr_auc_defined in (False, np.bool_(False)) and pd.isna(row.pr_auc)


def test_drift_uses_training_edges_and_handles_missing_constant_all_missing():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, np.nan], "constant": [1.0] * 4, "missing": [np.nan] * 4})
    validation = pd.DataFrame({"x": [-100.0, 1.0, 100.0, np.nan], "constant": [0.0, 1.0, 2.0, 1.0], "missing": [1.0, np.nan, np.nan, np.nan]})
    rows, refs = feature_drift(train, validation, split=1, bins=2)
    result = pd.DataFrame(rows).set_index("feature")
    assert refs["x"]["edges"][0] == -np.inf and refs["x"]["edges"][-1] == np.inf
    assert result.loc["x", "validation_below_train_range_count"] == 1
    assert result.loc["x", "validation_above_train_range_count"] == 1
    assert result.loc["constant", "train_reference_status"] == "constant_training"
    assert result.loc["missing", "train_reference_status"] == "all_missing_training"
    with pytest.raises(ValueError, match="infinity"):
        feature_drift(pd.DataFrame({"x": [1.0, np.inf]}), pd.DataFrame({"x": [1.0]}), split=1)


def test_strict_outputs_fixed_schema_and_input_immutability(tmp_path):
    source, before = frame(), frame()
    payload, performance, drift = evaluated(source)
    copies = copy.deepcopy(payload), performance.copy(deep=True), drift.copy(deep=True)
    paths = write_temporal_results(payload, performance, drift, tmp_path)
    parsed = json.loads(paths[0].read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert parsed["status"] == "complete"
    assert tuple(pd.read_csv(paths[1]).columns) == PERFORMANCE_COLUMNS
    assert tuple(pd.read_csv(paths[2]).columns) == DRIFT_COLUMNS
    assert payload == copies[0]
    pd.testing.assert_frame_equal(performance, copies[1])
    pd.testing.assert_frame_equal(drift, copies[2])
    pd.testing.assert_frame_equal(source, before)


def test_cli_smoke_help_and_bad_inputs(tmp_path):
    csv, output = tmp_path / "input.csv", tmp_path / "output"
    frame(60).to_csv(csv, index=False)
    command = [sys.executable, str(CLI), "--input-csv", str(csv), "--output-dir", str(output),
               "--exclude-column", "identifier", "--quick", "--n-splits", "2", "--drift-bins", "3"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    help_result = subprocess.run([sys.executable, str(CLI), "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert help_result.returncode == 0 and "--timestamp-column" in help_result.stdout
    assert all((output / name).is_file() for name in (
        "temporal_validation_results.json", "temporal_performance.csv", "temporal_feature_drift.csv"))
    bad = tmp_path / "bad.csv"
    frame(30).assign(timestamp="bad").to_csv(bad, index=False)
    invalid = subprocess.run([sys.executable, str(CLI), "--input-csv", str(bad), "--quick"], cwd=ROOT, capture_output=True, text=True)
    missing = subprocess.run([sys.executable, str(CLI), "--input-csv", str(tmp_path / "none.csv")], cwd=ROOT, capture_output=True, text=True)
    assert all(item.returncode != 0 and "Traceback" not in item.stderr for item in (invalid, missing))
