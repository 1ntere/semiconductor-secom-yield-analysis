import copy
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import average_precision_score, roc_auc_score

from src.fold_safe_modeling import (
    CSV_COLUMNS,
    METRICS,
    MODELS,
    STRATEGIES,
    CorrelationPruner,
    FoldPreprocessor,
    FoldSafeConfig,
    ModelBasedSelector,
    MutualInformationSelector,
    _metric_values,
    build_feature_pipeline,
    evaluate_fold_safe,
    write_fold_safe_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "scripts" / "run_fold_safe_modeling.py"


def make_frame(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    target = np.where(np.arange(rows) % 5 == 0, 1, -1)
    signal = (target == 1).astype(float) * 5 + rng.normal(0, 0.15, rows)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows).astype(str),
            "signal": signal,
            "signal_copy": signal,
            "noise_a": rng.normal(size=rows),
            "noise_b": rng.normal(size=rows),
            "some_missing": rng.normal(size=rows),
            "high_missing": rng.normal(size=rows),
            "constant": 1.0,
            "class": target,
        }
    )
    frame.loc[::7, "some_missing"] = np.nan
    frame.loc[: int(rows * 0.7), "high_missing"] = np.nan
    return frame


def quick_config(**overrides) -> FoldSafeConfig:
    values = {"n_splits": 5, "outer_holdout_size": 0.2, "random_state": 17}
    values.update(overrides)
    return FoldSafeConfig.quick(**values)


@lru_cache(maxsize=1)
def evaluated():
    return evaluate_fold_safe(
        make_frame(),
        config=quick_config(),
        strategies=("all_features", "mi_selected"),
        model_names=("dummy_prior", "logistic_balanced"),
    )


def test_preprocessor_learns_only_from_training_missingness():
    train = make_frame(40).drop(columns=["timestamp", "class"])
    validation = train.iloc[:5].copy()
    validation["noise_a"] = np.nan
    transformer = FoldPreprocessor(missing_threshold=0.4).fit(train)
    before = tuple(transformer.get_feature_names_out())

    transformed = transformer.transform(validation)

    assert tuple(transformer.get_feature_names_out()) == before
    assert "noise_a" in transformed.columns


def test_validation_extremes_do_not_change_correlation_selection():
    train = pd.DataFrame({"z": range(30), "a": range(30), "b": np.arange(30) ** 2})
    y = pd.Series([-1, 1] * 15)
    selector = CorrelationPruner(threshold=0.95, random_state=4).fit(train, y)
    names = tuple(selector.get_feature_names_out())
    validation = train.iloc[:4].copy()
    validation.loc[:, :] = [[999, -999, 0], [-999, 999, 1], [0, 0, 2], [5, 8, 3]]

    selector.transform(validation)

    assert tuple(selector.get_feature_names_out()) == names


def test_transform_before_fit_is_rejected():
    with pytest.raises(NotFittedError):
        FoldPreprocessor().transform(pd.DataFrame({"a": [1.0]}))


def test_extra_validation_columns_are_ignored_and_order_is_fitted_schema():
    train = pd.DataFrame({"b": [1.0, 2.0], "a": [3.0, 4.0]})
    transformer = FoldPreprocessor().fit(train)
    validation = pd.DataFrame({"extra": [9.0], "a": [5.0], "b": [6.0]})

    result = transformer.transform(validation)

    assert result.columns.tolist() == ["b", "a"]


def test_missing_fitted_validation_column_is_rejected():
    transformer = FoldPreprocessor().fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    with pytest.raises(ValueError, match="missing fitted feature"):
        transformer.transform(pd.DataFrame({"a": [5.0]}))


@pytest.mark.parametrize("bad", [pd.Series([True, False]), pd.Series(["1", "2"])])
def test_bool_and_string_features_are_rejected(bad):
    with pytest.raises(TypeError, match="numeric and non-boolean"):
        FoldPreprocessor().fit(pd.DataFrame({"bad": bad}))


def test_all_features_removed_is_clear_error():
    with pytest.raises(ValueError, match="All features were removed"):
        FoldPreprocessor().fit(pd.DataFrame({"constant": [1.0, 1.0], "empty": [np.nan, np.nan]}))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pipeline_returns_matching_feature_names_and_counts(strategy):
    frame = make_frame(80)
    X = frame.drop(columns=["timestamp", "class"])
    y = frame["class"]
    pipeline = build_feature_pipeline(strategy, quick_config())
    transformed = pipeline.fit_transform(X, y)

    assert transformed.columns.tolist() == list(pipeline.steps[-1][1].get_feature_names_out())
    assert transformed.shape[1] > 0


def test_each_fold_fits_fresh_mi_and_model_selectors(monkeypatch):
    counts = {"mi": 0, "model": 0}
    original_mi = MutualInformationSelector.fit
    original_model = ModelBasedSelector.fit

    def count_mi(self, X, y):
        counts["mi"] += 1
        return original_mi(self, X, y)

    def count_model(self, X, y):
        counts["model"] += 1
        return original_model(self, X, y)

    monkeypatch.setattr(MutualInformationSelector, "fit", count_mi)
    monkeypatch.setattr(ModelBasedSelector, "fit", count_model)
    evaluate_fold_safe(
        make_frame(80),
        config=quick_config(),
        strategies=("mi_selected", "model_selected"),
        model_names=("dummy_prior",),
    )

    assert counts == {"mi": 5, "model": 5}


def test_selectors_receive_training_fold_rows_and_targets_only(monkeypatch):
    fitted_indices = []
    original_fit = MutualInformationSelector.fit

    def record_fit(self, X, y):
        fitted_indices.append((tuple(X.index), tuple(y.index)))
        return original_fit(self, X, y)

    monkeypatch.setattr(MutualInformationSelector, "fit", record_fit)
    payload, _ = evaluate_fold_safe(
        make_frame(80),
        config=quick_config(),
        strategies=("mi_selected",),
        model_names=("dummy_prior",),
    )
    expected = [tuple(fold["train_indices"]) for fold in payload["cv"]["folds"]]

    assert len(fitted_indices) == 5
    assert [indices for indices, _ in fitted_indices] == expected
    assert all(feature_indices == target_indices for feature_indices, target_indices in fitted_indices)


def test_explicit_identifier_and_derived_columns_are_excluded():
    frame = make_frame(80)
    frame["wafer_id"] = np.arange(len(frame))
    frame["derived_result"] = (frame["class"] == 1).astype(int)

    payload, _ = evaluate_fold_safe(
        frame,
        excluded_columns=("wafer_id", "derived_result"),
        config=quick_config(),
        strategies=("all_features",),
        model_names=("dummy_prior",),
    )

    assert payload["dataset"]["excluded_columns"] == ["wafer_id", "derived_result"]
    assert all(
        name not in {"wafer_id", "derived_result"}
        for result in payload["results"]
        for fold in result["folds"]
        for name in fold["selected_features"]
    )


def test_unknown_explicit_exclusion_is_rejected():
    with pytest.raises(ValueError, match="Excluded columns were not found"):
        evaluate_fold_safe(
            make_frame(80),
            excluded_columns=("misspelled_identifier",),
            config=quick_config(),
        )


def test_input_frame_is_not_mutated():
    frame = make_frame(80)
    before = frame.copy(deep=True)
    evaluate_fold_safe(
        frame,
        config=quick_config(),
        strategies=("all_features",),
        model_names=("dummy_prior",),
    )
    pd.testing.assert_frame_equal(frame, before)


def test_same_random_state_is_reproducible():
    options = {
        "config": quick_config(),
        "strategies": ("all_features",),
        "model_names": ("logistic_balanced",),
    }
    first_payload, first_csv = evaluate_fold_safe(make_frame(80), **options)
    second_payload, second_csv = evaluate_fold_safe(make_frame(80), **options)

    assert first_payload == second_payload
    pd.testing.assert_frame_equal(first_csv, second_csv)


@pytest.mark.parametrize(
    ("strategies", "model_names", "message"),
    [
        (("unknown",), ("dummy_prior",), "Invalid feature strategies"),
        (("all_features",), ("unknown",), "Invalid models"),
        ((), ("dummy_prior",), "Invalid feature strategies"),
        (("all_features",), (), "Invalid models"),
    ],
)
def test_invalid_or_empty_combination_subset_is_rejected(strategies, model_names, message):
    with pytest.raises(ValueError, match=message):
        evaluate_fold_safe(
            make_frame(40),
            config=quick_config(),
            strategies=strategies,
            model_names=model_names,
        )


def test_requested_combination_order_is_deterministic():
    payload, _ = evaluate_fold_safe(
        make_frame(80),
        config=quick_config(),
        strategies=("model_selected", "all_features"),
        model_names=("logistic_balanced", "dummy_prior"),
    )
    combinations = [(result["strategy"], result["model"]) for result in payload["results"]]
    assert combinations == [
        ("model_selected", "logistic_balanced"),
        ("model_selected", "dummy_prior"),
        ("all_features", "logistic_balanced"),
        ("all_features", "dummy_prior"),
    ]


def test_all_combinations_share_one_fold_definition():
    payload, csv_frame = evaluated()
    assert len(payload["cv"]["folds"]) == 5
    assert csv_frame.groupby(["strategy", "model"])["fold"].apply(list).apply(lambda folds: folds == [1, 2, 3, 4, 5]).all()


def test_fold_rows_and_aggregates_are_complete_and_exact():
    payload, csv_frame = evaluated()
    assert len(csv_frame) == len(payload["metadata"]["strategies"]) * len(payload["metadata"]["models"]) * 5
    assert tuple(csv_frame.columns) == CSV_COLUMNS
    for result in payload["results"]:
        rows = csv_frame[(csv_frame.strategy == result["strategy"]) & (csv_frame.model == result["model"])]
        for metric in METRICS:
            assert result["metrics"][metric]["mean"] == pytest.approx(rows[metric].mean())
            assert result["metrics"][metric]["std"] == pytest.approx(rows[metric].std(ddof=1))


def test_score_metrics_are_not_computed_from_hard_predictions():
    y_true = pd.Series([-1, -1, 1, 1])
    predictions = np.array([-1, -1, -1, -1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    metrics = _metric_values(y_true, predictions, scores)

    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y_true, scores, pos_label=1))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, scores))
    assert metrics["pr_auc"] != average_precision_score(y_true, predictions, pos_label=1)
    assert metrics["roc_auc"] != roc_auc_score(y_true, predictions)


def test_positive_class_and_outer_holdout_are_explicit():
    payload, _ = evaluated()
    assert payload["metadata"]["positive_class"] == 1
    assert payload["metadata"]["outer_holdout_used"] is True
    assert payload["metadata"]["outer_holdout_evaluated"] is False


def test_strict_json_and_csv_outputs(tmp_path):
    payload, csv_frame = evaluated()
    payload_before = copy.deepcopy(payload)
    csv_before = csv_frame.copy(deep=True)
    json_path, csv_path = write_fold_safe_results(payload, csv_frame, tmp_path)

    parsed = json.loads(
        json_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    parsed_csv = pd.read_csv(csv_path)
    assert parsed["schema_version"] == "1.0"
    assert tuple(parsed_csv.columns) == CSV_COLUMNS
    assert len(parsed_csv) == len(csv_frame)
    assert payload == payload_before
    pd.testing.assert_frame_equal(csv_frame, csv_before)


def test_csv_cli_smoke_test(tmp_path):
    csv_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    make_frame(80).to_csv(csv_path, index=False)
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--input-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--quick",
            "--strategy",
            "all_features",
            "--model",
            "dummy_prior",
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "fold_safe_results.json").is_file()
    assert (output_dir / "fold_safe_fold_metrics.csv").is_file()
    assert "Best mean PR-AUC:" in result.stdout


def test_cli_inputs_are_mutually_exclusive(tmp_path):
    csv_path = tmp_path / "input.csv"
    make_frame(20).to_csv(csv_path, index=False)
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--input-csv", str(csv_path), "--uci-secom"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("case", ["missing_path", "missing_target"])
def test_cli_reports_input_errors_without_traceback(tmp_path, case):
    csv_path = tmp_path / "input.csv"
    if case == "missing_target":
        make_frame(30).drop(columns="class").to_csv(csv_path, index=False)
    else:
        csv_path = tmp_path / "absent.csv"
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--input-csv", str(csv_path), "--quick"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_output_files_stay_directly_under_requested_directory(tmp_path):
    payload, csv_frame = evaluated()
    output_dir = tmp_path / "chosen" / "nested"
    paths = write_fold_safe_results(payload, csv_frame, output_dir)
    assert all(path.parent.resolve() == output_dir.resolve() for path in paths)


def test_default_report_directory_is_git_ignored():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "reports/*" in gitignore
