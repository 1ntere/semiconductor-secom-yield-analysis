import copy
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from src.fold_safe_modeling import FoldSafeConfig, MODELS, STRATEGIES
from src.repeated_cv import (
    FOLD_COLUMNS,
    PAIRWISE_COLUMNS,
    REPEAT_COLUMNS,
    RepeatedCVConfig,
    evaluate_repeated_candidate,
    finalize_parts,
    load_validation_results,
    make_repeated_splits,
    prepare_repeated_data,
    select_candidates,
    split_fingerprint,
    validate_part,
    write_part,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_repeated_cv.py"


def frame(rows=60):
    rng = np.random.default_rng(7)
    y = np.where(np.arange(rows) % 4 == 0, 1, -1)
    signal = (y == 1).astype(float) * 4 + rng.normal(0, 0.2, rows)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=rows).astype(str),
        "signal": signal,
        "copy": signal,
        "noise": rng.normal(size=rows),
        "class": y,
    })


def validation_payload(overrides=None):
    overrides = overrides or {}
    results = []
    for s_index, strategy in enumerate(STRATEGIES):
        for m_index, model in enumerate(MODELS):
            mean = 0.10 + s_index * 0.01 + m_index * 0.02
            std = 0.05
            mean, std = overrides.get((strategy, model), (mean, std))
            results.append({
                "strategy": strategy,
                "model": model,
                "metrics": {"pr_auc": {"mean": mean, "std": std}},
            })
    return {"schema_version": "1.0", "results": results}


def small_configs():
    return RepeatedCVConfig(2, 2, 11), FoldSafeConfig.quick(
        random_state=11, n_splits=2, selector_rf_estimators=3, model_rf_estimators=3,
        mi_null_permutations=1, hist_max_iter=5,
    )


@lru_cache(maxsize=2)
def evaluated(candidate):
    repeated, modeling = small_configs()
    return evaluate_repeated_candidate(
        frame(), candidate, repeated_config=repeated, modeling_config=modeling,
    )


def test_shortlist_selection_is_deterministic_and_unique():
    payload = validation_payload({
        ("all_features", "random_forest_balanced"): (0.30, 0.05),
        ("correlation_pruned", "histgb_balanced"): (0.27, 0.04),
        ("mi_selected", "random_forest_balanced"): (0.26, 0.03),
    })
    first = select_candidates(payload)
    assert first == select_candidates(copy.deepcopy(payload))
    assert len(first) == len({(item["strategy"], item["model"]) for item in first}) == 4
    assert (first[0]["strategy"], first[0]["model"]) == ("all_features", "dummy_prior")
    assert first[1]["model"] == "random_forest_balanced"
    assert first[2]["model"] != first[1]["model"]


def test_shortlist_tie_breaks_by_std_then_canonical_order():
    payload = validation_payload({
        ("all_features", "logistic_balanced"): (0.50, 0.08),
        ("all_features", "random_forest_balanced"): (0.50, 0.03),
        ("correlation_pruned", "random_forest_balanced"): (0.50, 0.03),
    })
    shortlist = select_candidates(payload)
    assert (shortlist[1]["strategy"], shortlist[1]["model"]) == ("all_features", "random_forest_balanced")


@pytest.mark.parametrize(("splits", "repeats"), [(1, 2), (2, 0)])
def test_invalid_repeated_config_is_rejected(splits, repeats):
    with pytest.raises(ValueError):
        RepeatedCVConfig(splits, repeats)


def test_repeat_fold_split_ids_and_fingerprints_are_deterministic():
    config, _ = small_configs()
    X, y, _ = prepare_repeated_data(frame(), config)
    splits = make_repeated_splits(X, y, config)
    assert [(s["repeat_id"], s["fold_id"], s["split_id"]) for s in splits] == [
        (0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)
    ]
    assert len({s["validation_fingerprint"] for s in splits}) == 4
    assert split_fingerprint(splits) == split_fingerprint(make_repeated_splits(X, y, config))


def test_different_random_state_changes_split_fingerprint():
    first, _ = small_configs()
    second = RepeatedCVConfig(2, 2, 12)
    X1, y1, _ = prepare_repeated_data(frame(), first)
    X2, y2, _ = prepare_repeated_data(frame(), second)
    assert split_fingerprint(make_repeated_splits(X1, y1, first)) != split_fingerprint(make_repeated_splits(X2, y2, second))


@pytest.mark.parametrize("candidate", [("all_features", "dummy_prior"), ("correlation_pruned", "logistic_balanced")])
def test_candidate_has_four_shared_splits_and_fold_safe_metadata(candidate):
    payload, rows = evaluated(candidate)
    assert len(rows) == 4
    assert rows["split_id"].tolist() == [0, 1, 2, 3]
    assert payload["cv"]["split_count"] == 4
    assert payload["positive_class"] == 1
    assert payload["dataset"]["outer_holdout_evaluated"] is False
    assert all(detail["feature_count_selected"] == len(detail["selected_features"]) for detail in payload["folds"])


def test_candidates_share_identical_validation_fingerprints():
    _, first = evaluated(("all_features", "dummy_prior"))
    _, second = evaluated(("correlation_pruned", "logistic_balanced"))
    assert first["validation_fingerprint"].tolist() == second["validation_fingerprint"].tolist()


def test_same_random_state_reproduces_metrics_except_elapsed_time():
    repeated, modeling = small_configs()
    first_payload, first = evaluate_repeated_candidate(frame(), ("all_features", "dummy_prior"), repeated_config=repeated, modeling_config=modeling)
    second_payload, second = evaluate_repeated_candidate(frame(), ("all_features", "dummy_prior"), repeated_config=repeated, modeling_config=modeling)
    pd.testing.assert_frame_equal(first.drop(columns="elapsed_seconds"), second.drop(columns="elapsed_seconds"))
    assert first_payload["cv"] == second_payload["cv"]


def test_input_frame_is_not_mutated():
    source = frame()
    before = source.copy(deep=True)
    repeated, modeling = small_configs()
    evaluate_repeated_candidate(source, ("all_features", "dummy_prior"), repeated_config=repeated, modeling_config=modeling)
    pd.testing.assert_frame_equal(source, before)


def test_transformer_is_fresh_and_fit_only_on_each_training_split(monkeypatch):
    fitted = []

    class RecordingTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            fitted.append((self, tuple(X.index)))
            return self

        def transform(self, X):
            return X

        def get_feature_names_out(self, input_features=None):
            return self.feature_names_in_

    monkeypatch.setattr("src.repeated_cv.build_feature_pipeline", lambda strategy, config: RecordingTransformer())
    monkeypatch.setattr("src.repeated_cv._feature_audit", lambda transformer: {
        "feature_count_before": 3,
        "feature_count_after_missing": 3,
        "feature_count_after_constant": 3,
        "feature_count_after_correlation": 3,
        "feature_count_selected": 3,
        "selected_features": ["signal", "copy", "noise"],
    })
    repeated, modeling = small_configs()
    X, y, _ = prepare_repeated_data(frame(), repeated)
    expected = [tuple(split["train_indices"]) for split in make_repeated_splits(X, y, repeated)]
    evaluate_repeated_candidate(
        frame(), ("all_features", "dummy_prior"),
        repeated_config=repeated, modeling_config=modeling,
    )
    assert [indices for _, indices in fitted] == expected
    assert len({id(transformer) for transformer, _ in fitted}) == len(expected)


def write_two_parts(tmp_path):
    candidates = [
        {"strategy": "all_features", "model": "dummy_prior", "reason": "baseline"},
        {"strategy": "correlation_pruned", "model": "logistic_balanced", "reason": "candidate"},
    ]
    parts = tmp_path / "parts"
    for item in candidates:
        key = (item["strategy"], item["model"])
        payload, rows = evaluated(key)
        payload = copy.deepcopy(payload)
        payload["shortlist"] = candidates
        write_part(payload, rows, parts / f"{key[0]}__{key[1]}")
    return candidates, parts


def test_atomic_part_is_strict_json_and_stable_csv(tmp_path):
    payload, rows = evaluated(("all_features", "dummy_prior"))
    json_path, csv_path = write_part(payload, rows, tmp_path)
    parsed, parsed_rows = validate_part(tmp_path, expected_candidate=("all_features", "dummy_prior"))
    assert parsed["status"] == "complete"
    assert tuple(parsed_rows.columns) == FOLD_COLUMNS
    assert not list(tmp_path.glob("*.tmp"))
    json.loads(json_path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert len(pd.read_csv(csv_path)) == 4


def test_incomplete_part_is_rejected(tmp_path):
    (tmp_path / "part_results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_part(tmp_path)


def test_part_with_replaced_split_id_is_rejected(tmp_path):
    payload, rows = evaluated(("all_features", "dummy_prior"))
    changed = rows.copy()
    changed.loc[changed.index[-1], "split_id"] = 99
    write_part(payload, changed, tmp_path)
    with pytest.raises(ValueError, match="split IDs"):
        validate_part(tmp_path)


@pytest.mark.parametrize("mismatch", ["dataset", "split", "candidate"])
def test_part_metadata_mismatch_is_rejected(tmp_path, mismatch):
    payload, rows = evaluated(("all_features", "dummy_prior"))
    write_part(payload, rows, tmp_path)
    kwargs = {
        "expected_candidate": ("all_features", "dummy_prior"),
        "expected_dataset_fingerprint": payload["dataset"]["dataset_fingerprint"],
        "expected_split_fingerprint": payload["cv"]["split_fingerprint"],
    }
    if mismatch == "dataset": kwargs["expected_dataset_fingerprint"] = "wrong"
    if mismatch == "split": kwargs["expected_split_fingerprint"] = "wrong"
    if mismatch == "candidate": kwargs["expected_candidate"] = ("all_features", "logistic_balanced")
    with pytest.raises(ValueError, match="mismatch"):
        validate_part(tmp_path, **kwargs)


def test_finalize_aggregates_splits_repeats_pairs_and_ranks(tmp_path):
    candidates, parts = write_two_parts(tmp_path)
    result, folds, repeats, pairs = finalize_parts(parts, candidates, tmp_path / "final")
    assert len(folds) == 8
    assert len(repeats) == 4 and tuple(repeats.columns) == REPEAT_COLUMNS
    assert len(pairs) == 4 and tuple(pairs.columns) == PAIRWISE_COLUMNS
    assert result["rank_ties"] == "average rank"
    assert len(result["rank_stability"]) == 2
    assert result["paired_comparison"][0]["win_count"] + result["paired_comparison"][0]["tie_count"] + result["paired_comparison"][0]["loss_count"] == 4


def test_split_and_repeat_summaries_use_ddof_one(tmp_path):
    candidates, parts = write_two_parts(tmp_path)
    result, folds, repeats, _ = finalize_parts(parts, candidates, tmp_path / "final")
    key = "all_features__dummy_prior"
    subset = folds[(folds.strategy == "all_features") & (folds.model == "dummy_prior")]
    repeat_subset = repeats[(repeats.strategy == "all_features") & (repeats.model == "dummy_prior")]
    assert result["split_summary"][key]["pr_auc"]["std"] == pytest.approx(subset.pr_auc.std(ddof=1))
    assert result["repeat_summary"][key]["pr_auc"]["std"] == pytest.approx(repeat_subset.pr_auc.std(ddof=1))
    assert result["split_summary"][key]["pr_auc"]["iqr"] == pytest.approx(subset.pr_auc.quantile(.75)-subset.pr_auc.quantile(.25))


def test_finalize_rejects_duplicate_shortlist(tmp_path):
    item = {"strategy": "all_features", "model": "dummy_prior"}
    with pytest.raises(ValueError, match="duplicate"):
        finalize_parts(tmp_path, [item, item], tmp_path / "final")


def test_finalize_rejects_missing_part(tmp_path):
    with pytest.raises(ValueError, match="Incomplete"):
        finalize_parts(tmp_path, [{"strategy": "all_features", "model": "dummy_prior"}], tmp_path / "final")


def test_finalize_rejects_parts_with_different_modeling_config(tmp_path):
    candidates, parts = write_two_parts(tmp_path)
    directory = parts / "correlation_pruned__logistic_balanced"
    payload, rows = validate_part(directory)
    payload["modeling_config"]["mi_null_permutations"] += 1
    write_part(payload, rows, directory)
    with pytest.raises(ValueError, match="different modeling configurations"):
        finalize_parts(parts, candidates, tmp_path / "final")


def test_load_validation_rejects_noncanonical_results(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version":"1.0","results":[]}), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical 16"):
        load_validation_results(path)


def test_cli_smoke_resume_and_partial_status(tmp_path):
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps(validation_payload()), encoding="utf-8")
    csv = tmp_path / "input.csv"
    frame().to_csv(csv, index=False)
    output = tmp_path / "output"
    command = [sys.executable, str(CLI), "--from-validation-results", str(validation), "--input-csv", str(csv), "--candidate", "all_features:dummy_prior", "--output-dir", str(output), "--n-splits", "2", "--n-repeats", "2"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    second = subprocess.run([*command, "--resume"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0 and "Partial execution only" in first.stdout
    assert second.returncode == 0 and "Reused complete part" in second.stdout


def test_cli_rejects_unsafe_candidate_and_bad_path(tmp_path):
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps(validation_payload()), encoding="utf-8")
    result = subprocess.run([sys.executable, str(CLI), "--from-validation-results", str(validation), "--candidate", "../escape:dummy_prior"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "error:" in result.stderr and "Traceback" not in result.stderr
    missing = subprocess.run([sys.executable, str(CLI), "--from-validation-results", str(tmp_path / "missing.json"), "--finalize"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert missing.returncode != 0 and "error:" in missing.stderr and "Traceback" not in missing.stderr


def test_cli_rejects_output_directory_that_contains_project(tmp_path):
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps(validation_payload()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), "--from-validation-results", str(validation),
         "--finalize", "--output-dir", str(ROOT.parent)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "project root or one of its parents" in result.stderr
    assert "Traceback" not in result.stderr
