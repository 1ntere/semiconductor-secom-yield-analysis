from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.cost_threshold import (
    CSV_COLUMNS, CostScenario, CostThresholdConfig, confusion_and_cost,
    evaluate_cost_thresholds, select_cost_threshold, threshold_candidates,
    validate_probabilities, write_cost_threshold_results,
)
from src.fold_safe_modeling import FoldSafeConfig


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_cost_threshold.py"


def frame(rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    y = np.where(np.arange(rows) % 5 == 0, 1, -1)
    return pd.DataFrame({
        "signal": y + rng.normal(0, 1.2, rows), "noise": rng.normal(size=rows),
        "sometimes_missing": np.where(np.arange(rows) % 9 == 0, np.nan, rng.normal(size=rows)),
        "timestamp": pd.date_range("2025-01-01", periods=rows).astype(str), "class": y,
    })


def evaluated(random_state: int = 7):
    return evaluate_cost_thresholds(
        frame(), config=CostThresholdConfig(random_state=random_state, outer_splits=3, inner_splits=2),
        modeling_config=FoldSafeConfig.quick(random_state=random_state, n_splits=3),
    )


def test_cost_direction_and_confusion():
    metrics = confusion_and_cost([1, 1, -1, -1], [1, -1, 1, -1], 2, 7)
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (1, 1, 1, 1)
    assert metrics["total_cost"] == 9 and metrics["cost_units_per_1000_samples"] == 2250


def test_threshold_boundaries_duplicates_and_tie_rule():
    candidates = threshold_candidates([0, 0.2, 0.2, 1])
    assert 0 in candidates and 1 in candidates and candidates[-1] > 1
    chosen = select_cost_threshold([1, -1], [0.5, 0.5], 1, 1)
    assert chosen["threshold"] > 1
    assert chosen["candidate_count"] == 4


@pytest.mark.parametrize("value", [[np.nan], [np.inf], [-0.1], [1.1], []])
def test_invalid_probabilities_rejected(value):
    with pytest.raises(ValueError):
        validate_probabilities(value)


@pytest.mark.parametrize("fp,fn", [(0, 1), (-1, 1), (1, np.inf), (1, np.nan), (True, 1)])
def test_invalid_costs_rejected(fp, fn):
    with pytest.raises(ValueError):
        CostScenario(fp, fn)


def test_nested_results_baselines_scenarios_aggregates_and_holdout():
    payload, rows = evaluated()
    assert len(rows) == 3 * 3 * 4 and tuple(rows.columns) == CSV_COLUMNS
    assert set(rows.policy) == {"cost_selected", "threshold_0_5", "all_normal", "all_fail"}
    assert payload["metadata"]["outer_holdout_evaluated"] is False
    assert len(payload["split"]["outer_holdout_positions"]) == payload["dataset"]["outer_holdout_size"]
    assert set(payload["split"]["outer_holdout_positions"]).isdisjoint(
        payload["split"]["outer_train_positions"]
    )
    assert (rows.tp + rows.fp + rows.tn + rows.fn == rows.validation_size).all()
    normal, fail = rows[rows.policy == "all_normal"], rows[rows.policy == "all_fail"]
    assert (normal.fp == 0).all() and (normal.tp == 0).all()
    assert (fail.fn == 0).all() and (fail.tn == 0).all()
    assert len(payload["aggregate"]) == 12


def test_threshold_selection_receives_no_outer_validation_labels(monkeypatch):
    import src.cost_threshold as module
    original, seen_lengths = module.select_cost_threshold, []
    def spy(y, probabilities, cost_fp, cost_fn):
        seen_lengths.append(len(y))
        return original(y, probabilities, cost_fp, cost_fn)
    monkeypatch.setattr(module, "select_cost_threshold", spy)
    payload, rows = evaluated()
    assert all(length not in set(rows.validation_size) for length in seen_lengths)
    assert seen_lengths.count(payload["dataset"]["outer_train_size"]) == 3


def test_inner_outer_split_disjointness_and_oof_coverage():
    payload, _ = evaluated()
    for outer in payload["outer_folds"]:
        assert set(outer["validation_indices"]).isdisjoint(outer["train_indices"])
        covered = []
        for inner in outer["inner_splits"]:
            assert set(inner["train_indices"]).isdisjoint(inner["validation_indices"])
            covered.extend(inner["validation_indices"])
        assert sorted(covered) == list(range(len(outer["train_indices"])))


def test_reproducible_and_input_unchanged():
    source, before = frame(), frame()
    first_payload, first_rows = evaluated(11)
    second_payload, second_rows = evaluated(11)
    assert first_payload["split_fingerprint"] == second_payload["split_fingerprint"]
    pd.testing.assert_frame_equal(first_rows, second_rows)
    pd.testing.assert_frame_equal(source, before)


def test_strict_json_fixed_csv_and_writer_inputs_unchanged(tmp_path):
    payload, rows = evaluated()
    payload_before, rows_before = copy.deepcopy(payload), rows.copy(deep=True)
    json_path, csv_path = write_cost_threshold_results(payload, rows, tmp_path)
    parsed = json.loads(json_path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert parsed["status"] == "complete" and tuple(pd.read_csv(csv_path).columns) == CSV_COLUMNS
    assert payload == payload_before
    pd.testing.assert_frame_equal(rows, rows_before)


def test_synthetic_csv_cli_smoke_and_help(tmp_path):
    input_path, output = tmp_path / "input.csv", tmp_path / "output"
    frame(60).to_csv(input_path, index=False)
    command = [sys.executable, str(CLI), "--input-csv", str(input_path), "--output-dir", str(output),
               "--quick", "--outer-splits", "2", "--inner-splits", "2", "--cost-scenario", "7:2"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    help_result = subprocess.run([sys.executable, str(CLI), "--help"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert help_result.returncode == 0 and "FN:FP" in help_result.stdout
    assert (output / "cost_threshold_results.json").is_file()
    assert (output / "cost_threshold_fold_policies.csv").is_file()


def test_cli_rejects_bad_cost_path_and_target(tmp_path):
    missing = subprocess.run([sys.executable, str(CLI), "--input-csv", str(tmp_path / "none.csv")], cwd=ROOT, capture_output=True, text=True)
    bad_cost = subprocess.run([sys.executable, str(CLI), "--input-csv", "x", "--cost-scenario", "0:1"], cwd=ROOT, capture_output=True, text=True)
    csv = tmp_path / "bad.csv"
    frame(30).drop(columns="class").to_csv(csv, index=False)
    bad_target = subprocess.run([sys.executable, str(CLI), "--input-csv", str(csv), "--quick", "--outer-splits", "2", "--inner-splits", "2"], cwd=ROOT, capture_output=True, text=True)
    assert all(item.returncode != 0 and "Traceback" not in item.stderr for item in (missing, bad_cost, bad_target))
