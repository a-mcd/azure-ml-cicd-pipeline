"""Unit tests for the champion-challenger comparison script."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.errors import MergeError

from src.pipeline_scripts import select_winner as selection


@pytest.fixture
def challenger_csv(tmp_path: Path) -> Path:
    """Create a valid challenger predictions CSV."""
    path = tmp_path / "challenger.csv"

    pd.DataFrame(
        {
            "Store": [1, 2],
            "Date": ["02-11-2012", "02-11-2012"],
            "Next_Week_Date": ["09-11-2012", "09-11-2012"],
            "Weekly_Sales": [90.0, 190.0],
            "Actual_Next_Week_Sales": [100.0, 200.0],
            "Predicted_Weekly_Sales": [105.0, 190.0],
        }
    ).to_csv(path, index=False)

    return path


@pytest.fixture
def champion_json(tmp_path: Path) -> Path:
    """Create a valid champion predictions JSON file."""
    path = tmp_path / "champion.json"

    response = {
        "predictions": [
            {
                "Store": 1,
                "Feature_Date": "02-11-2012",
                "Prediction_Date": "09-11-2012",
                "Predicted_Weekly_Sales": 120.0,
            },
            {
                "Store": 2,
                "Feature_Date": "02-11-2012",
                "Prediction_Date": "09-11-2012",
                "Predicted_Weekly_Sales": 180.0,
            },
        ]
    }

    path.write_text(json.dumps(response), encoding="utf-8")
    return path


@pytest.fixture
def challenger_dataframe() -> pd.DataFrame:
    """Create challenger predictions ready for merging."""
    return pd.DataFrame(
        {
            "Store": [1, 2],
            "Feature_Date": pd.to_datetime(
                ["02-11-2012", "02-11-2012"]
            ),
            "Prediction_Date": pd.to_datetime(
                ["09-11-2012", "09-11-2012"]
            ),
            "Actual_Next_Week_Sales": [100.0, 200.0],
            "Challenger_Predicted_Weekly_Sales": [105.0, 190.0],
        }
    )


@pytest.fixture
def champion_dataframe() -> pd.DataFrame:
    """Create champion predictions ready for merging."""
    return pd.DataFrame(
        {
            "Store": [1, 2],
            "Feature_Date": pd.to_datetime(
                ["02-11-2012", "02-11-2012"]
            ),
            "Prediction_Date": pd.to_datetime(
                ["09-11-2012", "09-11-2012"]
            ),
            "Champion_Predicted_Weekly_Sales": [120.0, 180.0],
        }
    )


@pytest.fixture
def tied_metrics() -> dict[str, float]:
    """Return a set of model metrics."""
    return {
        "wape": 10.0,
        "rmse": 100.0,
        "mape": 5.0,
        "mae": 50.0,
    }


def test_parse_args_returns_expected_paths(mocker, tmp_path):
    challenger = tmp_path / "challenger.csv"
    champion = tmp_path / "champion.json"
    github_output = tmp_path / "github-output.txt"

    mocker.patch.object(
        sys,
        "argv",
        [
            "select_winner.py",
            "--challenger-predictions",
            str(challenger),
            "--champion-predictions",
            str(champion),
            "--github-output",
            str(github_output),
        ],
    )

    arguments = selection.parse_args()

    assert isinstance(arguments, argparse.Namespace)
    assert arguments.challenger_predictions == challenger
    assert arguments.champion_predictions == champion
    assert arguments.github_output == github_output


def test_parse_args_rejects_missing_required_arguments(mocker):
    mocker.patch.object(sys, "argv", ["select_winner.py"])

    with pytest.raises(SystemExit) as error:
        selection.parse_args()

    assert error.value.code == 2


def test_validate_file_accepts_existing_non_empty_file(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("data", encoding="utf-8")

    selection.validate_file(path, "Predictions")


def test_validate_file_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Predictions was not found",
    ):
        selection.validate_file(path, "Predictions")


def test_validate_file_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.touch()

    with pytest.raises(
        ValueError,
        match="Predictions is empty",
    ):
        selection.validate_file(path, "Predictions")


def test_load_challenger_predictions_transforms_columns(
    challenger_csv,
):
    dataframe = selection.load_challenger_predictions(challenger_csv)

    assert "Weekly_Sales" not in dataframe.columns
    assert "Date" not in dataframe.columns
    assert "Next_Week_Date" not in dataframe.columns
    assert "Predicted_Weekly_Sales" not in dataframe.columns

    assert "Feature_Date" in dataframe.columns
    assert "Prediction_Date" in dataframe.columns
    assert "Challenger_Predicted_Weekly_Sales" in dataframe.columns
    assert "Actual_Next_Week_Sales" in dataframe.columns

    assert dataframe["Store"].tolist() == [1, 2]
    assert pd.api.types.is_integer_dtype(dataframe["Store"])
    assert pd.api.types.is_datetime64_any_dtype(
        dataframe["Feature_Date"]
    )
    assert pd.api.types.is_datetime64_any_dtype(
        dataframe["Prediction_Date"]
    )


def test_load_challenger_predictions_rejects_missing_columns(
    tmp_path,
):
    path = tmp_path / "challenger.csv"

    pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2012-11-02"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Challenger CSV is missing columns",
    ):
        selection.load_challenger_predictions(path)


def test_load_challenger_predictions_rejects_invalid_store(
    challenger_csv,
):
    dataframe = pd.read_csv(challenger_csv)
    dataframe["Store"] = dataframe["Store"].astype(object)
    dataframe.loc[0, "Store"] = "invalid"
    dataframe.to_csv(challenger_csv, index=False)

    with pytest.raises(ValueError):
        selection.load_challenger_predictions(challenger_csv)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("Date", "invalid-date"),
        ("Next_Week_Date", "invalid-date"),
    ],
)
def test_load_challenger_predictions_rejects_invalid_dates(
    challenger_csv,
    column,
    invalid_value,
):
    dataframe = pd.read_csv(challenger_csv)
    dataframe.loc[0, column] = invalid_value
    dataframe.to_csv(challenger_csv, index=False)

    with pytest.raises(ValueError):
        selection.load_challenger_predictions(challenger_csv)


def test_load_champion_predictions_transforms_columns(
    champion_json,
):
    dataframe = selection.load_champion_predictions(champion_json)

    assert dataframe.columns.tolist() == [
        "Store",
        "Feature_Date",
        "Prediction_Date",
        "Champion_Predicted_Weekly_Sales",
    ]
    assert dataframe["Store"].tolist() == [1, 2]
    assert pd.api.types.is_integer_dtype(dataframe["Store"])
    assert pd.api.types.is_datetime64_any_dtype(
        dataframe["Feature_Date"]
    )
    assert pd.api.types.is_datetime64_any_dtype(
        dataframe["Prediction_Date"]
    )
    assert dataframe[
        "Champion_Predicted_Weekly_Sales"
    ].tolist() == [120.0, 180.0]


@pytest.mark.parametrize(
    "response",
    [
        [],
        "invalid",
        None,
    ],
)
def test_load_champion_predictions_requires_json_object(
    tmp_path,
    response,
):
    path = tmp_path / "champion.json"
    path.write_text(json.dumps(response), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Champion response must be a JSON object",
    ):
        selection.load_champion_predictions(path)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"predictions": None},
        {"predictions": {}},
        {"predictions": "invalid"},
    ],
)
def test_load_champion_predictions_requires_predictions_list(
    tmp_path,
    response,
):
    path = tmp_path / "champion.json"
    path.write_text(json.dumps(response), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="must contain a 'predictions' list",
    ):
        selection.load_champion_predictions(path)


def test_load_champion_predictions_rejects_missing_columns(
    tmp_path,
):
    path = tmp_path / "champion.json"
    response = {
        "predictions": [
            {
                "Store": 1,
                "Feature_Date": "02-11-2012",
            }
        ]
    }
    path.write_text(json.dumps(response), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Champion predictions are missing columns",
    ):
        selection.load_champion_predictions(path)


def test_load_champion_predictions_rejects_invalid_store(
    tmp_path,
):
    path = tmp_path / "champion.json"
    response = {
        "predictions": [
            {
                "Store": "invalid",
                "Feature_Date": "02-11-2012",
                "Prediction_Date": "09-11-2012",
                "Predicted_Weekly_Sales": 100.0,
            }
        ]
    }
    path.write_text(json.dumps(response), encoding="utf-8")

    with pytest.raises(ValueError):
        selection.load_champion_predictions(path)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("Feature_Date", "invalid-date"),
        ("Prediction_Date", "invalid-date"),
    ],
)
def test_load_champion_predictions_rejects_invalid_dates(
    tmp_path,
    column,
    invalid_value,
):
    path = tmp_path / "champion.json"
    prediction = {
        "Store": 1,
        "Feature_Date": "02-11-2012",
        "Prediction_Date": "09-11-2012",
        "Predicted_Weekly_Sales": 100.0,
    }
    prediction[column] = invalid_value

    path.write_text(
        json.dumps({"predictions": [prediction]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        selection.load_champion_predictions(path)


def test_merge_predictions_returns_matching_rows(
    challenger_dataframe,
    champion_dataframe,
):
    comparison = selection.merge_predictions(
        challenger_dataframe,
        champion_dataframe,
    )

    assert len(comparison) == 2
    assert comparison[
        "Challenger_Predicted_Weekly_Sales"
    ].tolist() == [105.0, 190.0]
    assert comparison[
        "Champion_Predicted_Weekly_Sales"
    ].tolist() == [120.0, 180.0]


def test_merge_predictions_rejects_unmatched_challenger_rows(
    challenger_dataframe,
    champion_dataframe,
):
    champion_dataframe = champion_dataframe.iloc[[0]].copy()

    with pytest.raises(
        ValueError,
        match="Not all challenger rows matched champion rows",
    ):
        selection.merge_predictions(
            challenger_dataframe,
            champion_dataframe,
        )


def test_merge_predictions_rejects_unmatched_champion_rows(
    challenger_dataframe,
    champion_dataframe,
):
    extra_row = champion_dataframe.iloc[[0]].copy()
    extra_row["Store"] = 3

    champion_dataframe = pd.concat(
        [champion_dataframe, extra_row],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Not all champion rows matched challenger rows",
    ):
        selection.merge_predictions(
            challenger_dataframe,
            champion_dataframe,
        )


def test_merge_predictions_rejects_duplicate_keys(
    challenger_dataframe,
    champion_dataframe,
):
    duplicate = challenger_dataframe.iloc[[0]].copy()
    challenger_dataframe = pd.concat(
        [challenger_dataframe, duplicate],
        ignore_index=True,
    )

    with pytest.raises(MergeError):
        selection.merge_predictions(
            challenger_dataframe,
            champion_dataframe,
        )


def test_calculate_metrics_returns_expected_values():
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = pd.Series([110.0, 190.0, 330.0])

    metrics = selection.calculate_metrics(actual, predicted)

    assert metrics["wape"] == pytest.approx(8.3333333333)
    assert metrics["rmse"] == pytest.approx(19.1485421551)
    assert metrics["mae"] == pytest.approx(16.6666666667)
    assert metrics["mape"] == pytest.approx(8.3333333333)


def test_calculate_metrics_converts_numeric_strings():
    actual = pd.Series(["100", "200"])
    predicted = pd.Series(["110", "190"])

    metrics = selection.calculate_metrics(actual, predicted)

    assert metrics["wape"] == pytest.approx(6.6666666667)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["mape"] == pytest.approx(7.5)


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        (
            pd.Series([100.0, np.nan]),
            pd.Series([100.0, 200.0]),
        ),
        (
            pd.Series([100.0, 200.0]),
            pd.Series([100.0, np.nan]),
        ),
    ],
)
def test_calculate_metrics_rejects_missing_values(
    actual,
    predicted,
):
    with pytest.raises(
        ValueError,
        match="Actual or predicted sales contain missing values",
    ):
        selection.calculate_metrics(actual, predicted)


def test_calculate_metrics_rejects_non_numeric_actual_values():
    actual = pd.Series(["invalid", "200"])
    predicted = pd.Series([100.0, 200.0])

    with pytest.raises(ValueError):
        selection.calculate_metrics(actual, predicted)


def test_calculate_metrics_rejects_non_numeric_predictions():
    actual = pd.Series([100.0, 200.0])
    predicted = pd.Series(["invalid", "200"])

    with pytest.raises(ValueError):
        selection.calculate_metrics(actual, predicted)


def test_calculate_metrics_rejects_zero_actual_total():
    actual = pd.Series([0.0, 0.0])
    predicted = pd.Series([10.0, 20.0])

    with pytest.raises(
        ValueError,
        match="total actual sales are zero",
    ):
        selection.calculate_metrics(actual, predicted)


def test_calculate_metrics_rejects_zero_actual_value():
    actual = pd.Series([0.0, 100.0])
    predicted = pd.Series([10.0, 90.0])

    with pytest.raises(
        ValueError,
        match="actual sales contain zero values",
    ):
        selection.calculate_metrics(actual, predicted)


@pytest.mark.parametrize(
    (
        "challenger_changes",
        "champion_changes",
        "expected_winner",
        "expected_reason",
    ),
    [
        (
            {"wape": 9.98},
            {},
            "challenger",
            "lower WAPE",
        ),
        (
            {"wape": 10.02},
            {},
            "champion",
            "lower WAPE",
        ),
        (
            {"rmse": 98.0},
            {},
            "challenger",
            "lower RMSE after WAPE tie",
        ),
        (
            {"rmse": 102.0},
            {},
            "champion",
            "lower RMSE after WAPE tie",
        ),
        (
            {"mape": 4.98},
            {},
            "challenger",
            "lower MAPE after WAPE and RMSE tie",
        ),
        (
            {"mape": 5.02},
            {},
            "champion",
            "lower MAPE after WAPE and RMSE tie",
        ),
        (
            {"mae": 48.0},
            {},
            "challenger",
            "lower MAE after other metrics tied",
        ),
        (
            {"mae": 52.0},
            {},
            "champion",
            "lower MAE after other metrics tied",
        ),
    ],
)
def test_select_winner_uses_metric_priority(
    tied_metrics,
    challenger_changes,
    champion_changes,
    expected_winner,
    expected_reason,
):
    challenger = tied_metrics | challenger_changes
    champion = tied_metrics | champion_changes

    winner, reason = selection.select_winner(
        challenger,
        champion,
    )

    assert winner == expected_winner
    assert reason == expected_reason


def test_select_winner_defaults_to_champion_when_metrics_tied(
    tied_metrics,
):
    winner, reason = selection.select_winner(
        tied_metrics.copy(),
        tied_metrics.copy(),
    )

    assert winner == "champion"
    assert reason == "metrics effectively tied"


def test_select_winner_treats_tolerance_boundary_as_tie(
    tied_metrics,
):
    challenger = tied_metrics.copy()
    champion = tied_metrics.copy()

    challenger["wape"] = 9.99
    champion["wape"] = 10.0

    winner, reason = selection.select_winner(
        challenger,
        champion,
    )

    assert winner == "champion"
    assert reason == "metrics effectively tied"


def test_print_metrics_prints_formatted_values(capsys):
    metrics = {
        "wape": 5.123456,
        "rmse": 123.456,
        "mae": 100.123,
        "mape": 4.987654,
    }

    selection.print_metrics("Challenger", metrics)

    output = capsys.readouterr().out

    assert "Challenger WAPE: 5.1235%" in output
    assert "Challenger RMSE: 123.46" in output
    assert "Challenger MAE:  100.12" in output
    assert "Challenger MAPE: 4.9877%" in output


def test_write_summary_file_creates_parent_and_summary(
    tmp_path,
    capsys,
):
    path = tmp_path / "reports" / "summary.txt"
    challenger = {
        "wape": 5.0,
        "rmse": 100.0,
        "mae": 80.0,
        "mape": 4.0,
    }
    champion = {
        "wape": 6.0,
        "rmse": 110.0,
        "mae": 90.0,
        "mape": 5.0,
    }

    selection.write_summary_file(
        path,
        "challenger",
        "lower WAPE",
        challenger,
        champion,
    )

    summary = path.read_text(encoding="utf-8")

    assert "# Champion–Challenger Comparison" in summary
    assert "**Winner:** `challenger`" in summary
    assert "**Reason:** lower WAPE" in summary
    assert "| WAPE | 5.0000% | 6.0000% |" in summary
    assert "| RMSE | 100.00 | 110.00 |" in summary
    assert "| MAE | 80.00 | 90.00 |" in summary
    assert "| MAPE | 4.0000% | 5.0000% |" in summary
    assert f"Summary written to: {path}" in capsys.readouterr().out


def test_write_github_output_creates_output_file(tmp_path):
    path = tmp_path / "github-output.txt"

    selection.write_github_output(path, "challenger")

    assert path.read_text(encoding="utf-8") == "winner=challenger\n"


def test_write_github_output_appends_to_existing_file(tmp_path):
    path = tmp_path / "github-output.txt"
    path.write_text("existing=value\n", encoding="utf-8")

    selection.write_github_output(path, "champion")

    assert path.read_text(encoding="utf-8") == (
        "existing=value\n"
        "winner=champion\n"
    )


def test_main_runs_complete_comparison(
    mocker,
    monkeypatch,
    tmp_path,
    challenger_csv,
    champion_json,
    capsys,
):
    github_output = tmp_path / "github-output.txt"

    monkeypatch.chdir(tmp_path)

    mocker.patch.object(
        sys,
        "argv",
        [
            "select_winner.py",
            "--challenger-predictions",
            str(challenger_csv),
            "--champion-predictions",
            str(champion_json),
            "--github-output",
            str(github_output),
        ],
    )

    selection.main()

    comparison_path = (
        tmp_path / "challenger_champion_comparison.csv"
    )
    summary_path = (
        tmp_path / "challenger_champion_summary.txt"
    )

    assert comparison_path.is_file()
    assert summary_path.is_file()
    assert github_output.is_file()

    comparison = pd.read_csv(comparison_path)

    assert len(comparison) == 2
    assert "Challenger_Predicted_Weekly_Sales" in comparison
    assert "Champion_Predicted_Weekly_Sales" in comparison

    summary = summary_path.read_text(encoding="utf-8")
    github_contents = github_output.read_text(encoding="utf-8")
    output = capsys.readouterr().out

    assert "**Winner:** `challenger`" in summary
    assert "winner=challenger\n" == github_contents
    assert "Challenger WAPE:" in output
    assert "Champion WAPE:" in output
    assert "Winner: challenger" in output
    assert "Reason: lower WAPE" in output