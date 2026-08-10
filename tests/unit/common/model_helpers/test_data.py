from pathlib import Path

import pandas as pd
import pytest

from src.common.model_helpers.data import (
    chronological_split,
    get_outputs_dir,
    load_dataset,
)


def test_get_outputs_dir_uses_local_outputs_when_not_in_azure(monkeypatch, tmp_path):
    # Purpose: Check that get_outputs_dir uses a local outputs directory
    # when the code is not running inside Azure ML.
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    monkeypatch.delenv("AZUREML_OUTPUTS_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    outputs_dir = get_outputs_dir()

    assert outputs_dir == Path("outputs")
    assert outputs_dir.exists()
    assert outputs_dir.is_dir()


def test_get_outputs_dir_uses_azure_outputs_path_when_set(monkeypatch, tmp_path):
    # Purpose: Check that get_outputs_dir uses AZUREML_OUTPUTS_PATH
    # when Azure provides an explicit outputs directory.
    azure_outputs_path = tmp_path / "azure_outputs"

    monkeypatch.setenv("AZUREML_OUTPUTS_PATH", str(azure_outputs_path))
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)

    outputs_dir = get_outputs_dir()

    assert outputs_dir == azure_outputs_path
    assert outputs_dir.exists()
    assert outputs_dir.is_dir()


def test_get_outputs_dir_uses_outputs_when_azure_run_id_set_but_no_outputs_path(
    monkeypatch,
    tmp_path,
):
    # Purpose: Check that get_outputs_dir falls back to a local outputs directory
    # when AZUREML_RUN_ID is set but AZUREML_OUTPUTS_PATH is missing.
    monkeypatch.setenv("AZUREML_RUN_ID", "test-run-id")
    monkeypatch.delenv("AZUREML_OUTPUTS_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    outputs_dir = get_outputs_dir()

    assert outputs_dir == Path("outputs")
    assert outputs_dir.exists()
    assert outputs_dir.is_dir()


def test_load_dataset_loads_csv_and_parses_dates_dayfirst(tmp_path, capsys):
    # Purpose: Check that load_dataset reads a CSV file and parses Date values
    # using day-first date format.
    csv_path = tmp_path / "walmart.csv"

    df = pd.DataFrame({
        "Store": [1, 2],
        "Date": ["01/02/2024", "15/03/2024"],
        "Weekly_Sales": [1000.0, 2000.0],
    })
    df.to_csv(csv_path, index=False)

    result = load_dataset(str(csv_path))

    assert len(result) == 2
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result.loc[0, "Date"] == pd.Timestamp("2024-02-01")
    assert result.loc[1, "Date"] == pd.Timestamp("2024-03-15")

    captured = capsys.readouterr()

    assert "All dates parsed successfully." in captured.out


def test_load_dataset_raises_value_error_for_invalid_dates(tmp_path, capsys):
    # Purpose: Check that load_dataset raises a ValueError when any Date value
    # cannot be parsed.
    csv_path = tmp_path / "walmart.csv"

    df = pd.DataFrame({
        "Store": [1, 2],
        "Date": ["01/02/2024", "not-a-date"],
        "Weekly_Sales": [1000.0, 2000.0],
    })
    df.to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match="Invalid or missing Date values found in the Walmart dataset: 1 rows.",
    ):
        load_dataset(str(csv_path))
    
    captured = capsys.readouterr()

    assert "Found unparsed or missing Date values:" in captured.out
    assert "Raw date strings causing issues:" in captured.out
    assert "not-a-date" in captured.out


def test_chronological_split_splits_latest_dates_keeps_rows_and_prints_summary(capsys):
    # Purpose: Check that chronological_split places the most recent dates
    # into the test set, earlier dates into the training set, keeps all rows
    # for each selected date, and prints a readable train/test summary.
    df = pd.DataFrame({
        "Store": [1, 2, 1, 2, 1, 2],
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
            "2024-01-08",
            "2024-01-08",
            "2024-01-15",
            "2024-01-15",
        ]),
        "Weekly_Sales": [100, 150, 200, 250, 300, 350],
    })

    train, test = chronological_split(df, test_horizon_weeks=1)

    assert set(train["Date"]) == {
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-08"),
    }
    assert set(test["Date"]) == {
        pd.Timestamp("2024-01-15"),
    }

    assert len(train) == 4
    assert len(test) == 2

    assert train["Date"].max() == pd.Timestamp("2024-01-08")
    assert test["Date"].min() == pd.Timestamp("2024-01-15")

    captured = capsys.readouterr()

    assert "Train end date: 2024-01-08" in captured.out
    assert "Train rows: 4" in captured.out
    assert "Test rows: 2" in captured.out


def test_chronological_split_raises_error_when_not_enough_unique_dates():
    # Purpose: Check that chronological_split raises a ValueError when there
    # are not enough unique dates to create the requested test horizon.
    df = pd.DataFrame({
        "Store": [1, 1],
        "Date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
        "Weekly_Sales": [100, 200],
    })

    with pytest.raises(
        ValueError,
        match="Not enough weeks to create the requested test horizon.",
    ):
        chronological_split(df, test_horizon_weeks=2)