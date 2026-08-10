# tests/test_outputs.py

import pandas as pd
import pytest
import numpy as np

from src.common.model_helpers.outputs import (
    output_dataset,
    output_feature_importance,
    output_train_test_sets,
    output_challenger_forecast_by_store,
)


def test_output_dataset_saves_csv_and_prints_summary(tmp_path, capsys):
    # Purpose: Check that output_dataset combines metadata, features, and target values,
    # saves them to the requested CSV file, and prints a saved-file summary.
    meta = pd.DataFrame({
        "Store": [1, 2],
        "Date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
    })

    X = pd.DataFrame({
        "Temperature": [20.0, 21.5],
        "Fuel_Price": [3.5, 3.6],
    })

    y = pd.Series([1000.0, 2000.0])

    output_dataset(
        X=X,
        y=y,
        meta=meta,
        outputs_dir=tmp_path,
        filename="train_set.csv",
    )

    output_path = tmp_path / "train_set.csv"

    assert output_path.exists()

    result = pd.read_csv(output_path)

    assert list(result.columns) == [
        "Store",
        "Date",
        "Temperature",
        "Fuel_Price",
        "Weekly_Sales_tplus1",
    ]

    assert len(result) == 2
    assert result.loc[0, "Store"] == 1
    assert result.loc[0, "Temperature"] == 20.0
    assert result.loc[0, "Weekly_Sales_tplus1"] == 1000.0
    assert result.loc[1, "Weekly_Sales_tplus1"] == 2000.0

    captured = capsys.readouterr()

    assert "Saved dataset to CSV:" in captured.out
    assert "train_set.csv" in captured.out
    assert "2 rows" in captured.out


def test_output_dataset_raises_error_when_metadata_and_features_indexes_do_not_match(tmp_path):
    # Purpose: Check that output_dataset raises a ValueError when metadata and
    # feature indexes are different, preventing rows from being incorrectly aligned.
    meta = pd.DataFrame({
        "Store": [1, 2],
        "Date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
    }, index=[0, 1])

    X = pd.DataFrame({
        "Temperature": [20.0, 21.5],
        "Fuel_Price": [3.5, 3.6],
    }, index=[10, 11])

    y = pd.Series([1000.0, 2000.0], index=[10, 11])

    with pytest.raises(
        ValueError,
        match="Metadata index does not match feature dataset index.",
    ):
        output_dataset(
            X=X,
            y=y,
            meta=meta,
            outputs_dir=tmp_path,
            filename="bad_set.csv",
        )



def test_output_train_test_sets_saves_train_and_test_csv_files(tmp_path):
    # Purpose: Check that output_train_test_sets writes both train_set.csv
    # and test_set.csv using the supplied metadata, features, and targets.
    X_train = pd.DataFrame({
        "Temperature": [20.0, 21.0],
        "Fuel_Price": [3.5, 3.6],
    })

    y_train = pd.Series([1000.0, 2000.0])

    train_meta = pd.DataFrame({
        "Store": [1, 2],
        "Date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
    })

    X_test = pd.DataFrame({
        "Temperature": [22.0],
        "Fuel_Price": [3.7],
    })

    y_test = pd.Series([3000.0])

    test_meta = pd.DataFrame({
        "Store": [3],
        "Date": pd.to_datetime(["2024-01-15"]),
    })

    output_train_test_sets(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_meta=train_meta,
        test_meta=test_meta,
        outputs_dir=tmp_path,
    )

    train_path = tmp_path / "train_set.csv"
    test_path = tmp_path / "test_set.csv"

    assert train_path.exists()
    assert test_path.exists()

    train_result = pd.read_csv(train_path)
    test_result = pd.read_csv(test_path)

    assert len(train_result) == 2
    assert len(test_result) == 1

    assert train_result.loc[0, "Weekly_Sales_tplus1"] == 1000.0
    assert train_result.loc[1, "Weekly_Sales_tplus1"] == 2000.0
    assert test_result.loc[0, "Weekly_Sales_tplus1"] == 3000.0


def test_output_feature_importance_saves_csv_plot_zero_gains_and_sorts_by_gain(tmp_path):
    # Purpose: Check that output_feature_importance saves the feature importance
    # CSV and plot, maps XGBoost feature IDs to column names, includes zero gain
    # for unused features, and sorts features by gain from highest to lowest.
    class FakeBooster:
        def get_score(self, importance_type):
            assert importance_type == "gain"
            return {
                "f0": 2.0,
                "f2": 10.0,
            }

    class FakeModel:
        def get_booster(self):
            return FakeBooster()

    X_train = pd.DataFrame({
        "Temperature": [20.0, 21.0],
        "Fuel_Price": [3.5, 3.6],
        "CPI": [220.0, 221.0],
    })

    output_feature_importance(
        model=FakeModel(),
        X_train=X_train,
        outputs_dir=tmp_path,
    )

    output_dir = tmp_path / "xgb_outputs"
    csv_path = output_dir / "feature_importance_gain_full.csv"
    plot_path = output_dir / "feature_importance_all.png"

    assert output_dir.exists()
    assert csv_path.exists()
    assert plot_path.exists()

    result = pd.read_csv(csv_path)

    assert list(result.columns) == ["feature", "gain"]

    assert list(result["feature"]) == [
        "CPI",
        "Temperature",
        "Fuel_Price",
    ]

    assert list(result["gain"]) == [
        10.0,
        2.0,
        0.0,
    ]


def test_output_challenger_forecast_by_store_saves_forecasts_and_prints_summary(
    tmp_path,
    capsys,
):
    # Purpose: Check that challenger predictions are combined with the held-out
    # test week, saved in the expected format, and summarised in stdout.
    test = pd.DataFrame({
        "Store": [2, 1],
        "Date": pd.to_datetime([
            "2024-01-22",
            "2024-01-15",
        ]),
        "Weekly_Sales": [2300.0, 1150.0],
        "Weekly_Sales_tplus1": [2450.0, 1250.0],
    })

    predictions = np.array([2400.0, 1200.0])

    output_challenger_forecast_by_store(
        test=test,
        predictions=predictions,
        outputs_dir=tmp_path,
    )

    output_path = tmp_path / "challenger_forecast_by_store.csv"

    assert output_path.exists()

    result = pd.read_csv(output_path, keep_default_na=False)

    assert list(result.columns) == [
        "Store",
        "Date",
        "Next_Week_Date",
        "Weekly_Sales",
        "Actual_Next_Week_Sales",
        "Predicted_Weekly_Sales",
    ]

    assert len(result) == 2

    # The output is sorted by Store.
    assert result["Store"].tolist() == [1, 2]

    assert result["Date"].tolist() == [
        "15-01-2024",
        "22-01-2024",
    ]

    assert result["Next_Week_Date"].tolist() == [
        "22-01-2024",
        "29-01-2024",
    ]

    assert result["Weekly_Sales"].tolist() == [
        1150.0,
        2300.0,
    ]

    assert result["Actual_Next_Week_Sales"].tolist() == [
        1250.0,
        2450.0,
    ]

    # Predictions remain associated with their original test rows before the
    # completed output is sorted by Store.
    assert result["Predicted_Weekly_Sales"].tolist() == [
        1200.0,
        2400.0,
    ]

    captured = capsys.readouterr()

    assert "Saved challenger forecast to:" in captured.out
    assert "challenger_forecast_by_store.csv" in captured.out
    assert "Forecast rows: 2" in captured.out
    assert "Feature date: 2024-01-15" in captured.out
    assert "Prediction date: 2024-01-22" in captured.out

def test_output_challenger_forecast_by_store_preserves_prediction_row_alignment(
    tmp_path,
):
    # Purpose: Check that each prediction remains associated with its original
    # test row when the final output is sorted by Store.
    test = pd.DataFrame({
        "Store": [2, 1],
        "Date": pd.to_datetime([
            "2024-01-22",
            "2024-01-15",
        ]),
        "Weekly_Sales": [2300.0, 1150.0],
        "Weekly_Sales_tplus1": [2450.0, 1250.0],
    })

    predictions = np.array([2400.0, 1200.0])

    output_challenger_forecast_by_store(
        test=test,
        predictions=predictions,
        outputs_dir=tmp_path,
    )

    result = pd.read_csv(
        tmp_path / "challenger_forecast_by_store.csv",
    )

    assert result["Store"].tolist() == [1, 2]
    assert result["Predicted_Weekly_Sales"].tolist() == [
        1200.0,
        2400.0,
    ]
