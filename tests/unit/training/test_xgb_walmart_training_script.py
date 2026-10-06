"""Unit tests for the Walmart XGBoost training script."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from walmart_ml.training import xgb_walmart_training_script as training_script


@pytest.fixture
def training_data():
    """Return the input, split, and encoded data used by the tests."""
    original = pd.DataFrame({
        "Store": [1, 1, 2, 2],
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-08",
            "2024-01-01",
            "2024-01-08",
        ]),
        "Weekly_Sales": [1000.0, 1100.0, 2000.0, 2200.0],
    })

    featured = original.assign(
        Temperature=[10.0, 11.0, 20.0, 21.0],
        Weekly_Sales_tplus1=[1100.0, 1200.0, 2200.0, 2300.0],
    )

    train = featured.iloc[[0, 2]].copy()
    test = featured.iloc[[1, 3]].copy()

    train_encoded = pd.DataFrame({
        "Temperature": [10.0, 20.0],
        "S_1": [1.0, 0.0],
        "S_2": [0.0, 1.0],
        "Weekly_Sales_tplus1": [1100.0, 2200.0],
    })

    test_encoded = pd.DataFrame({
        "Temperature": [11.0, 21.0],
        "S_1": [1.0, 0.0],
        "S_2": [0.0, 1.0],
        "Weekly_Sales_tplus1": [1200.0, 2300.0],
    })

    return {
        "original": original,
        "featured": featured,
        "train": train,
        "test": test,
        "train_encoded": train_encoded,
        "test_encoded": test_encoded,
    }


def configure_training_mocks(
    mocker,
    tmp_path,
    training_data,
):
    """Configure the common mocks required by the main-function tests."""
    mocker.patch.object(
        training_script.config,
        "METADATA_COLUMNS",
        ["Store", "Date", "Weekly_Sales"],
    )
    mocker.patch.object(
        training_script.config,
        "ROLL_WINDOWS",
        [3, 6, 12],
    )
    mocker.patch.object(
        training_script.config,
        "TEST_HORIZON_WEEKS",
        4,
    )
    mocker.patch.object(
        training_script.config,
        "CHALLENGER_HORIZON_WEEKS",
        1,
    )

    mock_get_outputs_dir = mocker.patch.object(
        training_script.data,
        "get_outputs_dir",
        return_value=tmp_path,
    )
    mock_load_dataset = mocker.patch.object(
        training_script.data,
        "load_dataset",
        return_value=training_data["original"],
    )
    mock_build_features = mocker.patch.object(
        training_script.features,
        "build_features",
        return_value=training_data["featured"],
    )
    mock_split = mocker.patch.object(
        training_script.data,
        "chronological_split",
        return_value=(
            training_data["train"],
            training_data["test"],
        ),
    )

    mock_prepare_encodings = mocker.patch.object(
        training_script.encoding,
        "prepare_encodings",
        side_effect=[
            training_data["train_encoded"],
            training_data["test_encoded"],
        ],
    )
    mock_get_feature_columns = mocker.patch.object(
        training_script.encoding,
        "get_feature_columns",
        return_value=["Temperature", "S_1", "S_2"],
    )
    mock_validate = mocker.patch.object(
        training_script.validation,
        "validate_missing_values",
    )

    mock_output_sets = mocker.patch.object(
        training_script.outputs,
        "output_train_test_sets",
    )
    mock_output_importance = mocker.patch.object(
        training_script.outputs,
        "output_feature_importance",
    )
    mock_output_challenger = mocker.patch.object(
        training_script.outputs,
        "output_challenger_forecast_by_store",
    )

    mock_configure_mlflow = mocker.patch.object(
        training_script.mlflow_utils,
        "configure_mlflow",
    )

    mock_start_run = mocker.patch.object(
        training_script.mlflow,
        "start_run",
    )
    mock_start_run.return_value.__enter__.return_value = (
        mocker.sentinel.mlflow_run
    )
    mock_start_run.return_value.__exit__.return_value = None

    mock_log_param = mocker.patch.object(
        training_script.mlflow,
        "log_param",
    )
    mock_log_params = mocker.patch.object(
        training_script.mlflow,
        "log_params",
    )

    predictions = np.array([1210.0, 2290.0])

    xgb_model = mocker.MagicMock()
    xgb_model.get_params.return_value = {
        "n_estimators": 600,
        "eta": 0.05,
        "max_depth": 8,
    }
    xgb_model.predict.return_value = predictions

    mock_create_model = mocker.patch.object(
        training_script.model,
        "create_xgb_model",
        return_value=xgb_model,
    )
    mock_metrics = mocker.patch.object(
        training_script.metrics,
        "calculate_regression_metrics",
    )

    return {
        "get_outputs_dir": mock_get_outputs_dir,
        "load_dataset": mock_load_dataset,
        "build_features": mock_build_features,
        "split": mock_split,
        "prepare_encodings": mock_prepare_encodings,
        "get_feature_columns": mock_get_feature_columns,
        "validate": mock_validate,
        "output_sets": mock_output_sets,
        "output_importance": mock_output_importance,
        "output_challenger": mock_output_challenger,
        "configure_mlflow": mock_configure_mlflow,
        "start_run": mock_start_run,
        "log_param": mock_log_param,
        "log_params": mock_log_params,
        "create_model": mock_create_model,
        "metrics": mock_metrics,
        "xgb_model": xgb_model,
        "predictions": predictions,
    }


def test_main_trains_and_evaluates_candidate_model(
    mocker,
    monkeypatch,
    tmp_path,
    training_data,
    capsys,
):
    # Purpose: Check the complete candidate-model training flow, including
    # feature preparation, validation, training, evaluation, and outputs.
    mocks = configure_training_mocks(
        mocker,
        tmp_path,
        training_data,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xgb_walmart_training_script.py",
            "--training_data",
            "data/walmart_sales.csv",
        ],
    )

    training_script.main()

    mocks["get_outputs_dir"].assert_called_once_with()
    mocks["load_dataset"].assert_called_once_with(
        "data/walmart_sales.csv",
    )
    mocks["build_features"].assert_called_once_with(
        training_data["original"],
    )
    mocks["split"].assert_called_once_with(
        training_data["featured"],
        4,
    )

    assert mocks["prepare_encodings"].call_count == 2

    pd.testing.assert_frame_equal(
        mocks["prepare_encodings"].call_args_list[0].args[0],
        training_data["train"],
    )
    pd.testing.assert_frame_equal(
        mocks["prepare_encodings"].call_args_list[1].args[0],
        training_data["test"],
    )

    mocks["get_feature_columns"].assert_called_once_with(
        training_data["train_encoded"],
    )

    required_columns = [
        "Temperature",
        "S_1",
        "S_2",
        "Weekly_Sales_tplus1",
    ]

    assert mocks["validate"].call_args_list == [
        mocker.call(
            training_data["train_encoded"],
            "training",
            required_columns,
        ),
        mocker.call(
            training_data["test_encoded"],
            "test",
            required_columns,
        ),
    ]

    output_call = mocks["output_sets"].call_args.kwargs

    pd.testing.assert_frame_equal(
        output_call["X_train"],
        training_data["train_encoded"][
            ["Temperature", "S_1", "S_2"]
        ],
    )
    pd.testing.assert_series_equal(
        output_call["y_train"],
        training_data["train_encoded"]["Weekly_Sales_tplus1"],
    )
    pd.testing.assert_frame_equal(
        output_call["X_test"],
        training_data["test_encoded"][
            ["Temperature", "S_1", "S_2"]
        ],
    )
    pd.testing.assert_series_equal(
        output_call["y_test"],
        training_data["test_encoded"]["Weekly_Sales_tplus1"],
    )

    assert output_call["outputs_dir"] == tmp_path

    mocks["configure_mlflow"].assert_called_once_with()
    mocks["start_run"].assert_called_once_with()

    mocks["log_param"].assert_any_call(
        "training_data",
        os.path.abspath("data/walmart_sales.csv"),
    )
    mocks["log_param"].assert_any_call("test_horizon_weeks", 4)
    mocks["log_param"].assert_any_call("challenger_model", False)
    mocks["log_param"].assert_any_call("roll_windows", "3,6,12")
    mocks["log_param"].assert_any_call("n_features", 3)
    mocks["log_param"].assert_any_call("n_train_rows", 2)
    mocks["log_param"].assert_any_call("n_test_rows", 2)

    created_args = mocks["create_model"].call_args.args[0]

    assert created_args.training_data == "data/walmart_sales.csv"
    assert created_args.n_estimators == 600
    assert created_args.eta == pytest.approx(0.05)
    assert created_args.max_depth == 8
    assert created_args.subsample == pytest.approx(0.8)
    assert created_args.colsample_bytree == pytest.approx(0.8)
    assert created_args.reg_alpha == pytest.approx(0.0)
    assert created_args.reg_lambda == pytest.approx(1.0)
    assert created_args.challenger_model is False

    mocks["log_params"].assert_called_once_with(
        mocks["xgb_model"].get_params.return_value,
    )

    fit_call = mocks["xgb_model"].fit.call_args
    pd.testing.assert_frame_equal(
        fit_call.args[0],
        training_data["train_encoded"][
            ["Temperature", "S_1", "S_2"]
        ],
    )
    pd.testing.assert_series_equal(
        fit_call.args[1],
        training_data["train_encoded"]["Weekly_Sales_tplus1"],
    )

    predict_call = mocks["xgb_model"].predict.call_args
    pd.testing.assert_frame_equal(
        predict_call.args[0],
        training_data["test_encoded"][
            ["Temperature", "S_1", "S_2"]
        ],
    )

    metric_call = mocks["metrics"].call_args
    pd.testing.assert_series_equal(
        metric_call.args[0],
        training_data["test_encoded"]["Weekly_Sales_tplus1"],
    )
    np.testing.assert_array_equal(
        metric_call.args[1],
        mocks["predictions"],
    )

    mocks["output_importance"].assert_called_once()
    mocks["output_challenger"].assert_not_called()

    captured = capsys.readouterr()

    assert "Candidate model: using 4 weeks for test set." in captured.out
    assert "Features: 4" in captured.out
    assert "Training rows: 2" in captured.out
    assert "Test rows: 2" in captured.out


def test_main_outputs_challenger_forecast(
    mocker,
    monkeypatch,
    tmp_path,
    training_data,
    capsys,
):
    # Purpose: Check that challenger training uses the challenger horizon and
    # writes the held-out forecast used for champion/challenger comparison.
    mocks = configure_training_mocks(
        mocker,
        tmp_path,
        training_data,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xgb_walmart_training_script.py",
            "--training_data",
            "data/walmart_sales.csv",
            "--challenger_model",
            "--n_estimators",
            "800",
            "--eta",
            "0.1",
            "--max_depth",
            "10",
            "--subsample",
            "0.9",
            "--colsample_bytree",
            "0.7",
            "--reg_alpha",
            "0.2",
            "--reg_lambda",
            "1.5",
        ],
    )

    training_script.main()

    mocks["split"].assert_called_once_with(
        training_data["featured"],
        1,
    )

    created_args = mocks["create_model"].call_args.args[0]

    assert created_args.challenger_model is True
    assert created_args.n_estimators == 800
    assert created_args.eta == pytest.approx(0.1)
    assert created_args.max_depth == 10
    assert created_args.subsample == pytest.approx(0.9)
    assert created_args.colsample_bytree == pytest.approx(0.7)
    assert created_args.reg_alpha == pytest.approx(0.2)
    assert created_args.reg_lambda == pytest.approx(1.5)

    mocks["log_param"].assert_any_call("test_horizon_weeks", 1)
    mocks["log_param"].assert_any_call("challenger_model", True)

    challenger_call = mocks["output_challenger"].call_args.kwargs

    pd.testing.assert_frame_equal(
        challenger_call["test"],
        training_data["test"],
    )
    np.testing.assert_array_equal(
        challenger_call["predictions"],
        mocks["predictions"],
    )
    assert challenger_call["outputs_dir"] == tmp_path

    captured = capsys.readouterr()

    assert "Challenger model: using 1 week(s) for test set." in captured.out


def test_main_requires_training_data_argument(
    monkeypatch,
):
    # Purpose: Check that the training-data argument cannot be omitted.
    monkeypatch.setattr(
        sys,
        "argv",
        ["xgb_walmart_training_script.py"],
    )

    with pytest.raises(SystemExit) as error:
        training_script.main()

    assert error.value.code == 2