"""Integration tests for the Azure ML batch scoring entry script."""

from pathlib import Path

import pandas as pd
import pytest
import xgboost as xgb

from src.score_scripts import walmart_batch_score as scoring


pytestmark = pytest.mark.integration


def _train_and_save_model(model_directory: Path) -> None:
    """Create and save a small real XGBoost model."""
    model_directory.mkdir(parents=True)

    training_data = pd.DataFrame(
        {
            "Temperature": [50.0, 60.0, 70.0, 80.0],
            "lag_1": [80000.0, 100000.0, 120000.0, 140000.0],
        }
    )
    targets = [85000.0, 105000.0, 125000.0, 145000.0]

    matrix = xgb.DMatrix(
        training_data,
        label=targets,
        feature_names=["Temperature", "lag_1"],
    )

    model = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 2},
        matrix,
        num_boost_round=5,
    )
    model.save_model(model_directory / "model.xgb")


def test_init_and_run_score_real_batch(
    mocker,
    monkeypatch,
    tmp_path: Path,
):
    """Load a real model and feature CSV, then score real batch files."""
    model_directory = tmp_path / "model"
    _train_and_save_model(model_directory)

    feature_csv = tmp_path / "materialized_features.csv"
    pd.DataFrame(
        [
            {
                "Store": 1,
                "Date": "26-10-2012",
                "Temperature": 60.0,
                "lag_1": 100000.0,
            },
            {
                "Store": 2,
                "Date": "09-11-2012",
                "Temperature": 70.0,
                "lag_1": 120000.0,
            },
        ]
    ).to_csv(feature_csv, index=False)

    first_batch_file = tmp_path / "batch-1.csv"
    pd.DataFrame(
        {
            "Store": [2],
            "Date": ["09-11-2012"],
        }
    ).to_csv(first_batch_file, index=False)

    second_batch_file = tmp_path / "batch-2.csv"
    pd.DataFrame(
        {
            "Store": [1],
            "Date": ["26-10-2012"],
        }
    ).to_csv(second_batch_file, index=False)

    monkeypatch.setenv("AZUREML_MODEL_DIR", str(model_directory))
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "test-subscription")
    monkeypatch.setenv("AZURE_RESOURCE_GROUP", "test-resource-group")
    monkeypatch.setenv("AZURE_WORKSPACE_NAME", "test-workspace")
    monkeypatch.setenv("MODEL_FILENAME", "model.xgb")
    monkeypatch.setenv("DATE_OUTPUT_FORMAT", "%d-%m-%Y")

    mock_download = mocker.patch.object(
        scoring,
        "download_data_asset_csv",
        return_value=feature_csv,
    )

    scoring.init()

    result = scoring.run(
        [
            str(first_batch_file),
            str(second_batch_file),
        ]
    )

    assert result.columns.tolist() == scoring.OUTPUT_COLUMNS
    assert result["Store"].tolist() == [2, 1]
    assert result["Feature_Date"].tolist() == [
        "09-11-2012",
        "26-10-2012",
    ]
    assert result["Prediction_Date"].tolist() == [
        "16-11-2012",
        "02-11-2012",
    ]
    assert len(result) == 2
    assert result["Predicted_Weekly_Sales"].notna().all()

    mock_download.assert_called_once()