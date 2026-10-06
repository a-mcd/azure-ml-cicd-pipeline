"""Integration test for the complete XGBoost training pipeline."""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from walmart_ml.training import xgb_walmart_training_script as training


pytestmark = pytest.mark.integration


@pytest.fixture
def training_dataset(tmp_path: Path) -> Path:
    """Create enough weekly history for lag-52 features."""
    dataset_path = tmp_path / "walmart_training_data.csv"

    rows = []
    dates = pd.date_range(
        start="2010-01-01",
        periods=70,
        freq="W-FRI",
    )

    for store in range(1, 6):
        base_sales = 100000.0 + store * 10000.0

        for week_number, date in enumerate(dates):
            seasonal_change = (
                5000.0 * np.sin(week_number / 6.0)
            )
            weekly_growth = week_number * 250.0

            rows.append(
                {
                    "Store": store,
                    "Date": date.strftime("%d-%m-%Y"),
                    "Weekly_Sales": (
                        base_sales
                        + seasonal_change
                        + weekly_growth
                    ),
                    "Holiday_Flag": int(
                        week_number % 13 == 0
                    ),
                    "Temperature": (
                        50.0
                        + 10.0 * np.sin(week_number / 8.0)
                        + store * 0.1
                    ),
                    "Fuel_Price": (
                        2.50
                        + week_number * 0.005
                        + store * 0.001
                    ),
                    "CPI": (
                        210.0
                        + week_number * 0.05
                        + store * 0.01
                    ),
                    "Unemployment": (
                        8.0
                        - week_number * 0.01
                        + store * 0.001
                    ),
                }
            )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(dataset_path, index=False)

    return dataset_path


@pytest.fixture
def local_mlflow_store(tmp_path: Path):
    """Configure an isolated local MLflow tracking store."""
    tracking_directory = tmp_path / "mlruns"
    tracking_directory.mkdir()

    mlflow.end_run()
    mlflow.set_tracking_uri(tracking_directory.as_uri())
    mlflow.set_experiment("training-integration-test")

    yield tracking_directory

    mlflow.end_run()


def test_challenger_training_pipeline_end_to_end(
    mocker,
    monkeypatch,
    tmp_path,
    training_dataset,
    local_mlflow_store,
    capsys,
):
    """Run the real challenger training pipeline end to end."""
    outputs_directory = tmp_path / "outputs"
    outputs_directory.mkdir()

    monkeypatch.chdir(tmp_path)

    # Use an isolated output directory rather than Azure ML's
    # mounted outputs directory.
    mocker.patch.object(
        training.data,
        "get_outputs_dir",
        return_value=outputs_directory,
    )

    # Tracking was configured by local_mlflow_store, so prevent
    # configure_mlflow() from replacing it with an Azure URI.
    mock_configure_mlflow = mocker.patch.object(
        training.mlflow_utils,
        "configure_mlflow",
    )

    mocker.patch.object(
        sys,
        "argv",
        [
            "xgb_walmart_model_script.py",
            "--training_data",
            str(training_dataset),
            "--n_estimators",
            "2",
            "--eta",
            "0.05",
            "--max_depth",
            "2",
            "--subsample",
            "0.8",
            "--colsample_bytree",
            "0.8",
            "--reg_alpha",
            "0.0",
            "--reg_lambda",
            "1.0",
            "--challenger_model",
        ],
    )

    training.main()

    output = capsys.readouterr().out
    output_files = [
        path
        for path in outputs_directory.rglob("*")
        if path.is_file()
    ]

    assert "Challenger model:" in output
    assert "Features:" in output
    assert "Training rows:" in output
    assert "Test rows:" in output

    mock_configure_mlflow.assert_called_once_with()

    # The pipeline should produce train/test, feature-importance,
    # and challenger forecast artifacts.
    assert output_files
    assert any(
        "feature" in path.name.lower()
        and "importance" in path.name.lower()
        for path in output_files
    )
    assert any(
        "challenger" in path.name.lower()
        and "forecast" in path.name.lower()
        for path in output_files
    )

    client = MlflowClient(
        tracking_uri=local_mlflow_store.as_uri()
    )
    experiment = client.get_experiment_by_name(
        "training-integration-test"
    )

    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) == 1

    run = runs[0]

    # Verify training metadata was logged.
    assert run.data.params["challenger_model"] == "True"
    assert run.data.params["n_estimators"] == "2"
    assert run.data.params["max_depth"] == "2"
    assert int(run.data.params["n_train_rows"]) > 0
    assert int(run.data.params["n_test_rows"]) > 0
    assert int(run.data.params["n_features"]) > 0

    # Verify evaluation metrics were calculated and logged.
    expected_metrics = {
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "Median_AE",
        "Bias",
        "WAPE",
    }

    assert expected_metrics.issubset(
        run.data.metrics.keys()
    )

    for metric_name in expected_metrics:
        assert np.isfinite(
            run.data.metrics[metric_name]
        )