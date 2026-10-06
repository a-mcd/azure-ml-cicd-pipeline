"""Integration tests for Walmart real-time scoring."""

from pathlib import Path
import shutil

import pandas as pd
import pytest
import xgboost as xgb

# Update this import if the scoring script has a different module name.
from walmart_ml.score_scripts import walmart_realtime_score as scoring


pytestmark = pytest.mark.integration

MODEL_FEATURES = [
    "Temperature",
    "Fuel_Price",
    "lag_1",
]


@pytest.fixture
def saved_xgboost_model(tmp_path: Path) -> tuple[Path, xgb.Booster]:
    """Train and save a small real XGBoost model."""
    model_directory = tmp_path / "model"
    model_directory.mkdir()

    training_features = pd.DataFrame(
        {
            "Temperature": [50.0, 60.0, 70.0, 80.0],
            "Fuel_Price": [3.20, 3.40, 3.60, 3.80],
            "lag_1": [80_000.0, 100_000.0, 120_000.0, 140_000.0],
        }
    )
    targets = pd.Series(
        [85_000.0, 105_000.0, 125_000.0, 145_000.0]
    )

    training_matrix = xgb.DMatrix(
        training_features,
        label=targets,
        feature_names=MODEL_FEATURES,
    )

    model = xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.3,
            "seed": 42,
            "nthread": 1,
        },
        training_matrix,
        num_boost_round=10,
    )

    model.save_model(model_directory / "model.xgb")

    return model_directory, model


@pytest.fixture
def feature_csv(tmp_path: Path) -> Path:
    """Create realistic materialised feature data."""
    csv_path = tmp_path / "materialized-features.csv"

    pd.DataFrame(
        [
            {
                "Store": 1,
                "Date": "26-10-2012",
                "Temperature": 60.0,
                "Fuel_Price": 3.40,
                "lag_1": 100_000.0,
            },
            {
                "Store": 2,
                "Date": "02-11-2012",
                "Temperature": 70.0,
                "Fuel_Price": 3.60,
                "lag_1": 120_000.0,
            },
        ]
    ).to_csv(csv_path, index=False)

    return csv_path


def test_initialises_and_scores_multiple_records(
    mocker,
    monkeypatch,
    saved_xgboost_model,
    feature_csv: Path,
):
    """Load real artifacts and score an endpoint request end to end."""
    model_directory, expected_model = saved_xgboost_model

    monkeypatch.setenv(
        "AML_SUBSCRIPTION_ID",
        "integration-subscription",
    )
    monkeypatch.setenv(
        "DATA_WORKSPACE_RG",
        "integration-resource-group",
    )
    monkeypatch.setenv(
        "DATA_WORKSPACE_NAME",
        "integration-workspace",
    )
    monkeypatch.setenv(
        "FEATURE_DATA_ASSET_NAME",
        "walmart_materialized_features",
    )
    monkeypatch.setenv(
        "FEATURE_DATA_ASSET_VERSION",
        "1",
    )
    monkeypatch.setenv(
        "AZUREML_MODEL_DIR",
        str(model_directory),
    )

    def copy_feature_asset(*, local_path: Path, **_) -> Path:
        """Simulate downloading the Azure ML data asset."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(feature_csv, local_path)
        return local_path

    mock_download = mocker.patch.object(
        scoring,
        "download_data_asset_csv",
        side_effect=copy_feature_asset,
    )

    # Prevent state from another test from being reused.
    monkeypatch.setattr(scoring, "model", None)
    monkeypatch.setattr(scoring, "feature_data", None)
    monkeypatch.setattr(scoring, "model_feature_names", [])

    scoring.init()

    result = scoring.run(
        {
            "input_data": [
                {
                    "Store": 2,
                    "Date": "02-11-2012",
                },
                {
                    "Store": 1,
                    "Date": "26-10-2012",
                },
            ]
        }
    )

    expected_input = pd.DataFrame(
        [
            {
                "Temperature": 70.0,
                "Fuel_Price": 3.60,
                "lag_1": 120_000.0,
            },
            {
                "Temperature": 60.0,
                "Fuel_Price": 3.40,
                "lag_1": 100_000.0,
            },
        ],
        columns=MODEL_FEATURES,
    )

    expected_predictions = expected_model.predict(
        xgb.DMatrix(
            expected_input,
            feature_names=MODEL_FEATURES,
        )
    )

    assert "error" not in result
    assert len(result["predictions"]) == 2

    first_prediction = result["predictions"][0]
    assert first_prediction == {
        "Store": 2,
        "Feature_Date": "02-11-2012",
        "Prediction_Date": "09-11-2012",
        "Predicted_Weekly_Sales": pytest.approx(
            float(expected_predictions[0])
        ),
    }

    second_prediction = result["predictions"][1]
    assert second_prediction == {
        "Store": 1,
        "Feature_Date": "26-10-2012",
        "Prediction_Date": "02-11-2012",
        "Predicted_Weekly_Sales": pytest.approx(
            float(expected_predictions[1])
        ),
    }

    assert scoring.model is not None
    assert scoring.feature_data is not None
    assert scoring.model_feature_names == MODEL_FEATURES

    mock_download.assert_called_once_with(
        subscription_id="integration-subscription",
        resource_group_name="integration-resource-group",
        workspace_name="integration-workspace",
        data_asset_name="walmart_materialized_features",
        data_asset_version="1",
        local_path=Path(
            "/tmp/walmart_materialized_features.csv"
        ),
    )