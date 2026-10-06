"""Unit tests for the Walmart Azure ML batch scoring script."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from walmart_ml.score_scripts import walmart_batch_score as batch_score


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalise_store_converts_numeric_values_to_int64():
    stores = pd.Series(["1", 2.0, "3"])

    result = batch_score._normalise_store(stores)

    assert result.tolist() == [1, 2, 3]
    assert str(result.dtype) == "int64"


def test_normalise_store_raises_for_null_values():
    stores = pd.Series([1, None])

    with pytest.raises(
        ValueError,
        match=r"Store contains null values\.",
    ):
        batch_score._normalise_store(stores)


def test_normalise_store_raises_for_fractional_values():
    stores = pd.Series([1, 2.5])

    with pytest.raises(
        ValueError,
        match=r"Store must contain integer IDs\.",
    ):
        batch_score._normalise_store(stores)


def test_normalise_store_raises_for_non_numeric_values():
    stores = pd.Series([1, "invalid"])

    with pytest.raises(ValueError):
        batch_score._normalise_store(stores)


def test_normalise_date_parses_strips_and_normalises_dates():
    dates = pd.Series([
        " 01-01-2024 ",
        "08-01-2024",
    ])

    result = batch_score._normalise_date(dates)

    expected = pd.Series(pd.to_datetime([
        "2024-01-01",
        "2024-01-08",
    ]))

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_date",
    [
        "2024-01-01",
        "32-01-2024",
        "not-a-date",
        None,
    ],
)
def test_normalise_date_raises_for_invalid_dates(invalid_date):
    with pytest.raises(ValueError):
        batch_score._normalise_date(pd.Series([invalid_date]))


# ---------------------------------------------------------------------------
# Feature asset download
# ---------------------------------------------------------------------------


def test_download_feature_asset_uses_default_asset_settings(
    mocker,
    monkeypatch,
):
    monkeypatch.delenv("FEATURE_DATA_ASSET_NAME", raising=False)
    monkeypatch.delenv("FEATURE_DATA_ASSET_VERSION", raising=False)
    monkeypatch.delenv("FEATURE_DATA_FILENAME", raising=False)

    mocker.patch.object(
        batch_score.tempfile,
        "mkdtemp",
        return_value="/tmp/walmart-worker",
    )

    environment_values = {
        "AZURE_SUBSCRIPTION_ID": "subscription-id",
        "AZURE_RESOURCE_GROUP": "test-rg",
        "AZURE_WORKSPACE_NAME": "test-workspace",
    }
    mock_required_environment = mocker.patch.object(
        batch_score,
        "get_required_environment_variable",
        side_effect=lambda name: environment_values[name],
    )

    expected_path = Path(
        "/tmp/walmart-worker/walmart_materialized_features.csv"
    )
    mock_download = mocker.patch.object(
        batch_score,
        "download_data_asset_csv",
        return_value=expected_path,
    )

    result = batch_score._download_feature_asset()

    assert result == expected_path

    assert mock_required_environment.call_args_list == [
        mocker.call("AZURE_SUBSCRIPTION_ID"),
        mocker.call("AZURE_RESOURCE_GROUP"),
        mocker.call("AZURE_WORKSPACE_NAME"),
    ]

    mock_download.assert_called_once_with(
        subscription_id="subscription-id",
        resource_group_name="test-rg",
        workspace_name="test-workspace",
        data_asset_name="walmart_materialized_features",
        data_asset_version="1",
        local_path=expected_path,
    )


def test_download_feature_asset_uses_configured_asset_settings(
    mocker,
    monkeypatch,
):
    monkeypatch.setenv(
        "FEATURE_DATA_ASSET_NAME",
        "custom-feature-data",
    )
    monkeypatch.setenv("FEATURE_DATA_ASSET_VERSION", "8")
    monkeypatch.setenv("FEATURE_DATA_FILENAME", "custom.csv")

    mocker.patch.object(
        batch_score.tempfile,
        "mkdtemp",
        return_value="/tmp/custom-worker",
    )

    environment_values = {
        "AZURE_SUBSCRIPTION_ID": "subscription-id",
        "AZURE_RESOURCE_GROUP": "resource-group",
        "AZURE_WORKSPACE_NAME": "workspace",
    }
    mocker.patch.object(
        batch_score,
        "get_required_environment_variable",
        side_effect=lambda name: environment_values[name],
    )

    expected_path = Path("/tmp/custom-worker/custom.csv")
    mock_download = mocker.patch.object(
        batch_score,
        "download_data_asset_csv",
        return_value=expected_path,
    )

    result = batch_score._download_feature_asset()

    assert result == expected_path

    mock_download.assert_called_once_with(
        subscription_id="subscription-id",
        resource_group_name="resource-group",
        workspace_name="workspace",
        data_asset_name="custom-feature-data",
        data_asset_version="8",
        local_path=expected_path,
    )


# ---------------------------------------------------------------------------
# Model feature resolution
# ---------------------------------------------------------------------------


def test_resolve_model_features_uses_booster_feature_names(mocker):
    booster = mocker.MagicMock()
    booster.feature_names = ["Temperature", "Fuel_Price"]

    result = batch_score._resolve_model_features(
        booster,
        ["Store", "Date", "Temperature", "Fuel_Price", "Unused"],
    )

    assert result == ["Temperature", "Fuel_Price"]


def test_resolve_model_features_uses_defaults_when_model_has_no_names(
    mocker,
):
    booster = mocker.MagicMock()
    booster.feature_names = None

    available_columns = [
        "Store",
        "Date",
        *batch_score.DEFAULT_MODEL_FEATURES,
    ]

    result = batch_score._resolve_model_features(
        booster,
        available_columns,
    )

    assert result == batch_score.DEFAULT_MODEL_FEATURES
    assert result is not batch_score.DEFAULT_MODEL_FEATURES


def test_resolve_model_features_raises_when_features_are_missing(mocker):
    booster = mocker.MagicMock()
    booster.feature_names = [
        "Temperature",
        "Fuel_Price",
        "lag_52",
    ]

    with pytest.raises(
        ValueError,
        match=(
            r"Materialised feature data does not contain all model "
            r"features\. Missing: \['lag_52'\]"
        ),
    ):
        batch_score._resolve_model_features(
            booster,
            ["Temperature", "Fuel_Price"],
        )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_loads_model_feature_data_and_model_feature_names(
    mocker,
    monkeypatch,
    tmp_path,
):
    model_directory = tmp_path / "model-directory"
    feature_csv = tmp_path / "features.csv"

    monkeypatch.setenv("AZUREML_MODEL_DIR", str(model_directory))
    monkeypatch.setenv("MODEL_FILENAME", "challenger.xgb")

    mock_model = mocker.MagicMock()
    mock_model.feature_names = ["Temperature", "Fuel_Price"]

    mock_load_model = mocker.patch.object(
        batch_score,
        "load_xgboost_model",
        return_value=mock_model,
    )
    mocker.patch.object(
        batch_score,
        "_download_feature_asset",
        return_value=feature_csv,
    )

    feature_data = pd.DataFrame({
        "Store": [2, 1],
        "Date": pd.to_datetime([
            "08-01-2024",
            "01-01-2024",
        ]),
        "Temperature": [20.0, 10.0],
        "Fuel_Price": [3.2, 3.1],
    })

    mock_load_features = mocker.patch.object(
        batch_score,
        "load_feature_data",
        return_value=feature_data,
    )

    batch_score.init()

    mock_load_model.assert_called_once_with(
        model_directory,
        "challenger.xgb",
    )
    mock_load_features.assert_called_once_with(feature_csv)

    assert batch_score.model is mock_model
    assert batch_score.model_feature_names == [
        "Temperature",
        "Fuel_Price",
    ]

    expected_lookup = (
        feature_data
        .set_index(["Store", "Date"])
        .sort_index()
    )
    pd.testing.assert_frame_equal(
        batch_score.feature_lookup,
        expected_lookup,
    )


def test_init_uses_default_model_filename(
    mocker,
    monkeypatch,
    tmp_path,
):
    model_directory = tmp_path / "model-directory"

    monkeypatch.setenv("AZUREML_MODEL_DIR", str(model_directory))
    monkeypatch.delenv("MODEL_FILENAME", raising=False)

    mock_model = mocker.MagicMock()
    mock_model.feature_names = ["Temperature"]

    mock_load_model = mocker.patch.object(
        batch_score,
        "load_xgboost_model",
        return_value=mock_model,
    )
    mocker.patch.object(
        batch_score,
        "_download_feature_asset",
        return_value=tmp_path / "features.csv",
    )
    mocker.patch.object(
        batch_score,
        "load_feature_data",
        return_value=pd.DataFrame({
            "Store": [1],
            "Date": pd.to_datetime(["2024-01-01"]),
            "Temperature": [10.0],
        }),
    )

    batch_score.init()

    mock_load_model.assert_called_once_with(
        model_directory,
        "model.xgb",
    )


# ---------------------------------------------------------------------------
# Batch input reading
# ---------------------------------------------------------------------------


def test_read_batch_files_reads_normalises_and_preserves_request_order(
    tmp_path,
):
    first_file = tmp_path / "first.csv"
    second_file = tmp_path / "second.csv"

    pd.DataFrame({
        "Store": ["2", "1"],
        "Date": ["08-01-2024", "01-01-2024"],
        "Ignored": ["a", "b"],
    }).to_csv(first_file, index=False)

    pd.DataFrame({
        "Store": [3],
        "Date": ["15-01-2024"],
    }).to_csv(second_file, index=False)

    result = batch_score._read_batch_files([
        str(first_file),
        str(second_file),
    ])

    assert result["Store"].tolist() == [2, 1, 3]
    assert str(result["Store"].dtype) == "int64"
    assert result["Date"].tolist() == list(pd.to_datetime([
        "2024-01-08",
        "2024-01-01",
        "2024-01-15",
    ]))
    assert result["_request_order"].tolist() == [0, 1, 2]
    assert list(result.columns) == [
        "Store",
        "Date",
        "_request_order",
    ]


def test_read_batch_files_returns_empty_dataframe_for_empty_batch():
    result = batch_score._read_batch_files([])

    assert result.empty
    assert list(result.columns) == [
        "Store",
        "Date",
        "_request_order",
    ]


@pytest.mark.parametrize(
    ("columns", "expected_missing"),
    [
        (
            {"Date": ["01-01-2024"]},
            "Store",
        ),
        (
            {"Store": [1]},
            "Date",
        ),
        (
            {"Temperature": [10.0]},
            "Date.*Store|Store.*Date",
        ),
    ],
)
def test_read_batch_files_raises_when_key_columns_are_missing(
    tmp_path,
    columns,
    expected_missing,
):
    input_path = tmp_path / "input.csv"
    pd.DataFrame(columns).to_csv(input_path, index=False)

    with pytest.raises(
        ValueError,
        match=rf"is missing columns:.*{expected_missing}",
    ):
        batch_score._read_batch_files([str(input_path)])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def create_feature_lookup():
    """Return a materialised feature lookup for scoring tests."""
    feature_data = pd.DataFrame({
        "Store": [1, 2],
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-08",
        ]),
        "Temperature": [10.0, 20.0],
        "Fuel_Price": [3.1, 3.2],
    })

    return (
        feature_data
        .set_index(["Store", "Date"])
        .sort_index()
    )


def test_run_scores_rows_and_preserves_request_order(
    mocker,
    monkeypatch,
):
    requests = pd.DataFrame({
        "Store": pd.Series([2, 1], dtype="int64"),
        "Date": pd.to_datetime([
            "2024-01-08",
            "2024-01-01",
        ]),
        "_request_order": [0, 1],
    })

    mocker.patch.object(
        batch_score,
        "_read_batch_files",
        return_value=requests,
    )

    monkeypatch.setattr(
        batch_score,
        "feature_lookup",
        create_feature_lookup(),
    )
    monkeypatch.setattr(
        batch_score,
        "model_feature_names",
        ["Temperature", "Fuel_Price"],
    )

    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array([2200.0, 1100.0])
    monkeypatch.setattr(batch_score, "model", mock_model)

    matrix = mocker.sentinel.matrix
    mock_dmatrix = mocker.patch.object(
        batch_score.xgb,
        "DMatrix",
        return_value=matrix,
    )

    result = batch_score.run(["first.csv", "second.csv"])

    assert list(result.columns) == batch_score.OUTPUT_COLUMNS
    assert result["Store"].tolist() == [2, 1]
    assert result["Feature_Date"].tolist() == [
        "08-01-2024",
        "01-01-2024",
    ]
    assert result["Prediction_Date"].tolist() == [
        "15-01-2024",
        "08-01-2024",
    ]
    assert result["Predicted_Weekly_Sales"].tolist() == [
        2200.0,
        1100.0,
    ]

    dmatrix_call = mock_dmatrix.call_args

    pd.testing.assert_frame_equal(
        dmatrix_call.args[0],
        pd.DataFrame({
            "Temperature": [20.0, 10.0],
            "Fuel_Price": [3.2, 3.1],
        }),
    )
    assert dmatrix_call.kwargs["feature_names"] == [
        "Temperature",
        "Fuel_Price",
    ]

    mock_model.predict.assert_called_once_with(matrix)


def test_run_uses_configured_date_output_format(
    mocker,
    monkeypatch,
):
    requests = pd.DataFrame({
        "Store": pd.Series([1], dtype="int64"),
        "Date": pd.to_datetime(["2024-01-01"]),
        "_request_order": [0],
    })

    mocker.patch.object(
        batch_score,
        "_read_batch_files",
        return_value=requests,
    )
    monkeypatch.setattr(
        batch_score,
        "feature_lookup",
        create_feature_lookup(),
    )
    monkeypatch.setattr(
        batch_score,
        "model_feature_names",
        ["Temperature", "Fuel_Price"],
    )

    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array([1100.0])
    monkeypatch.setattr(batch_score, "model", mock_model)

    mocker.patch.object(
        batch_score.xgb,
        "DMatrix",
    )
    monkeypatch.setenv("DATE_OUTPUT_FORMAT", "%Y-%m-%d")

    result = batch_score.run(["input.csv"])

    assert result["Feature_Date"].tolist() == ["2024-01-01"]
    assert result["Prediction_Date"].tolist() == ["2024-01-08"]


def test_run_returns_empty_output_for_empty_batch(
    mocker,
):
    mocker.patch.object(
        batch_score,
        "_read_batch_files",
        return_value=pd.DataFrame(
            columns=["Store", "Date", "_request_order"]
        ),
    )

    result = batch_score.run([])

    assert result.empty
    assert list(result.columns) == batch_score.OUTPUT_COLUMNS


def test_run_raises_when_materialised_features_are_not_found(
    mocker,
    monkeypatch,
):
    requests = pd.DataFrame({
        "Store": pd.Series([1, 99], dtype="int64"),
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-15",
        ]),
        "_request_order": [0, 1],
    })

    mocker.patch.object(
        batch_score,
        "_read_batch_files",
        return_value=requests,
    )
    monkeypatch.setattr(
        batch_score,
        "feature_lookup",
        create_feature_lookup(),
    )
    monkeypatch.setattr(
        batch_score,
        "model_feature_names",
        ["Temperature", "Fuel_Price"],
    )

    with pytest.raises(
        ValueError,
        match=r"No materialised engineered features were found for 1 input",
    ) as error:
        batch_score.run(["input.csv"])

    assert "'Store': 99" in str(error.value)
    assert "'Date': '15-01-2024'" in str(error.value)


def test_run_raises_when_materialised_model_features_contain_nulls(
    mocker,
    monkeypatch,
):
    requests = pd.DataFrame({
        "Store": pd.Series([1], dtype="int64"),
        "Date": pd.to_datetime(["2024-01-01"]),
        "_request_order": [0],
    })

    feature_lookup = pd.DataFrame({
        "Store": [1],
        "Date": pd.to_datetime(["2024-01-01"]),
        "Temperature": [10.0],
        "Fuel_Price": [np.nan],
    }).set_index(["Store", "Date"])

    mocker.patch.object(
        batch_score,
        "_read_batch_files",
        return_value=requests,
    )
    monkeypatch.setattr(
        batch_score,
        "feature_lookup",
        feature_lookup,
    )
    monkeypatch.setattr(
        batch_score,
        "model_feature_names",
        ["Temperature", "Fuel_Price"],
    )

    with pytest.raises(
        ValueError,
        match=r"Materialised rows contain null model features\.",
    ) as error:
        batch_score.run(["input.csv"])

    assert "'Store': 1" in str(error.value)
    assert "'Date': '01-01-2024'" in str(error.value)
    assert "'missing_features': ['Fuel_Price']" in str(error.value)