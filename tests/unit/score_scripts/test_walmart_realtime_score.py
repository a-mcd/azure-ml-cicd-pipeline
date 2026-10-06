"""Unit tests for the Walmart real-time scoring entry script."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from walmart_ml.score_scripts import walmart_realtime_score as realtime_score



@pytest.fixture(autouse=True)
def reset_cached_scoring_state(monkeypatch):
    """Reset module-level endpoint state before every test."""
    monkeypatch.setattr(realtime_score, "model", None)
    monkeypatch.setattr(realtime_score, "feature_data", None)
    monkeypatch.setattr(realtime_score, "model_feature_names", [])


@pytest.fixture
def engineered_features():
    """Return normalized engineered features for multiple stores and dates."""
    return pd.DataFrame({
        "Store": [1, 1, 2],
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-08",
            "2024-01-01",
        ]),
        "Temperature": [10.0, 11.0, 20.0],
        "Fuel_Price": [3.1, 3.2, 3.3],
    })


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_init_loads_feature_data_and_model(
    mocker,
    engineered_features,
):
    environment = {
        "AML_SUBSCRIPTION_ID": "subscription-id",
        "DATA_WORKSPACE_RG": "data-rg",
        "DATA_WORKSPACE_NAME": "data-workspace",
        "FEATURE_DATA_ASSET_NAME": "walmart-features",
        "FEATURE_DATA_ASSET_VERSION": "7",
        "AZUREML_MODEL_DIR": "/var/azureml/model",
    }

    mock_required = mocker.patch.object(
        realtime_score,
        "get_required_environment_variable",
        side_effect=lambda name: environment[name],
    )

    local_csv_path = Path(
        "/tmp/walmart_materialized_features.csv"
    )
    mock_download = mocker.patch.object(
        realtime_score,
        "download_data_asset_csv",
    )
    mock_load_features = mocker.patch.object(
        realtime_score,
        "load_feature_data",
        return_value=engineered_features,
    )

    mock_model = mocker.MagicMock()
    mock_model.feature_names = ["Temperature", "Fuel_Price"]

    mock_load_model = mocker.patch.object(
        realtime_score,
        "load_xgboost_model",
        return_value=mock_model,
    )

    realtime_score.init()

    assert mock_required.call_args_list == [
        mocker.call("AML_SUBSCRIPTION_ID"),
        mocker.call("DATA_WORKSPACE_RG"),
        mocker.call("DATA_WORKSPACE_NAME"),
        mocker.call("FEATURE_DATA_ASSET_NAME"),
        mocker.call("FEATURE_DATA_ASSET_VERSION"),
        mocker.call("AZUREML_MODEL_DIR"),
    ]

    mock_download.assert_called_once_with(
        subscription_id="subscription-id",
        resource_group_name="data-rg",
        workspace_name="data-workspace",
        data_asset_name="walmart-features",
        data_asset_version="7",
        local_path=local_csv_path,
    )
    mock_load_features.assert_called_once_with(local_csv_path)
    mock_load_model.assert_called_once_with(
        Path("/var/azureml/model"),
    )

    assert realtime_score.model is mock_model
    assert realtime_score.feature_data is engineered_features
    assert realtime_score.model_feature_names == [
        "Temperature",
        "Fuel_Price",
    ]


@pytest.mark.parametrize("feature_names", [None, []])
def test_init_raises_when_model_has_no_feature_names(
    mocker,
    engineered_features,
    feature_names,
):
    mocker.patch.object(
        realtime_score,
        "get_required_environment_variable",
        side_effect=[
            "subscription-id",
            "data-rg",
            "data-workspace",
            "feature-data",
            "1",
            "/var/azureml/model",
        ],
    )
    mocker.patch.object(
        realtime_score,
        "download_data_asset_csv",
    )
    mocker.patch.object(
        realtime_score,
        "load_feature_data",
        return_value=engineered_features,
    )

    mock_model = mocker.MagicMock()
    mock_model.feature_names = feature_names

    mocker.patch.object(
        realtime_score,
        "load_xgboost_model",
        return_value=mock_model,
    )

    with pytest.raises(
        RuntimeError,
        match=r"The XGBoost model does not contain feature names\.",
    ):
        realtime_score.init()


def test_init_raises_when_feature_csv_is_missing_model_features(
    mocker,
    engineered_features,
):
    mocker.patch.object(
        realtime_score,
        "get_required_environment_variable",
        side_effect=[
            "subscription-id",
            "data-rg",
            "data-workspace",
            "feature-data",
            "1",
            "/var/azureml/model",
        ],
    )
    mocker.patch.object(
        realtime_score,
        "download_data_asset_csv",
    )
    mocker.patch.object(
        realtime_score,
        "load_feature_data",
        return_value=engineered_features,
    )

    mock_model = mocker.MagicMock()
    mock_model.feature_names = [
        "Temperature",
        "Fuel_Price",
        "lag_52",
        "rollmean_12",
    ]

    mocker.patch.object(
        realtime_score,
        "load_xgboost_model",
        return_value=mock_model,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"The engineered-feature CSV does not contain the "
            r"following model features: lag_52, rollmean_12"
        ),
    ):
        realtime_score.init()


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def test_parse_date_parses_and_normalizes_valid_date():
    result = realtime_score.parse_date("15-01-2024")

    assert result == pd.Timestamp("2024-01-15")
    assert result.hour == 0
    assert result.minute == 0


def test_parse_date_normalizes_timestamp_with_time():
    result = realtime_score.parse_date(
        pd.Timestamp("2024-01-15 14:35:20")
    )

    assert result == pd.Timestamp("2024-01-15")


def test_parse_date_raises_for_null():
    with pytest.raises(
        ValueError,
        match=r"Date cannot be null\.",
    ):
        realtime_score.parse_date(None)


def test_parse_date_raises_for_multiple_dates():
    with pytest.raises(
        ValueError,
        match=r"Date must contain one date value\.",
    ):
        realtime_score.parse_date([
            "01-01-2024",
            "08-01-2024",
        ])


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-15",
        "32-01-2024",
        "invalid",
    ],
)
def test_parse_date_raises_for_invalid_date(value):
    with pytest.raises(ValueError):
        realtime_score.parse_date(value)


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def test_parse_request_accepts_input_data_list():
    result = realtime_score.parse_request({
        "input_data": [
            {
                "Store": 1,
                "Date": "01-01-2024",
            },
            {
                "Store": "45",
                "Date": "08-01-2024",
            },
        ]
    })

    assert result == [
        {
            "Store": 1,
            "Date": pd.Timestamp("2024-01-01"),
        },
        {
            "Store": 45,
            "Date": pd.Timestamp("2024-01-08"),
        },
    ]


def test_parse_request_accepts_single_input_data_object():
    result = realtime_score.parse_request({
        "input_data": {
            "Store": 2,
            "Date": "01-01-2024",
        }
    })

    assert result == [{
        "Store": 2,
        "Date": pd.Timestamp("2024-01-01"),
    }]


def test_parse_request_accepts_direct_record_object():
    result = realtime_score.parse_request({
        "Store": 3,
        "Date": "15-01-2024",
    })

    assert result == [{
        "Store": 3,
        "Date": pd.Timestamp("2024-01-15"),
    }]


def test_parse_request_accepts_json_string():
    raw_data = json.dumps({
        "input_data": [{
            "Store": 1,
            "Date": "01-01-2024",
        }]
    })

    result = realtime_score.parse_request(raw_data)

    assert result[0]["Store"] == 1
    assert result[0]["Date"] == pd.Timestamp("2024-01-01")


def test_parse_request_accepts_json_bytes():
    raw_data = json.dumps({
        "input_data": {
            "Store": 1,
            "Date": "01-01-2024",
        }
    }).encode("utf-8")

    result = realtime_score.parse_request(raw_data)

    assert result == [{
        "Store": 1,
        "Date": pd.Timestamp("2024-01-01"),
    }]


@pytest.mark.parametrize(
    "raw_data",
    [
        None,
        123,
        [],
    ],
)
def test_parse_request_rejects_non_object_request(raw_data):
    with pytest.raises(
        ValueError,
        match=r"The request must be a JSON object\.",
    ):
        realtime_score.parse_request(raw_data)


@pytest.mark.parametrize(
    "records",
    [
        [],
        None,
        "invalid",
    ],
)
def test_parse_request_rejects_empty_or_invalid_input_data(records):
    with pytest.raises(
        ValueError,
        match=(
            r"'input_data' must contain an object or a non-empty list\."
        ),
    ):
        realtime_score.parse_request({
            "input_data": records,
        })


def test_parse_request_rejects_non_object_record():
    with pytest.raises(
        ValueError,
        match=r"Input record 1 must be a JSON object\.",
    ):
        realtime_score.parse_request({
            "input_data": [
                {
                    "Store": 1,
                    "Date": "01-01-2024",
                },
                "invalid-record",
            ]
        })


def test_parse_request_requires_store():
    with pytest.raises(
        ValueError,
        match=r"Input record 0 does not contain 'Store'\.",
    ):
        realtime_score.parse_request({
            "input_data": [{
                "Date": "01-01-2024",
            }]
        })


def test_parse_request_requires_date():
    with pytest.raises(
        ValueError,
        match=r"Input record 0 does not contain 'Date'\.",
    ):
        realtime_score.parse_request({
            "input_data": [{
                "Store": 1,
            }]
        })


@pytest.mark.parametrize("store", [None, "one", "1.5"])
def test_parse_request_rejects_invalid_store_value(store):
    with pytest.raises(
        ValueError,
        match=r"Invalid Store value:",
    ):
        realtime_score.parse_request({
            "input_data": [{
                "Store": store,
                "Date": "01-01-2024",
            }]
        })


@pytest.mark.parametrize("store", [0, -1, 46, 100])
def test_parse_request_rejects_store_outside_supported_range(store):
    with pytest.raises(
        ValueError,
        match=(
            rf"Store must be between 1 and 45; received {store}\."
        ),
    ):
        realtime_score.parse_request({
            "input_data": [{
                "Store": store,
                "Date": "01-01-2024",
            }]
        })


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def test_select_engineered_features_returns_exact_match(
    monkeypatch,
    engineered_features,
):
    monkeypatch.setattr(
        realtime_score,
        "feature_data",
        engineered_features,
    )

    result = realtime_score.select_engineered_features(
        store_id=1,
        requested_date=pd.Timestamp("2024-01-08 12:30:00"),
    )

    assert result["Store"] == 1
    assert result["Date"] == pd.Timestamp("2024-01-08")
    assert result["Temperature"] == pytest.approx(11.0)
    assert result["Fuel_Price"] == pytest.approx(3.2)


def test_select_engineered_features_raises_when_not_initialized():
    with pytest.raises(
        RuntimeError,
        match=r"Engineered-feature data has not been initialized\.",
    ):
        realtime_score.select_engineered_features(
            store_id=1,
            requested_date=pd.Timestamp("2024-01-01"),
        )


def test_select_engineered_features_raises_when_store_does_not_exist(
    monkeypatch,
    engineered_features,
):
    monkeypatch.setattr(
        realtime_score,
        "feature_data",
        engineered_features,
    )

    with pytest.raises(
        ValueError,
        match=r"No engineered-feature rows exist for Store 45\.",
    ):
        realtime_score.select_engineered_features(
            store_id=45,
            requested_date=pd.Timestamp("2024-01-01"),
        )


def test_select_engineered_features_lists_available_dates_for_store(
    monkeypatch,
    engineered_features,
):
    monkeypatch.setattr(
        realtime_score,
        "feature_data",
        engineered_features,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"No engineered-feature row found for Store 1 and "
            r"Date 15-01-2024\. Available dates for this store: "
            r"01-01-2024, 08-01-2024\."
        ),
    ):
        realtime_score.select_engineered_features(
            store_id=1,
            requested_date=pd.Timestamp("2024-01-15"),
        )


def test_select_engineered_features_rejects_duplicate_matches(
    monkeypatch,
):
    duplicate_features = pd.DataFrame({
        "Store": [1, 1],
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
        ]),
        "Temperature": [10.0, 11.0],
        "Fuel_Price": [3.1, 3.2],
    })
    monkeypatch.setattr(
        realtime_score,
        "feature_data",
        duplicate_features,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Multiple engineered-feature rows found for Store 1 "
            r"and Date 01-01-2024\."
        ),
    ):
        realtime_score.select_engineered_features(
            store_id=1,
            requested_date=pd.Timestamp("2024-01-01"),
        )


# ---------------------------------------------------------------------------
# Model input
# ---------------------------------------------------------------------------


def test_build_model_input_selects_and_orders_model_features(
    monkeypatch,
):
    monkeypatch.setattr(
        realtime_score,
        "model_feature_names",
        ["Fuel_Price", "Temperature"],
    )

    selected_row = pd.Series({
        "Store": 1,
        "Date": pd.Timestamp("2024-01-01"),
        "Temperature": "10.5",
        "Fuel_Price": "3.25",
        "Unused": 999,
    })

    result = realtime_score.build_model_input(selected_row)

    assert list(result.columns) == [
        "Fuel_Price",
        "Temperature",
    ]
    assert result.iloc[0].tolist() == [3.25, 10.5]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes)


@pytest.mark.parametrize(
    ("temperature", "fuel_price", "expected_columns"),
    [
        (None, 3.2, "Temperature"),
        ("invalid", 3.2, "Temperature"),
        (10.0, None, "Fuel_Price"),
        ("invalid", "invalid", "Temperature, Fuel_Price"),
    ],
)
def test_build_model_input_rejects_null_or_nonnumeric_values(
    monkeypatch,
    temperature,
    fuel_price,
    expected_columns,
):
    monkeypatch.setattr(
        realtime_score,
        "model_feature_names",
        ["Temperature", "Fuel_Price"],
    )

    selected_row = pd.Series({
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
    })

    with pytest.raises(
        ValueError,
        match=(
            r"The selected row contains null or nonnumeric values for: "
            + expected_columns
        ),
    ):
        realtime_score.build_model_input(selected_row)


# ---------------------------------------------------------------------------
# Record scoring
# ---------------------------------------------------------------------------


def test_score_record_builds_matrix_and_returns_prediction(
    mocker,
    monkeypatch,
):
    requested_date = pd.Timestamp("2024-01-01")

    selected_row = pd.Series({
        "Store": 1,
        "Date": requested_date,
        "Temperature": 10.0,
        "Fuel_Price": 3.1,
    })
    model_input = pd.DataFrame({
        "Temperature": [10.0],
        "Fuel_Price": [3.1],
    })

    mock_select = mocker.patch.object(
        realtime_score,
        "select_engineered_features",
        return_value=selected_row,
    )
    mock_build = mocker.patch.object(
        realtime_score,
        "build_model_input",
        return_value=model_input,
    )

    monkeypatch.setattr(
        realtime_score,
        "model_feature_names",
        ["Temperature", "Fuel_Price"],
    )

    matrix = mocker.sentinel.matrix
    mock_dmatrix = mocker.patch.object(
        realtime_score.xgb,
        "DMatrix",
        return_value=matrix,
    )

    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array([1234.56])
    monkeypatch.setattr(
        realtime_score,
        "model",
        mock_model,
    )

    result = realtime_score.score_record(
        store_id=1,
        requested_date=requested_date,
    )

    assert result == {
        "Store": 1,
        "Feature_Date": "01-01-2024",
        "Prediction_Date": "08-01-2024",
        "Predicted_Weekly_Sales": pytest.approx(1234.56),
    }

    mock_select.assert_called_once_with(
        store_id=1,
        requested_date=requested_date,
    )
    mock_build.assert_called_once_with(selected_row)
    mock_dmatrix.assert_called_once_with(
        model_input,
        feature_names=["Temperature", "Fuel_Price"],
    )
    mock_model.predict.assert_called_once_with(matrix)


def test_score_record_raises_when_model_is_not_initialized():
    with pytest.raises(
        RuntimeError,
        match=r"The XGBoost model has not been initialized\.",
    ):
        realtime_score.score_record(
            store_id=1,
            requested_date=pd.Timestamp("2024-01-01"),
        )


# ---------------------------------------------------------------------------
# Endpoint run handler
# ---------------------------------------------------------------------------


def test_run_scores_every_parsed_record(mocker):
    records = [
        {
            "Store": 1,
            "Date": pd.Timestamp("2024-01-01"),
        },
        {
            "Store": 2,
            "Date": pd.Timestamp("2024-01-08"),
        },
    ]

    mock_parse = mocker.patch.object(
        realtime_score,
        "parse_request",
        return_value=records,
    )
    mock_score = mocker.patch.object(
        realtime_score,
        "score_record",
        side_effect=[
            {
                "Store": 1,
                "Feature_Date": "01-01-2024",
                "Prediction_Date": "08-01-2024",
                "Predicted_Weekly_Sales": 1100.0,
            },
            {
                "Store": 2,
                "Feature_Date": "08-01-2024",
                "Prediction_Date": "15-01-2024",
                "Predicted_Weekly_Sales": 2200.0,
            },
        ],
    )

    raw_data = {
        "input_data": [
            {"Store": 1, "Date": "01-01-2024"},
            {"Store": 2, "Date": "08-01-2024"},
        ]
    }

    result = realtime_score.run(raw_data)

    assert result == {
        "predictions": [
            {
                "Store": 1,
                "Feature_Date": "01-01-2024",
                "Prediction_Date": "08-01-2024",
                "Predicted_Weekly_Sales": 1100.0,
            },
            {
                "Store": 2,
                "Feature_Date": "08-01-2024",
                "Prediction_Date": "15-01-2024",
                "Predicted_Weekly_Sales": 2200.0,
            },
        ]
    }

    mock_parse.assert_called_once_with(raw_data)
    assert mock_score.call_args_list == [
        mocker.call(
            store_id=1,
            requested_date=pd.Timestamp("2024-01-01"),
        ),
        mocker.call(
            store_id=2,
            requested_date=pd.Timestamp("2024-01-08"),
        ),
    ]


def test_run_returns_structured_error_for_invalid_request(mocker):
    mocker.patch.object(
        realtime_score,
        "parse_request",
        side_effect=ValueError("Store must be between 1 and 45."),
    )

    result = realtime_score.run({
        "input_data": {
            "Store": 99,
            "Date": "01-01-2024",
        }
    })

    assert result == {
        "error": "ValueError",
        "message": "Store must be between 1 and 45.",
    }


def test_run_returns_error_for_invalid_json():
    result = realtime_score.run("{not-valid-json}")

    assert result["error"] == "JSONDecodeError"
    assert "Expecting property name" in result["message"]


def test_run_does_not_hide_unexpected_runtime_errors(mocker):
    mocker.patch.object(
        realtime_score,
        "parse_request",
        side_effect=RuntimeError("Model unavailable"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"Model unavailable",
    ):
        realtime_score.run({
            "Store": 1,
            "Date": "01-01-2024",
        })