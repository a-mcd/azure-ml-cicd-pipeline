"""Azure ML real-time scoring for Walmart weekly sales.

The script:

1. Reads the model and data-asset settings from environment variables.
2. Resolves the specified Azure ML data asset version.
3. Downloads the engineered-feature CSV during container initialization.
4. Selects a feature row using an exact Store and Date match.
5. Passes the engineered features to the XGBoost model.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any
import pandas as pd
import xgboost as xgb

# Azure ML loads this entry script directly and may not add the packaged code
# root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This import must follow the sys.path configuration above.
# pylint: disable=wrong-import-position
from walmart_ml.common.scoring_helpers.scoring_utils import (
    download_data_asset_csv,
    get_required_environment_variable,
    load_feature_data,
    load_xgboost_model,
)


LOGGER = logging.getLogger("walmart-scoring")
LOGGER.setLevel(logging.INFO)


# Initialized once when the inference container starts.
model: xgb.Booster | None = None
feature_data: pd.DataFrame | None = None
model_feature_names: list[str] = []


def init() -> None:
    """Load and cache the model and engineered-feature CSV."""

    global model, feature_data, model_feature_names # pylint: disable=global-statement

    LOGGER.info("Starting Walmart endpoint initialization.")

    subscription_id = get_required_environment_variable(
        "AML_SUBSCRIPTION_ID"
    )
    resource_group = get_required_environment_variable(
        "DATA_WORKSPACE_RG"
    )
    workspace_name = get_required_environment_variable(
        "DATA_WORKSPACE_NAME"
    )
    data_asset_name = get_required_environment_variable(
        "FEATURE_DATA_ASSET_NAME"
    )
    data_asset_version = get_required_environment_variable(
        "FEATURE_DATA_ASSET_VERSION"
    )
    azureml_model_directory = get_required_environment_variable(
        "AZUREML_MODEL_DIR"
    )

    LOGGER.info(
        "Workspace: subscription=%s, resource-group=%s, workspace=%s",
        subscription_id,
        resource_group,
        workspace_name,
    )

    LOGGER.info(
        "Engineered-feature asset: %s:%s",
        data_asset_name,
        data_asset_version,
    )

    local_csv_path = Path(
        "/tmp/walmart_materialized_features.csv"
    )

    download_data_asset_csv(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        data_asset_name=data_asset_name,
        data_asset_version=data_asset_version,
        local_path=local_csv_path,
    )

    feature_data = load_feature_data(local_csv_path)
    model = load_xgboost_model(Path(azureml_model_directory))

    model_feature_names = list(model.feature_names or [])

    if not model_feature_names:
        raise RuntimeError(
            "The XGBoost model does not contain feature names."
        )

    missing_model_features = [
        feature_name
        for feature_name in model_feature_names
        if feature_name not in feature_data.columns
    ]

    if missing_model_features:
        raise ValueError(
            "The engineered-feature CSV does not contain the "
            "following model features: "
            + ", ".join(missing_model_features)
        )

    LOGGER.info(
        "Model expects %d features: %s",
        len(model_feature_names),
        model_feature_names,
    )

    LOGGER.info("Walmart endpoint initialization completed.")


def parse_date(value: Any) -> pd.Timestamp:
    """Parse and normalize an input date."""

    if value is None:
        raise ValueError("Date cannot be null.")

    parsed_date = pd.to_datetime(
        value,
        format="%d-%m-%Y",
        errors="raise",
    )

    if isinstance(parsed_date, pd.DatetimeIndex):
        raise ValueError(
            "Date must contain one date value."
        )

    return pd.Timestamp(parsed_date).normalize()


def parse_request(raw_data: Any) -> list[dict[str, Any]]:
    """Parse and validate the endpoint request."""

    if isinstance(raw_data, bytes):
        raw_data = raw_data.decode("utf-8")

    if isinstance(raw_data, str):
        request = json.loads(raw_data)
    elif isinstance(raw_data, dict):
        request = raw_data
    else:
        raise ValueError(
            "The request must be a JSON object."
        )

    records = request.get("input_data", request)

    if isinstance(records, dict):
        records = [records]

    if not isinstance(records, list) or not records:
        raise ValueError(
            "'input_data' must contain an object or a non-empty list."
        )

    parsed_records = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Input record {index} must be a JSON object."
            )

        if "Store" not in record:
            raise ValueError(
                f"Input record {index} does not contain 'Store'."
            )

        if "Date" not in record:
            raise ValueError(
                f"Input record {index} does not contain 'Date'."
            )

        try:
            store_id = int(record["Store"])
        except (TypeError, ValueError) as exception:
            raise ValueError(
                f"Invalid Store value: {record['Store']}."
            ) from exception

        if not 1 <= store_id <= 45:
            raise ValueError(
                f"Store must be between 1 and 45; "
                f"received {store_id}."
            )

        requested_date = parse_date(record["Date"])

        parsed_records.append(
            {
                "Store": store_id,
                "Date": requested_date,
            }
        )

    return parsed_records


def select_engineered_features(
    store_id: int,
    requested_date: pd.Timestamp,
) -> pd.Series:
    """Select a feature row using an exact Store and Date match."""

    if feature_data is None:
        raise RuntimeError(
            "Engineered-feature data has not been initialized."
        )

    requested_date = requested_date.normalize()

    matching_rows = feature_data.loc[
        (feature_data["Store"] == store_id)
        & (feature_data["Date"] == requested_date)
    ]

    if matching_rows.empty:
        available_dates = (
            feature_data.loc[
                feature_data["Store"] == store_id,
                "Date",
            ]
            .dt.strftime("%d-%m-%Y")
            .tolist()
        )

        if not available_dates:
            raise ValueError(
                f"No engineered-feature rows exist for Store {store_id}."
            )

        raise ValueError(
            f"No engineered-feature row found for Store {store_id} "
            f"and Date {requested_date:%d-%m-%Y}. "
            f"Available dates for this store: "
            f"{', '.join(available_dates)}."
        )

    if len(matching_rows) > 1:
        raise ValueError(
            f"Multiple engineered-feature rows found for Store "
            f"{store_id} and Date {requested_date:%d-%m-%Y}."
        )

    return matching_rows.iloc[0].copy()


def build_model_input(
    selected_row: pd.Series,
) -> pd.DataFrame:
    """Build a DataFrame containing the model's expected features."""

    feature_values = {
        feature_name: selected_row[feature_name]
        for feature_name in model_feature_names
    }

    model_input = pd.DataFrame(
        [feature_values],
        columns=model_feature_names,
    )

    model_input = model_input.apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_columns = model_input.columns[
        model_input.isna().any()
    ].tolist()

    if invalid_columns:
        raise ValueError(
            "The selected row contains null or nonnumeric values for: "
            + ", ".join(invalid_columns)
        )

    return model_input


def score_record(
    store_id: int,
    requested_date: pd.Timestamp,
) -> dict[str, Any]:
    """Use one week's features to predict the following week's sales."""

    if model is None:
        raise RuntimeError(
            "The XGBoost model has not been initialized."
        )

    selected_row = select_engineered_features(
        store_id=store_id,
        requested_date=requested_date,
    )

    model_input = build_model_input(
        selected_row
    )

    prediction = model.predict(
        xgb.DMatrix(
            model_input,
            feature_names=model_feature_names,
        )
    )[0]

    prediction_date = requested_date + pd.Timedelta(weeks=1)

    return {
        "Store": store_id,
        "Feature_Date": requested_date.strftime("%d-%m-%Y"),
        "Prediction_Date": prediction_date.strftime("%d-%m-%Y"),
        "Predicted_Weekly_Sales": float(prediction),
    }


def run(raw_data: Any) -> dict[str, Any]:
    """Handle an endpoint scoring request."""

    try:
        records = parse_request(raw_data)

        predictions = [
            score_record(
                store_id=record["Store"],
                requested_date=record["Date"],
            )
            for record in records
        ]

        return {
            "predictions": predictions,
        }

    except ValueError as exception:
        LOGGER.exception(
            "Prediction request failed."
        )

        return {
            "error": type(exception).__name__,
            "message": str(exception),
        }
