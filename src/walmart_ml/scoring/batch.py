"""Azure ML batch scoring entry script for Walmart weekly sales.

Each file supplied to ``run(mini_batch)`` must be a CSV containing ``Store``
and ``Date``, where ``Date`` uses ``DD-MM-YYYY`` format.
During ``init()``, each worker downloads a registered URI-file
data asset containing materialized engineered features, then performs a
Store/Date lookup and scores the matching rows with XGBoost.

Required deployment environment variables:
    AZURE_SUBSCRIPTION_ID
    AZURE_RESOURCE_GROUP
    AZURE_WORKSPACE_NAME

Optional deployment environment variables:
    FEATURE_DATA_ASSET_NAME       default: walmart_materialized_features
    FEATURE_DATA_ASSET_VERSION    default: 1
    FEATURE_DATA_FILENAME         exact CSV filename when the asset is a folder
    MODEL_FILENAME                default: model.xgb
    DATE_OUTPUT_FORMAT            default: %d-%m-%Y
"""

from __future__ import annotations
from pathlib import Path
import logging
import os
import sys
import tempfile
from typing import Iterable
import pandas as pd
import xgboost as xgb

# Azure ML copies the deployment's code folder (src) into the container and
# loads this file by path, without putting src on sys.path, so
# "import walmart_ml" would fail. parents[2] is src:
# src/walmart_ml/scoring/batch.py -> scoring -> walmart_ml -> src.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# This import must follow the sys.path setup above.
# pylint: disable=wrong-import-position
from walmart_ml.scoring import scoring_utils as utils

LOGGER = logging.getLogger(__name__)

KEY_COLUMNS = ["Store", "Date"]
OUTPUT_COLUMNS = [
    "Store",
    "Feature_Date",
    "Prediction_Date",
    "Predicted_Weekly_Sales",
]

DEFAULT_MODEL_FEATURES = [
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "lag_1",
    "lag_2",
    "lag_52",
    "rollmean_3",
    "rollmean_6",
    "rollmean_12",
    "month",
    "week",
    "quarter",
    "year",
    *[f"S_{store_id}" for store_id in range(1, 46)],
]

model: xgb.Booster
feature_lookup: pd.DataFrame
model_feature_names: list[str]


def _normalise_store(series: pd.Series) -> pd.Series:
    """Convert Store values to int64 store IDs.

    Args:
        series: Raw Store values.

    Returns:
        pd.Series: Store IDs as int64.

    Raises:
        ValueError: If a value is null, non-numeric or not a whole number.
    """
    stores = pd.to_numeric(series, errors="raise")
    if stores.isna().any():
        raise ValueError("Store contains null values.")
    if (stores % 1 != 0).any():
        raise ValueError("Store must contain integer IDs.")
    return stores.astype("int64")


def _normalise_date(series: pd.Series) -> pd.Series:
    """Parse DD-MM-YYYY dates into timezone-naive midnight timestamps.

    Args:
        series: Raw Date values.

    Returns:
        pd.Series: Normalised datetime values.

    Raises:
        ValueError: If a value is null or not in DD-MM-YYYY format.
    """
    dates = pd.to_datetime(
        series.astype(str).str.strip(),
        format="%d-%m-%Y",
        errors="raise",
        utc=True,
    )

    if dates.isna().any():
        raise ValueError("Date contains null values.")

    return dates.dt.tz_convert(None).dt.normalize()


def _download_feature_asset() -> Path:
    """Download the engineered-feature data asset to a temporary folder.

    The asset name, version and filename come from environment variables, with
    the defaults listed in the module docstring.

    Returns:
        Path: Local path of the downloaded CSV.
    """
    asset_name = os.getenv(
        "FEATURE_DATA_ASSET_NAME", "walmart_materialized_features"
    )
    asset_version = os.getenv("FEATURE_DATA_ASSET_VERSION", "1")
    filename = os.getenv(
        "FEATURE_DATA_FILENAME",
        "walmart_materialized_features.csv",
    )
    worker_directory = Path(
        tempfile.mkdtemp(prefix="walmart_batch_features_")
    )

    return utils.download_data_asset_csv(
        subscription_id=utils.get_required_environment_variable(
            "AZURE_SUBSCRIPTION_ID"
        ),
        resource_group_name=utils.get_required_environment_variable(
            "AZURE_RESOURCE_GROUP"
        ),
        workspace_name=utils.get_required_environment_variable(
            "AZURE_WORKSPACE_NAME"
        ),
        data_asset_name=asset_name,
        data_asset_version=asset_version,
        local_path=worker_directory / filename,
    )


def _resolve_model_features(
    booster: xgb.Booster,
    available_columns: Iterable[str],
) -> list[str]:
    """Return the model's feature names, checking the feature data has them all.

    Args:
        booster: Loaded XGBoost model.
        available_columns: Columns in the engineered-feature data.

    Returns:
        list[str]: Feature names in the order the model expects. Falls back to
        DEFAULT_MODEL_FEATURES if the model has none saved.

    Raises:
        ValueError: If any model feature is missing from the data.
    """
    available = set(available_columns)
    names = booster.feature_names or DEFAULT_MODEL_FEATURES
    missing = set(names).difference(available)

    if missing:
        raise ValueError(
            "Materialized feature data does not contain all model features. "
            f"Missing: {sorted(missing)}"
        )
    return list(names)


def init() -> None:
    """Load the model and materialized features once per batch worker.

    Called by Azure ML when the worker starts.
    """
    global model, feature_lookup, model_feature_names # pylint: disable=global-statement

    model_directory = Path(
        utils.get_required_environment_variable("AZUREML_MODEL_DIR")
    )
    model = utils.load_xgboost_model(
        model_directory,
        os.getenv("MODEL_FILENAME", "model.xgb"),
    )

    feature_csv = _download_feature_asset()
    feature_lookup = (
        utils.load_feature_data(feature_csv)
        .set_index(KEY_COLUMNS)
        .sort_index()
    )
    model_feature_names = _resolve_model_features(model, feature_lookup.columns)

    LOGGER.info(
        "Initialisation complete: model=%s, feature_data=%s, rows=%d, features=%d",
        model_directory,
        feature_csv,
        len(feature_lookup),
        len(model_feature_names),
    )


def _read_batch_files(mini_batch: Iterable[str]) -> pd.DataFrame:
    """Read and combine the Store/Date rows from every file in a mini-batch.

    Args:
        mini_batch: Paths of the input CSV files.

    Returns:
        pd.DataFrame: Store, Date and ``_request_order`` columns, in input order.

    Raises:
        ValueError: If a file is missing Store or Date, or has invalid values.
    """
    frames: list[pd.DataFrame] = []

    for input_path in mini_batch:
        frame = pd.read_csv(input_path)
        missing_keys = set(KEY_COLUMNS).difference(frame.columns)
        if missing_keys:
            raise ValueError(
                f"Input file {input_path!r} is missing columns: "
                f"{sorted(missing_keys)}"
            )

        frame = frame.loc[:, KEY_COLUMNS].copy()
        frame["Store"] = _normalise_store(frame["Store"])
        frame["Date"] = _normalise_date(frame["Date"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS + ["_request_order"])

    requests = pd.concat(frames, ignore_index=True)
    requests["_request_order"] = range(len(requests))
    return requests


def run(mini_batch: list[str]) -> pd.DataFrame:
    """Score every Store/Date row in the current Azure ML mini-batch.

    Called by Azure ML once per mini-batch, after :func:`init`.

    Args:
        mini_batch: Paths of the input CSV files.

    Returns:
        pd.DataFrame: One prediction per input row, in input order, with
        OUTPUT_COLUMNS.

    Raises:
        ValueError: If a row has no matching engineered features, or the
            matching row has null model features.
    """
    requests = _read_batch_files(mini_batch)
    if requests.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    joined = requests.join(
        feature_lookup,
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    missing_lookup = ~pd.MultiIndex.from_frame(
        requests[KEY_COLUMNS]
    ).isin(feature_lookup.index)
    if missing_lookup.any():
        missing_rows = requests.loc[missing_lookup, KEY_COLUMNS].head(10).copy()
        missing_rows["Date"] = missing_rows["Date"].dt.strftime("%d-%m-%Y")
        raise ValueError(
            "No materialized engineered features were found for "
            f"{int(missing_lookup.sum())} input row(s). "
            f"Examples: {missing_rows.to_dict('records')}"
        )

    incomplete = joined[model_feature_names].isna()
    if incomplete.any().any():
        null_feature_rows: list[dict[str, object]] = []
        for index in incomplete.index[incomplete.any(axis=1)][:10]:
            null_feature_rows.append(
                {
                    "Store": int(joined.at[index, "Store"]),
                    "Date": joined.at[index, "Date"].strftime("%d-%m-%Y"),
                    "missing_features": incomplete.columns[
                        incomplete.loc[index]
                    ].tolist(),
                }
            )
        raise ValueError(
            "Materialized rows contain null model features. "
            f"Examples: {null_feature_rows}"
        )

    matrix = xgb.DMatrix(
        joined.loc[:, model_feature_names],
        feature_names=model_feature_names,
    )
    predictions = model.predict(matrix)

    output_format = os.getenv("DATE_OUTPUT_FORMAT", "%d-%m-%Y")
    result = pd.DataFrame(
        {
            "Store": joined["Store"].astype("int64"),
            "Feature_Date": joined["Date"].dt.strftime(output_format),
            "Prediction_Date": (
                joined["Date"] + pd.Timedelta(days=7)
            ).dt.strftime(output_format),
            "Predicted_Weekly_Sales": predictions,
            "_request_order": joined["_request_order"],
        }
    )

    result = result.sort_values("_request_order").drop(columns="_request_order")

    return result[OUTPUT_COLUMNS].reset_index(drop=True)
