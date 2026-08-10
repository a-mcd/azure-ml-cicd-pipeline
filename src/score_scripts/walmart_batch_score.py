"""Azure ML batch scoring entry script for Walmart weekly sales.

Each file supplied to ``run(mini_batch)`` must be a CSV containing ``Store``
and ``Date``, where ``Date`` uses ``DD-MM-YYYY`` format.
During ``init()``, each worker downloads a registered URI-file
data asset containing materialised engineered features, then performs a
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
import sys
import logging
import os
import tempfile
from typing import Iterable
import pandas as pd
import xgboost as xgb

# Azure ML loads this entry script directly and may not add the packaged code
# root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This import must follow the sys.path configuration above.
# pylint: disable=wrong-import-position
from src.common.scoring_helpers.scoring_utils import (
    download_data_asset_csv,
    get_required_environment_variable,
    load_feature_data,
    load_xgboost_model,
)

# Azure ML ParallelRun loads the entry script directly and does not always add
# the entry script's directory to sys.path. Add it so sibling modules packaged
# in the same code directory can be imported reliably.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

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
    stores = pd.to_numeric(series, errors="raise")
    if stores.isna().any():
        raise ValueError("Store contains null values.")
    if (stores % 1 != 0).any():
        raise ValueError("Store must contain integer IDs.")
    return stores.astype("int64")


def _normalise_date(series: pd.Series) -> pd.Series:
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

    return download_data_asset_csv(
        subscription_id=get_required_environment_variable(
            "AZURE_SUBSCRIPTION_ID"
        ),
        resource_group_name=get_required_environment_variable(
            "AZURE_RESOURCE_GROUP"
        ),
        workspace_name=get_required_environment_variable(
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
    available = set(available_columns)
    names = booster.feature_names or DEFAULT_MODEL_FEATURES
    missing = set(names).difference(available)

    if missing:
        raise ValueError(
            "Materialised feature data does not contain all model features. "
            f"Missing: {sorted(missing)}"
        )
    return list(names)


def init() -> None:
    """Load the model and materialised features once per batch worker."""
    global model, feature_lookup, model_feature_names # pylint: disable=global-statement

    model_directory = Path(
        get_required_environment_variable("AZUREML_MODEL_DIR")
    )
    model = load_xgboost_model(
        model_directory,
        os.getenv("MODEL_FILENAME", "model.xgb"),
    )

    feature_csv = _download_feature_asset()
    feature_lookup = (
        load_feature_data(feature_csv)
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
    """Score every Store/Date row in the current Azure ML mini-batch."""
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
        examples = requests.loc[missing_lookup, KEY_COLUMNS].head(10).copy()
        examples["Date"] = examples["Date"].dt.strftime("%d-%m-%Y")
        raise ValueError(
            "No materialised engineered features were found for "
            f"{int(missing_lookup.sum())} input row(s). "
            f"Examples: {examples.to_dict('records')}"
        )

    incomplete = joined[model_feature_names].isna()
    if incomplete.any().any():
        examples: list[dict[str, object]] = []
        for index in incomplete.index[incomplete.any(axis=1)][:10]:
            examples.append(
                {
                    "Store": int(joined.at[index, "Store"]),
                    "Date": joined.at[index, "Date"].strftime("%d-%m-%Y"),
                    "missing_features": incomplete.columns[
                        incomplete.loc[index]
                    ].tolist(),
                }
            )
        raise ValueError(
            "Materialised rows contain null model features. "
            f"Examples: {examples}"
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
