"""Shared utilities for Walmart real-time and batch scoring entry scripts."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
import fsspec
import pandas as pd
import xgboost as xgb
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential


LOGGER = logging.getLogger("walmart-scoring")


def get_required_environment_variable(name: str) -> str:
    """Return a required, non-empty environment variable.

    Args:
        name: Name of the environment variable.

    Returns:
        str: The value, with surrounding whitespace removed.

    Raises:
        RuntimeError: If the variable is missing or empty.
    """

    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set."
        )
    return value


def find_model_file(
    model_directory: Path,
    filename: str = "model.xgb",
) -> Path:
    """Find a uniquely named model file beneath AZUREML_MODEL_DIR.

    Checks ``<model_directory>/model/<filename>`` first, then searches the
    whole folder.

    Args:
        model_directory: Value of ``AZUREML_MODEL_DIR``.
        filename: Model file name.

    Returns:
        Path: Path of the model file.

    Raises:
        FileNotFoundError: If no matching file exists.
        RuntimeError: If more than one matching file exists.
    """

    expected_path = model_directory / "model" / filename
    if expected_path.is_file():
        return expected_path

    matches = sorted(
        path for path in model_directory.rglob(filename) if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} beneath {model_directory}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple {filename!r} files were found beneath "
            f"{model_directory}: {matches}"
        )
    return matches[0]


def download_feature_csv(remote_uri: str, local_path: Path) -> Path:
    """Download one engineered-feature CSV to local worker storage.

    Args:
        remote_uri: Any URI fsspec can open, such as an ``azureml://`` path.
        local_path: Where to save the file. Its parent folder is created.

    Returns:
        Path: ``local_path``, after the download.

    Raises:
        FileNotFoundError: If the file was not written.
        RuntimeError: If the downloaded file is empty.
    """

    LOGGER.info("Downloading engineered-feature CSV from %s", remote_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with fsspec.open(remote_uri, mode="rb") as remote_file:
        with local_path.open(mode="wb") as local_file:
            shutil.copyfileobj(remote_file, local_file)

    if not local_path.is_file():
        raise FileNotFoundError(
            f"The CSV was not downloaded to {local_path}."
        )
    if local_path.stat().st_size == 0:
        raise RuntimeError(f"The downloaded CSV is empty: {local_path}.")

    LOGGER.info(
        "Downloaded engineered-feature CSV to %s; size=%d bytes",
        local_path,
        local_path.stat().st_size,
    )
    return local_path


def download_data_asset_csv(
    *,
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    data_asset_name: str,
    data_asset_version: str,
    local_path: Path,
) -> Path:
    """Resolve an immutable Azure ML URI-file asset and download its CSV.

    Authenticates with the endpoint's managed identity.

    Args:
        subscription_id: Azure subscription ID.
        resource_group_name: Resource group of the workspace.
        workspace_name: Azure ML workspace holding the data asset.
        data_asset_name: Data asset name.
        data_asset_version: Data asset version.
        local_path: Where to save the CSV.

    Returns:
        Path: Local path of the downloaded CSV.

    Raises:
        RuntimeError: If the asset has no storage path or the download fails.
        ValueError: If the asset is a folder rather than a single file.
    """

    ml_client = MLClient(
        credential=ManagedIdentityCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )
    data_asset = ml_client.data.get(
        name=data_asset_name,
        version=data_asset_version,
    )

    if not data_asset.path:
        raise RuntimeError(
            f"Data asset {data_asset_name}:{data_asset_version} "
            "does not contain a storage path."
        )

    asset_type = str(data_asset.type).lower()
    if "folder" in asset_type:
        raise ValueError(
            f"Data asset {data_asset_name}:{data_asset_version} is "
            f"{data_asset.type!s}, but this downloader requires a URI-file "
            "asset that points directly to one CSV."
        )

    LOGGER.info(
        "Resolved data asset %s:%s; type=%s; path=%s",
        data_asset.name,
        data_asset.version,
        data_asset.type,
        data_asset.path,
    )
    return download_feature_csv(data_asset.path, local_path)


def load_feature_data(csv_path: Path) -> pd.DataFrame:
    """Read, normalize, and validate materialized engineered features.

    Args:
        csv_path: Engineered-feature CSV with Store and DD-MM-YYYY Date columns.

    Returns:
        pd.DataFrame: Features sorted by Store and Date, with int64 Store and
        datetime Date.

    Raises:
        ValueError: If Store or Date is missing or invalid, or a Store/Date pair
            appears more than once.
    """

    dataframe = pd.read_csv(csv_path)
    missing_columns = {"Store", "Date"}.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Engineered-feature CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe = dataframe.copy()
    dataframe["Store"] = pd.to_numeric(
        dataframe["Store"], errors="raise"
    )
    if dataframe["Store"].isna().any():
        raise ValueError("Store contains null values.")
    if (dataframe["Store"] % 1 != 0).any():
        raise ValueError("Store must contain integer IDs.")
    dataframe["Store"] = dataframe["Store"].astype("int64")

    dataframe["Date"] = pd.to_datetime(
        dataframe["Date"],
        format="%d-%m-%Y",
        errors="raise",
    ).dt.normalize()

    duplicate_rows = dataframe.duplicated(
        subset=["Store", "Date"],
        keep=False,
    )
    if duplicate_rows.any():
        examples = (
            dataframe.loc[duplicate_rows, ["Store", "Date"]]
            .drop_duplicates()
            .head(10)
            .assign(Date=lambda values: values["Date"].dt.strftime("%d-%m-%Y"))
            .to_dict("records")
        )
        raise ValueError(
            "The engineered-feature CSV contains duplicate Store/Date rows. "
            f"Examples: {examples}"
        )

    dataframe.sort_values(["Store", "Date"], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    LOGGER.info(
        "Loaded %d engineered-feature rows for %d stores; date range=%s to %s",
        len(dataframe),
        dataframe["Store"].nunique(),
        dataframe["Date"].min().strftime("%d-%m-%Y"),
        dataframe["Date"].max().strftime("%d-%m-%Y"),
    )
    return dataframe


def load_xgboost_model(
    model_directory: Path,
    filename: str = "model.xgb",
) -> xgb.Booster:
    """Find and load an XGBoost Booster.

    Args:
        model_directory: Value of ``AZUREML_MODEL_DIR``.
        filename: Model file name.

    Returns:
        xgb.Booster: The loaded model.
    """

    model_path = find_model_file(model_directory, filename)
    LOGGER.info("Loading XGBoost model from %s", model_path)

    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster
