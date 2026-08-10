"""Integration tests for exporting materialised Delta features."""

import os
from pathlib import Path

import pandas as pd
import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


# The production module validates these variables during import.
os.environ.setdefault(
    "FEATURE_SET_NAME",
    "walmart_sales_features",
)
os.environ.setdefault("FEATURE_SET_VERSION", "1")
os.environ.setdefault(
    "OFFLINE_STORAGE_ACCOUNT",
    "integrationstorage",
)
os.environ.setdefault(
    "OFFLINE_STORAGE_CONTAINER",
    "offline-store",
)

from src.pipeline_scripts import (  # noqa: E402
    create_all_weeks_features as exporter,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def delta_table_path(tmp_path: Path) -> Path:
    """Return the path used for the local Delta table."""
    return tmp_path / "materialized-features.delta"


@pytest.fixture
def local_delta_spark():
    """Create a local Spark session with Delta support."""
    builder = (
        SparkSession.builder
        .master("local[1]")
        .appName("create-all-weeks-integration-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(
        builder
    ).getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    yield spark

    # main() also stops Spark, but calling stop again is safe.
    spark.stop()


def test_exports_all_valid_delta_rows_to_csv(
    mocker,
    monkeypatch,
    tmp_path,
    local_delta_spark,
    delta_table_path,
    capsys,
):
    """Read a real Delta table, transform it, and export CSV."""
    source_rows = [
        {
            "Store": 1,
            "Date": "2012-10-26",
            "Temperature": 60.0,
            "Fuel_Price": 3.50,
            "CPI": 220.0,
            "Unemployment": 7.5,
            "lag_1": 100000.0,
        },
        {
            "Store": 1,
            "Date": "2012-11-02",
            "Temperature": 61.0,
            "Fuel_Price": 3.51,
            "CPI": 220.1,
            "Unemployment": 7.4,
            "lag_1": 101000.0,
        },
        {
            "Store": 2,
            "Date": "2012-11-09",
            "Temperature": 55.0,
            "Fuel_Price": 3.45,
            "CPI": 219.5,
            "Unemployment": 7.7,
            "lag_1": 90000.0,
        },
        {
            "Store": 2,
            "Date": "invalid-date",
            "Temperature": 56.0,
            "Fuel_Price": 3.46,
            "CPI": 219.6,
            "Unemployment": 7.6,
            "lag_1": 91000.0,
        },
    ]

    source_dataframe = local_delta_spark.createDataFrame(
        source_rows
    )

    (
        source_dataframe.write
        .format("delta")
        .mode("overwrite")
        .save(str(delta_table_path))
    )

    output_path = tmp_path / "exports" / "features.csv"

    monkeypatch.setenv(
        "AZURE_STORAGE_KEY",
        "integration-test-key",
    )

    mock_create_session = mocker.patch.object(
        exporter,
        "create_delta_spark_session",
        return_value=local_delta_spark,
    )
    mock_find_path = mocker.patch.object(
        exporter,
        "find_materialized_delta_path",
        return_value=str(delta_table_path),
    )
    mocker.patch.object(
        exporter,
        "OUTPUT_PATH",
        output_path,
    )

    exporter.main()

    assert output_path.is_file()

    exported = pd.read_csv(output_path)

    # The invalid date row should have been removed.
    assert len(exported) == 3

    assert exported["Store"].tolist() == [1, 1, 2]

    # Rows are ordered by Store ascending and Date descending.
    assert exported["Date"].tolist() == [
        "02-11-2012",
        "26-10-2012",
        "09-11-2012",
    ]

    assert exported["Temperature"].tolist() == [
        61.0,
        60.0,
        55.0,
    ]
    assert exported["lag_1"].tolist() == [
        101000.0,
        100000.0,
        90000.0,
    ]

    assert "_parsed_date" not in exported.columns

    mock_create_session.assert_called_once_with(
        application_name="export-walmart-all-weeks",
        storage_account="integrationstorage",
        storage_key="integration-test-key",
    )
    mock_find_path.assert_called_once_with()

    output = capsys.readouterr().out

    assert "Reading materialised features from:" in output
    assert str(delta_table_path) in output
    assert "Materialised feature schema:" in output
    assert (
        "Writing all materialised weeks for every store to CSV"
        in output
    )
    assert "CSV written to:" in output
    assert "Materialised date range:" in output
    assert "Number of exported rows per store:" in output