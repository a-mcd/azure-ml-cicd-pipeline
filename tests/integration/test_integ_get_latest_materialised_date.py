"""Integration test for the latest materialised date script."""

from pathlib import Path

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from src.pipeline_scripts import (
    get_latest_materialised_date as latest_date,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def local_delta_spark():
    """Create a local Spark session with Delta support."""
    builder = (
        SparkSession.builder
        .master("local[1]")
        .appName("latest-materialised-date-integration-test")
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


def test_reads_latest_date_from_delta_and_writes_github_env(
    mocker,
    monkeypatch,
    tmp_path: Path,
    local_delta_spark,
    capsys,
):
    """Find the latest Delta date and export it to GITHUB_ENV."""
    delta_path = tmp_path / "materialized-features.delta"
    github_env_path = tmp_path / "github-env.txt"
    github_env_path.touch()

    source_rows = [
        {
            "Store": 1,
            "Date": "2012-10-26",
            "Weekly_Sales": 100000.0,
        },
        {
            "Store": 1,
            "Date": "2012-11-02",
            "Weekly_Sales": 110000.0,
        },
        {
            "Store": 2,
            "Date": "2012-11-09",
            "Weekly_Sales": 200000.0,
        },
        {
            "Store": 2,
            "Date": "2012-10-19",
            "Weekly_Sales": 190000.0,
        },
    ]

    source_dataframe = local_delta_spark.createDataFrame(
        source_rows
    )

    (
        source_dataframe.write
        .format("delta")
        .mode("overwrite")
        .save(str(delta_path))
    )

    monkeypatch.setenv(
        "FEATURE_SET_NAME",
        "walmart_sales_features",
    )
    monkeypatch.setenv("FEATURE_SET_VERSION", "1")
    monkeypatch.setenv(
        "OFFLINE_STORAGE_ACCOUNT",
        "integrationstorage",
    )
    monkeypatch.setenv(
        "OFFLINE_STORAGE_CONTAINER",
        "offline-store",
    )
    monkeypatch.setenv(
        "AZURE_STORAGE_KEY",
        "integration-test-key",
    )
    monkeypatch.setenv(
        "GITHUB_ENV",
        str(github_env_path),
    )

    mock_find_path = mocker.patch.object(
        latest_date,
        "find_materialized_delta_path",
        return_value=str(delta_path),
    )
    mock_create_session = mocker.patch.object(
        latest_date,
        "create_delta_spark_session",
        return_value=local_delta_spark,
    )
    mock_stop = mocker.spy(local_delta_spark, "stop")

    latest_date.main()

    captured = capsys.readouterr()

    assert "09-11-2012" in captured.out

    github_environment = github_env_path.read_text(
        encoding="utf-8"
    )

    assert (
        "LATEST_PROCESSED_DATE=09-11-2012"
        in github_environment
    )

    mock_find_path.assert_called_once_with(
        storage_account="integrationstorage",
        storage_container="offline-store",
        feature_set_name="walmart_sales_features",
        feature_set_version="1",
    )

    mock_create_session.assert_called_once_with(
        application_name="get-latest-materialized-date",
        storage_account="integrationstorage",
        storage_key="integration-test-key",
    )

    mock_stop.assert_called_once_with()