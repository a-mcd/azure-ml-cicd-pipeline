"""Unit tests for materialized feature-store utilities."""

import subprocess

import pytest
from walmart_ml.common.pipeline_helpers import materialized_store



def test_list_storage_files_returns_clean_non_empty_paths(mocker):
    # Purpose: Check that Azure CLI output is split into clean file paths
    # while blank lines are ignored.
    mock_run = mocker.patch(
        "walmart_ml.common.pipeline_helpers.materialized_store.subprocess.run",
    )
    mock_run.return_value.stdout = (
        "features/table/_delta_log/00001.json\n"
        "\n"
        "  features/table/part-00001.parquet  \n"
        "   \n"
    )

    result = materialized_store.list_storage_files(
        storage_account="teststorage",
        storage_container="test-container",
    )

    assert result == [
        "features/table/_delta_log/00001.json",
        "features/table/part-00001.parquet",
    ]


def test_list_storage_files_runs_expected_azure_cli_command(mocker):
    # Purpose: Check that files are requested recursively from the specified
    # Azure Data Lake filesystem.
    mock_run = mocker.patch(
        "walmart_ml.common.pipeline_helpers.materialized_store.subprocess.run",
    )
    mock_run.return_value.stdout = ""

    result = materialized_store.list_storage_files(
        storage_account="walmartlake",
        storage_container="walmart-source",
    )

    mock_run.assert_called_once_with(
        [
            "az",
            "storage",
            "fs",
            "file",
            "list",
            "--account-name",
            "walmartlake",
            "--file-system",
            "walmart-source",
            "--recursive",
            "true",
            "--exclude-dir",
            "--num-results",
            "5000",
            "--query",
            "[].name",
            "--output",
            "tsv",
            "--only-show-errors",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result == []


def test_list_storage_files_propagates_azure_cli_failure(mocker):
    # Purpose: Check that Azure CLI failures are not silently ignored.
    mocker.patch(
        "walmart_ml.common.pipeline_helpers.materialized_store.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1,
            cmd=["az", "storage", "fs", "file", "list"],
        ),
    )

    with pytest.raises(subprocess.CalledProcessError):
        materialized_store.list_storage_files(
            storage_account="walmartlake",
            storage_container="walmart-source",
        )


def test_create_delta_spark_session_configures_and_creates_session(mocker):
    # Purpose: Check that the Spark builder receives the required Delta and
    # Azure Storage configuration before the session is created.
    spark_builder = mocker.MagicMock()
    spark_builder.master.return_value = spark_builder
    spark_builder.appName.return_value = spark_builder
    spark_builder.config.return_value = spark_builder

    mocker.patch.object(
        materialized_store.SparkSession,
        "builder",
        spark_builder,
    )

    configured_builder = mocker.MagicMock()
    expected_session = mocker.sentinel.spark_session
    configured_builder.getOrCreate.return_value = expected_session

    mock_configure_delta = mocker.patch(
        "walmart_ml.common.pipeline_helpers.materialized_store."
        "configure_spark_with_delta_pip",
        return_value=configured_builder,
    )

    result = materialized_store.create_delta_spark_session(
        application_name="Walmart feature export",
        storage_account="walmartlake",
        storage_key="secret-storage-key",
    )

    assert result is expected_session

    spark_builder.master.assert_called_once_with("local[*]")
    spark_builder.appName.assert_called_once_with(
        "Walmart feature export",
    )

    assert spark_builder.config.call_args_list == [
        mocker.call(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        ),
        mocker.call(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        ),
        mocker.call(
            (
                "spark.hadoop.fs.azure.account.auth.type."
                "walmartlake.dfs.core.windows.net"
            ),
            "SharedKey",
        ),
        mocker.call(
            (
                "spark.hadoop.fs.azure.account.key."
                "walmartlake.dfs.core.windows.net"
            ),
            "secret-storage-key",
        ),
    ]

    mock_configure_delta.assert_called_once_with(
        spark_builder,
        extra_packages=[
            "org.apache.hadoop:hadoop-azure:3.3.4",
        ],
    )
    configured_builder.getOrCreate.assert_called_once_with()


def test_find_delta_roots_returns_sorted_unique_roots():
    # Purpose: Check that unique Delta-table roots are extracted and sorted.
    storage_files = [
        "features/store-b/_delta_log/00000000000000000001.json",
        "features/store-a/_delta_log/00000000000000000000.json",
        "features/store-b/_delta_log/00000000000000000000.json",
        "features/store-a/part-00001.parquet",
    ]

    result = materialized_store.find_delta_roots(storage_files)

    assert result == [
        "features/store-a",
        "features/store-b",
    ]


def test_find_delta_roots_ignores_non_json_delta_log_files():
    # Purpose: Check that checkpoints and other Delta log artifacts are not
    # incorrectly treated as JSON transaction-log entries.
    storage_files = [
        "features/table/_delta_log/00000000000000000000.checkpoint.parquet",
        "features/table/_delta_log/_last_checkpoint",
        "features/table/_delta_log/00000000000000000001.crc",
        "features/table/part-00001.parquet",
    ]

    assert materialized_store.find_delta_roots(storage_files) == []


def test_find_delta_roots_returns_empty_list_when_no_delta_tables_exist():
    # Purpose: Check that storage without Delta transaction logs produces an
    # empty result.
    storage_files = [
        "raw/Walmart_Sales.csv",
        "features/part-00001.parquet",
    ]

    assert materialized_store.find_delta_roots(storage_files) == []