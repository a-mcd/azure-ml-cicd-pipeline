"""Unit tests for exporting all materialized feature-store weeks."""

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_NAME = "walmart_ml.pipeline.create_all_weeks_features"


@pytest.fixture
def export_script(monkeypatch):
    """Import the script after configuring its required environment."""
    monkeypatch.setenv(
        "FEATURE_SET_NAME",
        "walmart_sales_features",
    )
    monkeypatch.setenv("FEATURE_SET_VERSION", "1")
    monkeypatch.setenv(
        "OFFLINE_STORAGE_ACCOUNT",
        "walmartstorage",
    )
    monkeypatch.setenv(
        "OFFLINE_STORAGE_CONTAINER",
        "offline-store",
    )

    # Force a fresh import so module-level environment variables are read
    # from the environment configured for this test.
    sys.modules.pop(MODULE_NAME, None)
    module = importlib.import_module(MODULE_NAME)

    yield module

    sys.modules.pop(MODULE_NAME, None)


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------


def test_require_environment_variable_returns_value(export_script):
    result = export_script.require_environment_variable(
        "TEST_VARIABLE",
        "test-value",
    )

    assert result == "test-value"


@pytest.mark.parametrize("value", [None, ""])
def test_require_environment_variable_raises_for_missing_value(
    export_script,
    value,
):
    with pytest.raises(
        RuntimeError,
        match=r"TEST_VARIABLE is not set\.",
    ):
        export_script.require_environment_variable(
            "TEST_VARIABLE",
            value,
        )


# ---------------------------------------------------------------------------
# Delta path discovery
# ---------------------------------------------------------------------------


def test_find_materialized_delta_path_returns_matching_abfss_uri(
    mocker,
    export_script,
    capsys,
):
    storage_files = [
        (
            "azureml/"
            "walmart_sales_features/1/"
            "_delta_log/00000000000000000000.json"
        ),
        (
            "azureml/"
            "other_features/1/"
            "_delta_log/00000000000000000000.json"
        ),
    ]
    delta_roots = [
        "azureml/other_features/1",
        "azureml/walmart_sales_features/1",
    ]

    mock_list_files = mocker.patch.object(
        export_script,
        "list_storage_files",
        return_value=storage_files,
    )
    mock_find_roots = mocker.patch.object(
        export_script,
        "find_delta_roots",
        return_value=delta_roots,
    )

    result = export_script.find_materialized_delta_path()

    assert result == (
        "abfss://offline-store"
        "@walmartstorage.dfs.core.windows.net/"
        "azureml/walmart_sales_features/1"
    )

    mock_list_files.assert_called_once_with(
        "walmartstorage",
        "offline-store",
    )
    mock_find_roots.assert_called_once_with(storage_files)

    captured = capsys.readouterr()

    assert "Delta tables found in the offline store:" in captured.out
    assert "azureml/other_features/1" in captured.out
    assert "azureml/walmart_sales_features/1" in captured.out


def test_find_materialized_delta_path_matches_feature_name_case_insensitively(
    mocker,
    export_script,
):
    mocker.patch.object(
        export_script,
        "list_storage_files",
        return_value=["unused"],
    )
    mocker.patch.object(
        export_script,
        "find_delta_roots",
        return_value=[
            "azureml/WALMART_SALES_FEATURES/1",
        ],
    )

    result = export_script.find_materialized_delta_path()

    assert result.endswith(
        "/azureml/WALMART_SALES_FEATURES/1"
    )


def test_find_materialized_delta_path_raises_when_no_match_exists(
    mocker,
    export_script,
):
    mocker.patch.object(
        export_script,
        "list_storage_files",
        return_value=["unused"],
    )
    mocker.patch.object(
        export_script,
        "find_delta_roots",
        return_value=[
            "azureml/other_features/1",
            "azureml/walmart_sales_features/2",
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"No materialised Delta table was found for "
            r"walmart_sales_features version 1\."
        ),
    ):
        export_script.find_materialized_delta_path()


def test_find_materialized_delta_path_raises_for_multiple_matches(
    mocker,
    export_script,
):
    matching_roots = [
        "first/walmart_sales_features/1",
        "second/walmart_sales_features/1",
    ]

    mocker.patch.object(
        export_script,
        "list_storage_files",
        return_value=["unused"],
    )
    mocker.patch.object(
        export_script,
        "find_delta_roots",
        return_value=matching_roots,
    )

    with pytest.raises(
        RuntimeError,
        match=r"More than one matching Delta table was found:",
    ) as error:
        export_script.find_materialized_delta_path()

    assert matching_roots[0] in str(error.value)
    assert matching_roots[1] in str(error.value)


# ---------------------------------------------------------------------------
# Main export flow
# ---------------------------------------------------------------------------


def configure_spark_dataframe_mocks(mocker):
    """Create the Spark and DataFrame mocks used by main-flow tests."""
    spark = mocker.MagicMock()

    feature_df = mocker.MagicMock(name="feature_df")
    all_weeks_df = mocker.MagicMock(name="all_weeks_df")
    export_df = mocker.MagicMock(name="export_df")

    spark.read.format.return_value.load.return_value = feature_df

    feature_df.withColumn.return_value = all_weeks_df

    # withColumn -> filter -> drop -> withColumnRenamed -> orderBy
    all_weeks_df.filter.return_value = all_weeks_df
    all_weeks_df.drop.return_value = all_weeks_df
    all_weeks_df.withColumnRenamed.return_value = all_weeks_df
    all_weeks_df.orderBy.return_value = all_weeks_df

    all_weeks_df.limit.return_value.count.return_value = 1
    all_weeks_df.withColumn.return_value = export_df

    pandas_output = mocker.MagicMock(name="pandas_output")
    export_df.toPandas.return_value = pandas_output

    date_range_result = mocker.MagicMock(name="date_range_result")
    all_weeks_df.agg.return_value = date_range_result

    grouped = mocker.MagicMock(name="grouped")
    grouped_aggregation = mocker.MagicMock(
        name="grouped_aggregation"
    )
    ordered_groups = mocker.MagicMock(name="ordered_groups")

    all_weeks_df.groupBy.return_value = grouped
    grouped.agg.return_value = grouped_aggregation
    grouped_aggregation.orderBy.return_value = ordered_groups

    return {
        "spark": spark,
        "feature_df": feature_df,
        "all_weeks_df": all_weeks_df,
        "export_df": export_df,
        "pandas_output": pandas_output,
        "date_range_result": date_range_result,
        "grouped": grouped,
        "grouped_aggregation": grouped_aggregation,
        "ordered_groups": ordered_groups,
    }


def configure_spark_function_mocks(mocker, export_script):
    """Mock Spark column expressions without starting a Spark runtime."""
    date_column = mocker.MagicMock(name="date_column")
    parsed_date_column = mocker.MagicMock(name="parsed_date_column")
    store_column = mocker.MagicMock(name="store_column")
    formatted_date_column = mocker.MagicMock(
        name="formatted_date_column"
    )

    def column_side_effect(name):
        if name == "Date":
            return date_column
        if name == "_parsed_date":
            return parsed_date_column
        if name == "Store":
            return store_column
        return mocker.MagicMock(name=f"column_{name}")

    mock_col = mocker.patch.object(
        export_script.F,
        "col",
        side_effect=column_side_effect,
    )
    mock_to_date = mocker.patch.object(
        export_script.F,
        "to_date",
        return_value=parsed_date_column,
    )
    mock_date_format = mocker.patch.object(
        export_script.F,
        "date_format",
        return_value=formatted_date_column,
    )

    mocker.patch.object(
        export_script.F,
        "min",
        side_effect=lambda name: mocker.MagicMock(
            name=f"min_{name}"
        ),
    )
    mocker.patch.object(
        export_script.F,
        "max",
        side_effect=lambda name: mocker.MagicMock(
            name=f"max_{name}"
        ),
    )
    mocker.patch.object(
        export_script.F,
        "count",
        return_value=mocker.MagicMock(name="count"),
    )
    mocker.patch.object(
        export_script.F,
        "countDistinct",
        return_value=mocker.MagicMock(name="count_distinct"),
    )

    return {
        "date_column": date_column,
        "parsed_date_column": parsed_date_column,
        "formatted_date_column": formatted_date_column,
        "col": mock_col,
        "to_date": mock_to_date,
        "date_format": mock_date_format,
    }


def test_main_exports_all_materialized_rows(
    mocker,
    monkeypatch,
    tmp_path,
    export_script,
    capsys,
):
    monkeypatch.setenv("AZURE_STORAGE_KEY", "storage-key")
    monkeypatch.setattr(
        export_script,
        "OUTPUT_PATH",
        tmp_path / "output" / "all_weeks.csv",
    )

    mocks = configure_spark_dataframe_mocks(mocker)
    function_mocks = configure_spark_function_mocks(
        mocker,
        export_script,
    )

    mock_create_spark = mocker.patch.object(
        export_script,
        "create_delta_spark_session",
        return_value=mocks["spark"],
    )
    mock_find_path = mocker.patch.object(
        export_script,
        "find_materialized_delta_path",
        return_value=(
            "abfss://offline-store"
            "@walmartstorage.dfs.core.windows.net/"
            "features/walmart_sales_features/1"
        ),
    )

    export_script.main()

    mock_create_spark.assert_called_once_with(
        application_name="export-walmart-all-weeks",
        storage_account="walmartstorage",
        storage_key="storage-key",
    )
    mocks["spark"].sparkContext.setLogLevel.assert_called_once_with(
        "WARN"
    )
    mock_find_path.assert_called_once_with()

    mocks["spark"].read.format.assert_called_once_with("delta")
    mocks["spark"].read.format.return_value.load.assert_called_once_with(
        "abfss://offline-store"
        "@walmartstorage.dfs.core.windows.net/"
        "features/walmart_sales_features/1"
    )
    mocks["feature_df"].printSchema.assert_called_once_with()

    function_mocks["to_date"].assert_called_once_with(
        function_mocks["date_column"],
    )
    function_mocks["parsed_date_column"].isNotNull.assert_called_once_with()

    mocks["all_weeks_df"].drop.assert_called_once_with("Date")
    mocks["all_weeks_df"].withColumnRenamed.assert_called_once_with(
        "_parsed_date",
        "Date",
    )
    mocks["all_weeks_df"].limit.assert_called_once_with(1)

    function_mocks["date_format"].assert_called_once_with(
        function_mocks["date_column"],
        "dd-MM-yyyy",
    )

    mocks["pandas_output"].to_csv.assert_called_once_with(
        tmp_path / "output" / "all_weeks.csv",
        index=False,
    )

    mocks["date_range_result"].show.assert_called_once_with(
        truncate=False,
    )
    mocks["all_weeks_df"].groupBy.assert_called_once_with("Store")
    mocks["grouped_aggregation"].orderBy.assert_called_once_with(
        "Store"
    )
    mocks["ordered_groups"].show.assert_called_once_with(
        n=100,
        truncate=False,
    )

    mocks["spark"].stop.assert_called_once_with()

    captured = capsys.readouterr()

    assert "Reading materialised features from:" in captured.out
    assert "Materialised feature schema:" in captured.out
    assert "Writing all materialised weeks" in captured.out
    assert "CSV written to:" in captured.out
    assert "Materialised date range:" in captured.out
    assert "Number of exported rows per store:" in captured.out


def test_main_raises_when_no_valid_feature_rows_exist(
    mocker,
    monkeypatch,
    export_script,
):
    monkeypatch.setenv("AZURE_STORAGE_KEY", "storage-key")

    mocks = configure_spark_dataframe_mocks(mocker)
    configure_spark_function_mocks(mocker, export_script)

    mocks["all_weeks_df"].limit.return_value.count.return_value = 0

    mocker.patch.object(
        export_script,
        "create_delta_spark_session",
        return_value=mocks["spark"],
    )
    mocker.patch.object(
        export_script,
        "find_materialized_delta_path",
        return_value="abfss://container@account/path",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"No materialised feature rows with valid dates were found\."
        ),
    ):
        export_script.main()

    mocks["spark"].stop.assert_called_once_with()
    mocks["pandas_output"].to_csv.assert_not_called()


def test_main_stops_spark_when_export_fails(
    mocker,
    monkeypatch,
    export_script,
):
    monkeypatch.setenv("AZURE_STORAGE_KEY", "storage-key")

    spark = mocker.MagicMock()
    spark.read.format.return_value.load.side_effect = RuntimeError(
        "Delta read failed"
    )

    mocker.patch.object(
        export_script,
        "create_delta_spark_session",
        return_value=spark,
    )
    mocker.patch.object(
        export_script,
        "find_materialized_delta_path",
        return_value="abfss://container@account/path",
    )

    with pytest.raises(
        RuntimeError,
        match=r"Delta read failed",
    ):
        export_script.main()

    spark.stop.assert_called_once_with()


def test_main_requires_storage_key(
    monkeypatch,
    export_script,
):
    monkeypatch.delenv("AZURE_STORAGE_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match=r"AZURE_STORAGE_KEY is not set\.",
    ):
        export_script.main()