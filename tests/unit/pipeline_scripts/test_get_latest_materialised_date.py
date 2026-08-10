"""Tests for retrieving the latest materialized feature-store date."""

from datetime import date, datetime

import pytest

from src.pipeline_scripts import get_latest_materialised_date as latest_date


@pytest.fixture
def required_environment(monkeypatch):
    """Configure the environment required by main()."""
    values = {
        "FEATURE_SET_NAME": "walmart_sales_features",
        "FEATURE_SET_VERSION": "1",
        "OFFLINE_STORAGE_ACCOUNT": "walmartstorage",
        "OFFLINE_STORAGE_CONTAINER": "offline-store",
        "AZURE_STORAGE_KEY": "storage-key",
        "GITHUB_ENV": "/tmp/github-env",
    }

    for name, value in values.items():
        monkeypatch.setenv(name, value)

    return values


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------


def test_required_environment_variable_returns_value(monkeypatch):
    monkeypatch.setenv("TEST_VARIABLE", "test-value")

    result = latest_date.required_environment_variable(
        "TEST_VARIABLE",
    )

    assert result == "test-value"


@pytest.mark.parametrize("value", [None, ""])
def test_required_environment_variable_raises_when_missing(
    monkeypatch,
    value,
):
    if value is None:
        monkeypatch.delenv("TEST_VARIABLE", raising=False)
    else:
        monkeypatch.setenv("TEST_VARIABLE", value)

    with pytest.raises(
        RuntimeError,
        match=r"TEST_VARIABLE is not set\.",
    ):
        latest_date.required_environment_variable("TEST_VARIABLE")


# ---------------------------------------------------------------------------
# Delta path discovery
# ---------------------------------------------------------------------------


def test_find_materialized_delta_path_returns_matching_abfss_uri(
    mocker,
):
    storage_files = [
        (
            "features/walmart_sales_features/1/"
            "_delta_log/00000000000000000000.json"
        ),
        (
            "features/other_features/1/"
            "_delta_log/00000000000000000000.json"
        ),
    ]
    delta_roots = [
        "features/other_features/1",
        "features/walmart_sales_features/1",
    ]

    mock_list_files = mocker.patch.object(
        latest_date,
        "list_storage_files",
        return_value=storage_files,
    )
    mock_find_roots = mocker.patch.object(
        latest_date,
        "find_delta_roots",
        return_value=delta_roots,
    )

    result = latest_date.find_materialized_delta_path(
        storage_account="walmartstorage",
        storage_container="offline-store",
        feature_set_name="walmart_sales_features",
        feature_set_version="1",
    )

    assert result == (
        "abfss://offline-store"
        "@walmartstorage.dfs.core.windows.net/"
        "features/walmart_sales_features/1"
    )

    mock_list_files.assert_called_once_with(
        "walmartstorage",
        "offline-store",
    )
    mock_find_roots.assert_called_once_with(storage_files)


def test_find_materialized_delta_path_matches_name_case_insensitively(
    mocker,
):
    mocker.patch.object(
        latest_date,
        "list_storage_files",
        return_value=["unused"],
    )
    mocker.patch.object(
        latest_date,
        "find_delta_roots",
        return_value=[
            "features/WALMART_SALES_FEATURES/1",
        ],
    )

    result = latest_date.find_materialized_delta_path(
        storage_account="walmartstorage",
        storage_container="offline-store",
        feature_set_name="walmart_sales_features",
        feature_set_version="1",
    )

    assert result.endswith(
        "/features/WALMART_SALES_FEATURES/1"
    )


def test_find_materialized_delta_path_matches_exact_version(
    mocker,
):
    mocker.patch.object(
        latest_date,
        "list_storage_files",
        return_value=["unused"],
    )
    mocker.patch.object(
        latest_date,
        "find_delta_roots",
        return_value=[
            "features/walmart_sales_features/10",
            "features/walmart_sales_features/1",
        ],
    )

    result = latest_date.find_materialized_delta_path(
        storage_account="walmartstorage",
        storage_container="offline-store",
        feature_set_name="walmart_sales_features",
        feature_set_version="1",
    )

    assert result.endswith(
        "/features/walmart_sales_features/1"
    )


def test_find_materialized_delta_path_raises_when_no_match_exists(
    mocker,
):
    mocker.patch.object(
        latest_date,
        "list_storage_files",
        return_value=["unused"],
    )
    mocker.patch.object(
        latest_date,
        "find_delta_roots",
        return_value=[
            "features/other_features/1",
            "features/walmart_sales_features/2",
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"No materialized Delta table was found for "
            r"walmart_sales_features version 1\."
        ),
    ):
        latest_date.find_materialized_delta_path(
            storage_account="walmartstorage",
            storage_container="offline-store",
            feature_set_name="walmart_sales_features",
            feature_set_version="1",
        )


def test_find_materialized_delta_path_raises_for_multiple_matches(
    mocker,
):
    matching_roots = [
        "first/walmart_sales_features/1",
        "second/walmart_sales_features/1",
    ]

    mocker.patch.object(
        latest_date,
        "list_storage_files",
        return_value=["unused"],
    )
    mocker.patch.object(
        latest_date,
        "find_delta_roots",
        return_value=matching_roots,
    )

    with pytest.raises(
        RuntimeError,
        match=r"More than one matching Delta table was found:",
    ) as error:
        latest_date.find_materialized_delta_path(
            storage_account="walmartstorage",
            storage_container="offline-store",
            feature_set_name="walmart_sales_features",
            feature_set_version="1",
        )

    assert matching_roots[0] in str(error.value)
    assert matching_roots[1] in str(error.value)


# ---------------------------------------------------------------------------
# Date formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2024, 1, 5), "05-01-2024"),
        (datetime(2024, 12, 31, 14, 30), "31-12-2024"),
    ],
)
def test_format_latest_date(value, expected):
    assert latest_date.format_latest_date(value) == expected


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def configure_spark_mocks(
    mocker,
    *,
    columns=None,
    latest_row=None,
):
    """Return mocked Spark objects for main-function tests."""
    spark = mocker.MagicMock(name="spark")
    feature_df = mocker.MagicMock(name="feature_df")
    selected_df = mocker.MagicMock(name="selected_df")

    feature_df.columns = (
        [latest_date.DATE_COLUMN, "Store"]
        if columns is None
        else columns
    )
    feature_df.select.return_value = selected_df
    selected_df.first.return_value = latest_row

    spark.read.format.return_value.load.return_value = feature_df

    return spark, feature_df, selected_df


def configure_spark_function_mocks(mocker):
    """Mock the Spark expression used to calculate the maximum date."""
    date_column = mocker.MagicMock(name="date_column")
    timestamp_expression = mocker.MagicMock(
        name="timestamp_expression"
    )
    maximum_expression = mocker.MagicMock(
        name="maximum_expression"
    )
    aliased_expression = mocker.MagicMock(
        name="aliased_expression"
    )

    mock_col = mocker.patch.object(
        latest_date.F,
        "col",
        return_value=date_column,
    )
    mock_to_timestamp = mocker.patch.object(
        latest_date.F,
        "to_timestamp",
        return_value=timestamp_expression,
    )
    mock_max = mocker.patch.object(
        latest_date.F,
        "max",
        return_value=maximum_expression,
    )
    maximum_expression.alias.return_value = aliased_expression

    return {
        "date_column": date_column,
        "timestamp_expression": timestamp_expression,
        "maximum_expression": maximum_expression,
        "aliased_expression": aliased_expression,
        "col": mock_col,
        "to_timestamp": mock_to_timestamp,
        "max": mock_max,
    }


def test_main_prints_and_writes_latest_processed_date(
    mocker,
    required_environment,
    capsys,
):
    materialized_path = (
        "abfss://offline-store"
        "@walmartstorage.dfs.core.windows.net/"
        "features/walmart_sales_features/1"
    )
    expected_date = datetime(2024, 1, 26)

    mock_find_path = mocker.patch.object(
        latest_date,
        "find_materialized_delta_path",
        return_value=materialized_path,
    )

    spark, feature_df, selected_df = configure_spark_mocks(
        mocker,
        latest_row={"latest_date": expected_date},
    )
    expressions = configure_spark_function_mocks(mocker)

    mock_create_spark = mocker.patch.object(
        latest_date,
        "create_delta_spark_session",
        return_value=spark,
    )
    mock_write_environment = mocker.patch.object(
        latest_date,
        "write_environment_variables",
    )

    result = latest_date.main()

    assert result is None

    mock_find_path.assert_called_once_with(
        storage_account="walmartstorage",
        storage_container="offline-store",
        feature_set_name="walmart_sales_features",
        feature_set_version="1",
    )
    mock_create_spark.assert_called_once_with(
        application_name="get-latest-materialized-date",
        storage_account="walmartstorage",
        storage_key="storage-key",
    )
    spark.sparkContext.setLogLevel.assert_called_once_with("ERROR")

    spark.read.format.assert_called_once_with("delta")
    spark.read.format.return_value.load.assert_called_once_with(
        materialized_path,
    )

    expressions["col"].assert_called_once_with("Date")
    expressions["to_timestamp"].assert_called_once_with(
        expressions["date_column"],
    )
    expressions["max"].assert_called_once_with(
        expressions["timestamp_expression"],
    )
    expressions["maximum_expression"].alias.assert_called_once_with(
        "latest_date",
    )

    feature_df.select.assert_called_once_with(
        expressions["aliased_expression"],
    )
    selected_df.first.assert_called_once_with()

    mock_write_environment.assert_called_once_with({
        "LATEST_PROCESSED_DATE": "26-01-2024",
    })
    spark.stop.assert_called_once_with()

    captured = capsys.readouterr()

    assert captured.out == "26-01-2024\n"
    assert captured.err == ""


def test_main_raises_when_date_column_is_missing(
    mocker,
    required_environment,
):
    spark, _, _ = configure_spark_mocks(
        mocker,
        columns=["Store", "Temperature"],
    )

    mocker.patch.object(
        latest_date,
        "find_materialized_delta_path",
        return_value="abfss://container@account/path",
    )
    mocker.patch.object(
        latest_date,
        "create_delta_spark_session",
        return_value=spark,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"Column 'Date' was not found in the "
            r"materialized Delta table\."
        ),
    ):
        latest_date.main()

    spark.stop.assert_called_once_with()


@pytest.mark.parametrize(
    "latest_row",
    [
        None,
        {"latest_date": None},
    ],
)
def test_main_raises_when_no_valid_dates_exist(
    mocker,
    required_environment,
    latest_row,
):
    spark, _, _ = configure_spark_mocks(
        mocker,
        latest_row=latest_row,
    )
    configure_spark_function_mocks(mocker)

    mocker.patch.object(
        latest_date,
        "find_materialized_delta_path",
        return_value="abfss://container@account/path",
    )
    mocker.patch.object(
        latest_date,
        "create_delta_spark_session",
        return_value=spark,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"No valid dates were found in the "
            r"materialized Delta table\."
        ),
    ):
        latest_date.main()

    spark.stop.assert_called_once_with()


def test_main_raises_when_github_env_is_missing(
    mocker,
    monkeypatch,
    required_environment,
    capsys,
):
    monkeypatch.delenv("GITHUB_ENV", raising=False)

    spark, _, _ = configure_spark_mocks(
        mocker,
        latest_row={
            "latest_date": datetime(2024, 1, 26),
        },
    )
    configure_spark_function_mocks(mocker)

    mocker.patch.object(
        latest_date,
        "find_materialized_delta_path",
        return_value="abfss://container@account/path",
    )
    mocker.patch.object(
        latest_date,
        "create_delta_spark_session",
        return_value=spark,
    )
    mock_write_environment = mocker.patch.object(
        latest_date,
        "write_environment_variables",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"GITHUB_ENV is not set\. "
            r"This script must run inside GitHub Actions\."
        ),
    ):
        latest_date.main()

    mock_write_environment.assert_not_called()
    spark.stop.assert_called_once_with()

    captured = capsys.readouterr()
    assert captured.out == "26-01-2024\n"


def test_main_stops_spark_when_delta_read_fails(
    mocker,
    required_environment,
):
    spark = mocker.MagicMock()
    spark.read.format.return_value.load.side_effect = RuntimeError(
        "Delta read failed"
    )

    mocker.patch.object(
        latest_date,
        "find_materialized_delta_path",
        return_value="abfss://container@account/path",
    )
    mocker.patch.object(
        latest_date,
        "create_delta_spark_session",
        return_value=spark,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Delta read failed",
    ):
        latest_date.main()

    spark.stop.assert_called_once_with()


def test_main_fails_before_creating_spark_when_environment_is_missing(
    mocker,
    monkeypatch,
    required_environment,
):
    monkeypatch.delenv("FEATURE_SET_NAME")

    mock_create_spark = mocker.patch.object(
        latest_date,
        "create_delta_spark_session",
    )

    with pytest.raises(
        RuntimeError,
        match=r"FEATURE_SET_NAME is not set\.",
    ):
        latest_date.main()

    mock_create_spark.assert_not_called()